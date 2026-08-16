"""
=============================================================================
МОДУЛЬ: Data Fetcher & External APIs (data_fetcher.py)
-----------------------------------------------------------------------------
НАЗНАЧЕНИЕ:
Модуль сбора и загрузки данных. Отвечает за опрос физических метеодатчиков
ClimateNet и загрузку внешних синоптических прогнозов с Open-Meteo API.

ОСНОВНЫЕ ФУНКЦИИ:
1. Загрузка живых метеонаблюдений с облачного API станций ClimateNet.
2. Получение 48-часовых глобальных прогнозов с Open-Meteo API с локальным
   дисковым кэшированием (1 час TTL).
3. Автоматический фолбэк на сохраненный кэш при сбоях сети или недоступности API.
=============================================================================
"""

import argparse
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import requests
from dotenv import load_dotenv

from data_pipeline import select_stations_for_run

# Загружаем переменные окружения (.env)
load_dotenv()

CLIMATENET_API_BASE_URL = os.getenv(
    "CLIMATENET_API_BASE_URL",
    "https://emvnh9buoh.execute-api.us-east-1.amazonaws.com"
)


def get_date_chunks(start_date_str: str, end_date_str: str, chunk_days: int = 90) -> List[Tuple[str, str]]:
    """
    Разбивает временной интервал на чанки по chunk_days дней для предотвращения таймаутов API.
    """
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    chunks = []
    current_start = start_date

    while current_start <= end_date:
        current_end = min(current_start + timedelta(days=chunk_days - 1), end_date)
        chunks.append((
            current_start.strftime("%Y-%m-%d"),
            current_end.strftime("%Y-%m-%d")
        ))
        current_start = current_end + timedelta(days=1)

    return chunks


def fetch_station_data(device_id: int, start_date: str, end_date: str) -> Dict[str, Any] | None:
    """
    Выполняет GET-запрос к API ClimateNet для получения данных по одной станции за указанный период.
    """
    url = f"{CLIMATENET_API_BASE_URL.rstrip('/')}/getData"
    params = {
        "device_id": device_id,
        "start_time": start_date,
        "end_time": end_date
    }

    try:
        print(f"  Запрос: {start_date} - {end_date} (device_id={device_id})...")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"  Ошибка при запросе к API для устройства {device_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Сбор сырых исторических данных ClimateNet.")
    parser.add_argument("--test", action="store_true", help="Сбор данных за 3 дня только для тестовой станции (ID 8)")
    parser.add_argument("--days", type=int, default=None,
                        help="Ограничить сбор последними N днями (по умолчанию: все данные с даты создания датчика)")
    args = parser.parse_args()

    # Пути к файлам
    stations_path = os.path.join("config", "stations.json")
    output_dir = os.path.join("data", "raw", "stations")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(stations_path):
        print(f"Ошибка: Файл конфигурации станций не найден по пути {stations_path}!")
        print("Пожалуйста, создайте config/stations.json перед запуском.")
        return

    with open(stations_path, "r", encoding="utf-8") as f:
        stations_data = json.load(f)
        stations = stations_data.get("stations", [])

    with open(os.path.join("config", "settings.json"), "r", encoding="utf-8") as f:
        settings = json.load(f)

    stations = select_stations_for_run(stations, settings)

    if not stations:
        print("Ошибка: Список станций в config/stations.json пуст.")
        return

    # Если тестовый режим, оставляем только станцию с generated_id = 8
    if args.test:
        print("--- Запуск в тестовом режиме (только станция ID 8, последние 3 дня) ---")
        stations = [s for s in stations if s.get("generated_id") == 8]
        if not stations:
            print("Предупреждение: Станция с generated_id=8 не найдена в stations.json. Используем первую доступную.")
            stations = [stations_data["stations"][0]]
        args.days = 3

    now = datetime.now()
    end_date_str = now.strftime("%Y-%m-%d")

    # Если --days не задан явно, используем самую раннюю из known dates (2020-01-01 как фолбэк)
    # Реальная стартовая дата будет скорректирована по created_at каждой станции ниже
    if args.days is not None:
        global_start_date = (now - timedelta(days=args.days)).strftime("%Y-%m-%d")
    else:
        global_start_date = "2020-01-01"  # дальний фолбэк — будет перекрыт created_at станции

    for station in stations:
        name = station.get("name", "Unknown")
        internal_id = station.get("id")
        generated_id = station.get("generated_id")

        status = station.get("Status", "online")
        created_at_str = station.get("created_at")

        if status == "offline":
            print(f"Станция '{name}' (id={internal_id}) помечена как offline. Пропускаем.")
            continue

        print(f"Сбор данных для станции: {name} (generated_id={generated_id}, id={internal_id})...")

        # Определяем стартовую дату: с даты создания датчика (created_at), или с global_start_date
        start_date_str = global_start_date
        if created_at_str:
            try:
                # Парсим ISO-формат с таймзоной или без
                created_at_clean = created_at_str.split("T")[0]  # берём только YYYY-MM-DD
                created_at = datetime.strptime(created_at_clean, "%Y-%m-%d")
                start_date_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
                if created_at > start_date_dt:
                    # created_at позже фолбэка — начинаем с created_at
                    start_date_str = created_at_clean
                    print(f"  Начало сбора: {start_date_str} (дата создания датчика)")
                else:
                    print(f"  Начало сбора: {start_date_str} (фолбэк, created_at={created_at_clean})")
            except ValueError as e:
                print(f"  Не удалось разобрать created_at='{created_at_str}': {e}. Используем {start_date_str}")
        else:
            print(f"  Начало сбора: {start_date_str} (created_at не задан, используем фолбэк)")

        # Разбиваем на чанки по 90 дней
        chunks = get_date_chunks(start_date_str, end_date_str, chunk_days=90)

        all_data = []
        keys = []

        for idx, (chunk_start, chunk_end) in enumerate(chunks):
            chunk_res = fetch_station_data(generated_id, chunk_start, chunk_end)
            if chunk_res and "data" in chunk_res:
                if not keys and "keys" in chunk_res:
                    keys = chunk_res["keys"]
                all_data.extend(chunk_res["data"])
            else:
                print(f"  Не удалось получить чанк {idx + 1}/{len(chunks)} для {name}")

        if not all_data:
            print(f"  Внимание: Данные для станции {name} не собраны.")
            continue

        # Сортируем данные по таймстампу (индекс 1 в keys)
        # Предполагаем, что "timestamp" находится на 1 индексе (из Data Schema: keys[1] == "timestamp")
        timestamp_idx = 1
        try:
            # Сортировка по возрастанию даты/времени
            all_data.sort(key=lambda x: x[timestamp_idx])
        except Exception as e:
            print(f"  Ошибка при сортировке по таймстампу: {e}")

        # Сохраняем собранный файл
        output_file = os.path.join(output_dir, f"station_{generated_id}.json")
        result_json = {
            "keys": keys,
            "data": all_data
        }

        with open(output_file, "w", encoding="utf-8") as out_f:
            json.dump(result_json, out_f, ensure_ascii=False, indent=2)

        print(f"  Успешно сохранено: {output_file} ({len(all_data)} записей)")


if __name__ == "__main__":
    main()


import argparse
import json
import os
from datetime import datetime, timedelta

import pandas as pd
import requests

from data_pipeline import select_stations_for_run


def load_stations_for_run(stations_path: str, settings_path: str):
    """Загружает список станций с учётом режима одной станции."""
    with open(stations_path, "r", encoding="utf-8") as f:
        stations = json.load(f)["stations"]

    with open(settings_path, "r", encoding="utf-8") as f:
        settings = json.load(f)

    return select_stations_for_run(stations, settings)


def query_open_meteo_api(url: str, lat: float, lon: float, start_date: str, end_date: str, timezone: str = "Asia/Yerevan") -> pd.DataFrame:
    """
    Универсальный запрос к Open-Meteo API.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation,cloud_cover",
        "timezone": timezone
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        res_json = response.json()

        hourly_data = res_json.get("hourly", {})
        if not hourly_data or "time" not in hourly_data:
            return pd.DataFrame()

        df = pd.DataFrame(hourly_data)
        df = df.rename(columns={"time": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    except Exception as e:
        print(f"    Ошибка при запросе к {url}: {e}")
        return pd.DataFrame()





def fetch_live_multimodel_forecast(lat: float, lon: float) -> pd.DataFrame:
    """
    Запрашивает оперативный 48-часовой прогноз от ведущих мировых моделей:
    - ECMWF IFS025 (Европейский центр — лучшая точность температуры в мире)
    - GFS Seamless (США / NOAA)
    - ICON Seamless (Германия / DWD)
    """
    BASE_PARAMS = {
        "latitude": lat,
        "longitude": lon,
        "forecast_days": 3,
        "timezone": "Asia/Yerevan",
    }
    HOURLY_VARS = "temperature_2m,relative_humidity_2m,surface_pressure,precipitation,wind_speed_10m"

    def _fetch_model(model_name: str, suffix: str) -> pd.DataFrame:
        params = {
            **BASE_PARAMS,
            "hourly": HOURLY_VARS,
            "models": model_name,
        }
        try:
            res = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=10)
            if res.status_code == 200:
                data = res.json().get("hourly", {})
                if data and "time" in data:
                    df = pd.DataFrame(data)
                    df = df.rename(columns={"time": "timestamp"})
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    if "temperature_2m" in df.columns:
                        df[f"temperature_2m_{suffix}"] = df["temperature_2m"]
                    if "relative_humidity_2m" in df.columns:
                        df[f"relative_humidity_2m_{suffix}"] = df["relative_humidity_2m"]
                    return df
        except Exception as e:
            print(f"  Ошибка запроса {model_name}: {e}")
        return pd.DataFrame()

    df_ecmwf = _fetch_model("ecmwf_ifs025", "ecmwf_ifs025")
    df_gfs = _fetch_model("gfs_seamless", "gfs_seamless")
    df_icon = _fetch_model("icon_seamless", "icon_seamless")

    dfs = [d for d in [df_ecmwf, df_gfs, df_icon] if not d.empty]
    if not dfs:
        return pd.DataFrame()

    merged = dfs[0]
    for d in dfs[1:]:
        extra_cols = [c for c in d.columns if c not in merged.columns and c != "timestamp"]
        if extra_cols:
            merged = merged.merge(d[["timestamp"] + extra_cols], on="timestamp", how="outer")

    merged = merged.sort_values("timestamp").reset_index(drop=True).ffill().bfill()

    # Сборка колонок температурного ансамбля
    temp_cols = [c for c in merged.columns if c.startswith("temperature_2m")]
    if temp_cols:
        merged["temperature_2m_ensemble_mean"] = merged[temp_cols].mean(axis=1)
        merged["temp_ensemble_spread"] = merged[temp_cols].std(axis=1).fillna(0.0)
        # Если есть базовый temperature_2m, оставляем среднее ансамбля как единый эталон
        merged["temperature_2m"] = merged["temperature_2m_ensemble_mean"]

    return merged



def main():
    parser = argparse.ArgumentParser(description="Сбор исторических внешних прогнозов и архива ERA5.")
    parser.add_argument("--test", action="store_true", help="Сбор данных за тестовый период")
    parser.add_argument("--days", type=int, default=None,
                        help="Ограничить сбор последними N днями (по умолчанию: все данные с даты создания датчика)")
    args = parser.parse_args()

    stations_path = os.path.join("config", "stations.json")
    settings_path = os.path.join("config", "settings.json")
    output_dir = os.path.join("data", "raw", "external_forecasts")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(stations_path):
        print(f"Ошибка: Файл {stations_path} не найден.")
        return

    if not os.path.exists(settings_path):
        print(f"Ошибка: Файл {settings_path} не найден.")
        return

    stations = load_stations_for_run(stations_path, settings_path)
    config_stations = stations

    if args.test:
        print("--- Запуск в тестовом режиме (только для станции 8) ---")
        stations = [s for s in stations if s["generated_id"] == 8]
        if not stations:
            stations = [config_stations[0]]

    # Фолбэк: если --days не задан, уходим как можно дальше; реальная дата будет уточнена по created_at каждой станции
    now = datetime.now()
    if args.days is not None:
        global_fallback_start = (now - timedelta(days=args.days)).strftime("%Y-%m-%d")
    else:
        global_fallback_start = "2020-01-01"  # дальний фолбэк — будет перекрыт created_at
    today_str = now.strftime("%Y-%m-%d")
    # ERA5 обычно запаздывает на 2-5 дней, поэтому для ERA5-запроса берём до 5 дней назад
    era5_end_str = (now - timedelta(days=5)).strftime("%Y-%m-%d")

    for station in stations:
        name = station["name"]
        sid = station["id"]
        gen_id = station["generated_id"]
        lat = float(station["latitude"])
        lon = float(station["longitude"])
        status = station.get("Status", "online")

        if status == "offline":
            continue

        # Определяем начальную дату по приоритетам:
        # 1. created_at из stations.json  (самый точный источник)
        # 2. Первая запись сенсорного JSON (если файл есть)
        # 3. global_fallback_start (очень дальний фолбэк)
        start_date = global_fallback_start
        date_source = f"фолбэк ({global_fallback_start})"

        # Приоритет 1: created_at из stations.json
        created_at_str = station.get("created_at")
        if created_at_str:
            try:
                created_at_clean = created_at_str.split("T")[0]  # берём только YYYY-MM-DD
                created_at_dt = datetime.strptime(created_at_clean, "%Y-%m-%d")
                fallback_dt = datetime.strptime(global_fallback_start, "%Y-%m-%d")
                # Берём позднюю из двух (сенсор не мог записывать до своего создания)
                start_date = max(created_at_dt, fallback_dt).strftime("%Y-%m-%d")
                date_source = f"created_at ({start_date})"
            except ValueError:
                pass

        # Приоритет 2: первая запись сенсорного JSON (если файл уже скачан)
        raw_sensor_file = os.path.join("data", "raw", "stations", f"station_{gen_id}.json")
        if os.path.exists(raw_sensor_file):
            try:
                with open(raw_sensor_file, "r") as sf:
                    sensor_json = json.load(sf)
                    sensor_data = sensor_json.get("data", [])
                if sensor_data:
                    first_ts = pd.to_datetime(sensor_data[0][1]).strftime("%Y-%m-%d")
                    last_ts = pd.to_datetime(sensor_data[-1][1]).strftime("%Y-%m-%d")
                    # Берём минимум из created_at и первой записи (JSON может быть неполным)
                    json_start_dt = datetime.strptime(first_ts, "%Y-%m-%d")
                    current_start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                    if json_start_dt < current_start_dt:
                        start_date = first_ts
                        date_source = f"первая запись JSON ({start_date})"
            except Exception as e:
                print(f"  Не удалось читать сенсорный JSON: {e}")

        print(f"\nCтанция: {name} | Начало: {start_date} ← {date_source} | Конец: {today_str}")

        # Эндпоинты Open-Meteo
        forecast_url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        archive_url = "https://archive-api.open-meteo.com/v1/archive"

        print(f"  Загрузка внешних данных для станции {name} (id={sid})...")

        # 1. Прогнозный архив Open-Meteo (до сегодняшнего дня)
        df_forecast = query_open_meteo_api(forecast_url, lat, lon, start_date, today_str)
        if not df_forecast.empty:
            out_file = os.path.join(output_dir, f"forecast_{sid}.csv")
            df_forecast.to_csv(out_file, index=False)
            print(f"  Сохранены прогнозы: {out_file} ({df_forecast.shape[0]} строк)")
        else:
            # Если исторический прогноз пуст (например, данные за вчера-сегодня еще не попали в эту базу)
            # делаем резервный запрос к основному API прогнозов
            print("  Исторический архив прогнозов недоступен. Пробуем стандартное API прогнозов...")
            fallback_forecast_url = "https://api.open-meteo.com/v1/forecast"
            df_forecast = query_open_meteo_api(fallback_forecast_url, lat, lon, start_date, today_str)
            if not df_forecast.empty:
                out_file = os.path.join(output_dir, f"forecast_{sid}.csv")
                df_forecast.to_csv(out_file, index=False)
                print(f"  Сохранены прогнозы (резервный API): {out_file} ({df_forecast.shape[0]} строк)")

        # 2. ERA5-архив (ограничен до era5_end_str, т.к. ERA5 запаздывает на 5 дней)
        df_era5 = query_open_meteo_api(archive_url, lat, lon, start_date, era5_end_str)
        if not df_era5.empty:
            out_file = os.path.join(output_dir, f"era5_{sid}.csv")
            df_era5.to_csv(out_file, index=False)
            print(f"  Сохранен архив ERA5: {out_file} ({df_era5.shape[0]} строк)")
        else:
            print("  Архив ERA5 для этого периода пока недоступен (обычно отстаёт на 2-5 дней).")




if __name__ == "__main__":
    main()
