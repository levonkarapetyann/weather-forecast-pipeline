#!/usr/bin/env python3
# isort: skip_file
import os
import sys

# Ensure src/ and project root are in sys.path before any local imports
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from typing import Any, Dict, List
from data_pipeline import select_stations_for_run
from project_paths import load_settings, resolve_path
import requests
from datetime import datetime, timedelta
import time
import json
import csv
import argparse
import os
import sys

# Импорт gspread клиента (sheets_writer)
try:
    from sheets_writer import SheetsWriter
    _SHEETS_WRITER_AVAILABLE = True
except ImportError:
    _SHEETS_WRITER_AVAILABLE = False

# Импорт CatBoost для независимого бинарного классификатора осадков
try:
    from catboost import CatBoostClassifier
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False

try:
    import pandas as pd
except ImportError:
    pd = None


def fetch_live_sensors_with_fallback(generated_id: int) -> Any:
    """Запрашивает свежие показания датчиков с API ClimateNet или читает локальный JSON."""
    settings = load_settings()
    now = datetime.now()
    start_time = (now - timedelta(days=2)).strftime("%Y-%m-%d")
    end_time = now.strftime("%Y-%m-%d")

    url = f"{settings['paths'].get('climatenet_url', 'https://emvnh9buoh.execute-api.us-east-1.amazonaws.com')}/getData"
    params = {"device_id": generated_id, "start_time": start_time, "end_time": end_time}

    try:
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code == 200:
            res_json = resp.json()
            keys = res_json.get("keys", [])
            data = res_json.get("data", [])
            if keys and data and pd is not None:
                df = pd.DataFrame(data, columns=keys)
                df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
                return df
    except Exception:
        pass

    raw_file = resolve_path("data", "raw", "stations", f"station_{generated_id}.json")
    if os.path.exists(raw_file):
        try:
            with open(raw_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            keys = raw_data.get("keys", [])
            data = raw_data.get("data", [])
            if keys and data and pd is not None:
                df = pd.DataFrame(data, columns=keys)
                df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
                return df
        except Exception:
            pass

    return None






"""
forecast_saver.py
===================
Скрипт, который каждый час (или другой заданный интервал) собирает финальные
гиперлокальные прогнозы со всех активных станций и дописывает их в
итоговый CSV-файл с указанием времени запуска (run_timestamp).

Финальный прогноз — это тот же результат, который отображается в Streamlit:
  TFT-модель → PID-коррекция.

Скрипт работает в двух режимах (автоматически выбирает):
  1. API режим  : опрашивает запущенный сервер FastAPI (http://127.0.0.1:8000).
  2. Автономный : если сервер недоступен — загружает модель и считает прогноз
                  напрямую внутри текущего процесса.

Запуск:
    python src/forecast_saver.py                # бесконечный цикл, раз в час
    python src/forecast_saver.py --once         # один запуск (тест)
    python src/forecast_saver.py --interval 30  # раз в 30 минут
"""


# ── Пути ──────────────────────────────────────────────────────────────────────
# Директория скрипта (src/) и корень проекта

# Добавляем src/ в sys.path чтобы импортировать внутренние модули

# ВАЖНО: переключаем рабочую директорию в корень проекта, чтобы все
# относительные пути внутри app.py работали корректно
os.chdir(PROJECT_ROOT)


# ── Схема колонок итогового CSV ───────────────────────────────────────────────
FIELDNAMES = [
    "run_timestamp",
    "station_id",
    "station_name",
    "forecast_datetime",
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_direction_degrees",
    "rain",
    "rain_probability",
    "will_rain",
    "uv",
    "lux",
    "pm1",
    "pm2_5",
    "pm10",
]

# ── Кэш модели внутри процесса (чтобы не перезагружать её на каждой станции) ─
_local_get_forecast = None


def _ensure_local_model() -> bool:
    """Импортирует get_forecast из app.py один раз и кэширует."""
    global _local_get_forecast
    if _local_get_forecast is not None:
        return True
    try:
        print("🔄 Загрузка TFT-модели в локальный процесс...")
        from app import get_forecast as _gf  # noqa: PLC0415
        _local_get_forecast = _gf
        print("✅ Модель загружена.")
        return True
    except Exception as exc:
        print(f"❌ Не удалось загрузить модель: {exc}")
        import traceback
        traceback.print_exc()
        return False


def _pydantic_to_dict(obj) -> dict:
    """Конвертирует Pydantic-объект (v1 или v2) в plain dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):          # pydantic v2
        return obj.model_dump()
    if hasattr(obj, "dict"):                # pydantic v1
        return obj.dict()
    raise TypeError(f"Не могу конвертировать объект типа {type(obj)} в dict")


def get_station_forecast(station_id: int, api_url: str) -> dict | None:
    """
    Возвращает финальный прогноз для станции в виде dict.

    Порядок попыток:
      1. GET {api_url}/forecast/{station_id}   — если FastAPI запущен
      2. Прямой вызов get_forecast() внутри процесса — fallback
    """
    # ── 1. API ─────────────────────────────────────────────────────────────────
    url = f"{api_url.rstrip('/')}/forecast/{station_id}"
    try:
        print(f"📡 Запрос через API: {url}")
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            print(f"   ✅ API ответил ({len(data.get('forecast', []))} шагов)")
            return data
        else:
            print(f"   ⚠️  API вернул {resp.status_code}: {resp.text[:120]}")
    except requests.exceptions.ConnectionError:
        print("   ℹ️  API недоступен — переход к локальному инференсу.")
    except Exception as exc:
        print(f"   ⚠️  Ошибка запроса API: {exc}")

    # ── 2. Локальный инференс ──────────────────────────────────────────────────
    print(f"🔮 Локальный инференс для станции {station_id}...")
    if not _ensure_local_model():
        return None

    try:
        response_obj = _local_get_forecast(station_id)
        data = _pydantic_to_dict(response_obj)

        # Конвертируем вложенные ForecastItem-объекты, если они ещё не dict
        if "forecast" in data:
            data["forecast"] = [
                _pydantic_to_dict(item) if not isinstance(item, dict) else item
                for item in data["forecast"]
            ]

        n = len(data.get("forecast", []))
        print(f"   ✅ Локальный инференс завершён ({n} шагов)")
        return data

    except Exception as exc:
        print(f"   ❌ Ошибка локального инференса: {exc}")
        import traceback
        traceback.print_exc()
        return None


def save_forecast_to_csv(forecast_data: dict, csv_path: str, run_ts: str) -> int:
    """
    Дописывает прогноз в CSV.  Возвращает количество записанных строк.
    """
    if not forecast_data or "forecast" not in forecast_data:
        return 0

    station_id = forecast_data.get("station_id")
    station_name = forecast_data.get("station_name")

    rows = []
    for item in forecast_data["forecast"]:
        # item может быть plain dict (из API JSON или после model_dump)
        rows.append({
            "run_timestamp":        run_ts,
            "station_id":           station_id,
            "station_name":         station_name,
            "forecast_datetime":    item.get("timestamp"),
            "temperature":          item.get("temperature"),
            "humidity":             item.get("humidity"),
            "pressure":             item.get("pressure"),
            "wind_speed":           item.get("wind_speed"),
            "wind_direction_degrees": item.get("wind_direction_degrees"),
            "rain":                 item.get("rain"),
            "rain_probability":      item.get("rain_probability"),
            "will_rain":            item.get("will_rain"),
            "uv":                   item.get("uv"),
            "lux":                  item.get("lux"),
            "pm1":                  item.get("pm1"),
            "pm2_5":                item.get("pm2_5"),
            "pm10":                 item.get("pm10"),
        })

    if not rows:
        return 0

    # Создаём директорию при необходимости
    dir_path = os.path.dirname(csv_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    file_exists = os.path.exists(csv_path)
    size_before = os.path.getsize(csv_path) if file_exists else 0
    print(
        f"  📂 CSV до записи: {'существует' if file_exists else 'не существует'}, размер: {size_before} байт, путь: {csv_path}")
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    size_after = os.path.getsize(csv_path)
    print(f"  💾 Записано {len(rows)} строк для '{station_name}' → {csv_path} (размер: {size_before} → {size_after} байт)")
    return len(rows)


def run_cycle(api_url: str, csv_path: str, settings: dict, sheets=None) -> None:
    """Один полный цикл: генерация и сохранение прогнозов по всем активным станциям."""
    now = datetime.now()
    run_ts = now.strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'#' * 70}")
    print(f"# ЦИКЛ ЗАПИСИ ПРОГНОЗОВ — {run_ts}")
    print(f"{'#' * 70}")

    stations_config_path = resolve_path(settings["paths"]["stations_config"])
    if not os.path.exists(stations_config_path):
        print(f"❌ Файл конфигурации станций не найден: {stations_config_path}")
        return

    with open(stations_config_path, "r", encoding="utf-8") as f:
        stations_data = json.load(f)

    active_stations = select_stations_for_run(stations_data["stations"], settings)
    print(f"Активных станций: {len(active_stations)}")

    total_rows = 0
    failed = 0

    for station in active_stations:
        sid = station["id"]           # порядковый id (для запроса прогноза)
        generated_id = station["generated_id"]  # реальный номер устройства/станции
        sname = station["name"]
        print(f"\n▶ Станция: {sname} (ID устройства: {generated_id})")

        forecast = get_station_forecast(sid, api_url)
        if forecast:
            # Заменяем порядковый id на реальный номер станции (generated_id)
            forecast["station_id"] = generated_id

            total_rows += save_forecast_to_csv(forecast, csv_path, run_ts)
            if sheets is not None:
                try:
                    sheets.append_station_forecasts(forecast, run_ts)
                    time.sleep(1.2)  # Задержка 1.2с для соблюдения квоты Google API (макс 60 запросов/мин)
                except Exception as exc:
                    print(f"   ⚠️  Sheets.append_station_forecasts не удалось выполнить: {exc}")


        else:
            print(f"   ❌ Прогноз не получен, станция пропущена.")
            failed += 1

    elapsed = (datetime.now() - now).total_seconds()
    print(f"\n{'#' * 70}")
    print(f"# ЦИКЛ ЗАВЕРШЁН — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Итого: записано {total_rows} строк, ошибок: {failed}, время: {elapsed:.1f}с")
    print(f"{'#' * 70}\n")


def run_actuals_cycle(sheets, settings: dict, hours_back: int = 2) -> None:
    """
    Получает актуальные показания датчиков за последние hours_back часов
    и отправляет весь пакет измерений в лист Actuals в Google Sheets.
    """
    if sheets is None:
        return

    now = datetime.now()
    cutoff_time = now - timedelta(hours=hours_back)
    print(f"\n{'=' * 70}")
    print(f"# ЦИКЛ РЕАЛЬНЫХ ДАННЫХ DATCHIKOV (за последние {hours_back}ч) — {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")

    stations_config_path = resolve_path(settings["paths"]["stations_config"])
    if not os.path.exists(stations_config_path):
        print(f"❌ Файл конфигурации станций не найден: {stations_config_path}")
        return

    with open(stations_config_path, "r", encoding="utf-8") as f:
        stations_data = json.load(f)

    active_stations = select_stations_for_run(stations_data["stations"], settings)
    batch = []
    station_rows_count = {}

    for station in active_stations:
        generated_id = station["generated_id"]
        sname = station["name"]

        try:
            df_live = fetch_live_sensors_with_fallback(generated_id)
            if df_live is None or df_live.empty:
                continue

            df_live["timestamp"] = pd.to_datetime(df_live["timestamp"], errors="coerce")
            df_recent = df_live[df_live["timestamp"] >= cutoff_time].sort_values("timestamp")

            if df_recent.empty:
                # Если с момента cutoff_time ответа не было, берём 8 последних имеющихся записей (~2 часа)
                df_recent = df_live.sort_values("timestamp").tail(8)

            st_count = 0
            for _, row in df_recent.iterrows():
                ts_val = row.get("timestamp")
                if isinstance(ts_val, (pd.Timestamp, datetime)):
                    ts_str = ts_val.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    ts_str = str(ts_val)

                temp = row.get("temperature")
                rain = row.get("rain", 0.0)

                batch.append({
                    "timestamp": ts_str,
                    "station_id": generated_id,
                    "station_name": sname,
                    "temperature_actual": float(temp) if pd.notna(temp) else None,
                    "rain_actual": float(rain) if pd.notna(rain) else 0.0,
                })
                st_count += 1

            station_rows_count[sname] = st_count
            print(f"  ✅ {sname}: {st_count} измерений за последние {hours_back}ч")

        except Exception as exc:
            print(f"  ⚠️  Ошибка получения данных датчиков для {sname}: {exc}")

    if batch:
        try:
            sheets.append_actuals_batch(batch)
        except Exception as exc:
            print(f"  ⚠️  Sheets.append_actuals_batch не удалось выполнить: {exc}")

    elapsed = (datetime.now() - now).total_seconds()
    print(f"# Цикл Actuals завершён, всего добавлено {len(batch)} строк ({elapsed:.1f}с)")
    print(f"{'=' * 70}\n")


def _seconds_until_next_run(interval_minutes: int) -> float:
    """
    Возвращает секунды до следующего запуска, выровненного по сетке.

    Например, при interval_minutes=60 следующий запуск — начало следующего часа.
    При interval_minutes=30 — следующий кратный 30-минутный интервал и т.д.
    """
    now = datetime.now()
    # Сколько секунд от начала текущих суток прошло
    seconds_today = now.hour * 3600 + now.minute * 60 + now.second + now.microsecond / 1e6
    interval_sec = interval_minutes * 60
    # Ближайший будущий кратный момент
    next_boundary = (int(seconds_today / interval_sec) + 1) * interval_sec
    wait = next_boundary - seconds_today
    return wait if wait > 0 else interval_sec


def main():
    parser = argparse.ArgumentParser(
        description="Планировщик записи финальных прогнозов ClimateNet (TFT + PID + Google Sheets)."
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
        help="Базовый URL FastAPI-сервера (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--csv-path",
        default="",
        help="Путь к итоговому CSV (default: weather_data/model_forecasts.csv)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Выполнить один цикл и выйти (режим тестирования)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        metavar="MINUTES",
        help="Интервал между циклами в минутах (default: 60)",
    )
    parser.add_argument(
        "--no-sheets",
        action="store_true",
        help="Отключить экспорт в Google Sheets",
    )
    args = parser.parse_args()

    # Загружаем настройки проекта
    settings = load_settings()

    # Путь к CSV
    csv_path = args.csv_path if args.csv_path else resolve_path("weather_data", "model_forecasts.csv")

    # 1. Инициализируем Google Sheets writer
    sheets = None
    gs_cfg = settings.get("google_sheets", {})
    if gs_cfg.get("enabled", False) and not args.no_sheets:
        if not _SHEETS_WRITER_AVAILABLE:
            print("⚠️  sheets_writer.py не найден — Google Sheets отключены.")
        else:
            try:
                creds_file = resolve_path(gs_cfg["credentials_file"])
                sheets = SheetsWriter(
                    credentials_file=creds_file,
                    forecasts_spreadsheet_id=gs_cfg["forecasts_spreadsheet_id"],
                    actuals_spreadsheet_id=gs_cfg["actuals_spreadsheet_id"],
                    forecasts_sheet_name=gs_cfg.get("forecasts_sheet", "Forecasts"),
                    actuals_sheet_name=gs_cfg.get("actuals_sheet", "Actuals"),
                )
            except Exception as exc:
                print(f"[Google Sheets Warning] Не удалось подключиться: {exc}")
                sheets = None


    actuals_interval_hours = gs_cfg.get("actuals_interval_hours", 2)
    actuals_counter = actuals_interval_hours  # Первый раз запускаем сразу

    print("=" * 70)
    print("  ClimateNet — Планировщик записи прогнозов и Google Sheets")
    print("=" * 70)
    print(f"  Корень проекта  : {PROJECT_ROOT}")
    print(f"  API-сервер      : {args.api_url}")
    print(f"  CSV-файл        : {csv_path}")
    print(f"  Google Sheets   : {'включены' if sheets else 'отключены'}")
    print(f"  Actuals каждые  : {actuals_interval_hours} ч")
    print(f"  Режим           : {'однократный (--once)' if args.once else 'бесконечный цикл'}")
    print("=" * 70)

    # Первый запуск Forecasts
    run_cycle(args.api_url, csv_path, settings, sheets=sheets)

    # Первый запуск Actuals
    if sheets is not None:
        run_actuals_cycle(sheets, settings)
        actuals_counter = 0

    if args.once:
        print("✅ Однократный тестовый запуск завершён.")
        return

    # Бесконечный цикл
    while True:
        wait_sec = _seconds_until_next_run(args.interval)
        next_time = datetime.now() + timedelta(seconds=wait_sec)
        print(
            f"⏳ Следующий цикл: {next_time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"(через {wait_sec / 60:.1f} мин)"
        )
        try:
            time.sleep(wait_sec)
        except KeyboardInterrupt:
            print("\n\nОстановлено пользователем (Ctrl+C). До свидания!")
            sys.exit(0)

        run_cycle(args.api_url, csv_path, settings, sheets=sheets)

        if sheets is not None:
            actuals_counter += 1
            if actuals_counter >= actuals_interval_hours:
                run_actuals_cycle(sheets, settings)
                actuals_counter = 0


if __name__ == "__main__":
    main()