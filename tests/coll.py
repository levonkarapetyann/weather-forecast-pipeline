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
from data_pipeline import select_stations_for_run
from dotenv import load_dotenv
import requests
from typing import Any, Dict, List, Tuple
from datetime import datetime, timedelta
import json
import csv
import argparse
import os
import sys




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
    parser.add_argument("--test", action="store_true", help="Сбор данных только для тестовой станции (ID 8)")
    parser.add_argument("--days", type=int, default=730,
                        help="Количество исторических дней для загрузки (по умолчанию 2 года)")
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

    # Задаем целевой диапазон дат[cite: 2]
    start_date_calc = "2026-07-22"
    end_date_str = "2026-07-23"

    if args.test:
        print("--- Запуск в тестовом режиме (только станция ID 8) ---")
        stations = [s for s in stations if s.get("generated_id") == 8]
        if not stations:
            print("Предупреждение: Станция с generated_id=8 не найдена. Используем первую доступную.")
            stations = [stations_data["stations"][0]]
        print(f"Тестовый диапазон: {start_date_calc} - {end_date_str}")
    else:
        if args.days != 730:
            now = datetime.now()
            start_date_calc = (now - timedelta(days=args.days)).strftime("%Y-%m-%d")
            end_date_str = now.strftime("%Y-%m-%d")
            print(f"Динамический сбор за последние {args.days} дней: {start_date_calc} - {end_date_str}")
        else:
            print(f"Сбор данных за фиксированный период: {start_date_calc} - {end_date_str}")

    # Списки для объединения всех строк и хранения заголовков
    all_rows = []
    csv_headers = []

    for station in stations:
        name = station.get("name", "Unknown")
        internal_id = station.get("id")
        generated_id = station.get("generated_id")  # Номер станции (например: 8, 12...)

        status = station.get("Status", "online")
        created_at_str = station.get("created_at")

        if status == "offline":
            print(f"Станция '{name}' (id={internal_id}) помечена как offline. Пропускаем.")
            continue

        print(f"Сбор данных для станции: {name} (номер={generated_id})...")

        # Ограничиваем начало сбора датой создания станции[cite: 2]
        start_date_str = start_date_calc
        if created_at_str:
            try:
                created_at = datetime.strptime(created_at_str, "%Y-%m-%d")
                start_date_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
                if created_at > start_date_dt:
                    start_date_str = created_at_str
                    print(f"  Начальная дата скорректирована по дате создания станции: {start_date_str}")
            except ValueError:
                pass

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

        # Сортируем полученные данные по таймстампу
        timestamp_idx = 1
        try:
            all_data.sort(key=lambda x: x[timestamp_idx])
        except Exception as e:
            print(f"  Ошибка при сортировке по таймстампу: {e}")

        # Сохраняем структуру колонок (keys) от первой успешно ответившей станции
        if not csv_headers and keys:
            # Добавим колонку "station_id" в самое начало заголовков
            csv_headers = ["station_id"] + keys

        # Преобразуем каждую запись, добавляя номер станции в начало строки
        for record in all_data:
            row = [generated_id] + record
            all_rows.append(row)

        print(f"  Добавлено в общий пул: станция {generated_id} ({len(all_data)} записей)")

    # Записываем все накопленные данные в один итоговый CSV-файл
    if all_rows:
        output_file = os.path.join(output_dir, "all_stations_data.csv")
        try:
            with open(output_file, "w", encoding="utf-8", newline="") as csv_f:
                writer = csv.writer(csv_f)
                # Пишем заголовки колонок
                writer.writerow(csv_headers)
                # Пишем все строки данных
                writer.writerows(all_rows)
            print(f"\n[УСПЕХ] Все данные успешно объединены и сохранены в CSV: {output_file}")
        except Exception as e:
            print(f"\n[ОШИБКА] Не удалось сохранить итоговый CSV-файл: {e}")
    else:
        print("\n[ВНИМАНИЕ] Никаких данных не было собрано. CSV-файл не создан.")


if __name__ == "__main__":
    main()