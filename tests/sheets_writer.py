#!/usr/bin/env python3
"""
sheets_writer.py
================
Модуль записи прогнозов и реальных данных в Google Sheets.

Записывает каждый прогноз и реальные показания станции в её персональный лист,
названный точно так же, как метеостанция (например, Station_1).

Структура листа станции:
  - Столбцы A-F (Прогноз): run_timestamp, station_id, station_name, forecast_datetime, temperature, will_rain
  - Столбец G (Разделитель): Пустой
  - Столбцы H-K (Реальные показания): timestamp, station_name, temperature_actual, rain_actual
"""

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import gspread
    from google.oauth2.service_account import Credentials
    _GSPREAD_AVAILABLE = True
except ImportError:
    _GSPREAD_AVAILABLE = False

_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
]

FORECAST_STEPS_24H = 96
FORECAST_HEADERS = ["run_timestamp", "station_id", "station_name", "forecast_datetime", "temperature", "will_rain"]
ACTUALS_HEADERS = ["timestamp", "station_name", "temperature_actual", "rain_actual"]


class SheetsWriter:
    """
    Клиент для записи строк в персональные листы метеостанций в Google Sheets.
    """

    def __init__(
        self,
        credentials_file: str,
        forecasts_spreadsheet_id: str,
        actuals_spreadsheet_id: str = None,
        **kwargs
    ):
        if not _GSPREAD_AVAILABLE:
            raise RuntimeError(
                "gspread не установлен. Выполните: pip install gspread>=6.0.0"
            )

        creds = Credentials.from_service_account_file(credentials_file, scopes=_SCOPES)
        self._gc = gspread.authorize(creds)
        self.spreadsheet = self._gc.open_by_key(forecasts_spreadsheet_id)
        self._worksheets_cache = {}

        print(f"✅ Google Sheets подключена: {self.spreadsheet.title}")

    def _get_or_create_station_sheet(self, station_name: str):
        """Возвращает или создает персональный лист для станции station_name."""
        clean_name = str(station_name).strip()
        if clean_name in self._worksheets_cache:
            return self._worksheets_cache[clean_name]

        try:
            ws = self.spreadsheet.worksheet(clean_name)
        except Exception:
            # Создаем компактный лист на 100 строк и 15 колонок
            ws = self.spreadsheet.add_worksheet(title=clean_name, rows=100, cols=15)
            print(f"  ✏️  Создан новый персональный лист станции '{clean_name}'")

        self._ensure_station_headers(ws)
        self._worksheets_cache[clean_name] = ws
        return ws

    def _ensure_station_headers(self, ws):
        """Проверяет и записывает параллельные заголовки прогноза (A1:F1) и реальных данных (K1:N1)."""
        try:
            first_row = ws.row_values(1)
            if not first_row or len(first_row) < 6:
                # Записываем заголовки прогноза в A1:F1
                ws.update("A1:F1", [FORECAST_HEADERS], value_input_option="USER_ENTERED")
                # Записываем заголовки реальных показаний в K1:N1
                ws.update("K1:N1", [ACTUALS_HEADERS], value_input_option="USER_ENTERED")
                print(f"  ✏️  Двойные заголовки (Прогноз A-F | Факты K-N) инициализированы в '{ws.title}'")
        except Exception as exc:
            print(f"  ⚠️ Ошибка инициализации заголовков в '{ws.title}': {exc}")


    def append_station_forecasts(self, forecast_data: Any, run_ts: str) -> int:
        """Дописывает прогноз станции в столбцы A:F её персонального листа."""
        if forecast_data is None:
            return 0

        if not isinstance(forecast_data, dict):
            if hasattr(forecast_data, "model_dump"):
                forecast_data = forecast_data.model_dump()
            elif hasattr(forecast_data, "dict"):
                forecast_data = forecast_data.dict()

        if not isinstance(forecast_data, dict) or "forecast" not in forecast_data:
            return 0

        station_id = forecast_data.get("station_id", "")
        station_name = forecast_data.get("station_name", f"Station_{station_id}")
        steps = forecast_data.get("forecast", [])[:FORECAST_STEPS_24H]

        rows = []
        for item in steps:
            if not isinstance(item, dict):
                if hasattr(item, "model_dump"):
                    item = item.model_dump()
                elif hasattr(item, "dict"):
                    item = item.dict()
                else:
                    continue

            temp = item.get("temperature")
            will_rain = item.get("will_rain")
            forecast_dt = item.get("timestamp", "")

            if will_rain is True or will_rain == 1:
                will_rain_str = "TRUE"
            elif will_rain is False or will_rain == 0:
                will_rain_str = "FALSE"
            else:
                will_rain_str = str(will_rain) if will_rain is not None else ""

            forecast_dt_str = str(forecast_dt)
            if forecast_dt_str and not forecast_dt_str.startswith("'"):
                forecast_dt_str = f"'{forecast_dt_str}"

            run_ts_str = str(run_ts)
            if run_ts_str and not run_ts_str.startswith("'"):
                run_ts_str = f"'{run_ts_str}"

            rows.append([
                run_ts_str,
                str(station_id),
                str(station_name),
                forecast_dt_str,
                str(round(float(temp), 2)) if temp is not None else "",
                will_rain_str,
            ])

        # Вставляем новые строки прогноза сверху (сразу под заголовками A1:F1)
        empty_row = ["", "", "", "", "", ""]
        rows_to_insert = rows + [empty_row, empty_row, empty_row]

        import time
        for attempt in range(3):
            try:
                ws = self._get_or_create_station_sheet(station_name)
                ws.insert_rows(rows_to_insert, row=2, value_input_option="USER_ENTERED")
                print(f"  📊 Sheets [{station_name}] Прогноз: +{len(rows)} строк (вставлено сверху со строки A2)")
                return len(rows)
            except Exception as exc:
                if "exceeds grid limits" in str(exc).lower() or "range" in str(exc).lower():
                    try:
                        ws.add_rows(len(rows_to_insert) + 2)
                    except Exception:
                        pass
                if "429" in str(exc) or "Quota exceeded" in str(exc):
                    print(f"  ⏳ [Google API Quota Limit 429] Ожидание 5 сек перед повтором ({attempt+1}/3)...")
                    time.sleep(5)
                else:
                    print(f"  ⚠️  Ошибка записи прогноза для '{station_name}': {exc}")
                    return 0
        return 0


    def append_station_actuals(self, station_name: str, actuals_list: list) -> int:
        """Дописывает фактические данные станции в столбцы K:N её персонального листа (снизу под ранее записанными)."""
        if not actuals_list:
            return 0

        rows = []
        for r in actuals_list:
            temp = r.get("temperature_actual")
            rain = r.get("rain_actual")
            ts_str = str(r.get("timestamp", ""))
            if ts_str and not ts_str.startswith("'"):
                ts_str = f"'{ts_str}"

            rows.append([
                ts_str,
                str(station_name),
                str(round(float(temp), 2)) if temp is not None else "",
                str(round(float(rain), 4)) if rain is not None else "0.0",
            ])

        # Вставляем реальные данные сверху в столбцы K:N под заголовками
        empty_actual_row = ["", "", "", ""]
        rows_to_insert = rows + [empty_actual_row, empty_actual_row, empty_actual_row]

        import time
        for attempt in range(3):
            try:
                ws = self._get_or_create_station_sheet(station_name)
                target_end_row = 1 + len(rows_to_insert)

                if target_end_row > ws.row_count:
                    ws.add_rows(target_end_row - ws.row_count + 10)

                col_k_vals = ws.col_values(11)
                existing_k_count = len(col_k_vals) - 1 if len(col_k_vals) > 1 else 0

                if existing_k_count > 0:
                    existing_data = ws.get(f"K2:N{1 + existing_k_count}")
                    ws.update(f"K{2 + len(rows_to_insert)}:N{1 + existing_k_count + len(rows_to_insert)}", existing_data, value_input_option="USER_ENTERED")

                ws.update(f"K2:N{target_end_row}", rows_to_insert, value_input_option="USER_ENTERED")
                print(f"  📊 Sheets [{station_name}] Реальные замеры: +{len(rows)} строк (вставлено сверху со строки K2)")
                return len(rows)
            except Exception as exc:
                if "exceeds grid limits" in str(exc).lower() or "range" in str(exc).lower():
                    try:
                        ws.add_rows(len(rows_to_insert) + 2)
                    except Exception:
                        pass
                if "429" in str(exc) or "Quota exceeded" in str(exc):
                    print(f"  ⏳ [Google API Quota Limit 429] Ожидание 5 сек перед повтором ({attempt+1}/3)...")
                    time.sleep(5)
                else:
                    print(f"  ⚠️  Ошибка записи фактов для '{station_name}': {exc}")
                    return 0
        return 0




    # Обратная совместимость
    def append_forecasts(self, forecast_data: Any, run_ts: str) -> int:
        return self.append_station_forecasts(forecast_data, run_ts)

    def append_actuals_batch(self, rows_dicts: list) -> int:
        grouped = {}
        for r in rows_dicts:
            sname = r.get("station_name", "Unknown")
            grouped.setdefault(sname, []).append(r)

        total = 0
        for sname, items in grouped.items():
            total += self.append_station_actuals(sname, items)
        return total
