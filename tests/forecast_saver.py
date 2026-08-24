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

# Import gspread client (sheets_writer)
try:
    from sheets_writer import SheetsWriter
    _SHEETS_WRITER_AVAILABLE = True
except ImportError:
    _SHEETS_WRITER_AVAILABLE = False

# Import CatBoost for independent rain classifier
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
    """Queries fresh sensor readings from ClimateNet API or reads local JSON cache."""
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
Automated script that periodically ingests final hyperlocal forecasts
from all active stations and appends them to historical CSV/Sheets archives.


Final forecast corresponds to production output:
  TFT Model -> CatBoost Residuals -> PID Correction.

The script supports two execution modes (auto-selected):
  1. API Mode    : queries running FastAPI server (http://127.0.0.1:8000).
  2. Standalone  : loads models and computes forecasts in-process if server is down.
                  

Usage:
    python tests/forecast_saver.py                # periodic hourly loop
    python tests/forecast_saver.py --once         # single execution test
    python tests/forecast_saver.py --interval 30  # every 30 minutes
"""


# ── Paths ──────────────────────────────────────────────────────────────────────
# Script directory and project root

# Add src/ to sys.path to import internal modules

# Set working directory to project root so all relative paths resolve
# correctly across internal modules
os.chdir(PROJECT_ROOT)


# ── Output CSV Column Schema ───────────────────────────────────────────────────
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

# ── In-process Model Cache (prevents re-instantiation per station) ────────────
_local_get_forecast = None


def _ensure_local_model() -> bool:
    """Imports get_forecast from app.py once and caches it."""
    global _local_get_forecast
    if _local_get_forecast is not None:
        return True
    try:
        print("🔄 Loading TFT model into local process...")
        from app import get_forecast as _gf  # noqa: PLC0415
        _local_get_forecast = _gf
        print("✅ Model loaded.")
        return True
    except Exception as exc:
        print(f"❌ Failed to load model: {exc}")
        import traceback
        traceback.print_exc()
        return False


def _pydantic_to_dict(obj) -> dict:
    """Converts Pydantic object (v1 or v2) to plain dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):          # pydantic v2
        return obj.model_dump()
    if hasattr(obj, "dict"):                # pydantic v1
        return obj.dict()
    raise TypeError(f"Cannot convert object of type {type(obj)} to dict")


def get_station_forecast(station_id: int, api_url: str) -> dict | None:
    """
    Returns final forecast for station as dict.

    Fallback order:
      1. GET {api_url}/forecast/{station_id} - if FastAPI running
      2. Direct in-process get_forecast() - fallback
    """
    # ── 1. API ─────────────────────────────────────────────────────────────────
    url = f"{api_url.rstrip('/')}/forecast/{station_id}"
    try:
        print(f"📡 Querying via API: {url}")
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            print(f"   ✅ API responded ({len(data.get('forecast', []))} steps)")
            return data
        else:
            print(f"   ⚠️  API returned {resp.status_code}: {resp.text[:120]}")
    except requests.exceptions.ConnectionError:
        print("   ℹ️  API unavailable - falling back to local inference.")
    except Exception as exc:
        print(f"   ⚠️  API request error: {exc}")

    # ── 2. Local Inference ──────────────────────────────────────────────────────
    print(f"🔮 Local inference for station {station_id}...")
    if not _ensure_local_model():
        return None

    try:
        response_obj = _local_get_forecast(station_id)
        data = _pydantic_to_dict(response_obj)

        # Convert nested ForecastItem objects if not already dict
        if "forecast" in data:
            data["forecast"] = [
                _pydantic_to_dict(item) if not isinstance(item, dict) else item
                for item in data["forecast"]
            ]

        n = len(data.get("forecast", []))
        print(f"   ✅ Local inference completed ({n} steps)")
        return data

    except Exception as exc:
        print(f"   ❌ Local inference error: {exc}")
        import traceback
        traceback.print_exc()
        return None


def save_forecast_to_csv(forecast_data: dict, csv_path: str, run_ts: str) -> int:
    """
    Appends forecast to CSV. Returns count of written rows.
    """
    if not forecast_data or "forecast" not in forecast_data:
        return 0

    station_id = forecast_data.get("station_id")
    station_name = forecast_data.get("station_name")

    rows = []
    for item in forecast_data["forecast"]:
        # item can be plain dict (from API JSON or after model_dump)
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

    # Create directory if needed
    dir_path = os.path.dirname(csv_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    file_exists = os.path.exists(csv_path)
    size_before = os.path.getsize(csv_path) if file_exists else 0
    print(
        f"  📂 CSV before write: {'exists' if file_exists else 'not found'}, size: {size_before} bytes, path: {csv_path}")
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    size_after = os.path.getsize(csv_path)
    print(f"  💾 Appended {len(rows)} rows for '{station_name}' -> {csv_path} (size: {size_before} -> {size_after} bytes)")
    return len(rows)


def run_cycle(api_url: str, csv_path: str, settings: dict, sheets=None) -> None:
    """Single full cycle: generate and persist forecasts across all active stations."""
    now = datetime.now()
    run_ts = now.strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'#' * 70}")
    print(f"# FORECAST PERSISTENCE CYCLE - {run_ts}")
    print(f"{'#' * 70}")

    stations_config_path = resolve_path(settings["paths"]["stations_config"])
    if not os.path.exists(stations_config_path):
        print(f"❌ Station configuration file not found: {stations_config_path}")
        return

    with open(stations_config_path, "r", encoding="utf-8") as f:
        stations_data = json.load(f)

    active_stations = select_stations_for_run(stations_data["stations"], settings)
    print(f"Active stations: {len(active_stations)}")

    total_rows = 0
    failed = 0

    for station in active_stations:
        sid = station["id"]           # sequential id (for querying forecast)
        generated_id = station["generated_id"]  # actual device/station number
        sname = station["name"]
        print(f"\n▶ Station: {sname} (Device ID: {generated_id})")

        forecast = get_station_forecast(sid, api_url)
        if forecast:
            # Substitute sequential id with actual station generated_id
            forecast["station_id"] = generated_id

            total_rows += save_forecast_to_csv(forecast, csv_path, run_ts)
            if sheets is not None:
                try:
                    sheets.append_station_forecasts(forecast, run_ts)
                    time.sleep(1.2)  # 1.2s delay to comply with Google API quota (max 60 requests/min)
                except Exception as exc:
                    print(f"   ⚠️  Sheets.append_station_forecasts failed: {exc}")


        else:
            print(f"   ❌ Forecast not obtained, station skipped.")
            failed += 1

    elapsed = (datetime.now() - now).total_seconds()
    print(f"\n{'#' * 70}")
    print(f"# CYCLE COMPLETED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Summary: appended {total_rows} rows, errors: {failed}, elapsed: {elapsed:.1f}s")
    print(f"{'#' * 70}\n")


def run_actuals_cycle(sheets, settings: dict, hours_back: int = 2) -> None:
    """
    Fetches fresh sensor observations over the last hours_back hours
    and pushes measurement batch to Google Sheets.
    """
    if sheets is None:
        return

    now = datetime.now()
    cutoff_time = now - timedelta(hours=hours_back)
    print(f"\n{'=' * 70}")
    print(f"# LIVE SENSOR OBSERVATION CYCLE (last {hours_back}h) - {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")

    stations_config_path = resolve_path(settings["paths"]["stations_config"])
    if not os.path.exists(stations_config_path):
        print(f"❌ Station configuration file not found: {stations_config_path}")
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
                # If no response since cutoff_time, take last 8 available records (~2 hours)
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
            print(f"  ✅ {sname}: {st_count} observations over the last {hours_back}h")

        except Exception as exc:
            print(f"  ⚠️  Sensor data fetch error for {sname}: {exc}")

    if batch:
        try:
            sheets.append_actuals_batch(batch)
        except Exception as exc:
            print(f"  ⚠️  Sheets.append_actuals_batch failed: {exc}")

    elapsed = (datetime.now() - now).total_seconds()
    print(f"# Actuals cycle completed, total appended {len(batch)} rows ({elapsed:.1f}s")
    print(f"{'=' * 70}\n")


def _seconds_until_next_run(interval_minutes: int) -> float:
    """
    Returns seconds until next scheduled run aligned to grid.

    E.g. with interval_minutes=60, next run is top of the next hour.
    With interval_minutes=30, next run is 30-minute mark, etc.
    """
    now = datetime.now()
    # Seconds elapsed since midnight
    seconds_today = now.hour * 3600 + now.minute * 60 + now.second + now.microsecond / 1e6
    interval_sec = interval_minutes * 60
    # Next interval timestamp
    next_boundary = (int(seconds_today / interval_sec) + 1) * interval_sec
    wait = next_boundary - seconds_today
    return wait if wait > 0 else interval_sec


def main():
    parser = argparse.ArgumentParser(
        description="ClimateNet Forecast Persistence Scheduler (TFT + PID + Google Sheets)."
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
        help="Base FastAPI server URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--csv-path",
        default="",
        help="Path to output CSV (default: weather_data/model_forecasts.csv)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Execute single cycle and exit (test mode)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        metavar="MINUTES",
        help="Interval between cycles in minutes (default: 60)",
    )
    parser.add_argument(
        "--no-sheets",
        action="store_true",
        help="Disable Google Sheets export",
    )
    args = parser.parse_args()

    # Load project settings
    settings = load_settings()

    # Path to CSV
    csv_path = args.csv_path if args.csv_path else resolve_path("weather_data", "model_forecasts.csv")

    # 1. Initialize Google Sheets writer
    sheets = None
    gs_cfg = settings.get("google_sheets", {})
    if gs_cfg.get("enabled", False) and not args.no_sheets:
        if not _SHEETS_WRITER_AVAILABLE:
            print("⚠️  sheets_writer.py not found - Google Sheets disabled.")
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
                print(f"[Google Sheets Warning] Failed to connect: {exc}")
                sheets = None


    actuals_interval_hours = gs_cfg.get("actuals_interval_hours", 2)
    actuals_counter = actuals_interval_hours  # Initial execution immediately

    print("=" * 70)
    print("  ClimateNet - Forecast Persistence & Google Sheets Scheduler")
    print("=" * 70)
    print(f"  Project Root    : {PROJECT_ROOT}")
    print(f"  API Server      : {args.api_url}")
    print(f"  CSV File        : {csv_path}")
    print(f"  Google Sheets   : {'enabled' if sheets else 'disabled'}")
    print(f"  Actuals every   : {actuals_interval_hours} h")
    print(f"  Mode            : {'single run (--once)' if args.once else 'continuous loop'}")
    print("=" * 70)

    # Initial run: Forecasts
    run_cycle(args.api_url, csv_path, settings, sheets=sheets)

    # Initial run: Actuals
    if sheets is not None:
        run_actuals_cycle(sheets, settings)
        actuals_counter = 0

    if args.once:
        print("✅ Single test execution complete.")
        return

    # Continuous loop
    while True:
        wait_sec = _seconds_until_next_run(args.interval)
        next_time = datetime.now() + timedelta(seconds=wait_sec)
        print(
            f"⏳ Next cycle: {next_time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"(in {wait_sec / 60:.1f} min)"
        )
        try:
            time.sleep(wait_sec)
        except KeyboardInterrupt:
            print("\n\nStopped by user (Ctrl+C). Goodbye!")
            sys.exit(0)

        run_cycle(args.api_url, csv_path, settings, sheets=sheets)

        if sheets is not None:
            actuals_counter += 1
            if actuals_counter >= actuals_interval_hours:
                run_actuals_cycle(sheets, settings)
                actuals_counter = 0


if __name__ == "__main__":
    main()