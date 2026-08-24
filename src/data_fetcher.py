"""
=============================================================================
MODULE: Data Fetcher & External APIs (data_fetcher.py)
-----------------------------------------------------------------------------
PURPOSE:
Data ingestion module. Responsible for querying physical ClimateNet weather
station sensors and fetching external synoptic forecasts / reanalysis archives
from the Open-Meteo API.

KEY FUNCTIONS:
1. etch_all_sensor_data(): Ingests raw observations from ClimateNet Cloud API.
2. etch_all_external_forecasts(): Ingests historical synoptic forecasts & ERA5.
3. etch_live_multimodel_forecast(): Retrieves multi-model ensemble forecasts
   (ECMWF IFS, GFS, ICON) with ensemble mean and spread calculation.
=============================================================================
"""

import argparse
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from data_pipeline import select_stations_for_run
from project_paths import resolve_path, load_settings, load_stations_config

load_dotenv()

CLIMATENET_API_BASE_URL = os.getenv(
    "CLIMATENET_API_BASE_URL",
    "https://emvnh9buoh.execute-api.us-east-1.amazonaws.com"
)


def get_date_chunks(start_date_str: str, end_date_str: str, chunk_days: int = 90) -> List[Tuple[str, str]]:
    """Splits a time interval into chunks of chunk_days to prevent API request timeouts."""
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
    """Performs a GET request to the ClimateNet API to retrieve sensor data for a station."""
    url = f"{CLIMATENET_API_BASE_URL.rstrip('/')}/getData"
    params = {
        "device_id": device_id,
        "start_time": start_date,
        "end_time": end_date
    }

    try:
        print(f"  Request: {start_date} to {end_date} (device_id={device_id})...")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"  API request error for device {device_id}: {e}")
        return None


def load_stations_for_run(stations_path: str = None, settings_path: str = None) -> List[Dict[str, Any]]:
    """Loads stations list applying single-station or active station filters."""
    if stations_path is None:
        stations_path = resolve_path("config", "stations.json")
    if settings_path is None:
        settings_path = resolve_path("config", "settings.json")

    with open(stations_path, "r", encoding="utf-8") as f:
        stations = json.load(f).get("stations", [])

    with open(settings_path, "r", encoding="utf-8") as f:
        settings = json.load(f)

    return select_stations_for_run(stations, settings)


def query_open_meteo_api(
    url: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    timezone: str = "Asia/Yerevan"
) -> pd.DataFrame:
    """Generic query to Open-Meteo historical forecast or archive APIs."""
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
        print(f"    Error querying {url}: {e}")
        return pd.DataFrame()


def fetch_live_multimodel_forecast(lat: float, lon: float) -> pd.DataFrame:
    """
    Fetches operational 48-hour forecasts from global numerical weather prediction models:
    - ECMWF IFS025 (European Centre)
    - GFS Seamless (NOAA)
    - ICON Seamless (DWD)
    Computes ensemble mean and ensemble spread.
    """
    base_params = {
        "latitude": lat,
        "longitude": lon,
        "forecast_days": 3,
        "timezone": "Asia/Yerevan",
    }
    hourly_vars = "temperature_2m,relative_humidity_2m,surface_pressure,precipitation,wind_speed_10m"

    def _fetch_model(model_name: str, suffix: str) -> pd.DataFrame:
        params = {
            **base_params,
            "hourly": hourly_vars,
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
            print(f"  Error querying {model_name}: {e}")
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

    temp_cols = [c for c in merged.columns if c.startswith("temperature_2m")]
    if temp_cols:
        merged["temperature_2m_ensemble_mean"] = merged[temp_cols].mean(axis=1)
        merged["temp_ensemble_spread"] = merged[temp_cols].std(axis=1).fillna(0.0)
        merged["temperature_2m"] = merged["temperature_2m_ensemble_mean"]

    return merged


def fetch_all_sensor_data(stations: List[Dict[str, Any]], output_dir: str, days: int = None, is_test: bool = False) -> None:
    """Fetches and stores raw sensor readings from ClimateNet API for configured stations."""
    os.makedirs(output_dir, exist_ok=True)
    now = datetime.now()
    end_date_str = now.strftime("%Y-%m-%d")

    if days is not None:
        global_start_date = (now - timedelta(days=days)).strftime("%Y-%m-%d")
    else:
        global_start_date = "2020-01-01"

    for station in stations:
        name = station.get("name", "Unknown")
        internal_id = station.get("id")
        generated_id = station.get("generated_id")
        status = station.get("Status", "online")
        created_at_str = station.get("created_at")

        if status == "offline":
            print(f"Station '{name}' (id={internal_id}) is offline. Skipping.")
            continue

        print(f"\n[Sensors] Collecting station: {name} (generated_id={generated_id}, id={internal_id})...")

        start_date_str = global_start_date
        if created_at_str:
            try:
                created_at_clean = created_at_str.split("T")[0]
                created_at = datetime.strptime(created_at_clean, "%Y-%m-%d")
                start_date_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
                if created_at > start_date_dt:
                    start_date_str = created_at_clean
                    print(f"  Collection start: {start_date_str} (station creation date)")
                else:
                    print(f"  Collection start: {start_date_str} (fallback, created_at={created_at_clean})")
            except ValueError as e:
                print(f"  Could not parse created_at='{created_at_str}': {e}. Using {start_date_str}")
        else:
            print(f"  Collection start: {start_date_str} (using fallback)")

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
                print(f"  Failed to fetch chunk {idx + 1}/{len(chunks)} for {name}")

        if not all_data:
            print(f"  Warning: No data collected for station {name}.")
            continue

        timestamp_idx = 1
        try:
            all_data.sort(key=lambda x: x[timestamp_idx])
        except Exception as e:
            print(f"  Error sorting timestamps: {e}")

        output_file = os.path.join(output_dir, f"station_{generated_id}.json")
        result_json = {
            "keys": keys,
            "data": all_data
        }

        with open(output_file, "w", encoding="utf-8") as out_f:
            json.dump(result_json, out_f, ensure_ascii=False, indent=2)

        print(f"  Successfully saved: {output_file} ({len(all_data)} records)")


def fetch_all_external_forecasts(stations: List[Dict[str, Any]], output_dir: str, days: int = None, is_test: bool = False) -> None:
    """Fetches and stores external forecast archives and ERA5 reanalysis for configured stations."""
    os.makedirs(output_dir, exist_ok=True)
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    era5_end_str = (now - timedelta(days=5)).strftime("%Y-%m-%d")

    if days is not None:
        global_fallback_start = (now - timedelta(days=days)).strftime("%Y-%m-%d")
    else:
        global_fallback_start = "2020-01-01"

    for station in stations:
        name = station["name"]
        sid = station["id"]
        gen_id = station["generated_id"]
        lat = float(station["latitude"])
        lon = float(station["longitude"])
        status = station.get("Status", "online")

        if status == "offline":
            continue

        start_date = global_fallback_start
        date_source = f"fallback ({global_fallback_start})"

        created_at_str = station.get("created_at")
        if created_at_str:
            try:
                created_at_clean = created_at_str.split("T")[0]
                created_at_dt = datetime.strptime(created_at_clean, "%Y-%m-%d")
                fallback_dt = datetime.strptime(global_fallback_start, "%Y-%m-%d")
                start_date = max(created_at_dt, fallback_dt).strftime("%Y-%m-%d")
                date_source = f"created_at ({start_date})"
            except ValueError:
                pass

        raw_sensor_file = resolve_path("data", "raw", "stations", f"station_{gen_id}.json")
        if os.path.exists(raw_sensor_file):
            try:
                with open(raw_sensor_file, "r") as sf:
                    sensor_json = json.load(sf)
                    sensor_data = sensor_json.get("data", [])
                if sensor_data:
                    first_ts = pd.to_datetime(sensor_data[0][1]).strftime("%Y-%m-%d")
                    json_start_dt = datetime.strptime(first_ts, "%Y-%m-%d")
                    current_start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                    if json_start_dt < current_start_dt:
                        start_date = first_ts
                        date_source = f"first JSON record ({start_date})"
            except Exception as e:
                print(f"  Could not read sensor JSON: {e}")

        print(f"\n[External] Station: {name} | Start: {start_date} ({date_source}) | End: {today_str}")

        forecast_url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        archive_url = "https://archive-api.open-meteo.com/v1/archive"

        # 1. Historical Forecast Archive
        df_forecast = query_open_meteo_api(forecast_url, lat, lon, start_date, today_str)
        if not df_forecast.empty:
            out_file = os.path.join(output_dir, f"forecast_{sid}.csv")
            df_forecast.to_csv(out_file, index=False)
            print(f"  Saved forecast archive: {out_file} ({df_forecast.shape[0]} rows)")
        else:
            print("  Historical forecast archive empty. Trying fallback forecast API...")
            fallback_forecast_url = "https://api.open-meteo.com/v1/forecast"
            df_forecast = query_open_meteo_api(fallback_forecast_url, lat, lon, start_date, today_str)
            if not df_forecast.empty:
                out_file = os.path.join(output_dir, f"forecast_{sid}.csv")
                df_forecast.to_csv(out_file, index=False)
                print(f"  Saved fallback forecast: {out_file} ({df_forecast.shape[0]} rows)")

        # 2. ERA5 Reanalysis Archive
        df_era5 = query_open_meteo_api(archive_url, lat, lon, start_date, era5_end_str)
        if not df_era5.empty:
            out_file = os.path.join(output_dir, f"era5_{sid}.csv")
            df_era5.to_csv(out_file, index=False)
            print(f"  Saved ERA5 archive: {out_file} ({df_era5.shape[0]} rows)")
        else:
            print("  ERA5 archive unavailable for this period (typically lags by 2-5 days).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest raw historical ClimateNet observations and external Open-Meteo forecasts.")
    parser.add_argument("--source", choices=["sensors", "external", "all"], default="all", help="Data source to collect (default: all)")
    parser.add_argument("--test", action="store_true", help="Run in test mode (station ID 8, 3 days)")
    parser.add_argument("--days", type=int, default=None, help="Limit collection to the last N days")
    args = parser.parse_args()

    stations_path = resolve_path("config", "stations.json")
    settings_path = resolve_path("config", "settings.json")

    if not os.path.exists(stations_path):
        print(f"Error: Stations config not found at {stations_path}")
        return

    stations = load_stations_for_run(stations_path, settings_path)
    if not stations:
        print("Error: Stations list is empty.")
        return

    if args.test:
        print("--- Running in test mode (station ID 8, 3 days) ---")
        filtered = [s for s in stations if s.get("generated_id") == 8]
        stations = filtered if filtered else [stations[0]]
        args.days = 3

    if args.source in ("sensors", "all"):
        print("\n=== Fetching Local Station Sensor Data ===")
        sensor_dir = resolve_path("data", "raw", "stations")
        fetch_all_sensor_data(stations, sensor_dir, days=args.days, is_test=args.test)

    if args.source in ("external", "all"):
        print("\n=== Fetching External Forecasts & ERA5 Archives ===")
        external_dir = resolve_path("data", "raw", "external_forecasts")
        fetch_all_external_forecasts(stations, external_dir, days=args.days, is_test=args.test)

    print("\nData collection finished.")


if __name__ == "__main__":
    main()
