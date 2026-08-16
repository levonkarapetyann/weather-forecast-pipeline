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
from data_fetcher import fetch_live_multimodel_forecast
import requests
import os
import sys




"""Тест: проверка fetch_live_multimodel_forecast и реального ответа Open-Meteo."""


LAT, LON = 40.18, 44.51

# 1) Прямой тест ECMWF API
print("=== Прямой тест ECMWF через requests ===")
try:
    params = {
        "latitude": LAT,
        "longitude": LON,
        "forecast_days": 2,
        "hourly": "temperature_2m,relative_humidity_2m",
        "models": "ecmwf_ifs025",
        "timezone": "Asia/Yerevan",
    }
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=15)
    print(f"HTTP {r.status_code}: {r.url}")
    if r.status_code == 200:
        data = r.json().get("hourly", {})
        print("Keys:", list(data.keys()))
        t_ecmwf = data.get("temperature_2m", [])
        print(f"temperature_2m (ECMWF) first 3: {t_ecmwf[:3]}")
    else:
        print("Ответ:", r.text[:300])
except Exception as e:
    print(f"ОШИБКА: {e}")

# 2) Прямой тест GFS API
print("\n=== Прямой тест GFS через requests ===")
try:
    params = {
        "latitude": LAT,
        "longitude": LON,
        "forecast_days": 2,
        "hourly": "temperature_2m,relative_humidity_2m",
        "models": "gfs_seamless",
        "timezone": "Asia/Yerevan",
    }
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=15)
    print(f"HTTP {r.status_code}: {r.url}")
    if r.status_code == 200:
        data = r.json().get("hourly", {})
        print("Keys:", list(data.keys()))
        t_gfs = data.get("temperature_2m", [])
        print(f"temperature_2m (GFS) first 3: {t_gfs[:3]}")
    else:
        print("Ответ:", r.text[:300])
except Exception as e:
    print(f"ОШИБКА: {e}")

# 3) Тест fetch_live_multimodel_forecast
print("\n=== Тест fetch_live_multimodel_forecast ===")

df = fetch_live_multimodel_forecast(LAT, LON)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
for col in ["temperature_2m", "temperature_2m_ecmwf_ifs025", "temperature_2m_gfs_seamless"]:
    if col in df.columns:
        s = df[col]
        print(f"  {col}: min={s.min():.2f}  max={s.max():.2f}  NaN={s.isna().sum()}")