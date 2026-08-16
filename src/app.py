"""
=============================================================================
МОДУЛЬ: FastAPI Backend Server (app.py)
-----------------------------------------------------------------------------
НАЗНАЧЕНИЕ:
Основной сервер API приложения прогнозирования погоды. Принимает запросы
от веб-интерфейса Streamlit или внешних клиентов, опрашивает метеодатчики
и выдает готовые гиперлокальные прогнозы погоды на 48 часов.

ОСНОВНЫЕ ФУНКЦИИ И АЛГОРИТМЫ:
1. Загрузка весов гибридной модели TFT + CatBoost Residuals + RainCatBoost.
2. Реализация фильтра Калмана (KalmanStateObserver) для очистки сенсорных
   данных от шума перед расчетом невязок ошибки.
3. Адаптивная PID-коррекция ошибки реального времени на ближнем горизонте (0-10ч).
4. Линейное блендирование (10-14ч) с внешними синоптическими прогнозами Open-Meteo.
5. Физическая валидация осадков (Physical Rain Guardrails) по дефициту точки росы.
=============================================================================
"""

import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from data_fetcher import fetch_live_multimodel_forecast
from model import TFTForecaster
from data_pipeline import (EXTERNAL_FORECAST_COLUMNS, MODEL_SENSOR_COLUMNS,
                          MODEL_TARGET_COLUMNS, NORMALIZE_COLUMNS, apply_scalers,
                          format_multimodel_external_df, inverse_scalers,
                          load_raw_station_json, prepare_feature_frame,
                          prepare_model_inputs, select_stations_for_run)
from project_paths import load_settings, resolve_path
from residual_engine import (ResidualModelBundle, apply_residual_correction,
                             load_residual_models, residual_models_available)
from rain_engine import RainCatBoostClassifier, prepare_rain_features
from inversion import apply_inversion_correction


class KalmanStateObserver:
    """
    2D Фильтр Калмана состояния [значение, скорость изменения (дрейф)].
    Предназначен для очистки высокочастотного шума метеосенсоров перед вычислением невязок PID.
    """

    def __init__(self, process_noise_std: float = 0.05, measurement_noise_std: float = 0.15, dt: float = 1.0):
        self.dt = dt
        self.F = np.array([[1.0, self.dt], [0.0, 1.0]], dtype=np.float64)
        self.H = np.array([[1.0, 0.0]], dtype=np.float64)
        q_var = process_noise_std ** 2
        self.Q = np.array([
            [q_var * (self.dt ** 3) / 3.0, q_var * (self.dt ** 2) / 2.0],
            [q_var * (self.dt ** 2) / 2.0, q_var * self.dt]
        ], dtype=np.float64)
        self.R = np.array([[measurement_noise_std ** 2]], dtype=np.float64)
        self.x = np.array([[0.0], [0.0]], dtype=np.float64)
        self.P = np.eye(2, dtype=np.float64) * 1.0
        self.initialized = False

    def update(self, z: float, max_z_score: float = 3.5) -> tuple[float, float]:
        if not self.initialized:
            self.x[0, 0] = z
            self.x[1, 0] = 0.0
            self.initialized = True
            return float(self.x[0, 0]), float(self.x[1, 0])

        x_pred = self.F @ self.x
        P_pred = (self.F @ self.P @ self.F.T) + self.Q
        y = np.array([[z]], dtype=np.float64) - (self.H @ x_pred)
        S = (self.H @ P_pred @ self.H.T) + self.R
        
        # Z-score Anomaly Guardrail: Бракуем выброс датчика если |y| / sqrt(S) > max_z_score
        innovation_std = np.sqrt(S[0, 0])
        z_score = abs(y[0, 0]) / max(innovation_std, 1e-6)
        if z_score > max_z_score:
            # Игнорируем невалидный спайк датчика, используем априорный прогноз модели Калмана
            self.x = x_pred
            self.P = P_pred
            return float(self.x[0, 0]), float(self.x[1, 0])

        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + (K @ y)
        self.P = (np.eye(2) - (K @ self.H)) @ P_pred
        return float(self.x[0, 0]), float(self.x[1, 0])

    def filter_series(self, values: list | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        filtered_vals = []
        filtered_vels = []
        clean_vals = [float(v) for v in values if v is not None and not np.isnan(v)]
        if len(clean_vals) > 4:
            meas_std = max(0.05, float(np.std(np.diff(clean_vals))))
            self.R = np.array([[meas_std ** 2]], dtype=np.float64)

        for val in values:
            if val is None or np.isnan(val):
                val = filtered_vals[-1] if filtered_vals else 0.0
            smooth_val, smooth_vel = self.update(float(val))
            filtered_vals.append(smooth_val)
            filtered_vels.append(smooth_vel)
        return np.array(filtered_vals, dtype=np.float64), np.array(filtered_vels, dtype=np.float64)



SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


app = FastAPI(title="ClimateNet Hyperlocal Forecasting API (Production Ready)")

device = torch.device("cpu")

MAX_CORRECTION_REAL = {
    "temperature": 3.0,
    "pressure": 5.0,
    "humidity": 15.0,
    "rain": 2.0,
    "wind_u": 3.0,
    "wind_v": 3.0
}

# 1. Загружаем настройки из центрального конфига
settings_file = resolve_path("config", "settings.json")
if not os.path.exists(settings_file):
    print("Критическая ошибка: config/settings.json не найден.")
    exit(1)

settings = load_settings(settings_file)

# Настройка путей из конфига
STATIONS_CONFIG_PATH = resolve_path(settings["paths"]["stations_config"])
MODEL_PATH = resolve_path(settings["paths"]["models_dir"], settings["paths"].get("model_filename", "tft_model.pth"))
SCALERS_PATH = resolve_path(settings["paths"]["scalers_file"])
PID_PARAMS_PATH = resolve_path(settings["paths"]["pid_params_file"])
RESIDUAL_MODELS_DIR = resolve_path(settings["paths"].get("residual_models_dir", "models/residual_catboost"))
CACHE_DIR = resolve_path("data", "cache")

os.makedirs(CACHE_DIR, exist_ok=True)

# Загружаем модель с учетом Attention-архитектуры
model = None
model_status = {"loaded": False, "path": MODEL_PATH, "error": None}
MODEL_MTIME = None
residual_bundle: ResidualModelBundle | None = None
residual_status = {"loaded": False, "path": RESIDUAL_MODELS_DIR, "error": None}
rain_classifier = None


def load_rain_classifier():
    global rain_classifier
    try:
        classifier = RainCatBoostClassifier(model_dir=resolve_path("models"))
        classifier.load()
        rain_classifier = classifier
        print("[App] Автономная модель осадков RainCatBoostClassifier успешно загружена.")
    except Exception as e:
        rain_classifier = None
        print(f"[App] Предупреждение: не удалось загрузить RainCatBoostClassifier ({e}). Будет использован фолбэк.")



def load_model() -> None:
    global model, model_status
    global MODEL_MTIME
    if not os.path.exists(MODEL_PATH):
        model_status = {"loaded": False, "path": MODEL_PATH, "error": "model file not found"}
        print(f"Предупреждение: файл весов модели {MODEL_PATH} не найден.")
        return

    try:
        tft_cfg = settings.get("tft", {})
        model = TFTForecaster(
            hidden_size=tft_cfg.get("hidden_size", 128),
            num_heads=tft_cfg.get("num_heads", 4),
            num_lstm_layers=tft_cfg.get("num_lstm_layers", 2),
            dropout=tft_cfg.get("dropout", 0.1),
        )
        state_dict = torch.load(MODEL_PATH, map_location=device)
        if "output_projs.1.weight" in state_dict and "output_proj.weight" not in state_dict:
            state_dict["output_proj.weight"] = state_dict.pop("output_projs.1.weight")
            state_dict["output_proj.bias"] = state_dict.pop("output_projs.1.bias")
            for i in [0, 2]:
                state_dict.pop(f"output_projs.{i}.weight", None)
                state_dict.pop(f"output_projs.{i}.bias", None)
        elif "output_projs.0.weight" in state_dict and "output_proj.weight" not in state_dict:
            state_dict["output_proj.weight"] = state_dict.pop("output_projs.0.weight")
            state_dict["output_proj.bias"] = state_dict.pop("output_projs.0.bias")
            for i in [1, 2]:
                state_dict.pop(f"output_projs.{i}.weight", None)
                state_dict.pop(f"output_projs.{i}.bias", None)
        elif "output_proj.weight" in state_dict and hasattr(model, "output_projs"):
            w = state_dict.pop("output_proj.weight")
            b = state_dict.pop("output_proj.bias")
            for i in range(3):
                state_dict[f"output_projs.{i}.weight"] = w.clone()
                state_dict[f"output_projs.{i}.bias"] = b.clone()
        model.load_state_dict(state_dict)
        model.eval()
        model_status = {"loaded": True, "path": MODEL_PATH, "error": None}
        try:
            MODEL_MTIME = os.path.getmtime(MODEL_PATH)
        except Exception:
            MODEL_MTIME = None
        print("Успешно загружена TFT модель прогнозирования.")
    except Exception as exc:
        model = None
        model_status = {"loaded": False, "path": MODEL_PATH, "error": str(exc)}
        print(f"Не удалось загрузить модель: {exc}")


def load_residual_model_bundle() -> None:
    global residual_bundle, residual_status
    try:
        residual_bundle = load_residual_models(RESIDUAL_MODELS_DIR)
        residual_status = {
            "loaded": residual_models_available(residual_bundle),
            "path": RESIDUAL_MODELS_DIR,
            "error": None,
        }
    except Exception as exc:
        residual_bundle = None
        residual_status = {"loaded": False, "path": RESIDUAL_MODELS_DIR, "error": str(exc)}


@app.on_event("startup")
def startup_event():
    load_model()
    load_residual_model_bundle()
    load_rain_classifier()


@app.post("/reload_model")
def reload_model():
    """Триггерно перезагрузить модель в памяти (вызывается тренером после сохранения)."""
    try:
        load_model()
        load_residual_model_bundle()
        mtime = os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
        return {"reloaded": model_status["loaded"], "model_mtime": mtime, "residual_loaded": residual_status["loaded"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model_mtime")
def get_model_mtime():
    """Возвращает mtime текущего файла модели (float seconds) либо null."""
    if os.path.exists(MODEL_PATH):
        try:
            return {"model_mtime": os.path.getmtime(MODEL_PATH)}
        except Exception:
            return {"model_mtime": None}
    return {"model_mtime": None}


@app.get("/pid_mtime")
def get_pid_mtime():
    """Возвращает время последнего сохранения PID-параметров."""
    if os.path.exists(PID_PARAMS_PATH):
        try:
            return {"pid_mtime": os.path.getmtime(PID_PARAMS_PATH)}
        except Exception:
            return {"pid_mtime": None}
    return {"pid_mtime": None}


# Вспомогательные переменные
target_cols = MODEL_TARGET_COLUMNS


def load_pid_params_for_station(station_key: str) -> dict:
    """Читает PID-параметры станции; при битом JSON возвращает пустой словарь."""
    if not os.path.exists(PID_PARAMS_PATH):
        return {}
    try:
        with open(PID_PARAMS_PATH, "r", encoding="utf-8") as pf:
            data = json.load(pf)
        return data.get(station_key, {})
    except (json.JSONDecodeError, OSError) as err:
        print(f"[Ошибка] Не удалось прочитать {PID_PARAMS_PATH}: {err}")
        return {}


class ForecastItem(BaseModel):
    timestamp: str
    temperature: float
    humidity: float
    pressure: float
    wind_speed: float
    wind_direction_degrees: float
    rain: float
    rain_probability: float
    will_rain: bool
    uv: float
    lux: float
    pm1: float
    pm2_5: float
    pm10: float
    wind_gust: float | None = None
    frost_risk: bool | None = False
    fog_risk: bool | None = False
    baro_status: str | None = "Стабильное"


class ForecastResponse(BaseModel):
    station_id: int
    station_name: str
    generated_at: str
    actual_temperature: float | None = None
    feature_importance: dict[str, float] | None = None
    forecast: list[ForecastItem]


class ForecastComponentsResponse(BaseModel):
    station_id: int
    station_name: str
    generated_at: str
    timestamps: list[str]           # ISO timestamps for all 192 steps
    model_temperature: list[float]  # TFT raw output BEFORE PID correction
    model_humidity: list[float]
    model_rain: list[float]
    model_pressure: list[float]

# --- Улучшенные функции запросов к внешним сервисам ---


def fetch_live_sensors_with_fallback(generated_id: int) -> pd.DataFrame:
    """
    Запрашивает последние 24 часа данных датчиков у API.
    В случае сбоя переключается на локальный сырой файл в качестве имитации.
    """
    now = datetime.now()
    start_time = (now - timedelta(days=2)).strftime("%Y-%m-%d")
    end_time = now.strftime("%Y-%m-%d")

    url = f"{settings['paths'].get('climatenet_url', 'https://emvnh9buoh.execute-api.us-east-1.amazonaws.com')}/getData"
    params = {"device_id": generated_id, "start_time": start_time, "end_time": end_time}

    # 1. Попытка запроса к живому API датчиков
    try:
        response = requests.get(url, params=params, timeout=5)  # короткий таймаут 5 секунд
        response.raise_for_status()
        res_json = response.json()
        keys = res_json.get("keys", [])
        data = res_json.get("data", [])
        if keys and data:
            df = pd.DataFrame(data, columns=keys)
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")

            # Приведение типов для предотвращения "Series cannot interpolate with object dtype"
            float_cols = [
                "uv", "lux", "temperature", "pressure", "humidity",
                "pm1", "pm2_5", "pm10", "wind speed", "rain"
            ]
            for col in float_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
            if "id" in df.columns:
                df["id"] = pd.to_numeric(df["id"], errors="coerce").round().astype("Int64")

            return df
    except Exception as e:
        print(f"  [Предупреждение] Ошибка при запросе датчиков: {e}. Активация резервного сценария (локальный кэш).")

    # 2. Фолбэк: Читаем сохраненный при сборе данных файл
    local_file = os.path.join("data", "raw", "stations", f"station_{generated_id}.json")
    if os.path.exists(local_file):
        try:
            df = load_raw_station_json(local_file)
            # Отрезаем последние 24 часа от последней имеющейся в файле даты
            last_time = df["timestamp"].max()
            df_last_24h = df[df["timestamp"] >= (last_time - timedelta(hours=24))].copy()
            return df_last_24h
        except Exception as err:
            print(f"  [Ошибка] Не удалось прочитать локальный кэш датчиков: {err}")

    return pd.DataFrame()


def fetch_live_forecast_with_cache(lat: float, lon: float, station_id: int) -> pd.DataFrame:
    """
    Запрашивает свежий внешний прогноз (Open-Meteo, Meteostat) с учетом настроек в settings.json.
    """
    forecast_sources = settings.get("forecast_sources", {})
    use_open_meteo = forecast_sources.get("use_open_meteo", settings.get("use_open_meteo", True))
    use_meteostat = forecast_sources.get("use_meteostat", settings.get("use_meteostat", True))

    df_om = pd.DataFrame()
    df_ms = pd.DataFrame()

    horizon_hours = settings["model"]["horizon_steps"] * 15 / 60
    forecast_days = int(np.ceil(horizon_hours / 24)) + 2

    if use_open_meteo:
        cache_file = os.path.join(CACHE_DIR, f"forecast_cache_{station_id}.json")
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation,cloud_cover",
            "timezone": "Asia/Yerevan",
            "forecast_days": forecast_days
        }
        try:
            res = requests.get(url, params=params, timeout=5)
            if res.status_code == 200:
                hourly = res.json().get("hourly", {})
                if hourly and "time" in hourly:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(hourly, f)
                    df_om = pd.DataFrame(hourly).rename(columns={"time": "timestamp"})
                    df_om["timestamp"] = pd.to_datetime(df_om["timestamp"], format="mixed")
        except Exception as e:
            print(f"  [Предупреждение] Ошибка связи с Open-Meteo: {e}")

        if df_om.empty and os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                df_om = pd.DataFrame(cached).rename(columns={"time": "timestamp"})
                df_om["timestamp"] = pd.to_datetime(df_om["timestamp"], format="mixed")
            except Exception as err:
                print(f"  [Ошибка] Не удалось прочесть файл кэша прогнозов: {err}")

    if use_meteostat:
        try:
            today_str = datetime.now().strftime("%Y-%m-%d")
            future_str = (datetime.now() + timedelta(days=forecast_days)).strftime("%Y-%m-%d")
            from data_fetcher import query_meteostat_api
            df_ms = query_meteostat_api(lat, lon, today_str, future_str)
        except Exception as e:
            print(f"  [Предупреждение] Ошибка получения прогноза Meteostat: {e}")

    return format_multimodel_external_df(df_om, df_ms)

# --- Эндпоинты ---


@app.get("/health")
def health_check():
    return {
        "status": "ok" if model_status["loaded"] else "degraded",
        "model_loaded": model_status["loaded"],
        "model_path": model_status["path"],
        "error": model_status["error"],
        "residual_loaded": residual_status["loaded"],
        "residual_path": residual_status["path"],
        "residual_error": residual_status["error"],
    }


@app.get("/stations")
def get_stations():
    """Возвращает список станций, ограниченный режимом одной станции при необходимости."""
    if not os.path.exists(STATIONS_CONFIG_PATH):
        raise HTTPException(status_code=404, detail="Конфигурация станций не найдена.")
    with open(STATIONS_CONFIG_PATH, "r", encoding="utf-8") as f:
        stations = json.load(f)["stations"]
    return select_stations_for_run(stations, settings)


@app.get("/forecast/{station_id}", response_model=ForecastResponse)
def get_forecast(station_id: int):
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Модель прогнозирования не загружена на сервере.")

    # 1. Загрузка настроек метеостанции
    with open(STATIONS_CONFIG_PATH, "r", encoding="utf-8") as f:
        stations = json.load(f)["stations"]
    stations = select_stations_for_run(stations, settings)
    station_meta = next((s for s in stations if s["id"] == station_id), None)
    if not station_meta:
        raise HTTPException(status_code=404, detail=f"Станция с ID {station_id} не найдена в конфиге.")

    gen_id = station_meta["generated_id"]
    lat = float(station_meta["latitude"])
    lon = float(station_meta["longitude"])
    elevation = float(station_meta["elevation_m"])

    # 2. Запрос истории датчиков (с поддержкой фолбэка)
    df_sensors = fetch_live_sensors_with_fallback(gen_id)
    if df_sensors.empty or len(df_sensors) < 10:
        raise HTTPException(status_code=503, detail="Датчики ClimateNet недоступны, локальный архив отсутствует.")

    # Предобработка сырой истории через единый пайплайн
    df_sensors = prepare_feature_frame(df_sensors, station_meta)

    # Оставляем последние 24 часа (96 шагов)
    df_sensors = df_sensors.sort_values("timestamp")
    lookback = settings["model"]["lookback_steps"]
    if len(df_sensors) < lookback:
        # Если строк меньше, чем требуется, построим регулярную сетку из 96 шагов, заканчивающуюся последним timestamp
        last_ts = df_sensors["timestamp"].max() if not df_sensors.empty else datetime.now()
        timestamps = pd.date_range(end=last_ts, periods=lookback, freq="15min")
        df_sensors = df_sensors.set_index("timestamp").reindex(timestamps)
        df_sensors.index.name = "timestamp"
        df_sensors = df_sensors.reset_index().ffill().bfill()
    else:
        df_sensors = df_sensors.tail(lookback)

    # Защита: заменим inf/NaN на 0.0 для предотвращения появления NaN в выходах модели
    df_sensors = df_sensors.replace([np.inf, -np.inf], np.nan)
    df_sensors = df_sensors.ffill().fillna(0.0)

    # 3. Запрос прогноза Open-Meteo (с поддержкой кэша)
    df_forecast = fetch_live_forecast_with_cache(lat, lon, station_id)
    if df_forecast.empty:
        raise HTTPException(status_code=503, detail="Внешний прогноз недоступен, кэш отсутствует.")

    # Формируем словарь прогноза Open-Meteo по времени для блендинга осадков
    wapi_dict = {}
    if "timestamp" in df_forecast.columns:
        forecast_df_copy = df_forecast.copy()
        forecast_df_copy["timestamp"] = pd.to_datetime(forecast_df_copy["timestamp"], format="mixed")
        for _, f_row in forecast_df_copy.iterrows():
            t_str = f_row["timestamp"].strftime("%Y-%m-%d %H:00")
            precip = float(f_row.get("precipitation", 0.0))
            # Прикидываем rain_probability на основе precipitation если явного поля нет
            prob = float(f_row.get("precipitation_probability", 100.0 if precip > 0.1 else 0.0))
            wapi_dict[t_str] = {"rain_probability": prob, "precipitation": precip}

    # DIAGNOSTICS: логируем размеры и количество NaN в датасетах, чтобы понять отсутствие данных
    try:
        print(f"[DEBUG] station_id={station_id} df_sensors.shape={df_sensors.shape}")
        print(f"[DEBUG] df_sensors NaN counts: {df_sensors.isna().sum().to_dict()}")
        print(f"[DEBUG] df_sensors sample:\n{df_sensors.head(3).to_dict(orient='records')}")
    except Exception as ex:
        print(f"[DEBUG] Error printing df_sensors diagnostics: {ex}")

    try:
        print(f"[DEBUG] station_id={station_id} df_forecast.shape={df_forecast.shape}")
        print(f"[DEBUG] df_forecast NaN counts: {df_forecast.isna().sum().to_dict()}")
        print(f"[DEBUG] df_forecast sample:\n{df_forecast.head(3).to_dict(orient='records')}")
    except Exception as ex:
        print(f"[DEBUG] Error printing df_forecast diagnostics: {ex}")

    # 4. Нормализация входов с помощью scalers.json
    if not os.path.exists(SCALERS_PATH):
        raise HTTPException(status_code=500, detail="Файл скейлеров нормализации не найден.")
    with open(SCALERS_PATH, "r", encoding="utf-8") as sf:
        scalers = json.load(sf)

    station_key = f"station_{station_id}"
    if station_key not in scalers:
        raise HTTPException(
            status_code=500, detail=f"Масштабирующие коэффициенты для станции {station_id} отсутствуют.")

    try:
        print(f"[DEBUG] scalers for {station_key} keys: {list(scalers[station_key].keys())[:30]}")
    except Exception:
        print(f"[DEBUG] scalers for {station_key} not printable or missing details")

    station_scalers = scalers[station_key]
    enc_t, dec_t, future_timestamps = prepare_model_inputs(
        df_sensors,
        df_forecast,
        station_meta,
        station_scalers,
        lookback_steps=settings["model"]["lookback_steps"],
        horizon_steps=settings["model"]["horizon_steps"],
        sensor_columns=MODEL_SENSOR_COLUMNS,
        static_columns=["latitude", "longitude", "elevation_m"],
        external_columns=EXTERNAL_FORECAST_COLUMNS,
        normalize_columns=NORMALIZE_COLUMNS,
    )

    # 5. Прогон через модель
    # Перед прогоном убеждаемся, что входы — тензоры PyTorch на нужном устройстве
    if not isinstance(enc_t, torch.Tensor):
        enc_tensor = torch.from_numpy(enc_t).float().to(device)
    else:
        enc_tensor = enc_t.float().to(device)

    if not isinstance(dec_t, torch.Tensor):
        dec_tensor = torch.from_numpy(dec_t).float().to(device)
    else:
        dec_tensor = dec_t.float().to(device)

    with torch.no_grad():
        raw_pred = model(enc_tensor, dec_tensor).squeeze(0).cpu().numpy()
        num_targets = len(MODEL_TARGET_COLUMNS)
        if raw_pred.ndim == 2 and raw_pred.shape[1] == num_targets * 3:
            raw_pred = raw_pred.reshape(raw_pred.shape[0], num_targets, 3)
            preds_norm = raw_pred[:, :, 1]  # q0.5 (медиана)
        else:
            preds_norm = raw_pred

    preds_norm = apply_residual_correction(
        preds_norm=preds_norm,
        sensor_df=df_sensors,
        forecast_df=df_forecast,
        station_meta=station_meta,
        station_scalers=station_scalers,
        future_timestamps=future_timestamps,
        residual_bundle=residual_bundle,
    )

    # 6. PID Коррекция прогноза
    pid_params = load_pid_params_for_station(station_key)

    preds_corrected = preds_norm.copy()
    df_sensors_norm = apply_scalers(df_sensors, station_scalers, NORMALIZE_COLUMNS)
    last_actuals_norm = df_sensors_norm[target_cols].iloc[-1].values

    near_horizon_hours = settings.get("residual_boost", {}).get("near_horizon_hours", 6.0)
    split_step = int(near_horizon_hours * 4.0)
    blend_window_steps = int(settings.get("residual_boost", {}).get("blend_window_steps", 4))
    blend_scale = max(1.0, blend_window_steps / 2.0)

    for idx, var_name in enumerate(target_cols):
        # rain_binary — логит классификатора: без scaler и без PID-коррекции
        if var_name == "rain_binary":
            continue

        std_v = float(scalers[station_key][var_name]["std"])
        max_corr_real = MAX_CORRECTION_REAL.get(var_name, 5.0)
        max_corr_z = max_corr_real / std_v if std_v > 0 else max_corr_real
        params = pid_params.get(var_name, {"Kp": 0.0, "Ki": 0.0, "Kd": 0.0})

        Kp_short = float(params.get("Kp_short", params.get("Kp", 0.0)))
        Ki_short = float(params.get("Ki_short", params.get("Ki", 0.0)))
        Kd_short = float(params.get("Kd_short", params.get("Kd", 0.0)))
        alpha_short = float(params.get("alpha_short", params.get("alpha", 0.995)))

        Kp_long = float(params.get("Kp_long", Kp_short))
        Ki_long = float(params.get("Ki_long", Ki_short))
        Kd_long = float(params.get("Kd_long", Kd_short))
        alpha_long = float(params.get("alpha_long", alpha_short))

        # Weather Regime PID: Детекция резкого погодного фронта по изменению давления dP/dt
        is_front_active = False
        if "pressure_trend_3h" in df_sensors.columns:
            p_trend = float(df_sensors["pressure_trend_3h"].iloc[-1])
            is_front_active = abs(p_trend) > 1.5

        if is_front_active:
            # На погодных фронтах повышаем Kp на 40% для мгновенного отклика PID
            Kp_short *= 1.40
            Kd_short *= 1.25

        # Отключаем PID для осадков во избежание «призраков дождя» после реальных ливней
        if var_name == "rain":
            Kp_short = Ki_short = Kd_short = 0.0
            Kp_long = Ki_long = Kd_long = 0.0

        int_limit = float(params.get("int_limit", 10.0))

        # Фильтрация сенсорного шума Калманом (Kalman Observer) по всей 24-часовой истории
        kalman_obs = KalmanStateObserver(process_noise_std=0.05, measurement_noise_std=0.15)
        sensor_series = df_sensors_norm[var_name].values
        filtered_series, _ = kalman_obs.filter_series(sensor_series)
        smooth_actual_norm = float(filtered_series[-1])

        # Ошибка на текущем шаге t0 относительно очищенного Калманом показания
        e_t0 = preds_norm[0, idx] - smooth_actual_norm
        if np.isnan(e_t0):
            e_t0 = 0.0

        e_t0 = float(np.clip(e_t0, -3.0, 3.0))

        # Разделение ошибки на системный сдвиг (bias, 60%) и мгновенный шум (noise, 40%)
        # Системный сдвиг гасится медленно (alpha_long), а шум — быстро (alpha_short)
        e_bias = 0.6 * e_t0
        e_noise = 0.4 * e_t0

        e_prev_bias = e_bias
        e_prev_noise = e_noise
        integral_sum = e_t0
        e_prev2 = e_t0

        for t in range(settings["model"]["horizon_steps"]):
            # Плавный (sigmoid) переход параметров (bumpless transfer)
            w = 1.0 / (1.0 + np.exp(-(t - split_step) / blend_scale))

            Kp = (1.0 - w) * Kp_short + w * Kp_long
            Ki = (1.0 - w) * Ki_short + w * Ki_long
            Kd = (1.0 - w) * Kd_short + w * Kd_long
            alpha_s = (1.0 - w) * alpha_short + w * alpha_long
            alpha_l = alpha_long

            e_prev_noise = e_prev_noise * alpha_s
            e_prev_bias = e_prev_bias * alpha_l
            e_prev = e_prev_noise + e_prev_bias

            integral_sum = np.clip(integral_sum * alpha_s + e_prev, -int_limit, int_limit)
            e_deriv = e_prev - e_prev2

            correction = Kp * e_prev + Ki * integral_sum + Kd * e_deriv
            # Экспоненциальное затухание начальной ошибки по горизонту (Gain Decay):
            decay_factor = float(np.exp(-t / 12.0))
            correction = correction * decay_factor
            correction = float(np.clip(correction, -max_corr_z, max_corr_z))
            preds_corrected[t, idx] = preds_norm[t, idx] - correction
            e_prev2 = e_prev

    # 7. Денормализация
    preds_df = pd.DataFrame(preds_corrected, columns=target_cols)
    preds_df = inverse_scalers(preds_df, {col: scalers[station_key][col]
                               for col in target_cols if col in scalers[station_key]}, target_cols)
    preds_final = preds_df.to_numpy(dtype=np.float32)



    # 7.2. Каскадная физика Влажности RH(T) (Magnus-Tetens Equation Coupling)
    # Корректируем влажность RH на основе давления насыщенного пара при T_final
    t_idx = target_cols.index("temperature")
    h_idx = target_cols.index("humidity")

    temp_c = preds_final[:, t_idx]
    humidity_raw = preds_final[:, h_idx]
    # Аппроксимация точки росы из T и RH (формула Магнуса)
    dew_c = temp_c - ((100.0 - np.clip(humidity_raw, 1.0, 100.0)) / 5.0)

    # Насыщенное давление пара e_s(T) и фактическое e(Td)
    es_t = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
    e_td = 6.112 * np.exp((17.67 * dew_c) / (dew_c + 243.5))
    rh_physical = np.clip((e_td / np.maximum(es_t, 1e-6)) * 100.0, 5.0, 100.0)

    # Плавное взвешивание: 70% физическая влажность + 30% прямая модель
    preds_final[:, h_idx] = 0.7 * rh_physical + 0.3 * humidity_raw

    # 7.3. Модель Ночной Инверсии на базе зенитного угла Солнца (Solar Zenith Angle Inversion)
    station_lat = float(station_meta.get("latitude", 40.2))
    temp_before = preds_final[:, 2].copy()
    wind_u = preds_final[:, 9]
    wind_v = preds_final[:, 10]

    cloud_cover_vec = None
    if "cloud_cover" in df_forecast.columns and "timestamp" in df_forecast.columns:
        try:
            df_fc_cc = df_forecast.copy()
            df_fc_cc["timestamp"] = pd.to_datetime(df_fc_cc["timestamp"], format="mixed")
            cc_vals = df_fc_cc["cloud_cover"].values
            if len(cc_vals) > 0 and np.max(cc_vals) > 1.0:
                cc_vals = cc_vals / 100.0
            s_cc = pd.Series(cc_vals, index=df_fc_cc["timestamp"])
            s_cc_interp = s_cc.reindex(s_cc.index.union(future_timestamps)).interpolate(method="time").reindex(future_timestamps).fillna(0.0)
            cloud_cover_vec = s_cc_interp.values
        except Exception:
            cloud_cover_vec = None

    preds_final[:, 2] = apply_inversion_correction(
        temp_raw=temp_before,
        wind_u=wind_u,
        wind_v=wind_v,
        timestamps=future_timestamps,
        lat_deg=station_lat,
        scale=1.0,
        cloud_cover=cloud_cover_vec,
    )

    # 7.4. Тепловая Инерция Почвы (Solar Soil Inertia) - Плавный синусоидальный сдвиг пика 12:00-16:00
    for t in range(len(preds_final)):
        fut_dt = future_timestamps[t]
        hour_frac = fut_dt.hour + fut_dt.minute / 60.0
        if 12.0 <= hour_frac <= 16.0:
            # Плавная синусоида прогрева почвы с пиком в 14:00 (+0.4°C)
            soil_inertia_boost = 0.4 * np.sin(np.pi * (hour_frac - 12.0) / 4.0)
            preds_final[t, 2] += float(soil_inertia_boost)

    # 7.5. Physical Rate Clipping (Физические ограничения максимальной скорости прогрева/остывания)
    # СТРОГО В КОНЦЕ ПОСТОБРАБОТКИ для устранения любых оставшихся ступенек: <= 0.875°C за 15 мин
    max_step_delta = 0.875
    for t in range(1, len(preds_final)):
        delta = preds_final[t, 2] - preds_final[t - 1, 2]
        if abs(delta) > max_step_delta:
            clamped_delta = np.clip(delta, -max_step_delta, max_step_delta)
            preds_final[t, 2] = preds_final[t - 1, 2] + clamped_delta

    forecast_list = []
    for t in range(settings["model"]["horizon_steps"]):
        u = preds_final[t, 9]
        v = preds_final[t, 10]
        speed = float(np.sqrt(u**2 + v**2))

        # 7.6. Динамика порывов ветра (Wind Gust Dynamics)
        temp_val = float(preds_final[t, 2])
        humidity_val = float(preds_final[t, 4])
        dew_val = temp_val - ((100.0 - humidity_val) / 5.0)
        instability = max(0.0, temp_val - dew_val) / 10.0
        gust_speed = float(round(speed * (1.25 + 0.15 * instability), 1))

        rad = np.arctan2(u, v)
        deg = float(np.degrees(rad))
        if deg < 0:
            deg += 360.0

        rain_amount = float(np.clip(preds_final[t, 8], 0.0, 500.0))

        # Расчет вероятности осадков из 12-й головы TFT
        rain_binary_logit = preds_final[t, 11]
        rain_prob = 1.0 / (1.0 + np.exp(-rain_binary_logit))
        rain_probability_pct = float(np.clip(rain_prob * 100.0, 0.0, 100.0))

        # На дальнем горизонте (12-48ч) ансамблируем с Open-Meteo вероятностью
        fut_time = future_timestamps[t]
        time_key = fut_time.strftime("%Y-%m-%d %H:00")
        if 'wapi_dict' in locals() and time_key in wapi_dict:
            om_prob = float(wapi_dict[time_key].get("rain_probability", 0.0))
            if t >= 40: # С 10-го часа
                w_om = min(1.0, (t - 40) / 16.0)
                rain_probability_pct = (1.0 - w_om) * rain_probability_pct + w_om * om_prob

        # 7.7. Индикаторы Риска Заморозков и Тумана (Frost & Fog Event Predictor)
        frost_risk = bool(temp_val <= 2.0 and (temp_val - dew_val) <= 1.5)
        fog_risk = bool(humidity_val >= 94.0 and speed <= 1.2)

        # 7.8. Барометрическая Тенденция (Barometric Trend Classification)
        press_val = float(preds_final[t, 3])
        if t >= 12:
            p_diff_3h = press_val - float(preds_final[t - 12, 3])
        else:
            p_diff_3h = 0.0
            
        if p_diff_3h < -1.5:
            baro_status = "Быстрое падение (Приближение шторма)"
        elif p_diff_3h > 1.5:
            baro_status = "Быстрый рост (Улучшение погоды)"
        else:
            baro_status = "Стабильное"

        # Физические условия атмосферы (Physics Guardrails)
        step_temp = float(preds_final[t, 2])
        step_hum = float(preds_final[t, 4])
        dew_point_approx = step_temp - ((100.0 - step_hum) / 5.0)
        dew_deficit = step_temp - dew_point_approx

        # Физический фильтр: дождь физически возможен при высокой влажности и малом дефиците точки росы
        physics_favorable = (step_hum >= 70.0) and (dew_deficit <= 3.0)
        has_rain_amount = rain_amount > 0.1

        will_rain = bool(has_rain_amount or (rain_probability_pct >= 60.0 and physics_favorable))

        t_min, t_max = settings["anomaly"]["temperature_range"]
        pressure_val = float(preds_final[t, 3])
        p_dev = settings["anomaly"]["pressure_deviation"]
        p_expected = 1013.25 * (1 - 0.0000225577 * elevation) ** 5.25588

        item = ForecastItem(
            timestamp=future_timestamps[t].strftime("%Y-%m-%d %H:%M:%S"),
            temperature=float(np.clip(preds_final[t, 2], t_min, t_max)),
            humidity=float(np.clip(preds_final[t, 4], 0.0, 100.0)),
            pressure=float(np.clip(pressure_val, p_expected - p_dev, p_expected + p_dev)),
            wind_speed=round(speed, 2),
            wind_direction_degrees=round(deg, 1),
            rain=rain_amount,
            rain_probability=round(rain_probability_pct, 1),
            will_rain=will_rain,
            uv=float(np.clip(preds_final[t, 0], 0.0, 15.0)),
            lux=float(np.clip(preds_final[t, 1], 0.0, 65000.0)),
            pm1=float(np.clip(preds_final[t, 5], 0.0, 1000.0)),
            pm2_5=float(np.clip(preds_final[t, 6], 0.0, 1000.0)),
            pm10=float(np.clip(preds_final[t, 7], 0.0, 1000.0)),
            wind_gust=gust_speed,
            frost_risk=frost_risk,
            fog_risk=fog_risk,
            baro_status=baro_status
        )
        forecast_list.append(item)

    actual_temp = None
    if "temperature" in df_sensors.columns:
        valid_t = df_sensors["temperature"].dropna()
        if not valid_t.empty:
            actual_temp = float(valid_t.iloc[-1])

    # Извлекаем веса важности фичей из TFT внимания (Variable Selection Weights)
    feat_imp = None
    if hasattr(model, "last_enc_var_weights") and model.last_enc_var_weights is not None:
        try:
            # (B, T, num_vars) -> усредняем по батчу и времени
            avg_w = model.last_enc_var_weights.mean(dim=(0, 1)).cpu().numpy()
            feat_imp = {col: round(float(w), 4) for col, w in zip(MODEL_SENSOR_COLUMNS, avg_w)}
        except Exception as ex:
            print(f"[App] Error computing feature_importance: {ex}")

    return ForecastResponse(
        station_id=station_id,
        station_name=station_meta["name"],
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        actual_temperature=actual_temp,
        feature_importance=feat_imp,
        forecast=forecast_list
    )


@app.get("/forecast_components/{station_id}", response_model=ForecastComponentsResponse)
def get_forecast_components(station_id: int):
    """
    Компоненты для графиков: сырой TFT до PID-коррекции.
    Финальный PID-прогноз отдаёт /forecast.
    """
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Модель прогнозирования не загружена.")

    with open(STATIONS_CONFIG_PATH, "r", encoding="utf-8") as f:
        stations = json.load(f)["stations"]
    stations = select_stations_for_run(stations, settings)
    station_meta = next((s for s in stations if s["id"] == station_id), None)
    if not station_meta:
        raise HTTPException(status_code=404, detail=f"Станция с ID {station_id} не найдена.")

    gen_id = station_meta["generated_id"]
    lat = float(station_meta["latitude"])
    lon = float(station_meta["longitude"])

    # Fetch sensors and external forecast (same as main endpoint)
    df_sensors = fetch_live_sensors_with_fallback(gen_id)
    if df_sensors.empty or len(df_sensors) < 10:
        raise HTTPException(status_code=503, detail="Датчики ClimateNet недоступны.")
    df_sensors = prepare_feature_frame(df_sensors, station_meta)
    df_sensors = df_sensors.sort_values("timestamp")
    lookback = settings["model"]["lookback_steps"]
    if len(df_sensors) < lookback:
        # Если строк меньше, чем требуется, построим регулярную сетку из 96 шагов, заканчивающуюся последним timestamp
        last_ts = df_sensors["timestamp"].max() if not df_sensors.empty else datetime.now()
        timestamps = pd.date_range(end=last_ts, periods=lookback, freq="15min")
        df_sensors = df_sensors.set_index("timestamp").reindex(timestamps)
        df_sensors.index.name = "timestamp"
        df_sensors = df_sensors.reset_index().ffill().bfill()
    else:
        df_sensors = df_sensors.tail(lookback)

    df_sensors = df_sensors.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df_forecast_ext = fetch_live_forecast_with_cache(lat, lon, station_id)
    if df_forecast_ext.empty:
        raise HTTPException(status_code=503, detail="Внешний прогноз недоступен.")

    if not os.path.exists(SCALERS_PATH):
        raise HTTPException(status_code=500, detail="Файл скейлеров не найден.")
    with open(SCALERS_PATH, "r", encoding="utf-8") as sf:
        scalers = json.load(sf)

    station_key = f"station_{station_id}"
    if station_key not in scalers:
        raise HTTPException(status_code=500, detail=f"Скейлеры для станции {station_id} отсутствуют.")

    station_scalers = scalers[station_key]
    enc_t, dec_t, future_timestamps = prepare_model_inputs(
        df_sensors, df_forecast_ext, station_meta, station_scalers,
        lookback_steps=settings["model"]["lookback_steps"],
        horizon_steps=settings["model"]["horizon_steps"],
        sensor_columns=MODEL_SENSOR_COLUMNS,
        static_columns=["latitude", "longitude", "elevation_m"],
        external_columns=EXTERNAL_FORECAST_COLUMNS,
        normalize_columns=NORMALIZE_COLUMNS,
    )

    enc_tensor = torch.from_numpy(enc_t).float().to(device)
    dec_tensor = torch.from_numpy(dec_t).float().to(device)
    with torch.no_grad():
        raw_pred = model(enc_tensor, dec_tensor).squeeze(0).cpu().numpy()
        num_targets = len(MODEL_TARGET_COLUMNS)
        if raw_pred.ndim == 2 and raw_pred.shape[1] == num_targets * 3:
            raw_pred = raw_pred.reshape(raw_pred.shape[0], num_targets, 3)
            preds_norm = raw_pred[:, :, 1]  # q0.5 (медиана)
        else:
            preds_norm = raw_pred

    # Denormalize raw model output (before PID) for comparison charts

    preds_raw_df = pd.DataFrame(preds_norm, columns=target_cols)
    preds_raw_df = inverse_scalers(
        preds_raw_df,
        {col: scalers[station_key][col] for col in target_cols if col in scalers[station_key]},
        target_cols
    )
    preds_raw = preds_raw_df.to_numpy(dtype=np.float32)

    t_min, t_max = settings["anomaly"]["temperature_range"]
    p_dev = settings["anomaly"]["pressure_deviation"]
    elevation = float(station_meta["elevation_m"])
    p_expected = 1013.25 * (1 - 0.0000225577 * elevation) ** 5.25588

    model_temp = [float(np.clip(preds_raw[t, 2], t_min, t_max)) for t in range(len(future_timestamps))]
    model_hum = [float(np.clip(preds_raw[t, 4], 0.0, 100.0)) for t in range(len(future_timestamps))]
    model_rain = [float(np.clip(preds_raw[t, 8], 0.0, 500.0)) for t in range(len(future_timestamps))]
    model_pres = [float(np.clip(preds_raw[t, 3], p_expected - p_dev, p_expected + p_dev))
                  for t in range(len(future_timestamps))]

    return ForecastComponentsResponse(
        station_id=station_id,
        station_name=station_meta["name"],
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        timestamps=[ts.strftime("%Y-%m-%d %H:%M:%S") for ts in future_timestamps],
        model_temperature=model_temp,
        model_humidity=model_hum,
        model_rain=model_rain,
        model_pressure=model_pres,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
