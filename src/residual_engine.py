"""
=============================================================================
МОДУЛЬ: Residual Boosting Engine (residual_engine.py)
-----------------------------------------------------------------------------
НАЗНАЧЕНИЕ:
Модуль ансамблирования и машинного обучения для корректировки систематических
ошибок (Residual Correction) основной нейросети TFT.

ОСНОВНЫЕ ФУНКЦИИ И КЛАССЫ:
1. Класс `ResidualModelBundle`: хранение обученных регрессоров CatBoost для
   температуры, влажности и давления.
2. `apply_residual_correction`: коррекция выходов нейросети на основе текущих
   трендов ошибки и синоптических разностей.
3. Цикл генерации данных ошибок и обучения модели корректировки остатков.
=============================================================================
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data_pipeline import MODEL_TARGET_COLUMNS, NORMALIZE_COLUMNS, apply_scalers

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover - optional dependency fallback
    CatBoostRegressor = None


RESIDUAL_TARGETS = ["temperature", "humidity", "pressure"]
RESIDUAL_FEATURE_COLUMNS = [
    "station_id",
    "station_thermal_heating_index",
    "temp_velocity_3h",
    "latitude",
    "longitude",
    "elevation_m",
    "forecast_step_idx",
    "forecast_step_hours",
    "horizon_step",
    "horizon_step_log",
    "step_ratio",
    "forecast_hour",
    "forecast_dayofweek",
    "forecast_month",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "month_sin",
    "month_cos",
    "last_temperature_norm",
    "last_humidity_norm",
    "last_pressure_norm",
    "temperature_trend_1h_norm",
    "humidity_trend_1h_norm",
    "pressure_trend_1h_norm",
    "temperature_2m_norm",
    "relative_humidity_2m_norm",
    "surface_pressure_norm",
    "wind_speed_10m_norm",
    "precipitation_norm",
    "cloud_cover_norm",
    "wind_direction_sin",
    "wind_direction_cos",
    "tft_pred_temperature_norm",
    "tft_pred_humidity_norm",
    "tft_pred_pressure_norm",
    "diff_tft_ext_temp",
    "diff_tft_ext_humidity",
    "diff_tft_ext_pressure",
    "ext_temp_change_from_start",
    "ext_pressure_change_from_start",
    "ext_temp_step_delta",
    "tft_temp_trend_step",
    "tft_humidity_trend_step",
    "tft_pressure_trend_step",
    "temp_ensemble_spread",
    "rolling_mean_bias_24h",
    "last_3h_temp_slope",
    "diff_ecmwf_gfs_temp",
    "ext_temp_bias_24h",
    "temp_diff_24h",
    "openmeteo_temp_bias_3h",
]



@dataclass(frozen=True)
class ResidualModelBundle:
    models: Dict[str, object]
    feature_columns: List[str]


def _time_features(timestamp: pd.Timestamp) -> Dict[str, float]:
    hour = int(timestamp.hour)
    doy = int(timestamp.dayofyear)
    month = int(timestamp.month)
    weekday = int(timestamp.dayofweek)
    hour_rad = 2.0 * np.pi * hour / 24.0
    doy_rad = 2.0 * np.pi * doy / 366.0
    month_rad = 2.0 * np.pi * (month - 1) / 12.0

    return {
        "forecast_hour": float(hour),
        "forecast_dayofweek": float(weekday),
        "forecast_month": float(month),
        "hour_sin": float(np.sin(hour_rad)),
        "hour_cos": float(np.cos(hour_rad)),
        "doy_sin": float(np.sin(doy_rad)),
        "doy_cos": float(np.cos(doy_rad)),
        "month_sin": float(np.sin(month_rad)),
        "month_cos": float(np.cos(month_rad)),
    }


def _prepare_external_frame(
    forecast_df: pd.DataFrame,
    future_timestamps: pd.DatetimeIndex,
    station_scalers: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    forecast_frame = forecast_df.copy()
    forecast_frame["timestamp"] = pd.to_datetime(forecast_frame["timestamp"], format="mixed")
    forecast_frame = forecast_frame.sort_values("timestamp")
    forecast_frame = forecast_frame.set_index("timestamp").resample("15min").ffill().reset_index()

    future_df = pd.DataFrame({"timestamp": future_timestamps})
    aligned = pd.merge(future_df, forecast_frame, on="timestamp", how="left").ffill().bfill()

    if "temperature" in station_scalers and "temperature_2m" in aligned.columns:
        aligned["temperature_2m_norm"] = (
            aligned["temperature_2m"] - station_scalers["temperature"]["mean"]
        ) / station_scalers["temperature"]["std"]
    else:
        aligned["temperature_2m_norm"] = 0.0

    if "humidity" in station_scalers and "relative_humidity_2m" in aligned.columns:
        aligned["relative_humidity_2m_norm"] = (
            aligned["relative_humidity_2m"] - station_scalers["humidity"]["mean"]
        ) / station_scalers["humidity"]["std"]
    else:
        aligned["relative_humidity_2m_norm"] = 0.0

    if "pressure" in station_scalers and "surface_pressure" in aligned.columns:
        aligned["surface_pressure_norm"] = (
            aligned["surface_pressure"] - station_scalers["pressure"]["mean"]
        ) / station_scalers["pressure"]["std"]
    else:
        aligned["surface_pressure_norm"] = 0.0

    if "wind_speed_10m" in station_scalers and "wind_speed_10m" in aligned.columns:
        aligned["wind_speed_10m_norm"] = (
            aligned["wind_speed_10m"] - station_scalers["wind_speed_10m"]["mean"]
        ) / station_scalers["wind_speed_10m"]["std"]
    else:
        aligned["wind_speed_10m_norm"] = 0.0

    if "rain" in station_scalers and "precipitation" in aligned.columns:
        aligned["precipitation_norm"] = (
            aligned["precipitation"] - station_scalers["rain"]["mean"]
        ) / station_scalers["rain"]["std"]
    else:
        aligned["precipitation_norm"] = 0.0

    if "cloud_cover" in station_scalers and "cloud_cover" in aligned.columns:
        aligned["cloud_cover_norm"] = (
            aligned["cloud_cover"] - station_scalers["cloud_cover"]["mean"]
        ) / station_scalers["cloud_cover"]["std"]
    else:
        aligned["cloud_cover_norm"] = 0.0

    if "wind_direction_10m" in aligned.columns:
        wind_dir = pd.to_numeric(aligned["wind_direction_10m"], errors="coerce").fillna(0.0)
        wind_rad = np.deg2rad(wind_dir)
        aligned["wind_direction_sin"] = np.sin(wind_rad)
        aligned["wind_direction_cos"] = np.cos(wind_rad)
    else:
        aligned["wind_direction_sin"] = 0.0
        aligned["wind_direction_cos"] = 0.0

    return aligned


def build_residual_feature_frame(
    sensor_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    station_meta: Dict[str, float],
    station_scalers: Dict[str, Dict[str, float]],
    future_timestamps: pd.DatetimeIndex,
    preds_norm: np.ndarray,
    sample_stride: int = 1,
    sample_indices: List[int] | None = None,
) -> pd.DataFrame:
    """Строит признаки CatBoost residual-модели для всего горизонта."""
    sensor_norm = apply_scalers(sensor_df.copy(), station_scalers, NORMALIZE_COLUMNS)
    sensor_norm = sensor_norm.sort_values("timestamp").reset_index(drop=True)
    ext_norm = _prepare_external_frame(forecast_df, future_timestamps, station_scalers)

    last_row = sensor_norm.iloc[-1]
    prev_1h_idx = max(0, len(sensor_norm) - 5)
    prev_1h_row = sensor_norm.iloc[prev_1h_idx]

    last_temp_norm = float(last_row.get("temperature", 0.0))
    last_hum_norm = float(last_row.get("humidity", 0.0))
    last_press_norm = float(last_row.get("pressure", 0.0))

    rows: List[Dict[str, float]] = []
    station_id = str(int(station_meta.get("id", 0)))
    total_steps = max(1, len(future_timestamps))

    prev_ext_temp = last_temp_norm
    prev_ext_press = last_press_norm
    prev_tft_temp = last_temp_norm
    prev_tft_hum = last_hum_norm
    prev_tft_press = last_press_norm

    if sample_indices is not None:
        step_indices = [idx for idx in sample_indices if idx < len(future_timestamps)]
    else:
        step_indices = list(range(0, len(future_timestamps), max(1, int(sample_stride))))
    for step_idx in step_indices:
        ts = future_timestamps[step_idx]
        time_feats = _time_features(ts)
        ext_row = ext_norm.iloc[step_idx]
        pred_row = preds_norm[step_idx]

        tft_temp = float(pred_row[MODEL_TARGET_COLUMNS.index("temperature")])
        tft_hum = float(pred_row[MODEL_TARGET_COLUMNS.index("humidity")])
        tft_press = float(pred_row[MODEL_TARGET_COLUMNS.index("pressure")])

        ext_temp = float(ext_row.get("temperature_2m_norm", 0.0))
        ext_hum = float(ext_row.get("relative_humidity_2m_norm", 0.0))
        ext_press = float(ext_row.get("surface_pressure_norm", 0.0))

        step_ratio = float(step_idx / max(1, total_steps - 1))

        diff_tft_ext_temp = tft_temp - ext_temp
        diff_tft_ext_humidity = tft_hum - ext_hum
        diff_tft_ext_pressure = tft_press - ext_press

        ext_temp_change_from_start = ext_temp - last_temp_norm
        ext_pressure_change_from_start = ext_press - last_press_norm

        ext_temp_step_delta = ext_temp - prev_ext_temp
        ext_pressure_step_delta = ext_press - prev_ext_press

        tft_temp_trend_step = tft_temp - prev_tft_temp
        tft_humidity_trend_step = tft_hum - prev_tft_hum
        tft_pressure_trend_step = tft_press - prev_tft_press

        # Расчет признаков ансамбля моделей ECMWF / GFS
        temp_spread = float(ext_row.get("temp_ensemble_spread", 0.0)) if hasattr(ext_row, "get") and ext_row.get("temp_ensemble_spread") is not None else 0.0
        rolling_mean_bias = float(ext_row.get("rolling_mean_bias_24h", 0.0)) if hasattr(ext_row, "get") and ext_row.get("rolling_mean_bias_24h") is not None else 0.0
        last_3h_slope = float(ext_row.get("last_3h_temp_slope", 0.0)) if hasattr(ext_row, "get") and ext_row.get("last_3h_temp_slope") is not None else 0.0
        ecmwf_t = ext_row.get("temperature_2m_ecmwf_ifs025", None)
        gfs_t = ext_row.get("temperature_2m_gfs_seamless", None)
        if ecmwf_t is not None and gfs_t is not None:
            diff_models = float(ecmwf_t) - float(gfs_t)
        else:
            diff_models = 0.0

        rows.append({
            "station_id": station_id,
            "latitude": float(station_meta.get("latitude", 0.0)),
            "longitude": float(station_meta.get("longitude", 0.0)),
            "elevation_m": float(station_meta.get("elevation_m", 0.0)),
            "forecast_step_idx": float(step_idx),
            "forecast_step_hours": float(step_idx * 0.25),
            "horizon_step": float(step_idx),
            "horizon_step_log": float(np.log1p(step_idx)),
            "step_ratio": step_ratio,
            **time_feats,
            "last_temperature_norm": last_temp_norm,
            "last_humidity_norm": last_hum_norm,
            "last_pressure_norm": last_press_norm,
            "temperature_trend_1h_norm": float(last_temp_norm - prev_1h_row.get("temperature", 0.0)),
            "humidity_trend_1h_norm": float(last_hum_norm - prev_1h_row.get("humidity", 0.0)),
            "pressure_trend_1h_norm": float(last_press_norm - prev_1h_row.get("pressure", 0.0)),
            "temperature_2m_norm": ext_temp,
            "relative_humidity_2m_norm": ext_hum,
            "surface_pressure_norm": ext_press,
            "wind_speed_10m_norm": float(ext_row.get("wind_speed_10m_norm", 0.0)),
            "precipitation_norm": float(ext_row.get("precipitation_norm", 0.0)),
            "cloud_cover_norm": float(ext_row.get("cloud_cover_norm", 0.0)),
            "wind_direction_sin": float(ext_row.get("wind_direction_sin", 0.0)),
            "wind_direction_cos": float(ext_row.get("wind_direction_cos", 0.0)),
            "tft_pred_temperature_norm": tft_temp,
            "tft_pred_humidity_norm": tft_hum,
            "tft_pred_pressure_norm": tft_press,
            "diff_tft_ext_temp": diff_tft_ext_temp,
            "diff_tft_ext_humidity": diff_tft_ext_humidity,
            "diff_tft_ext_pressure": diff_tft_ext_pressure,
            "ext_temp_change_from_start": ext_temp_change_from_start,
            "ext_pressure_change_from_start": ext_pressure_change_from_start,
            "ext_temp_step_delta": ext_temp_step_delta,
            "ext_pressure_step_delta": ext_pressure_step_delta,
            "tft_temp_trend_step": tft_temp_trend_step,
            "tft_humidity_trend_step": tft_humidity_trend_step,
            "tft_pressure_trend_step": tft_pressure_trend_step,
            "temp_ensemble_spread": temp_spread,
            "rolling_mean_bias_24h": rolling_mean_bias,
            "last_3h_temp_slope": last_3h_slope,
            "diff_ecmwf_gfs_temp": diff_models,
            "ext_temp_bias_24h": float(sensor_norm.get("ext_temp_bias_24h", pd.Series(0.0, index=sensor_norm.index)).iloc[-1]),
            "forecast_timestamp": ts,
        })

        prev_ext_temp = ext_temp
        prev_ext_press = ext_press
        prev_tft_temp = tft_temp
        prev_tft_hum = tft_hum
        prev_tft_press = tft_press

    feature_frame = pd.DataFrame(rows)
    for column in RESIDUAL_FEATURE_COLUMNS:
        if column not in feature_frame.columns:
            feature_frame[column] = 0.0
    return feature_frame


def load_residual_models(models_dir: str) -> ResidualModelBundle:
    models: Dict[str, object] = {}
    metadata_path = os.path.join(models_dir, "metadata.json")
    feature_columns = RESIDUAL_FEATURE_COLUMNS.copy()

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            if isinstance(metadata.get("feature_columns"), list):
                feature_columns = metadata["feature_columns"]
        except Exception:
            pass

    if CatBoostRegressor is None:
        return ResidualModelBundle(models=models, feature_columns=feature_columns)

    for target in RESIDUAL_TARGETS:
        # 1. Загрузка сегментированных по горизонтам моделей (short=0-6ч, medium=6-24ч, long=24-48ч)
        for seg in ["short", "medium", "long"]:
            seg_path = os.path.join(models_dir, f"{target}_{seg}.cbm")
            if os.path.exists(seg_path):
                try:
                    m = CatBoostRegressor()
                    with SuppressStderrStdout():
                        m.load_model(seg_path)
                    models[f"{target}_{seg}"] = m
                except Exception:
                    pass

        # 2. Фолбэк базовая модель
        model_path = os.path.join(models_dir, f"{target}.cbm")
        if os.path.exists(model_path):
            try:
                model = CatBoostRegressor()
                model.load_model(model_path)
                models[target] = model
            except Exception:
                continue

    return ResidualModelBundle(models=models, feature_columns=feature_columns)


def apply_residual_correction(
    preds_norm: np.ndarray,
    sensor_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    station_meta: Dict[str, float],
    station_scalers: Dict[str, Dict[str, float]],
    future_timestamps: pd.DatetimeIndex,
    residual_bundle: ResidualModelBundle | None,
    decay_lambda: float = 0.0,
    return_std: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    std_norm = np.zeros_like(preds_norm, dtype=np.float32)

    if residual_bundle is None or not residual_bundle.models:
        return (preds_norm, std_norm) if return_std else preds_norm

    feature_frame = build_residual_feature_frame(
        sensor_df=sensor_df,
        forecast_df=forecast_df,
        station_meta=station_meta,
        station_scalers=station_scalers,
        future_timestamps=future_timestamps,
        preds_norm=preds_norm,
        sample_stride=1,
    )

    corrected = preds_norm.copy()
    num_steps = len(future_timestamps)
    step_ratios = np.linspace(0.0, 1.0, num_steps, dtype=np.float32)
    decay_weights = np.exp(-decay_lambda * step_ratios).astype(np.float32)

    model_features = feature_frame[residual_bundle.feature_columns].copy()
    for target in RESIDUAL_TARGETS:
        target_idx = MODEL_TARGET_COLUMNS.index(target)
        
        # Проверяем наличие сегментированных моделей по горизонтам
        has_segmented = (
            f"{target}_short" in residual_bundle.models and
            f"{target}_medium" in residual_bundle.models and
            f"{target}_long" in residual_bundle.models
        )

        if has_segmented:
            point_pred = np.zeros(num_steps, dtype=np.float32)
            m_short = residual_bundle.models[f"{target}_short"]
            m_medium = residual_bundle.models[f"{target}_medium"]
            m_long = residual_bundle.models[f"{target}_long"]

            if num_steps > 0:
                s_end = min(24, num_steps)
                if s_end > 0:
                    p_s = m_short.predict(model_features.iloc[:s_end], verbose=0)
                    point_pred[:s_end] = p_s[:, 0] if (isinstance(p_s, np.ndarray) and p_s.ndim == 2 and p_s.shape[1] >= 2) else (p_s.ravel() if isinstance(p_s, np.ndarray) else p_s)

                if num_steps > 24:
                    m_end = min(96, num_steps)
                    p_m = m_medium.predict(model_features.iloc[24:m_end], verbose=0)
                    point_pred[24:m_end] = p_m[:, 0] if (isinstance(p_m, np.ndarray) and p_m.ndim == 2 and p_m.shape[1] >= 2) else (p_m.ravel() if isinstance(p_m, np.ndarray) else p_m)

                if num_steps > 96:
                    p_l = m_long.predict(model_features.iloc[96:], verbose=0)
                    point_pred[96:] = p_l[:, 0] if (isinstance(p_l, np.ndarray) and p_l.ndim == 2 and p_l.shape[1] >= 2) else (p_l.ravel() if isinstance(p_l, np.ndarray) else p_l)
        else:
            model = residual_bundle.models.get(target)
            if model is None:
                continue
            raw_pred = model.predict(model_features)
            if isinstance(raw_pred, np.ndarray) and raw_pred.ndim == 2 and raw_pred.shape[1] >= 2:
                point_pred = raw_pred[:, 0].astype(np.float32)
                var_pred = np.maximum(0.0, raw_pred[:, 1].astype(np.float32))
                std_norm[:, target_idx] = np.sqrt(var_pred)
            else:
                point_pred = np.asarray(raw_pred, dtype=np.float32)

        # Context-aware Gating Network weighting:
        wind_speed_val = model_features.get("wind_speed_10m_norm", pd.Series(
            0.0, index=model_features.index)).to_numpy()
        precip_val = model_features.get("precipitation_norm", pd.Series(0.0, index=model_features.index)).to_numpy()
        step_ratio_val = model_features.get("step_ratio", pd.Series(0.0, index=model_features.index)).to_numpy()

        gate_logit = 1.2 * wind_speed_val + 1.8 * np.maximum(0.0, precip_val) - 0.4 * step_ratio_val
        tft_gating_alpha = 1.0 / (1.0 + np.exp(-np.clip(gate_logit, -5.0, 5.0)))
        catboost_weight = (1.0 - tft_gating_alpha).astype(np.float32)

        effective_correction = point_pred * decay_weights * catboost_weight
        corrected[:, target_idx] = corrected[:, target_idx] + effective_correction

        if target == "humidity" and "humidity" in station_scalers:
            mean = station_scalers["humidity"]["mean"]
            std = station_scalers["humidity"]["std"]
            actual_hum = corrected[:, target_idx] * std + mean
            actual_hum = np.clip(actual_hum, 0.0, 100.0)
            corrected[:, target_idx] = (actual_hum - mean) / std

    return (corrected, std_norm) if return_std else corrected


def residual_models_available(bundle: ResidualModelBundle | None) -> bool:
    return bool(bundle and bundle.models)
#!/usr/bin/env python3
"""Обучение CatBoost residual-моделей для temperature, humidity и pressure."""

import argparse
import json
import os
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from catboost import CatBoostRegressor, Pool
except Exception as exc:  # pragma: no cover - runtime guard
    raise SystemExit("CatBoost не установлен. Добавьте зависимость из requirements.txt и повторите запуск.") from exc

from model import ClimateDataset, TFTForecaster
from data_pipeline import (MODEL_TARGET_COLUMNS, NORMALIZE_COLUMNS, apply_scalers,
                          prepare_feature_frame, filter_station_files_for_run,
                          load_combined_external_forecast, select_stations_for_run)
from project_paths import load_settings, resolve_path


def _sample_window_positions(total_count: int, max_windows: int | None) -> list[int]:
    if max_windows is None or total_count <= max_windows:
        return list(range(total_count))
    return np.linspace(0, total_count - 1, max_windows, dtype=int).tolist()


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def _rolling_folds(
    n_samples: int,
    n_folds: int,
    min_train_ratio: float,
    gap_size: int,
    val_size: int | None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds for time-series CV."""
    if n_samples < 10:
        return []

    n_folds = max(2, int(n_folds))
    gap_size = max(0, int(gap_size))
    min_train_size = max(1, int(n_samples * float(min_train_ratio)))
    if min_train_size >= n_samples - 1:
        min_train_size = max(1, n_samples - 2)

    remaining = n_samples - min_train_size - gap_size
    if remaining < n_folds:
        n_folds = remaining
    if n_folds <= 0:
        return []

    if val_size is None:
        fold_val_size = max(1, remaining // n_folds)
    else:
        fold_val_size = max(1, int(val_size))

    folds: List[Tuple[np.ndarray, np.ndarray]] = []

    for fold_idx in range(n_folds):
        train_end = min_train_size + fold_idx * fold_val_size
        val_start = train_end + gap_size
        val_end = min(n_samples, val_start + fold_val_size)
        if fold_idx == n_folds - 1:
            val_end = n_samples
        if val_start >= val_end:
            continue

        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)
        if train_idx.size == 0 or val_idx.size == 0:
            continue
        folds.append((train_idx, val_idx))

    return folds


def _load_model(settings: dict, device: torch.device) -> TFTForecaster:
    tft_cfg = settings.get("tft", {})
    model = TFTForecaster(
        hidden_size=tft_cfg.get("hidden_size", 128),
        num_heads=tft_cfg.get("num_heads", 4),
        num_lstm_layers=tft_cfg.get("num_lstm_layers", 2),
        dropout=tft_cfg.get("dropout", 0.1),
    ).to(device)

    model_path = resolve_path(
        settings["paths"]["models_dir"],
        settings["paths"].get("model_filename", "tft_model.pth"),
    )
    state_dict = torch.load(model_path, map_location=device)
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
    return model


def clean_old_residual_models(models_dir: str):
    """Очищает устаревшие cbm файлы моделей CatBoost перед новым обучением."""
    if os.path.exists(models_dir):
        cleaned = 0
        for f in os.listdir(models_dir):
            if f.endswith(".cbm") or f.endswith(".json"):
                try:
                    os.remove(os.path.join(models_dir, f))
                    cleaned += 1
                except Exception:
                    pass
        print(f"🧹 Очищено {cleaned} старых файлов моделей CatBoost из {models_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение CatBoost residual-моделей для TFT-прогноза.")
    parser.add_argument("--max-windows-per-station", type=int, default=240, help="Ограничить число окон на станцию")
    parser.add_argument("--step-stride", type=int, default=4,
                        help="Брать каждый N-й шаг горизонта (для короткого горизонта)")
    parser.add_argument("--step-stride-long", type=int, default=1,
                        help="Брать каждый N-й шаг для горизонта > near_horizon_hours")
    parser.add_argument("--horizon-step-weight", type=float, default=1.5,
                        help="Вес признаков horizon_step и horizon_step_log при обучении CatBoost (начинать с 1.5)")
    parser.add_argument("--long-horizon-sample-weight", type=float, default=1.5,
                        help="Вес объектов с горизонтом > near_horizon_hours")
    parser.add_argument("--iterations", type=int, default=700, help="Итерации CatBoost")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate CatBoost")
    parser.add_argument("--depth", type=int, default=8, help="Глубина деревьев CatBoost")
    parser.add_argument("--l2-leaf-reg", type=float, default=5.0, help="L2 регуляризация CatBoost")
    parser.add_argument(
        "--uncertainty",
        action="store_true",
        help="Использовать RMSEWithUncertainty для оценки доверительных интервалов в CatBoost",
    )
    parser.add_argument(
        "--cv-mode",
        choices=["rolling", "holdout"],
        default="rolling",
        help="Режим валидации residual-моделей",
    )
    parser.add_argument("--cv-folds", type=int, default=5, help="Количество rolling-folds")
    parser.add_argument(
        "--cv-min-train-ratio",
        type=float,
        default=0.5,
        help="Минимальная доля train в первом rolling fold",
    )
    parser.add_argument(
        "--cv-gap-steps",
        type=int,
        default=4,
        help="Gap между train и valid в residual-строках (по умолчанию 4 ~ 1 час при шаге 15 мин)",
    )
    parser.add_argument(
        "--cv-val-size",
        type=int,
        default=None,
        help="Размер validation-блока на fold в residual-строках (если не задан, считается автоматически)",
    )
    args = parser.parse_args()

    settings = load_settings(resolve_path("config", "settings.json"))
    device = torch.device("cpu")

    near_horizon_hours = settings.get("residual_boost", {}).get("near_horizon_hours", 6.0)
    near_horizon_steps = int(near_horizon_hours * 4.0)

    processed_dir = resolve_path(settings["paths"]["processed_dir"])
    raw_ext_dir = resolve_path(settings["paths"]["external_dir"])
    stations_config = resolve_path(settings["paths"]["stations_config"])
    scalers_path = resolve_path(settings["paths"]["scalers_file"])
    residual_models_dir = resolve_path(settings["paths"].get("residual_models_dir", "models/residual_catboost"))
    os.makedirs(residual_models_dir, exist_ok=True)
    clean_old_residual_models(residual_models_dir)

    if not os.path.exists(stations_config):
        raise SystemExit(f"Stations config not found: {stations_config}")
    if not os.path.exists(scalers_path):
        raise SystemExit(f"Scalers file not found: {scalers_path}")

    with open(stations_config, "r", encoding="utf-8") as handle:
        stations = json.load(handle)["stations"]
    stations = select_stations_for_run(stations, settings)

    with open(scalers_path, "r", encoding="utf-8") as handle:
        all_scalers = json.load(handle)

    model = _load_model(settings, device)

    station_files = [name for name in os.listdir(processed_dir) if name.startswith(
        "station_") and name.endswith("_features.parquet")]
    station_files = filter_station_files_for_run(station_files, settings)

    feature_rows: list[dict] = []
    residual_targets: dict[str, list[float]] = {target: [] for target in RESIDUAL_TARGETS}

    print("--- Сбор residual-выборки ---")
    for file_name in station_files:
        sid = int(file_name.split("_")[1])
        station_meta = next((station for station in stations if station["id"] == sid), None)
        station_key = f"station_{sid}"
        parquet_path = os.path.join(processed_dir, file_name)

        if station_meta is None or station_key not in all_scalers:
            continue
        if not os.path.exists(parquet_path):
            continue

        df_forecast = load_combined_external_forecast(raw_ext_dir, sid, settings)
        if df_forecast.empty:
            continue

        df_features = pd.read_parquet(parquet_path)
        df_features["timestamp"] = pd.to_datetime(df_features["timestamp"], format="mixed")
        if "timestamp" in df_forecast.columns and not pd.api.types.is_datetime64_any_dtype(df_forecast["timestamp"]):
            df_forecast["timestamp"] = pd.to_datetime(df_forecast["timestamp"], format="mixed")
        df_features["latitude"] = float(station_meta["latitude"])
        df_features["longitude"] = float(station_meta["longitude"])
        df_features["elevation_m"] = float(station_meta["elevation_m"])

        features_norm = prepare_feature_frame(df_features, station_meta)
        features_norm = apply_scalers(features_norm, all_scalers[station_key], NORMALIZE_COLUMNS)
        dataset = ClimateDataset(features_norm, df_forecast, scalers=all_scalers[station_key])
        if len(dataset) == 0:
            continue

        selected_positions = _sample_window_positions(len(dataset), args.max_windows_per_station)
        lookback = dataset.lookback_steps
        horizon = dataset.horizon_steps

        # Dynamic step indices selection based on near_horizon_steps
        short_indices = list(range(0, min(near_horizon_steps, horizon), max(1, args.step_stride)))
        if horizon > near_horizon_steps:
            long_indices = list(range(near_horizon_steps, horizon, max(1, args.step_stride_long)))
        else:
            long_indices = []
        sample_indices = short_indices + long_indices

        for position in selected_positions:
            enc_x, dec_x, targets = dataset[position]
            with torch.no_grad():
                raw_pred = model(enc_x.unsqueeze(0), dec_x.unsqueeze(0)).squeeze(0).cpu().numpy()
            if raw_pred.ndim == 2 and raw_pred.shape[1] == 12 * 3:
                preds_norm = raw_pred.reshape(raw_pred.shape[0], 12, 3)[:, :, 1]
            else:
                preds_norm = raw_pred
            targets_norm = targets.cpu().numpy()
            start_idx = dataset.valid_indices[position]
            future_timestamps = pd.DatetimeIndex(
                features_norm["timestamp"].iloc[start_idx + lookback: start_idx +
                                                lookback + horizon].reset_index(drop=True)
            )

            feature_frame = build_residual_feature_frame(
                sensor_df=df_features.iloc[start_idx: start_idx + lookback].copy(),
                forecast_df=df_forecast,
                station_meta=station_meta,
                station_scalers=all_scalers[station_key],
                future_timestamps=future_timestamps,
                preds_norm=preds_norm,
                sample_indices=sample_indices,
            )

            sampled_step_indices = [idx for idx in sample_indices if idx < horizon][: len(feature_frame)]
            for row_idx, step_idx in enumerate(sampled_step_indices):
                feature_rows.append(feature_frame.iloc[row_idx].to_dict())
                for target in RESIDUAL_TARGETS:
                    target_idx = MODEL_TARGET_COLUMNS.index(target)
                    residual_targets[target].append(
                        float(targets_norm[step_idx, target_idx] - preds_norm[step_idx, target_idx]))

        print(f"  station_{sid}: {len(selected_positions)} окон, строк всего: {len(feature_rows)}")

    if not feature_rows:
        raise SystemExit("Residual training data is empty.")

    feature_df = pd.DataFrame(feature_rows)
    feature_df["station_id"] = feature_df["station_id"].astype(str)

    sort_order = feature_df.sort_values(["forecast_timestamp", "station_id"]).index.to_numpy()
    feature_df = feature_df.loc[sort_order].reset_index(drop=True)
    for target in RESIDUAL_TARGETS:
        residual_targets[target] = np.asarray(residual_targets[target], dtype=np.float32)[sort_order].tolist()

    feature_df_model = feature_df.drop(columns=["forecast_timestamp"])
    loss_func = "MAE"

    metadata = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feature_columns": RESIDUAL_FEATURE_COLUMNS,
        "step_stride": args.step_stride,
        "max_windows_per_station": args.max_windows_per_station,
        "loss_function": loss_func,
        "uncertainty_enabled": args.uncertainty,
        "cv_mode": args.cv_mode,
        "cv_folds": args.cv_folds,
        "cv_min_train_ratio": args.cv_min_train_ratio,
        "cv_gap_steps": args.cv_gap_steps,
        "cv_val_size": args.cv_val_size,
    }

    print(f"--- Обучение CatBoost residual models (loss: {loss_func}) ---")
    cat_features = ["station_id"]
    cv_metrics = {}
    feature_importance_report: dict[str, dict[str, float]] = {}
    oof_frames: list[pd.DataFrame] = []

    # Calculate sample weights (combining horizon weights + extreme physical event weights)
    step_indices_arr = feature_df_model["horizon_step"].to_numpy()
    sample_weights = np.where(step_indices_arr >= near_horizon_steps, args.long_horizon_sample_weight, 1.0)

    # Усиливаем вес экстремальных сценариев (инверсии, фён, высокая конвективная неустойчивость)
    if "inversion_risk" in feature_df_model.columns:
        inv_mask = feature_df_model["inversion_risk"].to_numpy() > 0.7
        sample_weights = sample_weights + np.where(inv_mask, 1.5, 0.0)
    if "foehn_index" in feature_df_model.columns:
        foehn_mask = feature_df_model["foehn_index"].to_numpy() > 2.0
        sample_weights = sample_weights + np.where(foehn_mask, 2.0, 0.0)
    if "pseudo_cape_index" in feature_df_model.columns:
        cape_mask = feature_df_model["pseudo_cape_index"].to_numpy() > 30.0
        sample_weights = sample_weights + np.where(cape_mask, 1.0, 0.0)

    feature_weights = {col: 1.0 for col in RESIDUAL_FEATURE_COLUMNS}
    feature_weights["horizon_step"] = args.horizon_step_weight
    feature_weights["horizon_step_log"] = args.horizon_step_weight

    for target in RESIDUAL_TARGETS:
        y = np.asarray(residual_targets[target], dtype=np.float32)
        
        # Асимметричное усиление веса недопрогноза дневного пика температуры (Asymmetric Peak Weighting)
        target_sample_weights = sample_weights.copy()
        if target == "temperature":
            hour_arr = feature_df_model["forecast_hour"].to_numpy() if "forecast_hour" in feature_df_model.columns else np.zeros(len(y))
            # 1. Дневной пик (11:00-16:00): штраф за недопрогноз дневного максимума
            peak_mask = (hour_arr >= 11) & (hour_arr <= 16) & (y > 0.1)
            # 2. Ночная коррекция (02:00-06:00): штраф за переохлаждение (недопрогноз ночной температуры)
            night_mask = (hour_arr >= 2) & (hour_arr <= 6) & (y > 0.1)
            target_sample_weights += np.where(peak_mask, 1.5, 0.0) + np.where(night_mask, 1.5, 0.0)

        if args.cv_mode == "rolling":
            folds = _rolling_folds(
                n_samples=len(feature_df_model),
                n_folds=args.cv_folds,
                min_train_ratio=args.cv_min_train_ratio,
                gap_size=args.cv_gap_steps,
                val_size=args.cv_val_size,
            )
            if not folds:
                raise SystemExit("Не удалось построить rolling folds. Увеличьте выборку или уменьшите cv-folds.")

            oof_pred = np.full(shape=len(y), fill_value=np.nan, dtype=np.float32)
            oof_fold_id = np.full(shape=len(y), fill_value=-1, dtype=np.int32)
            fold_stats = []
            for fold_no, (train_idx, val_idx) in enumerate(folds, start=1):
                train_features = feature_df_model.iloc[train_idx]
                val_features = feature_df_model.iloc[val_idx]
                y_train = y[train_idx]
                y_val = y[val_idx]

                train_pool = Pool(train_features[RESIDUAL_FEATURE_COLUMNS], y_train,
                                  cat_features=cat_features, weight=target_sample_weights[train_idx])
                val_pool = Pool(val_features[RESIDUAL_FEATURE_COLUMNS], y_val,
                                cat_features=cat_features, weight=target_sample_weights[val_idx])

                fold_model = CatBoostRegressor(
                    loss_function=loss_func,
                    iterations=args.iterations,
                    learning_rate=args.learning_rate,
                    depth=args.depth,
                    l2_leaf_reg=args.l2_leaf_reg,
                    feature_weights=feature_weights,
                    random_seed=42 + fold_no,
                    verbose=100,
                    early_stopping_rounds=50,
                )
                fold_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
                raw_pred = fold_model.predict(val_features[RESIDUAL_FEATURE_COLUMNS])
                val_pred = raw_pred[:, 0].astype(np.float32) if raw_pred.ndim == 2 else raw_pred.astype(np.float32)
                oof_pred[val_idx] = val_pred
                oof_fold_id[val_idx] = fold_no

                fold_rmse = _rmse(y_val, val_pred)
                fold_mae = _mae(y_val, val_pred)
                fold_stats.append({
                    "fold": fold_no,
                    "train_size": int(train_idx.size),
                    "val_size": int(val_idx.size),
                    "rmse": fold_rmse,
                    "mae": fold_mae,
                })
                print(
                    f"  {target} | fold {fold_no}/{len(folds)} "
                    f"train={train_idx.size} val={val_idx.size} "
                    f"RMSE={fold_rmse:.4f} MAE={fold_mae:.4f}"
                )

            valid_mask = ~np.isnan(oof_pred)
            overall_rmse = _rmse(y[valid_mask], oof_pred[valid_mask])
            overall_mae = _mae(y[valid_mask], oof_pred[valid_mask])

            tft_col = f"tft_pred_{target}_norm"
            if tft_col in feature_df.columns:
                tft_pred = feature_df[tft_col].to_numpy(dtype=np.float32)
            else:
                tft_pred = np.zeros_like(y)

            oof_frame = pd.DataFrame({
                "station_id": feature_df.loc[valid_mask, "station_id"].astype(str).to_numpy(),
                "forecast_timestamp": feature_df.loc[valid_mask, "forecast_timestamp"].to_numpy(),
                "forecast_step_idx": feature_df.loc[valid_mask, "forecast_step_idx"].to_numpy(),
                "fold_id": oof_fold_id[valid_mask],
                "target_name": target,
                "y_true": (tft_pred[valid_mask] + y[valid_mask]).astype(np.float32),
                "tft_pred": tft_pred[valid_mask].astype(np.float32),
                "residual_true": y[valid_mask].astype(np.float32),
                "catboost_oof_pred": oof_pred[valid_mask].astype(np.float32),
                "residual_after_cb": (y[valid_mask] - oof_pred[valid_mask]).astype(np.float32),
            })
            oof_frames.append(oof_frame)

            cv_metrics[target] = {
                "mode": "rolling",
                "coverage": float(valid_mask.mean()),
                "oof_rmse": overall_rmse,
                "oof_mae": overall_mae,
                "folds": fold_stats,
            }
            print(
                f"  {target} | OOF RMSE={overall_rmse:.4f} "
                f"OOF MAE={overall_mae:.4f} coverage={valid_mask.mean():.2%}"
            )
        else:
            split_idx = int(len(feature_df) * 0.8)
            split_idx = max(1, min(split_idx, len(feature_df_model) - 1))
            train_features = feature_df_model.iloc[:split_idx].copy()
            val_features = feature_df_model.iloc[split_idx:].copy()
            y_train = y[:split_idx]
            y_val = y[split_idx:]

            train_pool = Pool(train_features[RESIDUAL_FEATURE_COLUMNS], y_train,
                              cat_features=cat_features, weight=target_sample_weights[:split_idx])
            val_pool = Pool(val_features[RESIDUAL_FEATURE_COLUMNS], y_val,
                            cat_features=cat_features, weight=target_sample_weights[split_idx:])
            holdout_model = CatBoostRegressor(
                loss_function=loss_func,
                iterations=args.iterations,
                learning_rate=args.learning_rate,
                depth=args.depth,
                l2_leaf_reg=args.l2_leaf_reg,
                feature_weights=feature_weights,
                random_seed=42,
                verbose=100,
                early_stopping_rounds=50,
            )
            holdout_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
            raw_pred = holdout_model.predict(val_features[RESIDUAL_FEATURE_COLUMNS])
            val_pred = raw_pred[:, 0].astype(np.float32) if raw_pred.ndim == 2 else raw_pred.astype(np.float32)
            cv_metrics[target] = {
                "mode": "holdout",
                "rmse": _rmse(y_val, val_pred),
                "mae": _mae(y_val, val_pred),
                "train_size": int(len(train_features)),
                "val_size": int(len(val_features)),
            }
            print(
                f"  {target} | holdout RMSE={cv_metrics[target]['rmse']:.4f} "
                f"MAE={cv_metrics[target]['mae']:.4f}"
            )

        # Прод-модель обучаем на всем датасете residual-признаков
        full_pool = Pool(feature_df_model[RESIDUAL_FEATURE_COLUMNS], y,
                         cat_features=cat_features, weight=target_sample_weights)
        model_cb = CatBoostRegressor(
            loss_function=loss_func,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            l2_leaf_reg=args.l2_leaf_reg,
            feature_weights=feature_weights,
            random_seed=42,
            verbose=100,
        )
        model_cb.fit(full_pool)

        model_path = os.path.join(residual_models_dir, f"{target}.cbm")
        model_cb.save_model(model_path)
        print(f"  saved {target} -> {model_path}")

        # Обучаем 3 сегментированные по горизонтам модели (short=0-6ч, medium=6-24ч, long=24-48ч)
        segments = {
            "short": feature_df_model["forecast_step_idx"] < 24,
            "medium": (feature_df_model["forecast_step_idx"] >= 24) & (feature_df_model["forecast_step_idx"] < 96),
            "long": feature_df_model["forecast_step_idx"] >= 96,
        }

        for seg_name, mask in segments.items():
            if mask.sum() > 50:
                sub_features = feature_df_model[mask]
                sub_y = y[mask.to_numpy()]
                sub_weights = target_sample_weights[mask.to_numpy()]
                sub_pool = Pool(sub_features[RESIDUAL_FEATURE_COLUMNS], sub_y, cat_features=cat_features, weight=sub_weights)

                model_seg = CatBoostRegressor(
                    loss_function=loss_func,
                    iterations=min(args.iterations, 500),
                    learning_rate=args.learning_rate,
                    depth=min(args.depth, 7),
                    l2_leaf_reg=args.l2_leaf_reg,
                    feature_weights=feature_weights,
                    random_seed=42,
                    verbose=0,
                )
                model_seg.fit(sub_pool)
                seg_path = os.path.join(residual_models_dir, f"{target}_{seg_name}.cbm")
                model_seg.save_model(seg_path)
                print(f"  saved segmented model {target} ({seg_name}) -> {seg_path}")

        # Расчет Feature Importance
        imp_values = model_cb.get_feature_importance(full_pool)
        imp_sorted = sorted(
            zip(RESIDUAL_FEATURE_COLUMNS, [float(v) for v in imp_values]),
            key=lambda item: item[1],
            reverse=True,
        )
        feature_importance_report[target] = {k: round(v, 4) for k, v in imp_sorted}
        print(f"  [Top-5 Feature Importance: {target}]")
        for k, v in imp_sorted[:5]:
            print(f"    - {k}: {v:.2f}%")

    if oof_frames:
        oof_df = pd.concat(oof_frames, ignore_index=True)
        oof_path = os.path.join(residual_models_dir, "oof_predictions.parquet")
        oof_df.to_parquet(oof_path, index=False)
        metadata["oof_predictions_path"] = oof_path
        metadata["oof_rows"] = int(len(oof_df))

    imp_path = os.path.join(residual_models_dir, "feature_importance.json")
    with open(imp_path, "w", encoding="utf-8") as handle:
        json.dump(feature_importance_report, handle, indent=2)
    metadata["feature_importance_path"] = imp_path

    metadata["cv_metrics"] = cv_metrics
    with open(os.path.join(residual_models_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


if __name__ == "__main__":
    main()
