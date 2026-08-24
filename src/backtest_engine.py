import warnings
warnings.filterwarnings('ignore')
"""
=============================================================================
MODULE: Core Backtesting & Metric Calculation Engine (backtest_engine.py)
-----------------------------------------------------------------------------
PURPOSE:
Dedicated engine for rolling blind backtesting
and metric evaluation (MAE, RMSE, Bias) across forecast horizons.

KEY FUNCTIONS:
1. `run_station_backtest`: generates blind forecast trajectories over historical
   N days (1, 7, 30, 60) and benchmarks against ground truth sensor observations.
2. `calculate_metrics_by_horizons`: error metric evaluation across forecast horizons
   (1h, 6h, 12h, 18h, 24h, 36h, 48h).
=============================================================================
"""

import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Import resolution
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_fetcher import fetch_station_data
from data_pipeline import (EXTERNAL_FORECAST_COLUMNS, MODEL_SENSOR_COLUMNS,
                          MODEL_TARGET_COLUMNS, NORMALIZE_COLUMNS, apply_scalers,
                          load_combined_external_forecast, load_raw_station_json,
                          prepare_feature_frame, prepare_model_inputs,
                          select_stations_for_run)
from inversion import apply_inversion_correction, apply_rolling_bias_correction
from model import TFTForecaster
from project_paths import load_settings, resolve_path
from residual_engine import ResidualModelBundle, apply_residual_correction, load_residual_models


class KalmanStateObserver:
    """Kalman filter for sensor noise attenuation during backtesting."""
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
        
        innovation_std = np.sqrt(S[0, 0])
        z_score = abs(y[0, 0]) / max(innovation_std, 1e-6)
        if z_score > max_z_score:
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


MAX_CORRECTION_REAL = {
    "temperature": 3.0,
    "pressure": 5.0,
    "humidity": 15.0,
    "rain": 2.0,
    "wind_u": 3.0,
    "wind_v": 3.0
}


def calculate_metrics_by_horizons(
    df: pd.DataFrame,
    horizons_hours: List[int] = [1, 6, 12, 18, 24, 36, 48]
) -> Dict[str, Dict[str, Any]]:
    metrics_by_horizon = {}
    targets = ["temperature", "humidity", "pressure", "rain"]

    if df.empty:
        return metrics_by_horizon

    for h in horizons_hours:
        max_steps = int(h * 4)  # 15-minute steps (4 steps per hour)
        df_sub = df[df["horizon_step"] <= max_steps] if "horizon_step" in df.columns else df.copy()
        
        # Exact slice: 1-hour window at the horizon cutoff h
        min_exact = max(1, max_steps - 3)
        df_exact = df[(df["horizon_step"] >= min_exact) & (df["horizon_step"] <= max_steps)] if "horizon_step" in df.columns else df.copy()

        h_metrics = {}
        for tgt in targets:
            pred_col = f"{tgt}_pred"
            act_col = f"{tgt}_actual"
            if pred_col in df_sub.columns and act_col in df_sub.columns:
                valid = df_sub.dropna(subset=[pred_col, act_col])
                valid_exact = df_exact.dropna(subset=[pred_col, act_col])
                if not valid.empty:
                    y_pred = valid[pred_col].to_numpy().astype(float)
                    y_true = valid[act_col].to_numpy().astype(float)
                    mae = float(np.mean(np.abs(y_pred - y_true)))
                    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
                    bias = float(np.mean(y_pred - y_true))

                    exact_mae = mae
                    exact_rmse = rmse
                    exact_bias = bias
                    if not valid_exact.empty:
                        e_pred = valid_exact[pred_col].to_numpy().astype(float)
                        e_true = valid_exact[act_col].to_numpy().astype(float)
                        exact_mae = float(np.mean(np.abs(e_pred - e_true)))
                        exact_rmse = float(np.sqrt(np.mean((e_pred - e_true) ** 2)))
                        exact_bias = float(np.mean(e_pred - e_true))

                    h_metrics[tgt] = {
                        "MAE": mae,
                        "RMSE": rmse,
                        "Bias": bias,
                        "exact_MAE": exact_mae,
                        "exact_RMSE": exact_rmse,
                        "exact_Bias": exact_bias,
                        "count": len(valid)
                    }
        metrics_by_horizon[f"{h}h"] = h_metrics

    return metrics_by_horizon


def apply_full_correction_pipeline(
    preds_norm: np.ndarray,
    df_sensors: pd.DataFrame,
    df_ext_cutoff: pd.DataFrame,
    station: Dict[str, Any],
    station_scalers: Dict[str, Dict[str, float]],
    future_timestamps: pd.DatetimeIndex,
    residual_bundle: Optional[ResidualModelBundle],
    pid_params: Dict[str, Any],
    settings: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Full correction pipeline 1:1 identical to app.py production service:
    1. CatBoost Residual Boost (in normalized space)
    2. Kalman State Observer + 2-Stage PID Controller (in normalized space)
    3. Denormalization (Scaling Inversion)
    4. Magnus-Tetens Equation Coupling for humidity RH(T) (Block 7.2)
    5. Solar Zenith Angle Inversion Correction (Block 7.3)
    6. Rolling Bias Correction
    """
    target_cols = MODEL_TARGET_COLUMNS
    num_steps = len(future_timestamps)

    def _denorm(arr_norm):
        res_df = pd.DataFrame(arr_norm.copy(), columns=target_cols)
        for col in target_cols:
            if col in station_scalers:
                m_v = float(station_scalers[col]["mean"])
                s_v = float(station_scalers[col]["std"])
                res_df[col] = res_df[col] * s_v + m_v
        return res_df

    # 1. CatBoost Residual Correction (in normalized space)
    preds_corr_norm = preds_norm.copy()
    if residual_bundle is not None and residual_bundle.models:
        try:
            preds_corr_norm = apply_residual_correction(
                preds_norm=preds_norm,
                sensor_df=df_sensors.iloc[-96:] if len(df_sensors) >= 96 else df_sensors,
                forecast_df=df_ext_cutoff,
                station_meta=station,
                station_scalers=station_scalers,
                future_timestamps=future_timestamps,
                residual_bundle=residual_bundle,
            )
        except Exception:
            preds_corr_norm = preds_norm.copy()

    # 2. Kalman Observer + 2-Stage PID (in normalized space)
    df_sensors_norm = apply_scalers(df_sensors, station_scalers, NORMALIZE_COLUMNS)
    near_horizon_hours = settings.get("residual_boost", {}).get("near_horizon_hours", 6.0)
    split_step = int(near_horizon_hours * 4.0)
    blend_window_steps = int(settings.get("residual_boost", {}).get("blend_window_steps", 4))
    blend_scale = max(1.0, blend_window_steps / 2.0)

    for idx, var_name in enumerate(target_cols):
        if var_name == "rain_binary" or var_name not in station_scalers:
            continue

        std_v = float(station_scalers[var_name]["std"])
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

        # Weather Regime PID: Front detection based on barometric tendency dP/dt
        is_front_active = False
        if "pressure_trend_3h" in df_sensors.columns and len(df_sensors["pressure_trend_3h"]) > 0:
            p_trend = float(df_sensors["pressure_trend_3h"].iloc[-1])
            if not np.isnan(p_trend):
                is_front_active = abs(p_trend) > 1.5

        if is_front_active:
            Kp_short *= 1.40
            Kd_short *= 1.25

        if var_name == "rain":
            Kp_short = Ki_short = Kd_short = 0.0
            Kp_long = Ki_long = Kd_long = 0.0

        int_limit = float(params.get("int_limit", 10.0))

        # Kalman sensor noise filtering
        if var_name in df_sensors_norm.columns:
            kalman_obs = KalmanStateObserver(process_noise_std=0.05, measurement_noise_std=0.15)
            sensor_series = df_sensors_norm[var_name].values
            filtered_series, _ = kalman_obs.filter_series(sensor_series)
            smooth_actual_norm = float(filtered_series[-1])
        else:
            smooth_actual_norm = float(df_sensors_norm[var_name].iloc[-1]) if var_name in df_sensors_norm.columns else 0.0

        e_t0 = preds_corr_norm[0, idx] - smooth_actual_norm
        if np.isnan(e_t0):
            e_t0 = 0.0
        e_t0 = float(np.clip(e_t0, -1.5, 1.5))

        e_bias = 0.6 * e_t0
        e_noise = 0.4 * e_t0
        e_prev_bias = e_bias
        e_prev_noise = e_noise
        integral_sum = e_t0
        e_prev2 = e_t0

        for t in range(min(num_steps, preds_corr_norm.shape[0])):
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

            corr = Kp * e_prev + Ki * integral_sum + Kd * e_deriv
            soft_start = 1.0 - float(np.exp(-(t + 1) / 1.5))
            decay_factor = float(np.exp(-t / 12.0)) * soft_start
            corr = float(np.clip(corr * decay_factor, -max_corr_z, max_corr_z))
            preds_corr_norm[t, idx] = preds_corr_norm[t, idx] - corr
            e_prev2 = e_prev

    # 3. Denormalization
    preds_df = pd.DataFrame(preds_corr_norm, columns=target_cols)
    for col in target_cols:
        if col in station_scalers:
            m_v = float(station_scalers[col]["mean"])
            s_v = float(station_scalers[col]["std"])
            preds_df[col] = preds_df[col] * s_v + m_v

    # 4. Magnus-Tetens Humidity Coupling (Block 7.2)
    if "temperature" in preds_df.columns and "humidity" in preds_df.columns:
        temp_c = preds_df["temperature"].values
        humidity_raw = preds_df["humidity"].values
        dew_c = temp_c - ((100.0 - np.clip(humidity_raw, 1.0, 100.0)) / 5.0)
        es_t = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
        e_td = 6.112 * np.exp((17.67 * dew_c) / (dew_c + 243.5))
        rh_phys = np.clip((e_td / np.maximum(es_t, 1e-6)) * 100.0, 5.0, 100.0)
        preds_df["humidity"] = 0.7 * rh_phys + 0.3 * humidity_raw

    # 5. Solar Zenith Angle Inversion Correction (Block 7.3)
    if "temperature" in preds_df.columns:
        w_u = preds_df["wind_u"].values if "wind_u" in preds_df.columns else np.zeros(len(preds_df))
        w_v = preds_df["wind_v"].values if "wind_v" in preds_df.columns else np.zeros(len(preds_df))
        c_cover = df_ext_cutoff["cloud_cover"].values if "cloud_cover" in df_ext_cutoff.columns else None
        lat_deg = float(station.get("latitude", 40.2))
        preds_df["temperature"] = apply_inversion_correction(
            temp_raw=preds_df["temperature"].values,
            wind_u=w_u,
            wind_v=w_v,
            timestamps=future_timestamps.values,
            lat_deg=lat_deg,
            scale=1.0,
            cloud_cover=c_cover
        )

    # 6. Rolling Bias Correction (disabled offset inflation for TFT)
    pass

    pid_df = preds_df.copy()

    # 7. Diurnal Solar Peak Thermal Boost (Daytime T_max enhancement)
    if "temperature" in preds_df.columns:
        t_vals = preds_df["temperature"].values.copy()
        for i, ts in enumerate(future_timestamps):
            hr = ts.hour + ts.minute / 60.0
            if 11.0 <= hr <= 16.0:
                sun_boost = np.maximum(0.0, np.cos(np.radians(15.0 * (hr - 13.5)))) ** 1.5
                t_vals[i] += float(0.6 * sun_boost)
        preds_df["temperature"] = t_vals

    # 8. Night Warming Guard (Protects nocturnal temperature from dropping >2.5°C below NWP)
    if "temperature" in preds_df.columns and "temperature_2m" in df_ext_cutoff.columns:
        t_vals = preds_df["temperature"].values.copy()
        ext_t = df_ext_cutoff["temperature_2m"].values
        if len(ext_t) > 0 and len(ext_t) != len(t_vals):
            x_old = np.linspace(0, 1, len(ext_t))
            x_new = np.linspace(0, 1, len(t_vals))
            ext_t = np.interp(x_new, x_old, ext_t)
        min_len = min(len(t_vals), len(ext_t))
        for i in range(min_len):
            ts = future_timestamps[i]
            if 2 <= ts.hour <= 6:
                t_vals[i] = max(t_vals[i], float(ext_t[i]) - 2.5)
        preds_df["temperature"] = t_vals

    raw_df = _denorm(preds_norm)
    cb_df = _denorm(preds_corr_norm)
    pid_df = preds_df.copy()

    # 7. Diurnal Solar Peak Thermal Boost (Daytime T_max enhancement)
    if "temperature" in preds_df.columns:
        t_vals = preds_df["temperature"].values.copy()
        for i, ts in enumerate(future_timestamps):
            hr = ts.hour + ts.minute / 60.0
            if 11.0 <= hr <= 16.0:
                sun_boost = np.maximum(0.0, np.cos(np.radians(15.0 * (hr - 13.5)))) ** 1.5
                t_vals[i] += float(0.6 * sun_boost)
        preds_df["temperature"] = t_vals

    # 8. Night Warming Guard (Protects nocturnal temperature from dropping >2.5°C below NWP)
    if "temperature" in preds_df.columns and "temperature_2m" in df_ext_cutoff.columns:
        t_vals = preds_df["temperature"].values.copy()
        ext_t = df_ext_cutoff["temperature_2m"].values
        if len(ext_t) > 0 and len(ext_t) != len(t_vals):
            x_old = np.linspace(0, 1, len(ext_t))
            x_new = np.linspace(0, 1, len(t_vals))
            ext_t = np.interp(x_new, x_old, ext_t)
        min_len = min(len(t_vals), len(ext_t))
        for i in range(min_len):
            ts = future_timestamps[i]
            if 2 <= ts.hour <= 6:
                t_vals[i] = max(t_vals[i], float(ext_t[i]) - 2.5)
        preds_df["temperature"] = t_vals

    stage_dfs = {
        "raw": raw_df,
        "catboost": cb_df,
        "pid": pid_df,
        "final": preds_df
    }

    return preds_df, stage_dfs


class BacktestEngine:
    """In-sample inference engine for safe model loading and backtest execution."""

    def __init__(self):
        self.settings = load_settings()
        self.device = torch.device("cpu")
        self._model = None
        self._residual_bundle = None
        self._pid_params = {}
        self._scalers = {}
        self._stations = []
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return

        # 1. Load PID parameters
        pid_path = resolve_path(self.settings["paths"].get("pid_params_file", "config/pid_params.json"))
        if os.path.exists(pid_path):
            try:
                with open(pid_path, "r", encoding="utf-8") as pf:
                    self._pid_params = json.load(pf)
            except Exception as exc:
                print(f"BACKTEST_LOOP_EXC: {exc}")
                import traceback
                traceback.print_exc()

        # 2. Load station configuration and scalers
        stations_path = resolve_path(self.settings["paths"]["stations_config"])
        scalers_path = resolve_path(self.settings["paths"]["scalers_file"])

        if os.path.exists(stations_path):
            with open(stations_path, "r", encoding="utf-8") as f:
                self._stations = json.load(f).get("stations", [])

        if os.path.exists(scalers_path):
            with open(scalers_path, "r", encoding="utf-8") as f:
                self._scalers = json.load(f)

        # 3. Load TFT neural network
        model_filename = self.settings["paths"].get("model_filename", "tft_model.pth")
        model_path = resolve_path(self.settings["paths"]["models_dir"], model_filename)

        if os.path.exists(model_path):
            try:
                tft_cfg = self.settings.get("tft", {})
                state_dict = torch.load(model_path, map_location=self.device)
                num_dec_vars = len(EXTERNAL_FORECAST_COLUMNS)
                if "dec_vsn.softmax_grn.norm.weight" in state_dict:
                    num_dec_vars = state_dict["dec_vsn.softmax_grn.norm.weight"].shape[0]

                self._model = TFTForecaster(
                    num_decoder_vars=num_dec_vars,
                    hidden_size=tft_cfg.get("hidden_size", 128),
                    num_heads=tft_cfg.get("num_heads", 4),
                    num_lstm_layers=tft_cfg.get("num_lstm_layers", 2),
                    dropout=tft_cfg.get("dropout", 0.1)
                ).to(self.device)
                self._model.load_state_dict(state_dict)
                self._model.eval()
            except Exception as e:
                print(f"⚠️ Error loading TFT model for backtest: {e}")

        # 4. Load CatBoost Residuals bundle
        res_dir = resolve_path(self.settings["paths"].get("residual_models_dir", "models/residual_catboost"))
        if os.path.exists(res_dir):
            try:
                self._residual_bundle = load_residual_models(res_dir)
            except Exception as exc:
                print(f"BACKTEST_LOOP_EXC: {exc}")
                import traceback
                traceback.print_exc()

        self._initialized = True

    def run_backtest(
        self,
        station_id: int,
        days: int = 7,
        step_hours: int = 6,
        progress_callback: Optional[Any] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """
        Executes iterative rolling blind backtest over specified number of days
        with cutoff step (step_hours).
        """
        self.initialize()

        # Locate station metadata
        station = next((s for s in self._stations if s["id"] == station_id), None)
        if not station:
            return pd.DataFrame(), {}

        gen_id = station["generated_id"]
        sname = station["name"]
        station_key = f"station_{station_id}"

        processed_dir = resolve_path(self.settings["paths"]["processed_dir"])
        raw_stations_dir = resolve_path(self.settings["paths"]["raw_dir"])
        raw_ext_dir = resolve_path(self.settings["paths"]["external_dir"])

        parquet_file_id = os.path.join(processed_dir, f"station_{station_id}_features.parquet")
        parquet_file_gen = os.path.join(processed_dir, f"station_{gen_id}_features.parquet")
        raw_json_file_gen = os.path.join(raw_stations_dir, f"station_{gen_id}.json")
        raw_json_file_id = os.path.join(raw_stations_dir, f"station_{station_id}.json")

        df_sensors_all = pd.DataFrame()

        # 1. Fast loading from processed parquet (station_id or gen_id)
        if os.path.exists(parquet_file_id):
            try:
                df_sensors_all = pd.read_parquet(parquet_file_id)
            except Exception as exc:
                print(f"BACKTEST_LOOP_EXC: {exc}")
                import traceback
                traceback.print_exc()
        elif os.path.exists(parquet_file_gen):
            try:
                df_sensors_all = pd.read_parquet(parquet_file_gen)
            except Exception as exc:
                print(f"BACKTEST_LOOP_EXC: {exc}")
                import traceback
                traceback.print_exc()

        # 2. Fallback to raw JSON
        if df_sensors_all.empty and os.path.exists(raw_json_file_gen):
            try:
                df_raw = load_raw_station_json(raw_json_file_gen)
                if not df_raw.empty:
                    df_sensors_all = prepare_feature_frame(df_raw, station)
            except Exception as exc:
                print(f"BACKTEST_LOOP_EXC: {exc}")
                import traceback
                traceback.print_exc()

        if df_sensors_all.empty and os.path.exists(raw_json_file_id):
            try:
                df_raw = load_raw_station_json(raw_json_file_id)
                if not df_raw.empty:
                    df_sensors_all = prepare_feature_frame(df_raw, station)
            except Exception as exc:
                print(f"BACKTEST_LOOP_EXC: {exc}")
                import traceback
                traceback.print_exc()

        # 2. Fallback to ClimateNet API only if local files are missing
        if df_sensors_all.empty:
            now_dt = datetime.now()
            start_dt = now_dt - timedelta(days=int(days) + 7)
            try:
                api_res = fetch_station_data(gen_id, start_dt.strftime("%Y-%m-%d"), now_dt.strftime("%Y-%m-%d"))
                if api_res and "data" in api_res and "keys" in api_res and len(api_res["data"]) > 0:
                    df_raw_api = pd.DataFrame(api_res["data"], columns=api_res["keys"])
                    df_raw_api["timestamp"] = pd.to_datetime(df_raw_api["timestamp"], format="mixed")
                    df_sensors_all = prepare_feature_frame(df_raw_api, station)
            except Exception as exc:
                print(f"BACKTEST_LOOP_EXC: {exc}")
                import traceback
                traceback.print_exc()

        if df_sensors_all.empty:
            return pd.DataFrame(), {}

        df_sensors_all["timestamp"] = pd.to_datetime(df_sensors_all["timestamp"], format="mixed")
        df_sensors_all = df_sensors_all.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        df_ext_all = load_combined_external_forecast(raw_ext_dir, station_id, self.settings)
        if df_ext_all.empty:
            df_ext_all = load_combined_external_forecast(raw_ext_dir, gen_id, self.settings)
        if df_ext_all.empty:
            return pd.DataFrame(), {}

        df_ext_all["timestamp"] = pd.to_datetime(df_ext_all["timestamp"], format="mixed")

        max_ts = min(df_sensors_all["timestamp"].max(), df_ext_all["timestamp"].max())
        min_ts = max_ts - timedelta(days=int(days))

        lookback_steps = self.settings["model"]["lookback_steps"]
        horizon_steps = self.settings["model"]["horizon_steps"]
        station_scalers = self._scalers.get(station_key) or self._scalers.get(f'station_{gen_id}') or {}

        total_time_range = (max_ts - timedelta(hours=12) - min_ts).total_seconds()
        step_seconds = step_hours * 3600
        total_iterations = max(1, int(total_time_range / step_seconds))
        current_iter = 0

        results_records = []
        raw_records = []
        cb_records = []
        pid_records = []
        current_cutoff = min_ts

        while current_cutoff <= max_ts - timedelta(hours=12):
            if progress_callback:
                current_iter += 1
                progress_val = min(1.0, current_iter / total_iterations)
                progress_callback(progress_val)

            # Sensor history strictly prior to cutoff timestamp T
            df_sensor_cutoff = df_sensors_all[df_sensors_all["timestamp"] <= current_cutoff].tail(lookback_steps + 20).copy()
            if len(df_sensor_cutoff) < lookback_steps:
                current_cutoff += timedelta(hours=step_hours)
                continue

            # Future 48h external NWP forecast
            future_cutoff_end = current_cutoff + timedelta(hours=int(horizon_steps * 15 / 60))
            df_ext_cutoff = df_ext_all[
                (df_ext_all["timestamp"] >= current_cutoff) &
                (df_ext_all["timestamp"] <= future_cutoff_end)
            ].copy()

            if df_ext_cutoff.empty:
                current_cutoff += timedelta(hours=step_hours)
                continue

            try:
                enc_t, dec_t, future_timestamps = prepare_model_inputs(
                    df_sensor_cutoff,
                    df_ext_cutoff,
                    station,
                    station_scalers,
                    lookback_steps=lookback_steps,
                    horizon_steps=horizon_steps,
                    sensor_columns=MODEL_SENSOR_COLUMNS,
                    static_columns=["latitude", "longitude", "elevation_m"],
                    external_columns=EXTERNAL_FORECAST_COLUMNS,
                    normalize_columns=NORMALIZE_COLUMNS,
                )

                preds_norm = None
                if self._model is not None:
                    enc_tensor = torch.from_numpy(enc_t).float().to(self.device)
                    dec_tensor = torch.from_numpy(dec_t).float().to(self.device)

                    with torch.no_grad():
                        raw_pred = self._model(enc_tensor, dec_tensor).squeeze(0).cpu().numpy()
                        num_targets = len(MODEL_TARGET_COLUMNS)
                        if raw_pred.ndim == 2 and raw_pred.shape[1] == num_targets * 3:
                            preds_norm = raw_pred.reshape(raw_pred.shape[0], num_targets, 3)[:, :, 1]
                        else:
                            preds_norm = raw_pred

                    # Full correction cascade 1:1 identical to app.py production service
                    preds_df, stage_dfs = apply_full_correction_pipeline(
                        preds_norm=preds_norm,
                        df_sensors=df_sensor_cutoff,
                        df_ext_cutoff=df_ext_cutoff,
                        station=station,
                        station_scalers=station_scalers,
                        future_timestamps=future_timestamps,
                        residual_bundle=self._residual_bundle,
                        pid_params=self._pid_params.get(station_key) or self._pid_params.get(f'station_{gen_id}') or self._pid_params.get(sname) or {},
                        settings=self.settings,
                    )
                else:
                    preds_df = pd.DataFrame({
                        "temperature": df_ext_cutoff["temperature_2m"].values[:len(future_timestamps)],
                        "humidity": df_ext_cutoff["relative_humidity_2m"].values[:len(future_timestamps)],
                        "pressure": df_ext_cutoff["surface_pressure"].values[:len(future_timestamps)],
                        "rain": df_ext_cutoff["precipitation"].values[:len(future_timestamps)],
                    })

                # Extract ground truth observations
                df_actuals_horizon = df_sensors_all[
                    df_sensors_all["timestamp"].isin(future_timestamps)
                ].set_index("timestamp")

                for idx, ts in enumerate(future_timestamps):
                    rec = {
                        "timestamp": ts,
                        "cutoff_timestamp": current_cutoff,
                        "horizon_step": idx + 1,
                        "temperature_pred": float(preds_df["temperature"].iloc[idx]) if "temperature" in preds_df.columns else None,
                        "humidity_pred": float(preds_df["humidity"].iloc[idx]) if "humidity" in preds_df.columns else None,
                        "pressure_pred": float(preds_df["pressure"].iloc[idx]) if "pressure" in preds_df.columns else None,
                        "rain_pred": float(preds_df["rain"].iloc[idx]) if "rain" in preds_df.columns else 0.0,
                    }

                    if ts in df_actuals_horizon.index:
                        row_act = df_actuals_horizon.loc[ts]
                        if isinstance(row_act, pd.DataFrame):
                            row_act = row_act.iloc[0]
                        rec["temperature_actual"] = float(row_act["temperature"]) if "temperature" in row_act and pd.notna(row_act["temperature"]) else None
                        rec["humidity_actual"] = float(row_act["humidity"]) if "humidity" in row_act and pd.notna(row_act["humidity"]) else None
                        rec["pressure_actual"] = float(row_act["pressure"]) if "pressure" in row_act and pd.notna(row_act["pressure"]) else None
                        rec["rain_actual"] = float(row_act["rain"]) if "rain" in row_act and pd.notna(row_act["rain"]) else 0.0
                    else:
                        rec["temperature_actual"] = None
                        rec["humidity_actual"] = None
                        rec["pressure_actual"] = None
                        rec["rain_actual"] = None

                    results_records.append(rec)
                    
                    if "stage_dfs" in locals() and stage_dfs:
                        r_raw = rec.copy()
                        r_raw["temperature_pred"] = float(stage_dfs["raw"]["temperature"].iloc[idx]) if "temperature" in stage_dfs["raw"].columns else rec["temperature_pred"]
                        raw_records.append(r_raw)

                        r_cb = rec.copy()
                        r_cb["temperature_pred"] = float(stage_dfs["catboost"]["temperature"].iloc[idx]) if "temperature" in stage_dfs["catboost"].columns else rec["temperature_pred"]
                        cb_records.append(r_cb)

                        r_pid = rec.copy()
                        r_pid["temperature_pred"] = float(stage_dfs["pid"]["temperature"].iloc[idx]) if "temperature" in stage_dfs["pid"].columns else rec["temperature_pred"]
                        pid_records.append(r_pid)

            except Exception as exc:
                print(f"BACKTEST_LOOP_EXC: {exc}")
                import traceback
                traceback.print_exc()

            current_cutoff += timedelta(hours=step_hours)

        if not results_records:
            return pd.DataFrame(), {}

        df_res = pd.DataFrame(results_records)
        df_grouped = df_res.groupby("timestamp").mean(numeric_only=True).reset_index()
        metrics_by_horizon = calculate_metrics_by_horizons(df_res, horizons_hours=[1, 6, 12, 18, 24, 36, 48])

        # Component Impact Analysis across pipeline stages
        try:
            m_raw = calculate_metrics_by_horizons(pd.DataFrame(raw_records)) if raw_records else {}
            m_cb = calculate_metrics_by_horizons(pd.DataFrame(cb_records)) if cb_records else {}
            m_pid = calculate_metrics_by_horizons(pd.DataFrame(pid_records)) if pid_records else {}
            metrics_by_horizon["component_impact"] = {
                "raw_tft": m_raw,
                "catboost": m_cb,
                "pid": m_pid,
                "final": metrics_by_horizon
            }
        except Exception:
            pass

        return df_grouped, metrics_by_horizon


# Engine Singleton
_global_engine = BacktestEngine()

def run_backtest_service(station_id: int, days: int, step_hours: int, progress_callback: Optional[Any] = None):
    _global_engine.initialize()
    return _global_engine.run_backtest(station_id, days, step_hours, progress_callback)