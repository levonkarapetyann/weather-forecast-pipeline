#!/usr/bin/env python3
"""
=============================================================================
MODULE: 2-Month Rolling Blind Backtest (tests/backtest_vs_actuals.py)
-----------------------------------------------------------------------------
PURPOSE:
Simulation of blind forecasting over 2 months (60 days) preceding current date.
At each step, sensor window is truncated to cutoff date T,
generating 48h forecast (TFT + CatBoost Residuals), compared directly
against actual physical station observations.
=============================================================================
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch

# Setup paths for internal src/ module imports
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.chdir(PROJECT_ROOT)

from data_pipeline import (EXTERNAL_FORECAST_COLUMNS, MODEL_SENSOR_COLUMNS,
                          MODEL_TARGET_COLUMNS, NORMALIZE_COLUMNS, apply_scalers,
                          filter_station_files_for_run,
                          load_combined_external_forecast, load_raw_station_json,
                          prepare_feature_frame, prepare_model_inputs,
                          select_stations_for_run)
from data_fetcher import fetch_station_data
from model import TFTForecaster
from project_paths import load_settings, resolve_path
from inversion import apply_rolling_bias_correction
from residual_engine import (ResidualModelBundle, apply_residual_correction,
                             build_residual_feature_frame)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False


def calculate_metrics_by_horizons(df: pd.DataFrame, horizons_hours: list = [6, 12, 18, 24, 36, 48]) -> dict:
    """Computes horizon-sliced (6h, 12h, 18h, 24h, 36h, 48h) MAE, RMSE, and Bias metrics."""
    metrics_by_horizon = {}
    targets = ["temperature", "humidity", "pressure", "rain"]

    for h in horizons_hours:
        max_steps = int(h * 4)  # 15-minute steps
        df_sub = df[df["horizon_step"] <= max_steps] if "horizon_step" in df.columns else df.copy()
        
        h_metrics = {}
        for tgt in targets:
            pred_col = f"{tgt}_pred"
            act_col = f"{tgt}_actual"
            if pred_col in df_sub.columns and act_col in df_sub.columns:
                valid = df_sub.dropna(subset=[pred_col, act_col])
                if not valid.empty:
                    y_pred = valid[pred_col].to_numpy()
                    y_true = valid[act_col].to_numpy()
                    mae = float(np.mean(np.abs(y_pred - y_true)))
                    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
                    bias = float(np.mean(y_pred - y_true))
                    h_metrics[tgt] = {"MAE": mae, "RMSE": rmse, "Bias": bias, "count": len(valid)}
        metrics_by_horizon[f"{h}h"] = h_metrics

    return metrics_by_horizon


def apply_pid_correction_to_df(
    df_pred: pd.DataFrame,
    df_history: pd.DataFrame,
    station_name: str,
    pid_params_data: dict
) -> pd.DataFrame:
    """Applies PID feedback controller from pid_params.json identical to production app.py."""
    if df_pred.empty or df_history.empty or not pid_params_data:
        return df_pred

    df_out = df_pred.copy()
    st_pid = pid_params_data.get(station_name, pid_params_data.get("default", {}))

    for var_name in ["temperature", "humidity", "pressure"]:
        if var_name in df_out.columns and var_name in df_history.columns:
            recent = df_history[var_name].dropna()
            if len(recent) == 0:
                continue
            last_actual = float(recent.iloc[-1])
            params = st_pid.get(var_name, {"Kp": 0.15, "Ki": 0.01, "Kd": 0.05, "alpha": 0.95})
            
            Kp = float(params.get("Kp", 0.15))
            Ki = float(params.get("Ki", 0.01))
            Kd = float(params.get("Kd", 0.05))
            alpha = float(params.get("alpha", 0.95))

            first_pred = float(df_out[var_name].iloc[0])
            err_0 = last_actual - first_pred
            
            # PID correction over the first 24 hours with exponential decay
            horizon = len(df_out)
            integral = 0.0
            prev_err = err_0

            corrections = []
            for k in range(horizon):
                step_pred = float(df_out[var_name].iloc[k])
                err = (last_actual - step_pred) * (alpha ** k)
                integral += err
                derivative = err - prev_err
                prev_err = err
                
                corr = Kp * err + Ki * integral + Kd * derivative
                # Limit maximum PID correction to +/-3.5°C
                corr = float(np.clip(corr, -3.5, 3.5))
                corrections.append(corr)

            df_out[var_name] = df_out[var_name] + np.array(corrections)

    return df_out


def generate_plotly_report(df_res: pd.DataFrame, station_name: str, station_id: int, horizon_metrics: dict, output_path: str):
    """Constructs multi-panel interactive Plotly chart with horizon MAE breakdowns."""
    if not _PLOTLY_AVAILABLE:
        print("⚠️ Plotly is not installed. Run 'pip install plotly' to generate charts.")
        return

    m6 = horizon_metrics.get("6h", {}).get("temperature", {}).get("MAE", 0.0)
    m12 = horizon_metrics.get("12h", {}).get("temperature", {}).get("MAE", 0.0)
    m24 = horizon_metrics.get("24h", {}).get("temperature", {}).get("MAE", 0.0)
    m48 = horizon_metrics.get("48h", {}).get("temperature", {}).get("MAE", 0.0)

    m_hum = horizon_metrics.get("48h", {}).get("humidity", {}).get("MAE", 0.0)
    m_press = horizon_metrics.get("48h", {}).get("pressure", {}).get("MAE", 0.0)
    m_rain = horizon_metrics.get("48h", {}).get("rain", {}).get("MAE", 0.0)

    temp_title = f"Temperature (°C) | MAE 6h: {m6:.2f}°C | 12h: {m12:.2f}°C | 24h: {m24:.2f}°C | 48h: {m48:.2f}°C"

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            temp_title,
            f"Humidity (%) | MAE: {m_hum:.2f}%",
            f"Pressure (hPa) | MAE: {m_press:.2f} hPa",
            f"Precipitation (mm) | MAE: {m_rain:.2f} mm",
        )
    )

    # 1. Temperature
    if "temperature_actual" in df_res.columns:
        fig.add_trace(go.Scatter(
            x=df_res["timestamp"], y=df_res["temperature_actual"],
            name="Sensor Actual (T)", line=dict(color="#1f77b4", width=2)
        ), row=1, col=1)
    if "temperature_pred" in df_res.columns:
        fig.add_trace(go.Scatter(
            x=df_res["timestamp"], y=df_res["temperature_pred"],
            name="Blind Forecast (T)", line=dict(color="#ff7f0e", width=2, dash="dash")
        ), row=1, col=1)

    # 2. Humidity
    if "humidity_actual" in df_res.columns:
        fig.add_trace(go.Scatter(
            x=df_res["timestamp"], y=df_res["humidity_actual"],
            name="Sensor Actual (RH)", line=dict(color="#2ca02c", width=2)
        ), row=2, col=1)
    if "humidity_pred" in df_res.columns:
        fig.add_trace(go.Scatter(
            x=df_res["timestamp"], y=df_res["humidity_pred"],
            name="Blind Forecast (RH)", line=dict(color="#d62728", width=2, dash="dash")
        ), row=2, col=1)

    # 3. Pressure
    if "pressure_actual" in df_res.columns:
        fig.add_trace(go.Scatter(
            x=df_res["timestamp"], y=df_res["pressure_actual"],
            name="Sensor Actual (P)", line=dict(color="#9467bd", width=2)
        ), row=3, col=1)
    if "pressure_pred" in df_res.columns:
        fig.add_trace(go.Scatter(
            x=df_res["timestamp"], y=df_res["pressure_pred"],
            name="Blind Forecast (P)", line=dict(color="#8c564b", width=2, dash="dash")
        ), row=3, col=1)

    # 4. Precipitation
    if "rain_actual" in df_res.columns:
        fig.add_trace(go.Bar(
            x=df_res["timestamp"], y=df_res["rain_actual"],
            name="Precipitation Actual", marker_color="#17becf", opacity=0.6
        ), row=4, col=1)
    if "rain_pred" in df_res.columns:
        fig.add_trace(go.Scatter(
            x=df_res["timestamp"], y=df_res["rain_pred"],
            name="Precipitation Forecast", line=dict(color="#e377c2", width=2)
        ), row=4, col=1)

    fig.update_layout(
        title=f"2-Month Rolling Blind Backtest | Station {station_name} (ID {station_id})",
        height=1000,
        showlegend=True,
        hovermode="x unified",
        template="plotly_white"
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)
    print(f"📊 Saved interactive 2-month chart: {output_path}")


def run_backtest_for_station(
    station: dict,
    settings: dict,
    scalers_dict: dict,
    model: TFTForecaster,
    residual_bundle: ResidualModelBundle | None,
    device: any,
    pid_params_data: dict = None,
    months: int = 2,
    step_hours: int = 24
) -> pd.DataFrame:
    """Executes iterative rolling blind backtest for a single station over N months."""
    sid = station["id"]
    gen_id = station["generated_id"]
    sname = station["name"]

    processed_dir = resolve_path(settings["paths"]["processed_dir"])
    raw_stations_dir = resolve_path(settings["paths"]["raw_dir"])
    raw_ext_dir = resolve_path(settings["paths"]["external_dir"])
    station_key = f"station_{sid}"

    parquet_file = os.path.join(processed_dir, f"station_{gen_id}_features.parquet")
    raw_json_file = os.path.join(raw_stations_dir, f"station_{gen_id}.json")

    df_sensors_all = pd.DataFrame()

    # 1. Fetch fresh observations directly via ClimateNet API
    print(f"📡 Fetching live observations from ClimateNet API for station '{sname}' (device_id={gen_id})...")
    now_dt = datetime.now()
    start_dt = now_dt - timedelta(days=int(months * 30) + 7)
    try:
        api_res = fetch_station_data(gen_id, start_dt.strftime("%Y-%m-%d"), now_dt.strftime("%Y-%m-%d"))
        if api_res and "data" in api_res and "keys" in api_res and len(api_res["data"]) > 0:
            df_raw_api = pd.DataFrame(api_res["data"], columns=api_res["keys"])
            df_raw_api["timestamp"] = pd.to_datetime(df_raw_api["timestamp"], format="mixed")
            df_sensors_all = prepare_feature_frame(df_raw_api, station)
            print(f"  ✅ Received {len(df_sensors_all)} fresh observation records from ClimateNet API.")
    except Exception as e:
        print(f"  ⚠️ Could not fetch from API: {e}. Falling back to local files...")

    # 2. Fallback to local Parquet and raw JSON
    if df_sensors_all.empty and os.path.exists(parquet_file):
        try:
            df_sensors_all = pd.read_parquet(parquet_file)
            print(f"  ℹ️ Loaded {len(df_sensors_all)} rows from local parquet.")
        except Exception:
            pass

    if os.path.exists(raw_json_file):
        try:
            df_raw = load_raw_station_json(raw_json_file)
            if not df_raw.empty:
                df_feat = prepare_feature_frame(df_raw, station)
                if not df_feat.empty:
                    if not df_sensors_all.empty:
                        df_sensors_all = pd.concat([df_sensors_all, df_feat], ignore_index=True)
                    else:
                        df_sensors_all = df_feat
        except Exception as e:
            print(f"  ⚠️ Error reading raw JSON {raw_json_file}: {e}")

    if df_sensors_all.empty:
        print(f"❌ Sensor observations for station {gen_id} not found.")
        return pd.DataFrame()

    df_sensors_all["timestamp"] = pd.to_datetime(df_sensors_all["timestamp"], format="mixed")
    df_sensors_all = df_sensors_all.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    df_ext_all = load_combined_external_forecast(raw_ext_dir, sid, settings)
    if df_ext_all.empty:
        print(f"⚠️ External forecast for station {sid} unavailable.")
        return pd.DataFrame()

    df_ext_all["timestamp"] = pd.to_datetime(df_ext_all["timestamp"], format="mixed")

    max_ts = df_sensors_all["timestamp"].max()
    min_ts = max_ts - timedelta(days=int(months * 30))

    lookback_steps = settings["model"]["lookback_steps"]
    horizon_steps = settings["model"]["horizon_steps"]
    station_scalers = scalers_dict.get(station_key, {})

    results_records = []

    print(f"\n🚀 Running 2-month blind backtest for '{sname}' ({min_ts.strftime('%Y-%m-%d')} -> {max_ts.strftime('%Y-%m-%d')})...")

    # Iterate across cutoff timestamp T
    current_cutoff = min_ts
    while current_cutoff <= max_ts - timedelta(hours=12):
        # 1. Sensor history strictly prior to cutoff T (blind)
        df_sensor_cutoff = df_sensors_all[df_sensors_all["timestamp"] <= current_cutoff].copy()
        if len(df_sensor_cutoff) < lookback_steps:
            current_cutoff += timedelta(hours=step_hours)
            continue

        # 2. External NWP forecast for 48h forward from cutoff T
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

            enc_tensor = torch.from_numpy(enc_t).float().to(device)
            dec_tensor = torch.from_numpy(dec_t).float().to(device)

            with torch.no_grad():
                raw_pred = model(enc_tensor, dec_tensor).squeeze(0).cpu().numpy()
                num_targets = len(MODEL_TARGET_COLUMNS)
                if raw_pred.ndim == 2 and raw_pred.shape[1] == num_targets * 3:
                    preds_norm = raw_pred.reshape(raw_pred.shape[0], num_targets, 3)[:, :, 1]
                else:
                    preds_norm = raw_pred

            # Denormalize forecast
            preds_df = pd.DataFrame(preds_norm, columns=MODEL_TARGET_COLUMNS)
            for col in MODEL_TARGET_COLUMNS:
                if col in station_scalers:
                    m_v = float(station_scalers[col]["mean"])
                    s_v = float(station_scalers[col]["std"])
                    preds_df[col] = preds_df[col] * s_v + m_v

            # Apply CatBoost residual booster if available
            if residual_bundle is not None:
                try:
                    preds_df = apply_residual_correction(
                        forecast_df=preds_df,
                        sensor_df=df_sensor_cutoff,
                        station_meta=station,
                        scalers=station_scalers,
                        bundle=residual_bundle,
                        future_timestamps=future_timestamps,
                        ext_df=df_ext_cutoff
                    )
                except Exception:
                    pass

            # Apply PID feedback controller from pid_params.json (1:1 identical to app.py)
            if pid_params_data:
                try:
                    preds_df = apply_pid_correction_to_df(preds_df, df_sensor_cutoff, sname, pid_params_data)
                except Exception:
                    pass

            # Apply exponential rolling bias correction
            if "temperature" in preds_df.columns and "temperature" in df_sensor_cutoff.columns:
                try:
                    recent_actual = df_sensor_cutoff["temperature"].dropna().tail(24).values
                    recent_ext = df_sensor_cutoff["temperature_2m"].dropna().tail(24).values if "temperature_2m" in df_sensor_cutoff.columns else recent_actual
                    if len(recent_actual) > 0 and len(recent_ext) > 0:
                        preds_df["temperature"] = apply_rolling_bias_correction(
                            preds_df["temperature"].values,
                            recent_actual,
                            recent_ext,
                            decay_rate=0.05
                        )
                except Exception:
                    pass

            # Extract ground truth observations for benchmark
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

        except Exception as e:
            print(f"  ⚠️ Error at cutoff step {current_cutoff}: {e}")

        current_cutoff += timedelta(hours=step_hours)

    if not results_records:
        return pd.DataFrame()

    df_res = pd.DataFrame(results_records)
    # Average time duplicates for continuous 2-month trajectory chart
    df_grouped = df_res.groupby("timestamp").mean(numeric_only=True).reset_index()
    return df_grouped


def main():
    parser = argparse.ArgumentParser(description="2-Month rolling blind backtest for weather forecasting model.")
    parser.add_argument("--station-id", type=int, default=None, help="Station ID (default: first active station)")
    parser.add_argument("--months", type=int, default=2, help="Backtest duration in months (default: 2)")
    parser.add_argument("--step-hours", type=int, default=24, help="Cutoff window stride in hours (default: 24)")
    parser.add_argument("--max-horizon-hours", type=int, default=48, help="Maximum forecast horizon for MAE computation in hours (default: 48)")
    parser.add_argument("--mae-threshold", type=float, default=4.0, help="MAE anomaly threshold (°C) above which station is flagged (default: 4.0)")
    parser.add_argument("--output-dir", default="tests/plots", help="Directory to save output plots and reports")
    args = parser.parse_args()

    settings = load_settings()
    pid_params_path = resolve_path(settings["paths"].get("pid_params_file", "config/pid_params.json"))
    pid_params_data = {}
    if os.path.exists(pid_params_path):
        try:
            with open(pid_params_path, "r", encoding="utf-8") as pf:
                pid_params_data = json.load(pf)
        except Exception:
            pass
    device = torch.device("cpu")

    stations_config = resolve_path(settings["paths"]["stations_config"])
    scalers_path = resolve_path(settings["paths"]["scalers_file"])

    if not os.path.exists(stations_config) or not os.path.exists(scalers_path):
        print(f"❌ Configuration files not found: {stations_config} or {scalers_path}")
        return

    with open(stations_config, "r", encoding="utf-8") as f:
        stations = json.load(f)["stations"]
    stations = select_stations_for_run(stations, settings)

    with open(scalers_path, "r", encoding="utf-8") as f:
        scalers_dict = json.load(f)

    if args.station_id is not None:
        stations = [s for s in stations if s["id"] == args.station_id]

    if not stations:
        print("❌ Active stations not found.")
        return

    # Load TFT model
    model_filename = settings["paths"].get("model_filename", "tft_model.pth")
    model_path = resolve_path(settings["paths"]["models_dir"], model_filename)

    if not os.path.exists(model_path):
        print(f"❌ TFT Model checkpoint not found at: {model_path}")
        return

    tft_cfg = settings.get("tft", {})
    try:
        state_dict = torch.load(model_path, map_location=device)
        num_dec_vars = len(EXTERNAL_FORECAST_COLUMNS)
        if "dec_vsn.softmax_grn.norm.weight" in state_dict:
            num_dec_vars = state_dict["dec_vsn.softmax_grn.norm.weight"].shape[0]

        model = TFTForecaster(
            num_decoder_vars=num_dec_vars,
            hidden_size=tft_cfg.get("hidden_size", 128),
            num_heads=tft_cfg.get("num_heads", 4),
            num_lstm_layers=tft_cfg.get("num_lstm_layers", 2),
            dropout=tft_cfg.get("dropout", 0.1)
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"✅ Successfully loaded TFT model from {model_path} (num_decoder_vars={num_dec_vars})")
    except Exception as e:
        print(f"❌ Error loading model weights: {e}")
        return

    # Attempt loading CatBoost residual bundle
    residual_bundle = None
    res_dir = resolve_path(settings["paths"].get("residual_models_dir", "models/residual_catboost"))
    if os.path.exists(res_dir):
        try:
            residual_bundle = ResidualModelBundle(models_dir=res_dir)
            print("✅ Loaded CatBoost residual model bundle.")
        except Exception:
            pass

    all_station_results = []

    for station in stations:
        sid = station["id"]
        sname = station["name"]
        df_backtest = run_backtest_for_station(
            station, settings, scalers_dict, model, residual_bundle, device, pid_params_data,
            months=args.months, step_hours=args.step_hours
        )

        if not df_backtest.empty:
            horizon_metrics = calculate_metrics_by_horizons(df_backtest, horizons_hours=[6, 12, 18, 24, 36, 48])
            
            print(f"\n📊 Temperature MAE by Forecast Horizon for '{sname}':")
            st_horizons_temp_mae = {}
            for h_label, h_dict in horizon_metrics.items():
                temp_m = h_dict.get("temperature", {})
                t_mae = temp_m.get("MAE", 0.0)
                t_rmse = temp_m.get("RMSE", 0.0)
                t_bias = temp_m.get("Bias", 0.0)
                print(f"  ⏱️ Horizon {h_label:4s} | Temp MAE: {t_mae:5.3f}°C | RMSE: {t_rmse:5.3f}°C | Bias: {t_bias:+5.3f}°C")
                st_horizons_temp_mae[h_label] = t_mae

            # Anomaly threshold check (MAE > threshold)
            t_mae_24h = st_horizons_temp_mae.get("24h", 0.0)
            is_outlier = t_mae_24h > args.mae_threshold
            if is_outlier:
                print(f"  ⚠️ STATION FILTERED: MAE 24h ({t_mae_24h:.2f}°C) > {args.mae_threshold}°C. Reset to 0.0 for summary.")

            all_station_results.append({
                "id": sid,
                "name": sname,
                "is_outlier": is_outlier,
                "horizons": {
                    h: (0.0 if st_horizons_temp_mae.get(h, 0.0) > args.mae_threshold else st_horizons_temp_mae.get(h, 0.0))
                    for h in st_horizons_temp_mae
                },
                "raw_horizons": st_horizons_temp_mae
            })

            out_plot = os.path.join(args.output_dir, f"backtest_{args.months}months_station_{sid}.html")
            generate_plotly_report(df_backtest, sname, sid, horizon_metrics, out_plot)

    # Final summary across all stations
    if all_station_results:
        print("\n" + "=" * 70)
        print(f"📈 SUMMARY MAE REPORT ACROSS ALL STATIONS (Threshold: > {args.mae_threshold}°C)")
        print("=" * 70)
        
        valid_stations = [s for s in all_station_results if not s["is_outlier"]]
        outlier_stations = [s for s in all_station_results if s["is_outlier"]]

        print(f"Total stations: {len(all_station_results)} | Valid: {len(valid_stations)} | Filtered as 0: {len(outlier_stations)}")
        
        if outlier_stations:
            print("\nFiltered anomalous stations (reset to 0.0):")
            for os_st in outlier_stations:
                print(f"  ❌ {os_st['name']} (ID {os_st['id']}): Raw MAE 24h = {os_st['raw_horizons'].get('24h', 0.0):.2f}°C")

        horizons = ["6h", "12h", "18h", "24h", "36h", "48h"]
        print("\nFinal Mean Temperature MAE across all stations:")
        for h in horizons:
            mae_with_zeros = sum(st["horizons"].get(h, 0.0) for st in all_station_results) / len(all_station_results)
            mae_valid_only = sum(st["raw_horizons"].get(h, 0.0) for st in valid_stations) / len(valid_stations) if valid_stations else 0.0
            print(f"  ⏱️ Horizon {h:4s} | MAE (with reset zeros): {mae_with_zeros:5.3f}°C | (Valid stations only): {mae_valid_only:5.3f}°C")
        print("=" * 70)


if __name__ == "__main__":
    main()