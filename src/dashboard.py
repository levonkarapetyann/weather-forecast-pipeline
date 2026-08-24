import time
"""
=============================================================================
MODULE: Streamlit Web Frontend Dashboard (dashboard.py)
-----------------------------------------------------------------------------
PURPOSE:
Interactive user web interface for weather forecast visualization
and multi-station accuracy analytics.

KEY FUNCTIONS:
1. Forecast charts for temperature, humidity, pressure, wind, and precipitation
   with 15-minute, 1-hour, and 6-hour granularity.
2. Tab "🎯 Accuracy Benchmark": statistical error metric calculation
   (MAE, RMSE, Bias) comparing historical forecasts against actual ground truth.
3. Meteorological KPI cards, station connectivity status, and precipitation indicators.
=============================================================================
"""

import json
import os
import sys
from datetime import datetime, timedelta

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium

from project_paths import resolve_path
from backtest_engine import run_backtest_service

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# Utility: safe numeric formatting (returns '—' on None/NaN)
def safe_fmt(val, fmt="{:.1f}", na_str="—"):
    try:
        if val is None:
            return na_str
        # pandas NA handling
        if isinstance(val, (float,)) and pd.isna(val):
            return na_str
        return fmt.format(float(val))
    except Exception:
        return na_str


# Page configuration
st.set_page_config(
    page_title="ClimateNet Armenia Dashboard",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# UI styling (modern sleek responsive layout)
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 2px;
    }
    .subtitle {
        font-size: 16px;
        color: #64748b;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 5px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

API_BASE_URL = "http://127.0.0.1:8000"
REFRESH_SECONDS = 5

FORECAST_HISTORY_CSV = resolve_path("weather_data", "model_forecasts.csv")
ACTUAL_DATA_CSV = resolve_path("data", "raw", "stations", "all_stations_data.csv")

COMPARE_VARIABLES = {
    "temperature": {"label": "Temperature", "unit": "°C", "actual_col": "temperature"},
    "humidity": {"label": "Humidity", "unit": "%", "actual_col": "humidity"},
    "pressure": {"label": "Pressure", "unit": "hPa", "actual_col": "pressure"},
    "wind_speed": {"label": "Wind Speed", "unit": "m/s", "actual_col": "wind_speed"},
    "rain": {"label": "Precipitation", "unit": "mm", "actual_col": "rain"},
    "uv": {"label": "UV Index", "unit": "", "actual_col": "uv"},
    "lux": {"label": "Illuminance", "unit": "lux", "actual_col": "lux"},
    "pm2_5": {"label": "PM2.5", "unit": "µg/m³", "actual_col": "pm2_5"},
}

FORECAST_VALUE_COLUMNS = [
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_direction_degrees",
    "rain",
    "uv",
    "lux",
    "pm1",
    "pm2_5",
    "pm10",
]


st.cache_data(ttl=300)


def load_forecast_history(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    # Handles malformed lines gracefully
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    df["station_id"] = pd.to_numeric(df.get("station_id"), errors="coerce").astype("Int64")
    df["run_timestamp"] = pd.to_datetime(df["run_timestamp"], format="mixed", errors="coerce")
    df["forecast_datetime"] = pd.to_datetime(df["forecast_datetime"], format="mixed", errors="coerce")
    return df


@st.cache_data(ttl=300)
def load_actual_observations(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    # Handles malformed lines gracefully
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    df["station_id"] = pd.to_numeric(df.get("station_id"), errors="coerce").astype("Int64")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")
    df = df.rename(columns={"wind speed": "wind_speed"})
    return df


def _normalize_datetime_for_join(series: pd.Series) -> pd.Series:
    """
    Standardizes datetime keys for joining:
    - removes timezone,
    - rounds to 15-minute grid.
    """
    dt = pd.to_datetime(series, errors="coerce")
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)
    return dt.dt.floor("15min")


def _circular_mean_degrees(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    radians = np.deg2rad(values.astype(float))
    return float(np.rad2deg(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())) % 360)


FORECAST_CHART_INTERVALS = {
    "15 minutes": "15min",
    "1 hour": "1h",
    "6 hours per day": "6h_period",
}

FORECAST_CHART_VALUE_COLUMNS = ["temperature", "humidity", "pressure", "rain", "will_rain"]
FORECAST_MODEL_VALUE_COLUMNS = ["temperature", "humidity", "pressure", "rain"]


def _six_hour_period_start(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    date = ts.dt.normalize()
    period_hour = (ts.dt.hour // 6) * 6
    return date + pd.to_timedelta(period_hour, unit="h")


def _format_six_hour_period_label(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return ""
    period_end = (ts + pd.Timedelta(hours=6)).strftime("%H:%M")
    return f"{ts.strftime('%d.%m.%Y')} {ts.strftime('%H:%M')}–{period_end}"


def _aggregate_series_value(series: pd.Series, column: str):
    if column == "will_rain":
        flags = series.map(lambda value: value in (True, "True", "true", 1, "1")).astype(float)
        return bool(flags.mean() >= 0.5) if not flags.empty else False
    return pd.to_numeric(series, errors="coerce").mean()


def aggregate_timeseries_by_interval(
    df: pd.DataFrame,
    mode: str,
    value_columns: list[str],
    datetime_col: str = "timestamp",
    extra_aggregations: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Aggregates time series: 15-min (as is), 1-hour (mean), 6-hour blocks per day (mean)."""
    if df.empty or mode == "15min":
        return df.copy()

    work = df.copy()
    work[datetime_col] = pd.to_datetime(work[datetime_col], errors="coerce")
    if mode == "1h":
        work["bucket"] = work[datetime_col].dt.floor("h")
    elif mode == "6h_period":
        work["bucket"] = _six_hour_period_start(work[datetime_col])
    else:
        return work

    rows = []
    for bucket, group in work.groupby("bucket", sort=True):
        row = {datetime_col: bucket}
        for col in value_columns:
            if col in group.columns:
                row[col] = _aggregate_series_value(group[col], col)
        if extra_aggregations:
            for col, agg in extra_aggregations.items():
                if col not in group.columns:
                    continue
                series = group[col]
                if agg == "max":
                    row[col] = series.max()
                elif agg == "min":
                    row[col] = series.min()
                else:
                    row[col] = pd.to_numeric(series, errors="coerce").mean()
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_forecast_timeseries(
    df: pd.DataFrame,
    mode: str,
    value_columns: list[str],
) -> pd.DataFrame:
    return aggregate_timeseries_by_interval(df, mode, value_columns, datetime_col="timestamp")


def aggregate_compare_timeseries(
    df: pd.DataFrame,
    mode: str,
    forecast_col: str,
    actual_col: str,
) -> pd.DataFrame:
    value_columns = [col for col in (forecast_col, actual_col) if col in df.columns]
    extra = {}
    if "forecast_run_count" in df.columns:
        extra["forecast_run_count"] = "mean"
    if "run_timestamp_last" in df.columns:
        extra["run_timestamp_last"] = "max"
    return aggregate_timeseries_by_interval(
        df,
        mode,
        value_columns,
        datetime_col="forecast_datetime",
        extra_aggregations=extra or None,
    )


def _interval_step_label(mode: str) -> str:
    if mode == "1h":
        return "1 hour"
    if mode == "6h_period":
        return "6 hours"
    return "15 min"


def _interval_chart_caption(mode: str) -> str:
    if mode == "1h":
        return "Average value for each hour."
    if mode == "6h_period":
        return (
            "Average value for daily intervals: "
            "00:00–06:00, 06:00–12:00, 12:00–18:00, 18:00–24:00."
        )
    return "Native forecast step is 15 minutes."


def render_forecast_timeseries_charts(
    df_final: pd.DataFrame,
    df_model: pd.DataFrame,
    has_model: bool,
    interval_mode: str,
    actual_temp: float | None = None,
) -> None:
    df_final_plot = aggregate_forecast_timeseries(
        df_final, interval_mode, FORECAST_CHART_VALUE_COLUMNS
    )
    df_model_plot = (
        aggregate_forecast_timeseries(df_model, interval_mode, FORECAST_MODEL_VALUE_COLUMNS)
        if has_model
        else pd.DataFrame()
    )

    st.caption(_interval_chart_caption(interval_mode))

    COLOR_MODEL = "#94a3b8"
    COLOR_FINAL = "#f43f5e"

    xaxis_extra = {}
    if interval_mode == "6h_period" and not df_final_plot.empty:
        tickvals = df_final_plot["timestamp"].tolist()
        ticktext = [_format_six_hour_period_label(ts) for ts in tickvals]
        xaxis_extra = dict(tickmode="array", tickvals=tickvals, ticktext=ticktext, tickangle=-25)

    # ── Chart 1: Temperature ─────────────────────────────────────────
    fig_temp = go.Figure()
    if has_model:
        fig_temp.add_trace(go.Scatter(
            x=df_model_plot["timestamp"], y=df_model_plot["temperature"],
            name="TFT Model (pre-PID)", line=dict(color=COLOR_MODEL, width=2, dash="dot")))
    fig_temp.add_trace(go.Scatter(
        x=df_final_plot["timestamp"], y=df_final_plot["temperature"],
        name="Final Forecast (PID)", line=dict(color=COLOR_FINAL, width=3)))
    if actual_temp is not None and not df_final_plot.empty:
        fig_temp.add_trace(go.Scatter(
            x=[df_final_plot["timestamp"].iloc[0]],
            y=[actual_temp],
            mode="markers+text",
            name="Sensor (Ground Truth)",
            text=[f"Sensor: {actual_temp:.1f}°C"],
            textposition="top left",
            marker=dict(color="#10b981", size=14, symbol="diamond")
        ))
    fig_temp.update_layout(
        title="🌡️ Temperature (°C)",
        xaxis_title="Time", yaxis_title="°C",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        xaxis=xaxis_extra or None,
    )
    st.plotly_chart(fig_temp, width='stretch')

    # ── Chart 2: Precipitation ──────────────────────────────────────
    fig_rain = go.Figure()
    if has_model:
        fig_rain.add_trace(go.Scatter(
            x=df_model_plot["timestamp"], y=df_model_plot["rain"],
            name="TFT Model (pre-PID)", line=dict(color=COLOR_MODEL, width=2, dash="dot")))
    fig_rain.add_trace(go.Scatter(
        x=df_final_plot["timestamp"], y=df_final_plot["rain"],
        name="Final Forecast (PID)", line=dict(color=COLOR_FINAL, width=3)))
    fig_rain.add_trace(go.Scatter(
        x=df_final_plot["timestamp"],
        y=df_final_plot.get("will_rain", pd.Series([0] * len(df_final_plot))).astype(int),
        name="Rain Expected (Yes=1)", mode="lines",
        line=dict(color="#14b8a6", width=1, dash="longdash"), line_shape="hv",
        yaxis="y2",
    ))
    fig_rain.update_layout(
        title="🌧️ Precipitation (mm)",
        xaxis_title="Time",
        yaxis=dict(title="mm"),
        yaxis2=dict(title="Rain (Yes/No)", overlaying="y", side="right", range=[-0.2, 2], showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        xaxis=xaxis_extra or None,
    )
    st.plotly_chart(fig_rain, width='stretch')

    # ── Chart 3: Humidity ───────────────────────────────────────────
    fig_hum = go.Figure()
    if has_model:
        fig_hum.add_trace(go.Scatter(
            x=df_model_plot["timestamp"], y=df_model_plot["humidity"],
            name="TFT Model (pre-PID)", line=dict(color=COLOR_MODEL, width=2, dash="dot")))
    fig_hum.add_trace(go.Scatter(
        x=df_final_plot["timestamp"], y=df_final_plot["humidity"],
        name="Final Forecast (PID)", line=dict(color="#10b981", width=3)))
    fig_hum.update_layout(
        title="💧 Relative Humidity (%)",
        xaxis_title="Time", yaxis_title="%",
        yaxis=dict(range=[0, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        xaxis=xaxis_extra or None,
    )
    st.plotly_chart(fig_hum, width='stretch')

    # ── Chart 4: Pressure ───────────────────────────────────────────
    fig_pres = go.Figure()
    if has_model:
        fig_pres.add_trace(go.Scatter(
            x=df_model_plot["timestamp"], y=df_model_plot["pressure"],
            name="TFT Model (pre-PID)", line=dict(color=COLOR_MODEL, width=2, dash="dot")))
    fig_pres.add_trace(go.Scatter(
        x=df_final_plot["timestamp"], y=df_final_plot["pressure"],
        name="Final Forecast (PID)", line=dict(color="#8b5cf6", width=3)))
    fig_pres.update_layout(
        title="📊 Atmospheric Pressure (hPa)",
        xaxis_title="Time", yaxis_title="hPa",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        xaxis=xaxis_extra or None,
    )
    st.plotly_chart(fig_pres, width='stretch')


def render_forecast_vs_actual_comparison(
    df_compare: pd.DataFrame,
    var_meta: dict,
    forecast_value_col: str,
    actual_value_col: str,
    interval_mode: str,
) -> None:
    df_plot = aggregate_compare_timeseries(
        df_compare, interval_mode, forecast_value_col, actual_value_col
    )
    if df_plot.empty:
        st.info("No data available for selected interval.")
        return

    period_start = df_plot["forecast_datetime"].min().strftime("%Y-%m-%d %H:%M")
    period_end = df_plot["forecast_datetime"].max().strftime("%Y-%m-%d %H:%M")
    avg_runs = (
        int(df_plot["forecast_run_count"].mean())
        if "forecast_run_count" in df_plot.columns
        else 1
    )
    st.markdown(
        f"**Comparison Period:** {period_start} - {period_end} "
        f"({len(df_plot)} points, step {_interval_step_label(interval_mode)}, "
        f"avg {avg_runs} forecast(s) per point)"
    )
    st.caption(_interval_chart_caption(interval_mode))

    actual_series = pd.to_numeric(df_plot[actual_value_col], errors="coerce")
    forecast_series = pd.to_numeric(df_plot[forecast_value_col], errors="coerce")
    timestamps_series = df_plot.get("forecast_datetime")
    metrics = compute_point_metrics(actual_series, forecast_series, timestamps_series)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Comparison Points", metrics["n"])
    with m2:
        st.metric("MAE", safe_fmt(metrics["mae"], "{:.2f}"))
    with m3:
        st.metric("RMSE", safe_fmt(metrics["rmse"], "{:.2f}"))
    with m4:
        st.metric("Bias", safe_fmt(metrics["bias"], "{:+.2f}"))

    unit_str = var_meta.get("unit", "")
    unit_fmt = f" {unit_str}" if unit_str else ""
    e1, e2 = st.columns(2)
    with e1:
        min_val_str = safe_fmt(metrics["min_err"], "{:.2f}") + unit_fmt
        min_time_str = f"at lead hour {metrics['min_err_time']}" if metrics["min_err_time"] else ""
        st.metric("Minimum Error", min_val_str, delta=min_time_str, delta_color="normal")
    with e2:
        max_val_str = safe_fmt(metrics["max_err"], "{:.2f}") + unit_fmt
        max_time_str = f"at lead hour {metrics['max_err_time']}" if metrics["max_err_time"] else ""
        st.metric("Maximum Error", max_val_str, delta=max_time_str, delta_color="inverse")

    unit_suffix = f" {var_meta['unit']}" if var_meta["unit"] else ""
    xaxis_extra = {}
    if interval_mode == "6h_period":
        tickvals = df_plot["forecast_datetime"].tolist()
        ticktext = [_format_six_hour_period_label(ts) for ts in tickvals]
        xaxis_extra = dict(tickmode="array", tickvals=tickvals, ticktext=ticktext, tickangle=-25)

    fig_compare = go.Figure()
    fig_compare.add_trace(go.Scatter(
        x=df_plot["forecast_datetime"],
        y=forecast_series,
        name="Forecast (mean, TFT + PID)",
        line=dict(color="#f43f5e", width=3),
    ))
    fig_compare.add_trace(go.Scatter(
        x=df_plot["forecast_datetime"],
        y=actual_series,
        name="Actual (Sensors)",
        line=dict(color="#2563eb", width=2),
    ))
    fig_compare.update_layout(
        title=f"{var_meta['label']}: Forecast vs Actual{unit_suffix}",
        xaxis_title="Time",
        yaxis_title=f"{var_meta['label']}{unit_suffix}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        xaxis=xaxis_extra or None,
    )
    st.plotly_chart(fig_compare, width='stretch')

    with st.expander("Comparison Table", expanded=False):
        table_cols = ["forecast_datetime", forecast_value_col, actual_value_col]
        if "forecast_run_count" in df_plot.columns:
            table_cols.insert(1, "forecast_run_count")
        if "run_timestamp_last" in df_plot.columns:
            table_cols.insert(2, "run_timestamp_last")
        table = df_plot[table_cols].copy()
        table["error"] = forecast_series - actual_series
        rename_map = {
            "forecast_datetime": "Time",
            "forecast_run_count": "Forecast Runs",
            "run_timestamp_last": "Last Run",
            forecast_value_col: "Forecast (Mean)",
            actual_value_col: "Actual",
            "error": "Error",
        }
        table = table.rename(columns=rename_map)
        st.dataframe(table, width='stretch', hide_index=True)


def average_forecasts_by_datetime(df_forecasts: pd.DataFrame) -> pd.DataFrame:
    """Averages all forecasts generated prior to target timestamp."""
    if df_forecasts.empty:
        return df_forecasts

    prior = df_forecasts[df_forecasts["run_timestamp"] <= df_forecasts["forecast_datetime"]]
    if prior.empty:
        prior = df_forecasts

    rows = []
    for forecast_datetime, group in prior.groupby("forecast_datetime", sort=True):
        row = {"forecast_datetime": forecast_datetime}
        for col in FORECAST_VALUE_COLUMNS:
            if col not in group.columns:
                continue
            if col == "wind_direction_degrees":
                row[col] = _circular_mean_degrees(group[col])
            else:
                row[col] = pd.to_numeric(group[col], errors="coerce").mean()

        if "will_rain" in group.columns:
            rain_flags = group["will_rain"].map(
                lambda value: value in (True, "True", "true", 1, "1")
            ).astype(float)
            row["will_rain"] = bool(rain_flags.mean() >= 0.5)

        row["forecast_run_count"] = int(len(group))
        row["run_timestamp_first"] = group["run_timestamp"].min()
        row["run_timestamp_last"] = group["run_timestamp"].max()
        rows.append(row)

    return pd.DataFrame(rows).sort_values("forecast_datetime").reset_index(drop=True)


def build_forecast_vs_actual(
    df_forecasts: pd.DataFrame,
    df_actual: pd.DataFrame,
    generated_id: int,
) -> pd.DataFrame:
    target_id = pd.to_numeric(pd.Series([generated_id]), errors="coerce").astype("Int64").iloc[0]
    station_fc = df_forecasts[df_forecasts["station_id"] == target_id].copy()
    station_act = df_actual[df_actual["station_id"] == target_id].copy()
    if station_fc.empty or station_act.empty:
        return pd.DataFrame()

    station_fc["join_datetime"] = _normalize_datetime_for_join(station_fc["forecast_datetime"])
    station_act["join_datetime"] = _normalize_datetime_for_join(station_act["timestamp"])

    averaged_fc = average_forecasts_by_datetime(station_fc)
    averaged_fc["join_datetime"] = _normalize_datetime_for_join(averaged_fc["forecast_datetime"])
    merged = averaged_fc.merge(
        station_act,
        on="join_datetime",
        how="inner",
        suffixes=("_forecast", "_actual"),
    )
    return merged.sort_values("forecast_datetime").reset_index(drop=True)


def compute_point_metrics(actual: pd.Series, forecast: pd.Series, timestamps: pd.Series = None) -> dict:
    mask = actual.notna() & forecast.notna()
    act = actual[mask].astype(float)
    pred = forecast[mask].astype(float)
    if act.empty:
        return {
            "mae": None, "rmse": None, "bias": None, "n": 0,
            "min_err": None, "min_err_time": None,
            "max_err": None, "max_err_time": None
        }

    abs_err = np.abs(pred - act)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((pred - act) ** 2)))
    bias = float(np.mean(pred - act))

    min_idx = abs_err.idxmin()
    max_idx = abs_err.idxmax()
    min_err = float(abs_err.loc[min_idx])
    max_err = float(abs_err.loc[max_idx])

    min_err_time = ""
    max_err_time = ""
    if timestamps is not None and not timestamps.empty and min_idx in timestamps.index and max_idx in timestamps.index:
        ts_min = pd.to_datetime(timestamps.loc[min_idx])
        ts_max = pd.to_datetime(timestamps.loc[max_idx])
        min_err_time = ts_min.strftime("%H:%M (%d.%m)") if pd.notna(ts_min) else ""
        max_err_time = ts_max.strftime("%H:%M (%d.%m)") if pd.notna(ts_max) else ""

    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "n": int(len(act)),
        "min_err": min_err,
        "min_err_time": min_err_time,
        "max_err": max_err,
        "max_err_time": max_err_time,
    }


def _safe_response_json(response):
    try:
        return response.json()
    except (ValueError, requests.exceptions.JSONDecodeError):
        return {}


def get_stations():
    """Queries station list from API or reads directly from JSON on failure."""
    try:
        res = requests.get(f"{API_BASE_URL}/stations", timeout=3)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass

    # Fallback file
    config_path = os.path.join("config", "stations.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)["stations"]
    return []


# Sidebar
st.sidebar.image("https://img.icons8.com/clouds/150/000000/weather.png", width=100)
st.sidebar.title("ClimateNet Armenia")
st.sidebar.markdown("Hyperlocal hybrid 48-hour weather forecasting system.")

# Auto refresh & model status
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Auto-refresh:** every {REFRESH_SECONDS}s")
components.html(
    f"<script>setTimeout(function() {{ window.location.reload(); }}, {REFRESH_SECONDS * 1000});</script>",
    height=0,
)
pid_mtime = None
model_mtime = None
try:
    mtime_resp = requests.get(f"{API_BASE_URL}/model_mtime", timeout=3)
    if mtime_resp.status_code == 200:
        mtime_json = mtime_resp.json()
        model_mtime = mtime_json.get("model_mtime")

    pid_resp = requests.get(f"{API_BASE_URL}/pid_mtime", timeout=3)
    if pid_resp.status_code == 200:
        pid_json = pid_resp.json()
        pid_mtime = pid_json.get("pid_mtime")

    if model_mtime:
        st.sidebar.markdown(
            f"**Model last loaded:** {datetime.fromtimestamp(model_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.sidebar.markdown("**Model last loaded:** unknown")

    if pid_mtime:
        st.sidebar.markdown(
            f"**PID last loaded:** {datetime.fromtimestamp(pid_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.sidebar.markdown("**PID last loaded:** unknown")
except Exception:
    st.sidebar.markdown("**Model/PID last loaded:** query error")

stations = get_stations()

if not stations:
    st.error("Failed to load station configuration. Check config/stations.json.")
else:
    # Header
    st.markdown('<div class="main-title">⛈️ ClimateNet Hyperlocal Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Interactive Weather Monitoring & Hyperlocal Forecasting Dashboard</div>', unsafe_allow_html=True)

    # 1. Interactive map at top of page
    st.subheader("Weather Stations Map")

    # Center map on Armenia
    m = folium.Map(location=[40.1872, 44.5152], zoom_start=8, tiles="CartoDB positron")

    # Add station markers
    for s in stations:
        color = "green" if s.get("Status", "online") == "online" else "red"
        popup_text = f"<b>{s['name']}</b><br>Elevation: {s.get('elevation_m', 'N/A')} m<br>Status: {s.get('Status', 'online')}"

        folium.Marker(
            location=[float(s["latitude"]), float(s["longitude"])],
            popup=popup_text,
            tooltip=s["name"],
            icon=folium.Icon(color=color, icon="info-sign")
        ).add_to(m)

    # Render map in Streamlit
    map_data = st_folium(m, height=400, width="100%")

    # Interactive station selection (map or dropdown)
    selected_station_name = stations[0]["name"]

    # If user clicked map marker, select it
    if map_data and map_data.get("last_object_clicked"):
        clicked_coords = map_data["last_object_clicked"]
        # Find station by coordinates
        for s in stations:
            if abs(float(s["latitude"]) - clicked_coords["lat"]) < 0.001 and abs(float(s["longitude"]) - clicked_coords["lng"]) < 0.001:
                selected_station_name = s["name"]
                break

    # Sidebar station selection dropdown (synced with map click)
    station_names = [s["name"] for s in stations]
    selected_index = station_names.index(selected_station_name) if selected_station_name in station_names else 0

    station_select = st.sidebar.selectbox("Select Weather Station:", station_names, index=selected_index)
    selected_station = next(s for s in stations if s["name"] == station_select)

    st.sidebar.markdown(f"""
    **Station Information:**
    * **Name:** {selected_station['name']}
    * **Region:** {selected_station.get('parent_name_en', 'Armenia')}
    * **Elevation:** {selected_station.get('elevation_m', 'N/A')} m
    * **Coordinates:** {float(selected_station['latitude']):.4f}, {float(selected_station['longitude']):.4f}
    * **Sensor Status:**
      * BME280 (Weather): `{selected_station.get('BME280', 'valid')}`
      * LTR390 (Light): `{selected_station.get('LTR390', 'valid')}`
      * PMS5003 (Particulates): `{selected_station.get('PMS5003', 'valid')}`
      * Wind (Wind): `{selected_station.get('Wind', 'valid')}`
    """)

    # Top-level navigation tabs
    st.write("---")
    tab_live, tab_backtest, tab_accuracy = st.tabs([
        "📡 Live Operational Forecast",
        "🧪 Blind Backtest",
        "🎯 History & Accuracy Benchmark"
    ])

    with tab_live:
        st.header(f"48-Hour Weather Forecast: {selected_station['name']}")

        with st.spinner("Loading forecast and computing PID corrections..."):
            try:
                # Primary forecast (final PID-corrected)
                res = requests.get(f"{API_BASE_URL}/forecast/{selected_station['id']}", timeout=40)
                # Components: raw model prior to PID (for chart comparison)
                res_comp = requests.get(f"{API_BASE_URL}/forecast_components/{selected_station['id']}", timeout=40)

                if res.status_code == 200:
                    forecast_data = res.json()
                    df_final = pd.DataFrame(forecast_data["forecast"])
                    df_final["timestamp"] = pd.to_datetime(df_final["timestamp"])

                    df_model = pd.DataFrame()
                    if res_comp.status_code == 200:
                        comp_data = res_comp.json()
                        df_model = pd.DataFrame({
                            "timestamp": pd.to_datetime(comp_data["timestamps"]),
                            "temperature": comp_data["model_temperature"],
                            "humidity":    comp_data["model_humidity"],
                            "rain":        comp_data["model_rain"],
                            "pressure":    comp_data["model_pressure"],
                        })
                    else:
                        st.warning(
                            f"⚠️ Failed to retrieve forecast components (code {res_comp.status_code}). "
                            f"Raw model chart unavailable. "
                            f"Restart `python main.py serve` and refresh the page."
                        )

                    has_model = not df_model.empty

                    with st.expander("🔍 API Request Diagnostics", expanded=False):
                        st.write(f"• `/forecast` → HTTP {res.status_code}")
                        st.write(f"• `/forecast_components` → HTTP {res_comp.status_code}")
                        st.write(
                            f"• TFT Model data loaded: **{'Yes' if has_model else 'No'}** ({len(df_model)} rows)")

                    actual_temp_val = forecast_data.get("actual_temperature")
                    now_data = df_final.iloc[0]
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        if actual_temp_val is not None:
                            st.metric(
                                "Temperature (Forecast)",
                                safe_fmt(now_data.get("temperature"), "{:.1f} °C"),
                                delta=f"Sensor (Actual): {actual_temp_val:.1f} °C",
                                delta_color="off"
                            )
                        else:
                            st.metric("Temperature", safe_fmt(now_data.get("temperature"), "{:.1f} °C"))
                    with col2:
                        st.metric("Humidity", safe_fmt(now_data.get("humidity"), "{:.0f} %"))
                    with col3:
                        st.metric("Pressure", safe_fmt(now_data.get("pressure"), "{:.0f} hPa"))
                    with col4:
                        gust = now_data.get("wind_gust")
                        gust_str = f" (gusts up to {gust:.1f} m/s)" if gust else ""
                        st.metric("Wind Speed", safe_fmt(now_data.get("wind_speed"), "{:.1f} m/s"), delta=gust_str, delta_color="off")
                    with col5:
                        df_rain = df_final[df_final["will_rain"] == True] if "will_rain" in df_final.columns else pd.DataFrame()
                        if not df_rain.empty:
                            first_rain = df_rain.iloc[0]
                            rain_time = pd.to_datetime(first_rain["timestamp"])
                            now_dt = datetime.now()
                            if rain_time.date() == now_dt.date():
                                time_str = f"Today at {rain_time.strftime('%H:%M')}"
                            elif rain_time.date() == (now_dt + timedelta(days=1)).date():
                                time_str = f"Tomorrow at {rain_time.strftime('%H:%M')}"
                            else:
                                time_str = rain_time.strftime("%b %d at %H:%M")
                            rain_state = f"Yes ({time_str})"
                            prob = first_rain.get("rain_probability", 0.0)
                            amount = first_rain.get("rain", 0.0)
                            rain_delta = f"Confidence {prob:.0f}% ({amount:.1f} mm)"
                        else:
                            rain_state = "No"
                            max_prob = df_final["rain_probability"].max() if "rain_probability" in df_final.columns else 0.0
                            rain_delta = f"No precipitation (max: {max_prob:.0f}%)"

                        st.metric("Rain Expected", rain_state, delta=rain_delta)

                    # --- COMPONENT CONTRIBUTION METRICS (LIVE DELTA CARDS) ---
                    if has_model and "temperature" in df_final.columns and "temperature" in df_model.columns and len(df_final) > 0 and len(df_model) > 0:
                        st.markdown("##### ⚡ Component Impact (CatBoost, PID, Peak Boost)")
                        raw_temp = float(df_model["temperature"].iloc[0])
                        final_temp = float(df_final["temperature"].iloc[0])
                        diff_temp = final_temp - raw_temp

                        c1_live, c2_live, c3_live = st.columns(3)
                        with c1_live:
                            st.metric(
                                "🟢 CatBoost + PID Contribution",
                                f"{final_temp:.1f} °C",
                                delta=f"{diff_temp:+.2f} °C vs raw model",
                                delta_color="normal"
                            )
                        with c2_live:
                            st.metric(
                                "🔵 Pre-correction Baseline (Raw TFT)",
                                f"{raw_temp:.1f} °C",
                                delta="TFT Baseline",
                                delta_color="off"
                            )
                        with c3_live:
                            if actual_temp_val is not None:
                                err_raw = abs(raw_temp - float(actual_temp_val))
                                err_final = abs(final_temp - float(actual_temp_val))
                                err_gain = err_raw - err_final
                                st.metric(
                                    "🎯 Sensor Accuracy",
                                    f"{err_final:.2f} °C error",
                                    delta=f"-{err_gain:.2f} °C error gain" if err_gain >= 0 else f"+{abs(err_gain):.2f} °C",
                                    delta_color="normal"
                                )
                            else:
                                st.metric(
                                    "☀️ Thermal Peak Boost",
                                    "Active",
                                    delta="Diurnal solar thermal boost",
                                    delta_color="normal"
                                )

                    # Hazard risk and barometric status alerts
                    baro = now_data.get("baro_status", "Stable")
                    is_frost = any(df_final.get("frost_risk", [False]))
                    is_fog = any(df_final.get("fog_risk", [False]))

                    cols_warn = st.columns(3)
                    with cols_warn[0]:
                        st.info(f"📊 Barometric Trend: **{baro}**")
                    with cols_warn[1]:
                        if is_frost:
                            st.error("❄️ **WARNING: High Frost Risk in next 48h!**")
                        else:
                            st.success("🌱 Frost Risk: NONE")
                    with cols_warn[2]:
                        if is_fog:
                            st.warning("🌫️ **WARNING: Elevated Fog Risk on horizon!**")
                        else:
                            st.success("☀️ Fog Risk: NONE")

                    # Render TFT Attention Weights Feature Importance
                    feat_imp = forecast_data.get("feature_importance")
                    if feat_imp and isinstance(feat_imp, dict):
                        with st.expander("🧠 TFT Feature Importance Analysis (Attention Weights)", expanded=False):
                            df_imp = pd.DataFrame(list(feat_imp.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True).tail(10)
                            fig_imp = go.Figure(go.Bar(
                                x=df_imp["Importance"],
                                y=df_imp["Feature"],
                                orientation='h',
                                marker=dict(color='#3b82f6')
                            ))
                            fig_imp.update_layout(
                                title="Top-10 Most Influential Features for this Forecast",
                                xaxis_title="Attention Weight",
                                height=300,
                                margin=dict(l=20, r=20, t=40, b=20)
                            )
                            st.plotly_chart(fig_imp, use_container_width=True)

                    st.subheader("Forecast Time Series Charts")

                    tab_15m, tab_1h, tab_6h = st.tabs(list(FORECAST_CHART_INTERVALS.keys()))

                    with tab_15m:
                        render_forecast_timeseries_charts(
                            df_final, df_model, has_model, FORECAST_CHART_INTERVALS["15 minutes"], actual_temp=actual_temp_val
                        )
                    with tab_1h:
                        render_forecast_timeseries_charts(
                            df_final, df_model, has_model, FORECAST_CHART_INTERVALS["1 hour"], actual_temp=actual_temp_val
                        )
                    with tab_6h:
                        render_forecast_timeseries_charts(
                            df_final, df_model, has_model, FORECAST_CHART_INTERVALS["6 hours per day"], actual_temp=actual_temp_val
                        )

                else:
                    err_body = _safe_response_json(res)
                    detail = err_body.get("detail") if isinstance(err_body, dict) else None
                    if not detail:
                        detail = (res.text or "").strip() or f"HTTP {res.status_code}"
                    st.warning(f"API Error retrieving forecast: {detail}")
                    st.info("Please ensure the FastAPI server is running (`python main.py serve`).")
            except Exception as e:
                st.error(f"Error handling charts or API: {e}")
                st.info("Ensure FastAPI server is running: `python main.py serve` on port 8000.")

    with tab_backtest:
        st.header(f"🧪 Blind Forecast Backtesting: {selected_station['name']}")
        st.markdown(
            "This mode executes **blind forecast simulation** over historical observation series. "
            "At each iteration, sensor history is truncated strictly to cutoff timestamp $T$, "
            "generating a 48-hour forecast (TFT + CatBoost Residuals + PID), benchmarked directly "
            "against actual physical station measurements to calculate exact **MAE** error metrics."
        )

        c1, c2, c3 = st.columns([2, 2, 2])
        with c1:
            days_label = st.selectbox(
                "Backtest Period:",
                options=["1 day (24 hours)", "7 days (1 week)", "30 days (1 month)", "60 days (2 months)"],
                index=1,
                key="backtest_days_select"
            )
            days_map = {
                "1 day (24 hours)": 1,
                "7 days (1 week)": 7,
                "30 days (1 month)": 30,
                "60 days (2 months)": 60
            }
            sel_days = days_map[days_label]

        with c2:
            step_label = st.selectbox(
                "Cutoff step (simulation stride):",
                options=["1 hour (max accuracy)", "6 hours (fast & overview)", "12 hours", "24 hours"],
                index=1,
                key="backtest_step_select"
            )
            step_map = {
                "1 hour (max accuracy)": 1,
                "6 hours (fast & overview)": 6,
                "12 hours": 12,
                "24 hours": 24
            }
            sel_step = step_map[step_label]

        with c3:
            st.write("")
            st.write("")
            run_btn = st.button("🚀 Run Backtest", use_container_width=True, type="primary")

        cache_key = f"bt_{selected_station['id']}_{sel_days}_{sel_step}"

        if run_btn:
            t_start = time.time()
            progress_bar = st.progress(0.0, text="🚀 Initializing backtest...")
            last_p = [0.0]

            def update_progress(val):
                if val - last_p[0] >= 0.03 or val >= 1.0:
                    last_p[0] = val
                    elapsed = time.time() - t_start
                    eta = (elapsed / val * (1.0 - val)) if val > 0.05 else 0.0
                    progress_bar.progress(
                        val,
                        text=f"⏳ Computing blind forecast: {int(val*100)}% | Elapsed: {int(elapsed)}s | Remaining: ~{int(eta)}s"
                    )

            try:
                df_bt, h_metrics = run_backtest_service(
                    station_id=selected_station["id"],
                    days=sel_days,
                    step_hours=sel_step,
                    progress_callback=update_progress
                )
                progress_bar.empty()
                st.session_state[cache_key] = (df_bt, h_metrics)
            except FileNotFoundError as fnf:
                progress_bar.empty()
                st.error(f"❌ Data files not found: {fnf}. Please collect historical station data.")
            except Exception as e:
                progress_bar.empty()
                st.error(f"❌ Error executing backtest: {e}")
                st.info("💡 Note: Verify models in `models/` and processed features in `data/processed/`.")

        if cache_key in st.session_state:
            df_bt, h_metrics = st.session_state[cache_key]
            if df_bt.empty:
                st.warning("No sensor readings or external forecasts found for selected period.")
            else:
                st.success(f"✅ Blind backtest complete! Found {len(df_bt)} comparison time points.")

                # --- MAE METRICS SUMMARY ---
                st.subheader("📊 Mean Absolute Error (MAE) - Average model error:")

                m24 = h_metrics.get("24h", {})
                m48 = h_metrics.get("48h", {})

                t_mae_24 = m24.get("temperature", {}).get("MAE")
                h_mae_24 = m24.get("humidity", {}).get("MAE")
                p_mae_24 = m24.get("pressure", {}).get("MAE")
                r_mae_24 = m24.get("rain", {}).get("MAE")

                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.metric("Temperature MAE (24h)", safe_fmt(t_mae_24, "{:.2f} °C"), help="Average temperature deviation over 24h horizon")
                with mc2:
                    st.metric("Humidity MAE (24h)", safe_fmt(h_mae_24, "{:.2f} %"), help="Average humidity deviation over 24h horizon")
                with mc3:
                    st.metric("Pressure MAE (24h)", safe_fmt(p_mae_24, "{:.2f} hPa"), help="Average atmospheric pressure deviation")
                with mc4:
                    st.metric("Precipitation MAE (24h)", safe_fmt(r_mae_24, "{:.2f} mm"), help="Average precipitation error")

                # --- METRICS BY FORECAST HORIZON ---
                st.markdown("#### ⏱️ Error Metrics by Forecast Horizon (MAE, RMSE, Bias)")
                st.caption("• **Exact MAE**: error at exact lead time. • **Cumul MAE**: cumulative error from 0 to N hours.")
                rows_metrics = []
                for h_key, h_data in h_metrics.items():
                    t_info = h_data.get("temperature", {})
                    h_info = h_data.get("humidity", {})
                    p_info = h_data.get("pressure", {})
                    r_info = h_data.get("rain", {})
                    rows_metrics.append({
                        "Horizon": h_key,
                        "Temp Exact MAE (°C)": safe_fmt(t_info.get("exact_MAE", t_info.get("MAE")), "{:.2f}"),
                        "Temp Cumul MAE (°C)": safe_fmt(t_info.get("MAE"), "{:.2f}"),
                        "Temp RMSE (°C)": safe_fmt(t_info.get("exact_RMSE", t_info.get("RMSE")), "{:.2f}"),
                        "Temp Bias (°C)": safe_fmt(t_info.get("exact_Bias", t_info.get("Bias")), "{:+.2f}"),
                        "Humidity MAE (%)": safe_fmt(h_info.get("exact_MAE", h_info.get("MAE")), "{:.2f}"),
                        "Pressure MAE (hPa)": safe_fmt(p_info.get("exact_MAE", p_info.get("MAE")), "{:.2f}"),
                        "Rain MAE (mm)": safe_fmt(r_info.get("exact_MAE", r_info.get("MAE")), "{:.2f}"),
                        "Count": t_info.get("count", 0)
                    })
                if rows_metrics:
                    st.dataframe(pd.DataFrame(rows_metrics), use_container_width=True, hide_index=True)

                # --- COMPONENT IMPACT BREAKDOWN ---
                impact_data = h_metrics.get("component_impact", {})
                if impact_data:
                    st.markdown("#### ⚡ Component Impact Analysis (CatBoost & PID Impact)")
                    m_raw_24 = impact_data.get("raw_tft", {}).get("24h", {}).get("temperature", {}).get("MAE")
                    m_cb_24 = impact_data.get("catboost", {}).get("24h", {}).get("temperature", {}).get("MAE")
                    m_pid_24 = impact_data.get("pid", {}).get("24h", {}).get("temperature", {}).get("MAE")
                    m_fin_24 = impact_data.get("final", {}).get("24h", {}).get("temperature", {}).get("MAE")

                    c_cb, c_pid, c_tot = st.columns(3)
                    with c_cb:
                        if m_raw_24 and m_cb_24:
                            g_cb = m_raw_24 - m_cb_24
                            pct_cb = (g_cb / m_raw_24) * 100.0 if m_raw_24 > 0 else 0.0
                            st.metric("🟢 CatBoost Contribution (24h)", f"{m_cb_24:.2f} °C", delta=f"-{g_cb:.2f} °C (-{pct_cb:.1f}%)")
                        else:
                            st.metric("🟢 CatBoost Contribution (24h)", "Active")

                    with c_pid:
                        if m_cb_24 and m_pid_24:
                            g_pid = m_cb_24 - m_pid_24
                            pct_pid = (g_pid / m_cb_24) * 100.0 if m_cb_24 > 0 else 0.0
                            st.metric("🔵 PID Controller Contribution", f"{m_pid_24:.2f} °C", delta=f"-{g_pid:.2f} °C (-{pct_pid:.1f}%)")
                        else:
                            st.metric("🔵 PID Controller Contribution", "Active")

                    with c_tot:
                        if m_raw_24 and m_fin_24:
                            g_tot = m_raw_24 - m_fin_24
                            pct_tot = (g_tot / m_raw_24) * 100.0 if m_raw_24 > 0 else 0.0
                            st.metric("🏆 Total Cascade Improvement", f"{m_fin_24:.2f} °C", delta=f"-{g_tot:.2f} °C (-{pct_tot:.1f}%)")
                        else:
                            st.metric("🏆 Total Cascade Improvement", "Active")

                # --- INTERACTIVE PLOTLY CHARTS ---
                st.markdown("#### 📈 Charts: Blind Forecast (Dashed) vs Ground Truth Actuals (Solid)")

                fig_bt = make_subplots(
                    rows=4, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=(
                        "🌡️ Temperature (°C)",
                        "💧 Relative Humidity (%)",
                        "📊 Atmospheric Pressure (hPa)",
                        "🌧️ Precipitation (mm)"
                    )
                )

                # Temperature
                if "temperature_actual" in df_bt.columns:
                    fig_bt.add_trace(go.Scatter(x=df_bt["timestamp"], y=df_bt["temperature_actual"], name="Sensor Actual (Temp)", line=dict(color="#1f77b4", width=2)), row=1, col=1)
                if "temperature_pred" in df_bt.columns:
                    fig_bt.add_trace(go.Scatter(x=df_bt["timestamp"], y=df_bt["temperature_pred"], name="Blind Forecast (Temp)", line=dict(color="#ff7f0e", width=2, dash="dash")), row=1, col=1)

                # Humidity
                if "humidity_actual" in df_bt.columns:
                    fig_bt.add_trace(go.Scatter(x=df_bt["timestamp"], y=df_bt["humidity_actual"], name="Sensor Actual (Humidity)", line=dict(color="#2ca02c", width=2)), row=2, col=1)
                if "humidity_pred" in df_bt.columns:
                    fig_bt.add_trace(go.Scatter(x=df_bt["timestamp"], y=df_bt["humidity_pred"], name="Blind Forecast (Humidity)", line=dict(color="#d62728", width=2, dash="dash")), row=2, col=1)

                # Pressure
                if "pressure_actual" in df_bt.columns:
                    fig_bt.add_trace(go.Scatter(x=df_bt["timestamp"], y=df_bt["pressure_actual"], name="Sensor Actual (Pressure)", line=dict(color="#9467bd", width=2)), row=3, col=1)
                if "pressure_pred" in df_bt.columns:
                    fig_bt.add_trace(go.Scatter(x=df_bt["timestamp"], y=df_bt["pressure_pred"], name="Blind Forecast (Pressure)", line=dict(color="#8c564b", width=2, dash="dash")), row=3, col=1)

                # Rain
                if "rain_actual" in df_bt.columns:
                    fig_bt.add_trace(go.Bar(x=df_bt["timestamp"], y=df_bt["rain_actual"], name="Sensor Actual (Rain)", marker_color="#17becf", opacity=0.6), row=4, col=1)
                if "rain_pred" in df_bt.columns:
                    fig_bt.add_trace(go.Scatter(x=df_bt["timestamp"], y=df_bt["rain_pred"], name="Forecast (Rain)", line=dict(color="#e377c2", width=2)), row=4, col=1)

                fig_bt.update_layout(
                    height=900,
                    showlegend=True,
                    hovermode="x unified",
                    template="plotly_white"
                )
                st.plotly_chart(fig_bt, use_container_width=True)
        else:
            st.info("Select period settings and click '🚀 Run Backtest' to compute MAE and generate charts.")

    with tab_accuracy:
        st.header("Forecast vs Ground Truth Actuals Benchmark")
        st.caption(
            "Comparison of final forecasts (TFT + PID) from `weather_data/model_forecasts.csv` "
            "against actual station observations from `data/raw/stations/all_stations_data.csv`. "
            "For each target timestamp, the mean across runs is evaluated."
        )

        df_forecast_history = load_forecast_history(FORECAST_HISTORY_CSV)
        df_actual_obs = load_actual_observations(ACTUAL_DATA_CSV)

        if df_forecast_history.empty:
            st.info(
                "Forecast history file not found. Run `python tests/forecast_saver.py --once` "
                "or wait for the periodic recording cycle."
            )
        elif df_actual_obs.empty:
            st.info(
                "Ground truth observations file not found. Ingest data using "
                "`python main.py collect`."
            )
        else:
            generated_id = selected_station.get("generated_id")
            df_compare = build_forecast_vs_actual(df_forecast_history, df_actual_obs, generated_id)

            if df_compare.empty:
                st.info(
                    f"For station '{selected_station['name']}' (device #{generated_id}) "
                    "no overlap between forecasts and actual observations for period."
                )
            else:
                compare_var = st.selectbox(
                    "Variable for comparison:",
                    options=list(COMPARE_VARIABLES.keys()),
                    format_func=lambda key: (
                        f"{COMPARE_VARIABLES[key]['label']} ({COMPARE_VARIABLES[key]['unit']})"
                        if COMPARE_VARIABLES[key]["unit"]
                        else COMPARE_VARIABLES[key]["label"]
                    ),
                    key="forecast_vs_actual_variable",
                )
                var_meta = COMPARE_VARIABLES[compare_var]
                actual_col = var_meta["actual_col"]
                forecast_value_col = (
                    compare_var if compare_var in df_compare.columns else f"{compare_var}_forecast"
                )
                actual_value_col = (
                    actual_col if actual_col in df_compare.columns else f"{actual_col}_actual"
                )

                compare_tab_15m, compare_tab_1h, compare_tab_6h = st.tabs(
                    list(FORECAST_CHART_INTERVALS.keys())
                )

                with compare_tab_15m:
                    render_forecast_vs_actual_comparison(
                        df_compare,
                        var_meta,
                        forecast_value_col,
                        actual_value_col,
                        FORECAST_CHART_INTERVALS["15 minutes"],
                    )
                with compare_tab_1h:
                    render_forecast_vs_actual_comparison(
                        df_compare,
                        var_meta,
                        forecast_value_col,
                        actual_value_col,
                        FORECAST_CHART_INTERVALS["1 hour"],
                    )
                with compare_tab_6h:
                    render_forecast_vs_actual_comparison(
                        df_compare,
                        var_meta,
                        forecast_value_col,
                        actual_value_col,
                        FORECAST_CHART_INTERVALS["6 hours per day"],
                    )

        # Accuracy Benchmark Section
        st.write("---")
        st.header("Forecast Accuracy Evaluation (Open-Meteo Comparison)")

        benchmark_file = os.path.join("data", "processed", "provider_benchmark.csv")
        if os.path.exists(benchmark_file):
            df_bench = pd.read_csv(benchmark_file)
            df_bench_station = df_bench[df_bench["station_id"] == selected_station["id"]]

            if not df_bench_station.empty:
                def fmt_num_3(v):
                    try:
                        if v is None or pd.isna(v):
                            return "—"
                        return f"{float(v):.3f}"
                    except Exception:
                        return "—"

                def fmt_signed_3(v):
                    try:
                        if v is None or pd.isna(v):
                            return "—"
                        return f"{float(v):+.3f}"
                    except Exception:
                        return "—"

                st.dataframe(
                    df_bench_station[["variable", "mae", "rmse", "bias"]]
                    .style.format({
                        "mae": fmt_num_3,
                        "rmse": fmt_num_3,
                        "bias": fmt_signed_3
                    }),
                    width='stretch',
                )
            else:
                st.info("Comparison metrics for this station are not yet computed. Run benchmark.")
        else:
            st.info("Comparison benchmark table not found. Run evaluation first via `python main.py benchmark`.")
