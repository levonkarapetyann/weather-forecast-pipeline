import time
"""
=============================================================================
МОДУЛЬ: Streamlit Web Frontend Dashboard (dashboard.py)
-----------------------------------------------------------------------------
НАЗНАЧЕНИЕ:
Интерактивный пользовательский веб-интерфейс для визуализации прогнозов погоды
и аналитики точности модели по метеостанциям.

ОСНОВНЫЕ ФУНКЦИИ:
1. Отрисовка графиков прогнозов температуры, влажности, давления, ветра и осадков
   с детализацией на 15 минут, 1 час и 6 часов.
2. Вкладка "🎯 Анализ точности": вычисление статистических метрик ошибки
   (MAE, RMSE, Bias) и сравнение прогнозов из таблицы Forecasts с фактом.
3. Отображение метеорологических метрик, статуса подключенных станций и дождя.
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


# Утилита: безопасное форматирование чисел (возвращает '—' при None/NaN)
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


# Настройка страницы
st.set_page_config(
    page_title="ClimateNet Armenia Dashboard",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стилизация интерфейса (С sleek темной/светлой темой)
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
    "temperature": {"label": "Температура", "unit": "°C", "actual_col": "temperature"},
    "humidity": {"label": "Влажность", "unit": "%", "actual_col": "humidity"},
    "pressure": {"label": "Давление", "unit": "hPa", "actual_col": "pressure"},
    "wind_speed": {"label": "Скорость ветра", "unit": "м/с", "actual_col": "wind_speed"},
    "rain": {"label": "Осадки", "unit": "мм", "actual_col": "rain"},
    "uv": {"label": "УФ-индекс", "unit": "", "actual_col": "uv"},
    "lux": {"label": "Освещённость", "unit": "lux", "actual_col": "lux"},
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
    # ИСПРАВЛЕНО: добавлен параметр on_bad_lines='skip'
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    df["station_id"] = pd.to_numeric(df.get("station_id"), errors="coerce").astype("Int64")
    df["run_timestamp"] = pd.to_datetime(df["run_timestamp"], format="mixed", errors="coerce")
    df["forecast_datetime"] = pd.to_datetime(df["forecast_datetime"], format="mixed", errors="coerce")
    return df


@st.cache_data(ttl=300)
def load_actual_observations(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    # ИСПРАВЛЕНО: добавлен параметр on_bad_lines='skip'
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    df["station_id"] = pd.to_numeric(df.get("station_id"), errors="coerce").astype("Int64")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")
    df = df.rename(columns={"wind speed": "wind_speed"})
    return df


def _normalize_datetime_for_join(series: pd.Series) -> pd.Series:
    """
    Делает ключ времени устойчивым для join:
    - убирает timezone (если есть),
    - округляет до сетки 15 минут.
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
    "15 минут": "15min",
    "1 час": "1h",
    "6 часов за день": "6h_period",
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
    """Агрегирует временной ряд: 15 мин (как есть), 1 ч (среднее), 6 ч блоки за день (среднее)."""
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
        return "1 час"
    if mode == "6h_period":
        return "6 часов"
    return "15 мин"


def _interval_chart_caption(mode: str) -> str:
    if mode == "1h":
        return "Среднее значение за каждый час."
    if mode == "6h_period":
        return (
            "Среднее значение за суточные интервалы: "
            "00:00–06:00, 06:00–12:00, 12:00–18:00, 18:00–24:00."
        )
    return "Исходный шаг прогноза — 15 минут."


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

    # ── График 1: Температура ───────────────────────────────────────
    fig_temp = go.Figure()
    if has_model:
        fig_temp.add_trace(go.Scatter(
            x=df_model_plot["timestamp"], y=df_model_plot["temperature"],
            name="Модель TFT (до PID)", line=dict(color=COLOR_MODEL, width=2, dash="dot")))
    fig_temp.add_trace(go.Scatter(
        x=df_final_plot["timestamp"], y=df_final_plot["temperature"],
        name="Финальный прогноз (PID)", line=dict(color=COLOR_FINAL, width=3)))
    if actual_temp is not None and not df_final_plot.empty:
        fig_temp.add_trace(go.Scatter(
            x=[df_final_plot["timestamp"].iloc[0]],
            y=[actual_temp],
            mode="markers+text",
            name="Датчик (Настоящие данные)",
            text=[f"Датчик: {actual_temp:.1f}°C"],
            textposition="top left",
            marker=dict(color="#10b981", size=14, symbol="diamond")
        ))
    fig_temp.update_layout(
        title="🌡️ Температура (°C)",
        xaxis_title="Время", yaxis_title="°C",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        xaxis=xaxis_extra or None,
    )
    st.plotly_chart(fig_temp, width='stretch')

    # ── График 2: Осадки ────────────────────────────────────────────
    fig_rain = go.Figure()
    if has_model:
        fig_rain.add_trace(go.Scatter(
            x=df_model_plot["timestamp"], y=df_model_plot["rain"],
            name="Модель TFT (до PID)", line=dict(color=COLOR_MODEL, width=2, dash="dot")))
    fig_rain.add_trace(go.Scatter(
        x=df_final_plot["timestamp"], y=df_final_plot["rain"],
        name="Финальный прогноз (PID)", line=dict(color=COLOR_FINAL, width=3)))
    fig_rain.add_trace(go.Scatter(
        x=df_final_plot["timestamp"],
        y=df_final_plot.get("will_rain", pd.Series([0] * len(df_final_plot))).astype(int),
        name="Дождь? (Да=1)", mode="lines",
        line=dict(color="#14b8a6", width=1, dash="longdash"), line_shape="hv",
        yaxis="y2",
    ))
    fig_rain.update_layout(
        title="🌧️ Осадки (мм)",
        xaxis_title="Время",
        yaxis=dict(title="мм"),
        yaxis2=dict(title="Дождь (Да/Нет)", overlaying="y", side="right", range=[-0.2, 2], showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        xaxis=xaxis_extra or None,
    )
    st.plotly_chart(fig_rain, width='stretch')

    # ── График 3: Влажность ─────────────────────────────────────────
    fig_hum = go.Figure()
    if has_model:
        fig_hum.add_trace(go.Scatter(
            x=df_model_plot["timestamp"], y=df_model_plot["humidity"],
            name="Модель TFT (до PID)", line=dict(color=COLOR_MODEL, width=2, dash="dot")))
    fig_hum.add_trace(go.Scatter(
        x=df_final_plot["timestamp"], y=df_final_plot["humidity"],
        name="Финальный прогноз (PID)", line=dict(color="#10b981", width=3)))
    fig_hum.update_layout(
        title="💧 Относительная влажность (%)",
        xaxis_title="Время", yaxis_title="%",
        yaxis=dict(range=[0, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        xaxis=xaxis_extra or None,
    )
    st.plotly_chart(fig_hum, width='stretch')

    # ── График 4: Давление ─────────────────────────────────────────
    fig_pres = go.Figure()
    if has_model:
        fig_pres.add_trace(go.Scatter(
            x=df_model_plot["timestamp"], y=df_model_plot["pressure"],
            name="Модель TFT (до PID)", line=dict(color=COLOR_MODEL, width=2, dash="dot")))
    fig_pres.add_trace(go.Scatter(
        x=df_final_plot["timestamp"], y=df_final_plot["pressure"],
        name="Финальный прогноз (PID)", line=dict(color="#8b5cf6", width=3)))
    fig_pres.update_layout(
        title="📊 Атмосферное давление (hPa)",
        xaxis_title="Время", yaxis_title="hPa",
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
        st.info("Нет данных для выбранного интервала.")
        return

    period_start = df_plot["forecast_datetime"].min().strftime("%Y-%m-%d %H:%M")
    period_end = df_plot["forecast_datetime"].max().strftime("%Y-%m-%d %H:%M")
    avg_runs = (
        int(df_plot["forecast_run_count"].mean())
        if "forecast_run_count" in df_plot.columns
        else 1
    )
    st.markdown(
        f"**Период сравнения:** {period_start} — {period_end} "
        f"({len(df_plot)} точек, шаг {_interval_step_label(interval_mode)}, "
        f"в среднем {avg_runs} прогноз(ов) на точку)"
    )
    st.caption(_interval_chart_caption(interval_mode))

    actual_series = pd.to_numeric(df_plot[actual_value_col], errors="coerce")
    forecast_series = pd.to_numeric(df_plot[forecast_value_col], errors="coerce")
    timestamps_series = df_plot.get("forecast_datetime")
    metrics = compute_point_metrics(actual_series, forecast_series, timestamps_series)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Точек сравнения", metrics["n"])
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
        min_time_str = f"на час {metrics['min_err_time']}" if metrics["min_err_time"] else ""
        st.metric("Минимальная ошибка", min_val_str, delta=min_time_str, delta_color="normal")
    with e2:
        max_val_str = safe_fmt(metrics["max_err"], "{:.2f}") + unit_fmt
        max_time_str = f"на час {metrics['max_err_time']}" if metrics["max_err_time"] else ""
        st.metric("Максимальная ошибка", max_val_str, delta=max_time_str, delta_color="inverse")

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
        name="Прогноз (среднее, TFT + PID)",
        line=dict(color="#f43f5e", width=3),
    ))
    fig_compare.add_trace(go.Scatter(
        x=df_plot["forecast_datetime"],
        y=actual_series,
        name="Факт (датчики)",
        line=dict(color="#2563eb", width=2),
    ))
    fig_compare.update_layout(
        title=f"{var_meta['label']}: прогноз vs факт{unit_suffix}",
        xaxis_title="Время",
        yaxis_title=f"{var_meta['label']}{unit_suffix}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        xaxis=xaxis_extra or None,
    )
    st.plotly_chart(fig_compare, width='stretch')

    with st.expander("Таблица сравнения", expanded=False):
        table_cols = ["forecast_datetime", forecast_value_col, actual_value_col]
        if "forecast_run_count" in df_plot.columns:
            table_cols.insert(1, "forecast_run_count")
        if "run_timestamp_last" in df_plot.columns:
            table_cols.insert(2, "run_timestamp_last")
        table = df_plot[table_cols].copy()
        table["error"] = forecast_series - actual_series
        rename_map = {
            "forecast_datetime": "Время",
            "forecast_run_count": "Число прогнозов",
            "run_timestamp_last": "Последний запуск",
            forecast_value_col: "Прогноз (среднее)",
            actual_value_col: "Факт",
            "error": "Ошибка",
        }
        table = table.rename(columns=rename_map)
        st.dataframe(table, width='stretch', hide_index=True)


def average_forecasts_by_datetime(df_forecasts: pd.DataFrame) -> pd.DataFrame:
    """Для каждого целевого времени усредняем все прогнозы, сделанные до него."""
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
    """Запрашивает список станций уAPI или читает напрямую из json в случае сбоя"""
    try:
        res = requests.get(f"{API_BASE_URL}/stations", timeout=3)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass

    # Резервный файл
    config_path = os.path.join("config", "stations.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)["stations"]
    return []


# Sidebar
st.sidebar.image("https://img.icons8.com/clouds/150/000000/weather.png", width=100)
st.sidebar.title("ClimateNet Armenia")
st.sidebar.markdown("Гиперлокальная гибридная система прогнозирования погоды на 48 часов.")

# Auto refresh & model status
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Автообновление:** каждые {REFRESH_SECONDS} секунд")
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
            f"**Последняя загрузка модели:** {datetime.fromtimestamp(model_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.sidebar.markdown("**Последняя загрузка модели:** неизвестна")

    if pid_mtime:
        st.sidebar.markdown(
            f"**Последняя загрузка PID:** {datetime.fromtimestamp(pid_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.sidebar.markdown("**Последняя загрузка PID:** неизвестна")
except Exception:
    st.sidebar.markdown("**Последняя загрузка модели/PID:** ошибка запроса")

stations = get_stations()

if not stations:
    st.error("Не удалось загрузить конфигурацию станций. Проверьте config/stations.json.")
else:
    # Заголовок
    st.markdown('<div class="main-title">⛈️ ClimateNet Hyperlocal Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Интерактивная панель мониторинга и прогнозирования погоды по станциям Армении</div>', unsafe_allow_html=True)

    # 1. Создаем интерактивную карту в верхней части страницы
    st.subheader("Карта метеостанций Армении")

    # Центрируем карту по Армении
    m = folium.Map(location=[40.1872, 44.5152], zoom_start=8, tiles="CartoDB positron")

    # Добавляем маркеры станций
    for s in stations:
        color = "green" if s.get("Status", "online") == "online" else "red"
        popup_text = f"<b>{s['name']}</b><br>Высота: {s.get('elevation_m', 'N/A')} м<br>Статус: {s.get('Status', 'online')}"

        folium.Marker(
            location=[float(s["latitude"]), float(s["longitude"])],
            popup=popup_text,
            tooltip=s["name"],
            icon=folium.Icon(color=color, icon="info-sign")
        ).add_to(m)

    # Рендерим карту в Streamlit
    map_data = st_folium(m, height=400, width="100%")

    # Интерактивный выбор станции (карта или выпадающий список)
    selected_station_name = stations[0]["name"]

    # Если пользователь кликнул маркер на карте, выбираем его
    if map_data and map_data.get("last_object_clicked"):
        clicked_coords = map_data["last_object_clicked"]
        # Находим станцию по координатам
        for s in stations:
            if abs(float(s["latitude"]) - clicked_coords["lat"]) < 0.001 and abs(float(s["longitude"]) - clicked_coords["lng"]) < 0.001:
                selected_station_name = s["name"]
                break

    # Выпадающий список выбора станции в сайдбаре (с синхронизацией клика по карте)
    station_names = [s["name"] for s in stations]
    selected_index = station_names.index(selected_station_name) if selected_station_name in station_names else 0

    station_select = st.sidebar.selectbox("Выберите метеостанцию:", station_names, index=selected_index)
    selected_station = next(s for s in stations if s["name"] == station_select)

    st.sidebar.markdown(f"""
    **Информация о станции:**
    * **Имя:** {selected_station['name']}
    * **Область:** {selected_station.get('parent_name_en', 'Armenia')}
    * **Высота:** {selected_station.get('elevation_m', 'N/A')} м
    * **Координаты:** {float(selected_station['latitude']):.4f}, {float(selected_station['longitude']):.4f}
    * **Статус сенсоров:**
      * BME280 (Погода): `{selected_station.get('BME280', 'valid')}`
      * LTR390 (Свет): `{selected_station.get('LTR390', 'valid')}`
      * PMS5003 (Пыль): `{selected_station.get('PMS5003', 'valid')}`
      * Wind (Ветер): `{selected_station.get('Wind', 'valid')}`
    """)

    # Главные вкладки верхнего уровня
    st.write("---")
    tab_live, tab_backtest, tab_accuracy = st.tabs([
        "📡 Оперативный прогноз (Live)",
        "🧪 Слепой Бэктест (Backtest)",
        "🎯 История и Оценка точности"
    ])

    with tab_live:
        st.header(f"Прогноз погоды на 48 часов: {selected_station['name']}")

        with st.spinner("Загрузка прогноза и вычисления PID..."):
            try:
                # Основной прогноз (финальный — только PID)
                res = requests.get(f"{API_BASE_URL}/forecast/{selected_station['id']}", timeout=40)
                # Компоненты: модель до PID (для сравнения на графиках)
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
                            f"⚠️ Не удалось получить компоненты прогноза (код {res_comp.status_code}). "
                            f"График модели до PID недоступен. "
                            f"Перезапустите `python src/app.py` и обновите страницу."
                        )

                    has_model = not df_model.empty

                    with st.expander("🔍 Статус запросов API", expanded=False):
                        st.write(f"• `/forecast` → HTTP {res.status_code}")
                        st.write(f"• `/forecast_components` → HTTP {res_comp.status_code}")
                        st.write(
                            f"• Данные модели TFT загружены: **{'Да' if has_model else 'Нет'}** ({len(df_model)} строк)")

                    actual_temp_val = forecast_data.get("actual_temperature")
                    now_data = df_final.iloc[0]
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        if actual_temp_val is not None:
                            st.metric(
                                "Температура (Прогноз)",
                                safe_fmt(now_data.get("temperature"), "{:.1f} °C"),
                                delta=f"Датчик (Факт): {actual_temp_val:.1f} °C",
                                delta_color="off"
                            )
                        else:
                            st.metric("Температура", safe_fmt(now_data.get("temperature"), "{:.1f} °C"))
                    with col2:
                        st.metric("Влажность", safe_fmt(now_data.get("humidity"), "{:.0f} %"))
                    with col3:
                        st.metric("Давление", safe_fmt(now_data.get("pressure"), "{:.0f} hPa"))
                    with col4:
                        gust = now_data.get("wind_gust")
                        gust_str = f" (порывы до {gust:.1f} м/с)" if gust else ""
                        st.metric("Ветер", safe_fmt(now_data.get("wind_speed"), "{:.1f} м/с"), delta=gust_str, delta_color="off")
                    with col5:
                        df_rain = df_final[df_final["will_rain"] == True] if "will_rain" in df_final.columns else pd.DataFrame()
                        if not df_rain.empty:
                            first_rain = df_rain.iloc[0]
                            rain_time = pd.to_datetime(first_rain["timestamp"])
                            now_dt = datetime.now()
                            if rain_time.date() == now_dt.date():
                                time_str = f"Сегодня в {rain_time.strftime('%H:%M')}"
                            elif rain_time.date() == (now_dt + timedelta(days=1)).date():
                                time_str = f"Завтра в {rain_time.strftime('%H:%M')}"
                            else:
                                time_str = rain_time.strftime("%d.%m в %H:%M")
                            rain_state = f"Да ({time_str})"
                            prob = first_rain.get("rain_probability", 0.0)
                            amount = first_rain.get("rain", 0.0)
                            rain_delta = f"Уверенность {prob:.0f}% ({amount:.1f} мм)"
                        else:
                            rain_state = "Нет"
                            max_prob = df_final["rain_probability"].max() if "rain_probability" in df_final.columns else 0.0
                            rain_delta = f"Без осадков (макс: {max_prob:.0f}%)"

                        st.metric("Будет дождь", rain_state, delta=rain_delta)

                    # --- МЕТРИЧЕСКИЕ КАРТОЧКИ ВКЛАДА КОМПОНЕНТОВ В ОПЕРАТИВНОМ ПРОГНОЗЕ (LIVE DELTA CARDS) ---
                    if has_model and "temperature" in df_final.columns and "temperature" in df_model.columns and len(df_final) > 0 and len(df_model) > 0:
                        st.markdown("##### ⚡ Вклад компонентов коррекции (CatBoost, PID, Peak Boost)")
                        raw_temp = float(df_model["temperature"].iloc[0])
                        final_temp = float(df_final["temperature"].iloc[0])
                        diff_temp = final_temp - raw_temp

                        c1_live, c2_live, c3_live = st.columns(3)
                        with c1_live:
                            st.metric(
                                "🟢 Вклад CatBoost + PID",
                                f"{final_temp:.1f} °C",
                                delta=f"{diff_temp:+.2f} °C от сырой модели",
                                delta_color="normal"
                            )
                        with c2_live:
                            st.metric(
                                "🔵 Прогноз до коррекций (Raw TFT)",
                                f"{raw_temp:.1f} °C",
                                delta="Базовый вектор TFT",
                                delta_color="off"
                            )
                        with c3_live:
                            if actual_temp_val is not None:
                                err_raw = abs(raw_temp - float(actual_temp_val))
                                err_final = abs(final_temp - float(actual_temp_val))
                                err_gain = err_raw - err_final
                                st.metric(
                                    "🎯 Точность к датчику",
                                    f"{err_final:.2f} °C ошибка",
                                    delta=f"-{err_gain:.2f} °C точность" if err_gain >= 0 else f"+{abs(err_gain):.2f} °C",
                                    delta_color="normal"
                                )
                            else:
                                st.metric(
                                    "☀️ Thermal Peak Boost",
                                    "Активен",
                                    delta="Дневной термо-прогрев",
                                    delta_color="normal"
                                )

                    # Предупреждения про барометрический статус, заморозки и туман
                    baro = now_data.get("baro_status", "Стабильное")
                    is_frost = any(df_final.get("frost_risk", [False]))
                    is_fog = any(df_final.get("fog_risk", [False]))

                    cols_warn = st.columns(3)
                    with cols_warn[0]:
                        st.info(f"📊 Динамика давления: **{baro}**")
                    with cols_warn[1]:
                        if is_frost:
                            st.error("❄️ **ВНИМАНИЕ: Риск заморозков в ближайшие 48ч!**")
                        else:
                            st.success("🌱 Риск заморозков: Отсутствует")
                    with cols_warn[2]:
                        if is_fog:
                            st.warning("🌫️ **ВНИМАНИЕ: Вероятность тумана на горизонте!**")
                        else:
                            st.success("☀️ Риск тумана: Отсутствует")

                    # Отображение важности фичей TFT (Attention Weights Feature Importance)
                    feat_imp = forecast_data.get("feature_importance")
                    if feat_imp and isinstance(feat_imp, dict):
                        with st.expander("🧠 Анализ важности признаков нейросети (TFT Feature Importance)", expanded=False):
                            df_imp = pd.DataFrame(list(feat_imp.items()), columns=["Признак", "Важность"]).sort_values("Важность", ascending=True).tail(10)
                            fig_imp = go.Figure(go.Bar(
                                x=df_imp["Важность"],
                                y=df_imp["Признак"],
                                orientation='h',
                                marker=dict(color='#3b82f6')
                            ))
                            fig_imp.update_layout(
                                title="Топ-10 самых влиятельных признаков для этого прогноза",
                                xaxis_title="Вес внимания (Attention Weight)",
                                height=300,
                                margin=dict(l=20, r=20, t=40, b=20)
                            )
                            st.plotly_chart(fig_imp, use_container_width=True)

                    st.subheader("Временные ряды прогнозов")

                    tab_15m, tab_1h, tab_6h = st.tabs(list(FORECAST_CHART_INTERVALS.keys()))

                    with tab_15m:
                        render_forecast_timeseries_charts(
                            df_final, df_model, has_model, FORECAST_CHART_INTERVALS["15 минут"], actual_temp=actual_temp_val
                        )
                    with tab_1h:
                        render_forecast_timeseries_charts(
                            df_final, df_model, has_model, FORECAST_CHART_INTERVALS["1 час"], actual_temp=actual_temp_val
                        )
                    with tab_6h:
                        render_forecast_timeseries_charts(
                            df_final, df_model, has_model, FORECAST_CHART_INTERVALS["6 часов за день"], actual_temp=actual_temp_val
                        )

                else:
                    err_body = _safe_response_json(res)
                    detail = err_body.get("detail") if isinstance(err_body, dict) else None
                    if not detail:
                        detail = (res.text or "").strip() or f"HTTP {res.status_code}"
                    st.warning(f"Ошибка API при получении прогноза: {detail}")
                    st.info("Пожалуйста, убедитесь, что сервер FastAPI запущен (`python src/app.py`).")
            except Exception as e:
                st.error(f"Ошибка при работе с графиками или API: {e}")
                st.info("Убедитесь, что вы запустили сервер FastAPI командой: `python src/app.py` на порту 8000.")

    with tab_backtest:
        st.header(f"🧪 Слепой Бэктест прогнозов: {selected_station['name']}")
        st.markdown(
            "В этом режиме выполняется **«слепая» симуляция прогнозирования** на прошедших данных. "
            "На каждом шаге входные данные датчиков обрезаются строго до даты отсечки $T$, "
            "строится 48-часовой прогноз (TFT + CatBoost Residuals + PID), который затем сопоставляется "
            "с реальными измерениями сенсоров станции для расчета точных ошибок **MAE** (на сколько ошибается модель)."
        )

        c1, c2, c3 = st.columns([2, 2, 2])
        with c1:
            days_label = st.selectbox(
                "Период бэктестирования:",
                options=["1 день (24 часа)", "7 дней (1 неделя)", "30 дней (1 месяц)", "60 дней (2 месяца)"],
                index=1,
                key="backtest_days_select"
            )
            days_map = {
                "1 день (24 часа)": 1,
                "7 дней (1 неделя)": 7,
                "30 дней (1 месяц)": 30,
                "60 дней (2 месяца)": 60
            }
            sel_days = days_map[days_label]

        with c2:
            step_label = st.selectbox(
                "Шаг отсечки (окно симуляции):",
                options=["1 час (макс. точность)", "6 часов (быстро & наглядно)", "12 часов", "24 часа"],
                index=1,
                key="backtest_step_select"
            )
            step_map = {
                "1 час (макс. точность)": 1,
                "6 часов (быстро & наглядно)": 6,
                "12 часов": 12,
                "24 часа": 24
            }
            sel_step = step_map[step_label]

        with c3:
            st.write("")
            st.write("")
            run_btn = st.button("🚀 Запустить бэктест", use_container_width=True, type="primary")

        cache_key = f"bt_{selected_station['id']}_{sel_days}_{sel_step}"

        if run_btn:
            t_start = time.time()
            progress_bar = st.progress(0.0, text="🚀 Инициализация бэктеста...")
            last_p = [0.0]

            def update_progress(val):
                if val - last_p[0] >= 0.03 or val >= 1.0:
                    last_p[0] = val
                    elapsed = time.time() - t_start
                    eta = (elapsed / val * (1.0 - val)) if val > 0.05 else 0.0
                    progress_bar.progress(
                        val,
                        text=f"⏳ Расчет слепого прогноза: {int(val*100)}% | Прошло: {int(elapsed)}с | Осталось: ~{int(eta)}с"
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
                st.error(f"❌ Файлы данных не найдены: {fnf}. Соберите исторические данные станций.")
            except Exception as e:
                progress_bar.empty()
                st.error(f"❌ Ошибка при выполнении бэктеста: {e}")
                st.info("💡 Рекомендация: Проверьте наличие моделей в `models/` и сохраненных признаков в `data/processed/`.")

        if cache_key in st.session_state:
            df_bt, h_metrics = st.session_state[cache_key]
            if df_bt.empty:
                st.warning("За выбранный период показаний датчиков или внешних прогнозов не найдено.")
            else:
                st.success(f"✅ Слепой бэктест завершен! Найдено {len(df_bt)} временных точек сравнения.")

                # --- СВОДКА МЕТРИК MAE ---
                st.subheader("📊 Средняя абсолютная ошибка (MAE) — На сколько ошибается модель:")

                m24 = h_metrics.get("24h", {})
                m48 = h_metrics.get("48h", {})

                t_mae_24 = m24.get("temperature", {}).get("MAE")
                h_mae_24 = m24.get("humidity", {}).get("MAE")
                p_mae_24 = m24.get("pressure", {}).get("MAE")
                r_mae_24 = m24.get("rain", {}).get("MAE")

                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.metric("MAE Температуры (24h)", safe_fmt(t_mae_24, "{:.2f} °C"), help="Среднее отклонение температуры на 24ч горизонте")
                with mc2:
                    st.metric("MAE Влажности (24h)", safe_fmt(h_mae_24, "{:.2f} %"), help="Среднее отклонение влажности на 24ч горизонте")
                with mc3:
                    st.metric("MAE Давления (24h)", safe_fmt(p_mae_24, "{:.2f} hPa"), help="Среднее отклонение атмосферного давления")
                with mc4:
                    st.metric("MAE Осадков (24h)", safe_fmt(r_mae_24, "{:.2f} мм"), help="Средняя ошибка количества осадков")

                # --- ДЕТАЛЬНАЯ ТАБЛИЦА МЕТРИК ПО ГОРИЗОНТАМ ---
                st.markdown("#### ⏱️ Детализация ошибок MAE, RMSE и Bias по горизонтам времени")
                st.caption("• **Exact MAE**: ошибка непосредственно на указанной часу прогноза. • **Cumul MAE**: накопительная ошибка от 0 до N часов.")
                rows_metrics = []
                for h_key, h_data in h_metrics.items():
                    t_info = h_data.get("temperature", {})
                    h_info = h_data.get("humidity", {})
                    p_info = h_data.get("pressure", {})
                    r_info = h_data.get("rain", {})
                    rows_metrics.append({
                        "Горизонт": h_key,
                        "Temp Exact MAE (°C)": safe_fmt(t_info.get("exact_MAE", t_info.get("MAE")), "{:.2f}"),
                        "Temp Cumul MAE (°C)": safe_fmt(t_info.get("MAE"), "{:.2f}"),
                        "Temp RMSE (°C)": safe_fmt(t_info.get("exact_RMSE", t_info.get("RMSE")), "{:.2f}"),
                        "Temp Bias (°C)": safe_fmt(t_info.get("exact_Bias", t_info.get("Bias")), "{:+.2f}"),
                        "Humidity MAE (%)": safe_fmt(h_info.get("exact_MAE", h_info.get("MAE")), "{:.2f}"),
                        "Pressure MAE (hPa)": safe_fmt(p_info.get("exact_MAE", p_info.get("MAE")), "{:.2f}"),
                        "Rain MAE (мм)": safe_fmt(r_info.get("exact_MAE", r_info.get("MAE")), "{:.2f}"),
                        "Точек": t_info.get("count", 0)
                    })
                if rows_metrics:
                    st.dataframe(pd.DataFrame(rows_metrics), use_container_width=True, hide_index=True)

                # --- ВЗВЕШЕННЫЙ ВКЛАД КОМПОНЕНТОВ (COMPONENT IMPACT) ---
                impact_data = h_metrics.get("component_impact", {})
                if impact_data:
                    st.markdown("#### ⚡ Вклад компонентов каскада коррекций (CatBoost & PID Impact)")
                    m_raw_24 = impact_data.get("raw_tft", {}).get("24h", {}).get("temperature", {}).get("MAE")
                    m_cb_24 = impact_data.get("catboost", {}).get("24h", {}).get("temperature", {}).get("MAE")
                    m_pid_24 = impact_data.get("pid", {}).get("24h", {}).get("temperature", {}).get("MAE")
                    m_fin_24 = impact_data.get("final", {}).get("24h", {}).get("temperature", {}).get("MAE")

                    c_cb, c_pid, c_tot = st.columns(3)
                    with c_cb:
                        if m_raw_24 and m_cb_24:
                            g_cb = m_raw_24 - m_cb_24
                            pct_cb = (g_cb / m_raw_24) * 100.0 if m_raw_24 > 0 else 0.0
                            st.metric("🟢 Вклад CatBoost (24h)", f"{m_cb_24:.2f} °C", delta=f"-{g_cb:.2f} °C (-{pct_cb:.1f}%)")
                        else:
                            st.metric("🟢 Вклад CatBoost (24h)", "Активен")

                    with c_pid:
                        if m_cb_24 and m_pid_24:
                            g_pid = m_cb_24 - m_pid_24
                            pct_pid = (g_pid / m_cb_24) * 100.0 if m_cb_24 > 0 else 0.0
                            st.metric("🔵 Вклад PID-регулятора", f"{m_pid_24:.2f} °C", delta=f"-{g_pid:.2f} °C (-{pct_pid:.1f}%)")
                        else:
                            st.metric("🔵 Вклад PID-регулятора", "Активен")

                    with c_tot:
                        if m_raw_24 and m_fin_24:
                            g_tot = m_raw_24 - m_fin_24
                            pct_tot = (g_tot / m_raw_24) * 100.0 if m_raw_24 > 0 else 0.0
                            st.metric("🏆 Общее улучшение каскада", f"{m_fin_24:.2f} °C", delta=f"-{g_tot:.2f} °C (-{pct_tot:.1f}%)")
                        else:
                            st.metric("🏆 Общее улучшение каскада", "Активно")

                # --- ИНТЕРАКТИВНЫЕ ГРАФИКИ PLOTLY ---
                st.markdown("#### 📈 Графики: Слепой прогноз (пунктир) vs Фактические показания датчиков (сплошная)")

                fig_bt = make_subplots(
                    rows=4, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=(
                        "🌡️ Температура (°C)",
                        "💧 Относительная влажность (%)",
                        "📊 Атмосферное давление (hPa)",
                        "🌧️ Осадки (мм)"
                    )
                )

                # Temperature
                if "temperature_actual" in df_bt.columns:
                    fig_bt.add_trace(go.Scatter(x=df_bt["timestamp"], y=df_bt["temperature_actual"], name="Факт датчика (Temp)", line=dict(color="#1f77b4", width=2)), row=1, col=1)
                if "temperature_pred" in df_bt.columns:
                    fig_bt.add_trace(go.Scatter(x=df_bt["timestamp"], y=df_bt["temperature_pred"], name="Слепой прогноз (Temp)", line=dict(color="#ff7f0e", width=2, dash="dash")), row=1, col=1)

                # Humidity
                if "humidity_actual" in df_bt.columns:
                    fig_bt.add_trace(go.Scatter(x=df_bt["timestamp"], y=df_bt["humidity_actual"], name="Факт датчика (Humidity)", line=dict(color="#2ca02c", width=2)), row=2, col=1)
                if "humidity_pred" in df_bt.columns:
                    fig_bt.add_trace(go.Scatter(x=df_bt["timestamp"], y=df_bt["humidity_pred"], name="Слепой прогноз (Humidity)", line=dict(color="#d62728", width=2, dash="dash")), row=2, col=1)

                # Pressure
                if "pressure_actual" in df_bt.columns:
                    fig_bt.add_trace(go.Scatter(x=df_bt["timestamp"], y=df_bt["pressure_actual"], name="Факт датчика (Pressure)", line=dict(color="#9467bd", width=2)), row=3, col=1)
                if "pressure_pred" in df_bt.columns:
                    fig_bt.add_trace(go.Scatter(x=df_bt["timestamp"], y=df_bt["pressure_pred"], name="Слепой прогноз (Pressure)", line=dict(color="#8c564b", width=2, dash="dash")), row=3, col=1)

                # Rain
                if "rain_actual" in df_bt.columns:
                    fig_bt.add_trace(go.Bar(x=df_bt["timestamp"], y=df_bt["rain_actual"], name="Факт осадков (Rain)", marker_color="#17becf", opacity=0.6), row=4, col=1)
                if "rain_pred" in df_bt.columns:
                    fig_bt.add_trace(go.Scatter(x=df_bt["timestamp"], y=df_bt["rain_pred"], name="Прогноз осадков (Rain)", line=dict(color="#e377c2", width=2)), row=4, col=1)

                fig_bt.update_layout(
                    height=900,
                    showlegend=True,
                    hovermode="x unified",
                    template="plotly_white"
                )
                st.plotly_chart(fig_bt, use_container_width=True)
        else:
            st.info("Выберите параметры периода и нажмите «🚀 Запустить бэктест» для визуализации и расчета MAE.")

    with tab_accuracy:
        st.header("Сравнение прогноза с реальными данными")
        st.caption(
            "Сопоставление финальных прогнозов (TFT + PID) из `weather_data/model_forecasts.csv` "
            "с фактическими показаниями станций из `data/raw/stations/all_stations_data.csv`. "
            "Для каждого целевого времени берётся среднее по всем часовым запускам до этого момента."
        )

        df_forecast_history = load_forecast_history(FORECAST_HISTORY_CSV)
        df_actual_obs = load_actual_observations(ACTUAL_DATA_CSV)

        if df_forecast_history.empty:
            st.info(
                "Файл истории прогнозов не найден. Запустите `python src/forecast_saver.py --once` "
                "или дождитесь первого часового цикла записи."
            )
        elif df_actual_obs.empty:
            st.info(
                "Файл фактических наблюдений не найден. Соберите данные командой "
                "`python src/coll.py` или `python src/collect_data.py`."
            )
        else:
            generated_id = selected_station.get("generated_id")
            df_compare = build_forecast_vs_actual(df_forecast_history, df_actual_obs, generated_id)

            if df_compare.empty:
                st.info(
                    f"Для станции «{selected_station['name']}» (устройство #{generated_id}) "
                    "нет пересечения прогнозов и фактических наблюдений за доступный период."
                )
            else:
                compare_var = st.selectbox(
                    "Показатель для сравнения:",
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
                        FORECAST_CHART_INTERVALS["15 минут"],
                    )
                with compare_tab_1h:
                    render_forecast_vs_actual_comparison(
                        df_compare,
                        var_meta,
                        forecast_value_col,
                        actual_value_col,
                        FORECAST_CHART_INTERVALS["1 час"],
                    )
                with compare_tab_6h:
                    render_forecast_vs_actual_comparison(
                        df_compare,
                        var_meta,
                        forecast_value_col,
                        actual_value_col,
                        FORECAST_CHART_INTERVALS["6 часов за день"],
                    )

        # Раздел оценки точности (Бенчмарк)
        st.write("---")
        st.header("Оценка точности прогнозов (Сравнение с Open-Meteo)")

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
                st.info("Метрики сравнения для данной станции еще не рассчитаны. Запустите benchmark.py.")
        else:
            st.info("Таблица сравнения точности отсутствует. Сначала проведите оценку точности с помощью `python src/benchmark.py`.")

