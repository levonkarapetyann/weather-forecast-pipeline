"""
=============================================================================
МОДУЛЬ: Offline Model Evaluation & Metrics (evaluate.py)
-----------------------------------------------------------------------------
НАЗНАЧЕНИЕ:
Скрипт расчёта оффлайн-метрик точности модели прогнозирования на тестовых выборках.

ОСНОВНЫЕ ФУНКЦИИ:
1. Вычисление метрик MAE, RMSE, MAPE и Bias по метеостанциям и горизонтам прогнозирования.
2. Детализация ошибки по времени суток (ночь, утро, день, вечер).
3. Формирование сводного отчёта в консоли и сохранение результатов в JSON.
=============================================================================
"""

import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model import ClimateDataset, TFTForecaster
from project_paths import resolve_path
from data_pipeline import (filter_station_files_for_run,
                          load_combined_external_forecast,
                          select_stations_for_run)
from inversion import apply_inversion_correction

HOUR_BUCKETS = {
    "night_00_06": (0, 6),
    "morning_06_12": (6, 12),
    "day_12_18": (12, 18),
    "evening_18_24": (18, 24)
}


def de_normalize(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Обратное Z-score масштабирование"""
    return data * std + mean


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Расчет MAE, RMSE и Bias (смещения) с исключением NaN"""
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    act_clean = actual[mask]
    pred_clean = predicted[mask]

    if len(act_clean) == 0:
        return {"mae": 0.0, "rmse": 0.0, "bias": 0.0}

    mae = np.mean(np.abs(pred_clean - act_clean))
    rmse = np.sqrt(np.mean((pred_clean - act_clean) ** 2))
    bias = np.mean(pred_clean - act_clean)

    return {"mae": float(mae), "rmse": float(rmse), "bias": float(bias)}


def calculate_rain_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Расчет метрик бинарной классификации осадков (ROC-AUC, F1-Score, Precision, Recall, Brier, CSI)."""
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss

    mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
    yt = y_true[mask]
    yp = y_prob[mask]

    if len(yt) == 0 or len(np.unique(yt)) < 2:
        return {"roc_auc": 0.5, "f1": 0.0, "precision": 0.0, "recall": 0.0, "brier": 0.0, "csi": 0.0}

    y_pred = (yp >= threshold).astype(int)

    # Critical Success Index (CSI / Threat Score) = TP / (TP + FP + FN)
    tp = np.sum((yt == 1) & (y_pred == 1))
    fp = np.sum((yt == 0) & (y_pred == 1))
    fn = np.sum((yt == 1) & (y_pred == 0))
    csi = tp / max(tp + fp + fn, 1)

    return {
        "roc_auc": float(roc_auc_score(yt, yp)),
        "f1": float(f1_score(yt, y_pred, zero_division=0)),
        "precision": float(precision_score(yt, y_pred, zero_division=0)),
        "recall": float(recall_score(yt, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(yt, yp)),
        "csi": float(csi),
    }



def main():
    # Загружаем настройки с абсолютным разрешением пути
    with open(resolve_path("config", "settings.json"), "r", encoding="utf-8") as f:
        settings = json.load(f)

    processed_dir = resolve_path(settings["paths"]["processed_dir"])
    raw_ext_dir = resolve_path(settings["paths"]["external_dir"])
    model_filename = settings["paths"].get("model_filename", "tft_model.pth")
    model_path = resolve_path(settings["paths"]["models_dir"], model_filename)
    scalers_path = resolve_path(settings["paths"]["scalers_file"])
    stations_config = resolve_path(settings["paths"]["stations_config"])

    if not os.path.exists(model_path):
        print(f"Ошибка: файл модели {model_path} не найден. Запустите сначала train.py")
        return

    if not os.path.exists(scalers_path):
        print(f"Ошибка: скейлеры {scalers_path} не найдены.")
        return

    with open(stations_config, "r", encoding="utf-8") as f:
        stations = json.load(f)["stations"]

    stations = select_stations_for_run(stations, settings)

    with open(scalers_path, "r", encoding="utf-8") as f:
        scalers = json.load(f)

    # Инициализация модели
    device = torch.device("cpu")  # Оценку делаем на CPU
    tft_cfg = settings.get("tft", {})
    model = TFTForecaster(
        hidden_size=tft_cfg.get("hidden_size", 128),
        num_heads=tft_cfg.get("num_heads", 4),
        num_lstm_layers=tft_cfg.get("num_lstm_layers", 2),
        dropout=tft_cfg.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    station_files = [f for f in os.listdir(processed_dir) if f.startswith(
        "station_") and f.endswith("_features.parquet")]
    station_files = filter_station_files_for_run(station_files, settings)
    target_cols = ["uv", "lux", "temperature", "pressure", "humidity",
                   "pm1", "pm2_5", "pm10", "rain", "wind_u", "wind_v"]

    evaluation_results = {}

    print("--- Запуск строгого тестирования модели на отложенной выборке (Test Split) ---")

    for sf in station_files:
        sid = int(sf.split("_")[1])
        parquet_path = os.path.join(processed_dir, sf)
        df_features = pd.read_parquet(parquet_path)
        df_forecast = load_combined_external_forecast(raw_ext_dir, sid, settings)

        if df_forecast.empty:
            continue

        df_features["timestamp"] = pd.to_datetime(df_features["timestamp"], format="mixed")
        if "timestamp" in df_forecast.columns and not pd.api.types.is_datetime64_any_dtype(df_forecast["timestamp"]):
            df_forecast["timestamp"] = pd.to_datetime(df_forecast["timestamp"], format="mixed")

        station_meta = next((s for s in stations if s["id"] == sid), None)
        if not station_meta:
            continue

        # Добавляем статические фичи
        df_features["latitude"] = float(station_meta["latitude"])
        df_features["longitude"] = float(station_meta["longitude"])
        df_features["elevation_m"] = float(station_meta["elevation_m"])

        # Определяем тест-выборку
        min_ts = df_features["timestamp"].min()
        max_ts = df_features["timestamp"].max()
        days_span = (max_ts - min_ts).days

        if days_span < 30:
            # Тестовый режим: берем последние 15% записей как тест
            n = len(df_features)
            idx_test_start = int(n * 0.85)
            df_test = df_features.iloc[idx_test_start:].copy()
        else:
            # Рабочий режим: строго данные с 2026 года
            df_test = df_features[df_features["timestamp"] >= "2026-01-01"].copy()

        if len(df_test) < (settings["model"]["lookback_steps"] + settings["model"]["horizon_steps"]):
            print(f"  Станция {station_meta['name']}: недостаточно данных для тестирования. Пропуск.")
            continue

        # Применяем Z-score коэффициенты, сохраненные при обучении (scalers.json)
        station_key = f"station_{sid}"
        if station_key not in scalers:
            print(f"  Станция {station_meta['name']} ({sid}): скейлеры не найдены, пропускаем.")
            continue

        station_scalers = scalers[station_key]
        for col in target_cols:
            if col not in station_scalers:
                print(f"  Станция {station_meta['name']} ({sid}): пропущен колонки {col} в скейлерах.")
                continue
            mean_v = station_scalers[col]["mean"]
            std_v = station_scalers[col]["std"]
            if std_v == 0.0:
                std_v = 1.0
            df_test[col] = (df_test[col] - mean_v) / std_v

        # Создаем Dataset и DataLoader
        dataset = ClimateDataset(df_test, df_forecast)
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

        enc_x, dec_x, targets_norm = next(iter(loader))

        # Получаем предсказания модели
        with torch.no_grad():
            preds_norm = model(enc_x, dec_x).numpy()  # (N_windows, 192, 11)
            targets_norm = targets_norm.numpy()

        hours_full = df_test["timestamp"].dt.hour.values
        ts_full = df_test["timestamp"].values
        lookback = dataset.lookback_steps
        horizon = dataset.horizon_steps
        hours_matrix = np.array([
            hours_full[s + lookback: s + lookback + horizon]
            for s in dataset.valid_indices
        ])
        timestamps_matrix = np.array([
            ts_full[s + lookback: s + lookback + horizon]
            for s in dataset.valid_indices
        ])

        station_results = {}

        print(f"\nСтанция: {station_meta['name']} (ID={sid}) | Тестовых окон: {preds_norm.shape[0]}")

        # Считаем ошибки по каждой переменной
        for idx, var_name in enumerate(target_cols):
            mean_v = scalers[station_key][var_name]["mean"]
            std_v = scalers[station_key][var_name]["std"]

            # Денормализуем предсказания и таргеты обратно в реальные единицы (°C, %, гПа)
            preds_raw = de_normalize(preds_norm[:, :, idx], mean_v, std_v)
            targets_raw = de_normalize(targets_norm[:, :, idx], mean_v, std_v)

            # 1. Общая ошибка по всему окну 48 часов
            overall_metrics = calculate_metrics(targets_raw, preds_raw)

            # 2. Ошибки на конкретных шагах прогноза (6ч, 12ч, 24ч, 48ч)
            # Шаг 24 (6ч), 48 (12ч), 96 (24ч), 192 (48ч)
            step_metrics = {}
            for step_name, step_idx in [("6h", 23), ("12h", 47), ("24h", 95), ("48h", 191)]:
                step_act = targets_raw[:, step_idx]
                step_pred = preds_raw[:, step_idx]
                step_metrics[step_name] = calculate_metrics(step_act, step_pred)

            hourly_metrics = {}
            for bucket_name, (h_start, h_end) in HOUR_BUCKETS.items():
                bucket_mask = (hours_matrix >= h_start) & (hours_matrix < h_end)
                if bucket_mask.sum() == 0:
                    continue
                bucket_act = targets_raw[bucket_mask]
                bucket_pred = preds_raw[bucket_mask]
                hourly_metrics[bucket_name] = calculate_metrics(bucket_act, bucket_pred)

            station_results[var_name] = {
                "overall": overall_metrics,
                "horizons": step_metrics,
                "by_hour_of_day": hourly_metrics
            }

            # Выводим отчет по важным переменным (температура, влажность, давление)
            if var_name in ["temperature", "humidity", "pressure"]:
                print(
                    f"  {var_name.capitalize():11s} | Overall MAE: {overall_metrics['mae']:.2f} | 6h MAE: {step_metrics['6h']['mae']:.2f} | 24h MAE: {step_metrics['24h']['mae']:.2f} | 48h MAE: {step_metrics['48h']['mae']:.2f}")

            if var_name == "temperature":
                u_mean = scalers[station_key]["wind_u"]["mean"]
                u_std = scalers[station_key]["wind_u"]["std"]
                v_mean = scalers[station_key]["wind_v"]["mean"]
                v_std = scalers[station_key]["wind_v"]["std"]

                wind_u_raw = de_normalize(preds_norm[:, :, target_cols.index("wind_u")], u_mean, u_std)
                wind_v_raw = de_normalize(preds_norm[:, :, target_cols.index("wind_v")], v_mean, v_std)
                lat_deg = float(station_meta.get("latitude", 40.2))

                preds_no_inv = preds_raw
                preds_half_inv = apply_inversion_correction(preds_raw, wind_u_raw, wind_v_raw, timestamps_matrix, lat_deg, scale=0.5)
                preds_full_inv = apply_inversion_correction(preds_raw, wind_u_raw, wind_v_raw, timestamps_matrix, lat_deg, scale=1.0)

                print("    🧪 Сравнение вариантов поправки 7.3 (Инверсия температуры):")
                print("      ┌───────────────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐")
                print("      │ Вариант                   │ Ночь MAE     │ Ночь Bias    │ Общий MAE    │ Общий Bias   │")
                print("      ├───────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤")

                night_mask = (hours_matrix >= 0) & (hours_matrix < 6)

                for lbl, p_var in [
                    ("raw_no_correction (scale=0.0)", preds_no_inv),
                    ("half_correction   (scale=0.5)", preds_half_inv),
                    ("full_correction   (scale=1.0)", preds_full_inv),
                ]:
                    m_tot = calculate_metrics(targets_raw, p_var)
                    m_ngt = calculate_metrics(targets_raw[night_mask], p_var[night_mask]) if night_mask.sum() > 0 else {"mae": 0, "bias": 0}
                    print(f"      │ {lbl:25s} │ {m_ngt['mae']:12.2f} │ {m_ngt['bias']:+12.2f} │ {m_tot['mae']:12.2f} │ {m_tot['bias']:+12.2f} │")
                print("      └───────────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘")
        evaluation_results[station_key] = station_results

    # Сохраняем подробный JSON отчет
    report_file = os.path.join(processed_dir, "evaluation_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2)

    print(f"\nДетальный отчет об ошибках сохранен в: {report_file}")


if __name__ == "__main__":
    main()


def run_benchmark() -> None:
    """Вычисляет метрики точности внешнего прогноза Open-Meteo (Бенчмарк)."""
    from project_paths import load_settings, load_stations_config, resolve_path

    settings = load_settings()
    processed_dir = resolve_path(settings["paths"]["processed_dir"])
    raw_ext_dir = resolve_path(settings["paths"]["external_dir"])
    out_dir = resolve_path("data", "processed")
    os.makedirs(out_dir, exist_ok=True)

    stations = load_stations_config()
    stations = select_stations_for_run(stations, settings)

    results = []
    print("=== Оценка точности провайдеров прогнозов (Open-Meteo vs Станции) ===")

    for station in stations:
        name = station["name"]
        sid = station["id"]
        gen_id = station["generated_id"]

        cn_file = os.path.join(processed_dir, f"station_{gen_id}_features.parquet")
        if not os.path.exists(cn_file):
            continue

        df_ext = load_combined_external_forecast(raw_ext_dir, sid, settings)
        if df_ext.empty:
            continue

        df_cn = pd.read_parquet(cn_file)
        df_cn["timestamp"] = pd.to_datetime(df_cn["timestamp"])
        if "timestamp" in df_ext.columns and not pd.api.types.is_datetime64_any_dtype(df_ext["timestamp"]):
            df_ext["timestamp"] = pd.to_datetime(df_ext["timestamp"])

        sensor_cols = ["temperature", "humidity", "pressure"]
        df_cn_hourly = df_cn.set_index("timestamp")[sensor_cols].resample("1h").mean().reset_index()

        df_merged = pd.merge(df_cn_hourly, df_ext, on="timestamp")
        if df_merged.empty:
            continue

        for var, ext_col in [("temperature", "temperature_2m"), ("humidity", "relative_humidity_2m"), ("pressure", "surface_pressure")]:
            if var in df_merged.columns and ext_col in df_merged.columns:
                m = calculate_metrics(df_merged[var].to_numpy(), df_merged[ext_col].to_numpy())
                m.update({"station_name": name, "station_id": sid, "variable": var, "provider": "Open-Meteo"})
                results.append(m)

    if results:
        df_res = pd.DataFrame(results)
        bench_out = os.path.join(out_dir, "provider_benchmark.csv")
        df_res.to_csv(bench_out, index=False)
        print(f"Бенчмарк успешно сохранен в: {bench_out}")
    else:
        print("Данные для расчета бенчмарка отсутствуют.")
