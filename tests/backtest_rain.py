import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from project_paths import resolve_path, load_settings
from data_pipeline import prepare_feature_frame, prepare_model_inputs, inverse_scalers, MODEL_TARGET_COLUMNS
from model import TFTForecaster


def load_tft_model(settings: dict, model_path: str):
    tft_cfg = settings.get("tft", {})
    model = TFTForecaster(
        hidden_size=tft_cfg.get("hidden_size", 128),
        num_heads=tft_cfg.get("num_heads", 4),
        num_lstm_layers=tft_cfg.get("num_lstm_layers", 2),
        dropout=tft_cfg.get("dropout", 0.1),
    )
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    if "output_projs.0.weight" in state_dict and "output_proj.weight" not in state_dict:
        state_dict["output_proj.weight"] = state_dict["output_projs.0.weight"]
        state_dict["output_proj.bias"] = state_dict["output_projs.0.bias"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def run_historical_rain_backtest(station_id: int, settings: dict, model, max_samples: int = 50):
    stations_cfg_path = resolve_path("config", "stations.json")
    with open(stations_cfg_path, "r", encoding="utf-8") as f:
        stations = json.load(f)["stations"]
    station_meta = next((s for s in stations if s["id"] == station_id), None)
    if not station_meta:
        print(f"❌ Станция ID {station_id} не найдена в stations.json")
        return None

    gen_id = station_meta["generated_id"]
    station_name = station_meta["name"]
    parquet_path = resolve_path("data", "processed", f"station_{gen_id}_features.parquet")

    if not os.path.exists(parquet_path):
        print(f"⚠️ Файл данных {parquet_path} не найден.")
        return None

    df_full = pd.read_parquet(parquet_path)
    if df_full.empty or len(df_full) < 300:
        print(f"⚠️ Недостаточно данных в {parquet_path} для оценки.")
        return None

    df_full = df_full.sort_values("timestamp").reset_index(drop=True)

    # Загружаем скалеры
    scalers_path = resolve_path("config", "scalers.json")
    with open(scalers_path, "r", encoding="utf-8") as f:
        scalers_dict = json.load(f)
    station_key = f"station_{gen_id}"
    station_scalers = scalers_dict.get(station_key, {})

    # Отбираем индексы с реальным дождем и без него
    rain_mask = df_full["rain"] > 0.1
    rain_indices = df_full.index[rain_mask & (df_full.index >= 96) & (df_full.index <= len(df_full) - 96)].tolist()
    dry_indices = df_full.index[(~rain_mask) & (df_full.index >= 96) & (df_full.index <= len(df_full) - 96)].tolist()

    if not rain_indices:
        print(f"ℹ️ В истории станции {station_name} (ID {station_id}) не найдено периодов с осадками > 0.1мм.")
        return None

    # Формируем сбалансированную выборку
    np.random.seed(42)
    selected_rain_idx = np.random.choice(rain_indices, min(len(rain_indices), max_samples // 2), replace=False)
    selected_dry_idx = np.random.choice(dry_indices, min(len(dry_indices), max_samples // 2), replace=False)
    test_indices = sorted(list(selected_rain_idx) + list(selected_dry_idx))

    y_true_rain = []
    y_pred_will_rain = []
    y_pred_prob_pct = []

    print(f"\n📊 Прогон бэктестирования для метеостанции '{station_name}' (выборка: {len(test_indices)} окон)...")

    for idx in test_indices:
        # Окно истории 96 шагов (24ч)
        df_window = df_full.iloc[idx - 96 : idx].copy()
        if "timestamp" not in df_window.columns and isinstance(df_window.index, pd.DatetimeIndex):
            df_window = df_window.reset_index()

        # Фактическое наличие дождя в следующие 24 часа
        actual_future_rain = float(df_full.iloc[idx : idx + 96]["rain"].max())
        actual_has_rain = actual_future_rain > 0.1

        try:
            # Инференс TFT
            forecast_df_dummy = pd.DataFrame({
                "timestamp": pd.date_range(start=df_window["timestamp"].iloc[-1], periods=192, freq="15min"),
                "temperature_2m": 20.0,
                "relative_humidity_2m": 50.0,
                "surface_pressure": 900.0,
                "wind_speed_10m": 5.0,
                "precipitation": 0.0,
                "cloud_cover": 20.0
            })
            encoder_inputs, decoder_inputs, future_ts = prepare_model_inputs(
                df_window, forecast_df_dummy, station_meta, station_scalers, 96, 192
            )
            with torch.no_grad():
                preds_norm = model(torch.from_numpy(encoder_inputs), torch.from_numpy(decoder_inputs)).numpy()[0]

            preds_df = pd.DataFrame(preds_norm, columns=MODEL_TARGET_COLUMNS)
            preds_df = inverse_scalers(preds_df, {col: station_scalers[col] for col in MODEL_TARGET_COLUMNS if col in station_scalers}, MODEL_TARGET_COLUMNS)
            preds_final = preds_df.to_numpy(dtype=np.float32)

            # Оценка первых 12 часов (48 шагов)
            has_pred_rain = False
            max_prob_pct = 0.0

            for t_step in range(48):
                rain_amount = float(np.clip(preds_final[t_step, 8], 0.0, 500.0))
                rain_binary_logit = preds_final[t_step, 11]
                rain_prob = 1.0 / (1.0 + np.exp(-rain_binary_logit))
                step_prob_pct = float(np.clip(rain_prob * 100.0, 0.0, 100.0))
                max_prob_pct = max(max_prob_pct, step_prob_pct)

                step_temp = float(preds_final[t_step, 2])
                step_hum = float(preds_final[t_step, 4])
                dew_point_approx = step_temp - ((100.0 - step_hum) / 5.0)
                dew_deficit = step_temp - dew_point_approx

                physics_favorable = (step_hum >= 75.0) and (dew_deficit <= 2.5)
                adaptive_threshold = 30.0 if physics_favorable else 60.0

                step_will_rain = bool(rain_amount > 0.1 or (step_prob_pct >= adaptive_threshold and physics_favorable))
                if step_will_rain:
                    has_pred_rain = True
                    break

            y_true_rain.append(actual_has_rain)
            y_pred_will_rain.append(has_pred_rain)
            y_pred_prob_pct.append(max_prob_pct)

        except Exception as exc:
            print(f"  ⚠️ Ошибка инференса для индекса {idx}: {exc}")
            continue

    y_true = np.array(y_true_rain, dtype=bool)
    y_pred = np.array(y_pred_will_rain, dtype=bool)

    tp = np.sum(y_true & y_pred)
    fp = np.sum((~y_true) & y_pred)
    tn = np.sum((~y_true) & (~y_pred))
    fn = np.sum(y_true & (~y_pred))

    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("=" * 60)
    print(f"  Результаты ретроспективы: {station_name}")
    print("=" * 60)
    print(f"  Всего протестировано окон   : {len(y_true)}")
    print(f"  Истинно дождливых окон (TP) : {tp}")
    print(f"  Ложные тревоги (FP)         : {fp}")
    print(f"  Истинно сухих окон (TN)     : {tn}")
    print(f"  Пропущенные дожди (FN)      : {fn}")
    print("-" * 60)
    print(f"  🎯 Accuracy (Общая точность): {accuracy*100:.1f}%")
    print(f"  🎯 Precision (Точность совпадений): {precision*100:.1f}%")
    print(f"  🎯 Recall (Полнота детекции): {recall*100:.1f}%")
    print(f"  🛡️ False Alarm Rate (Ложная тревога): {false_alarm_rate*100:.1f}%")
    print("=" * 60)

    return {
        "station_name": station_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "false_alarm_rate": false_alarm_rate,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
    }


def main():
    parser = argparse.ArgumentParser(description="Бэктест алгоритма осадков на исторических данных")
    parser.add_argument("--station_id", type=int, default=None, help="ID конкретной метеостанции")
    parser.add_argument("--all_stations", action="store_true", help="Прогнать бэктест по всем метеостанциям")
    args = parser.parse_args()

    settings = load_settings(resolve_path("config", "settings.json"))
    model_path = resolve_path("models", "tft_model.pth")
    model = load_tft_model(settings, model_path)

    stations_cfg_path = resolve_path("config", "stations.json")
    with open(stations_cfg_path, "r", encoding="utf-8") as f:
        stations = json.load(f)["stations"]

    if args.all_stations or args.station_id is None:
        print("\n🚀 Запуск ретроспективного тестирования осадков ПО ВСЕМ МЕТЕОСТАНЦИЯМ...")
        results = []
        for st in stations:
            res = run_historical_rain_backtest(st["id"], settings, model)
            if res:
                results.append(res)

        if results:
            total_tp = sum(r["tp"] for r in results)
            total_fp = sum(r["fp"] for r in results)
            total_tn = sum(r["tn"] for r in results)
            total_fn = sum(r["fn"] for r in results)
            total_windows = total_tp + total_fp + total_tn + total_fn

            global_acc = (total_tp + total_tn) / total_windows if total_windows > 0 else 0
            global_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            global_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            global_far = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0

            print("\n" + "=" * 70)
            print("  🏆 СВОДНЫЙ ИТОГОВЫЙ ОТЧЕТ БЭКТЕСТИРОВАНИЯ ОСАДКОВ ПО ВСЕМ СТАНЦИЯМ")
            print("=" * 70)
            print(f"  Протестировано станций       : {len(results)}")
            print(f"  Всего временных окон         : {total_windows}")
            print(f"  Истинно дождливых окон (TP)  : {total_tp}")
            print(f"  Ложные тревоги (FP)          : {total_fp}")
            print(f"  Истинно сухих окон (TN)      : {total_tn}")
            print(f"  Пропущенные дожди (FN)       : {total_fn}")
            print("-" * 70)
            print(f"  🎯 Сводный Accuracy (Общая точность): {global_acc*100:.1f}%")
            print(f"  🎯 Сводный Precision (Точность совпадений): {global_prec*100:.1f}%")
            print(f"  🎯 Сводный Recall (Полнота детекции): {global_rec*100:.1f}%")
            print(f"  🛡️ Сводный False Alarm Rate (Ложная тревога): {global_far*100:.1f}%")
            print("=" * 70)
    else:
        run_historical_rain_backtest(args.station_id, settings, model)


if __name__ == "__main__":
    main()
