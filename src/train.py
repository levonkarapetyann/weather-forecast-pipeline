"""
=============================================================================
МОДУЛЬ: TFT Model Training Pipeline (train.py)
-----------------------------------------------------------------------------
НАЗНАЧЕНИЕ:
Основной скрипт обучения и валидации нейронной сети Temporal Fusion Transformer (TFT).

ОСНОВНЫЕ ФУНКЦИИ:
1. Загрузка обработанных датасетов станций и сборка глобальных ConcatDataset.
2. Цикл обучения (Training Loop) с оптимизатором AdamW и взвешиванием градиентов.
3. Валидация модели и сохранение лучшего чекпоинта весов `models/tft_model.pth`.
=============================================================================
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
import requests
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from model import (NUM_QUANTILES, QUANTILES, ClimateDataset, HomoscedasticUncertaintyLoss,
                   TFTForecaster, masked_asymmetric_huber_loss,
                   masked_huber_loss, masked_mse_loss, masked_pinball_loss,
                   masked_rain_event_loss, masked_slope_loss)
from data_pipeline import (EXTERNAL_FORECAST_COLUMNS, MODEL_TARGET_COLUMNS,
                          NORMALIZE_COLUMNS, apply_scalers, build_scalers,
                          filter_station_files_for_run,
                          load_combined_external_forecast, select_stations_for_run)
from project_paths import load_settings, resolve_path


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    uncertainty_loss_fn: nn.Module = None,
    weights: torch.Tensor = None,
    rain_event_weight: float = 0.5,
    pos_weight: float = None,
    rain_binary_index: int = 11,
) -> tuple[float, dict]:
    """
    Вычисляет лосс и метеорологические метрики на валидационной выборке:
    Precision, Recall, F1, CSI (для прогноза дождя).
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    # Метрики бинарного прогноза дождя
    tp = fp = fn = tn = 0

    with torch.no_grad():
        for enc_x, dec_x, targets in val_loader:
            enc_x, dec_x, targets = enc_x.to(device), dec_x.to(device), targets.to(device)
            preds = model(enc_x, dec_x)

            if uncertainty_loss_fn is not None:
                task_losses = []
                for t_idx in range(12):
                    if t_idx == rain_binary_index:
                        t_loss = masked_rain_event_loss(
                            preds, targets,
                            rain_binary_index=rain_binary_index,
                            weight=rain_event_weight,
                            pos_weight=pos_weight
                        )
                    else:
                        t_loss = masked_huber_loss(preds[..., t_idx:t_idx+1], targets[..., t_idx:t_idx+1])
                    task_losses.append(t_loss)
                task_losses_tensor = torch.stack(task_losses)
                huber_loss, _ = uncertainty_loss_fn(task_losses_tensor)
            else:
                huber_loss = masked_huber_loss(preds, targets, weights=weights)

            slope_loss = masked_slope_loss(preds, targets, weights=weights)
            loss = huber_loss + 0.3 * slope_loss
            total_loss += loss.item()
            num_batches += 1

            # Считаем Precision / Recall / F1 / CSI по порогу 50%
            rain_pred_bin = (torch.sigmoid(preds[..., rain_binary_index]) >= 0.5).float()

            rain_true = torch.nan_to_num(targets[..., rain_binary_index], nan=0.0)
            tp += ((rain_pred_bin == 1) & (rain_true == 1)).sum().item()
            fp += ((rain_pred_bin == 1) & (rain_true == 0)).sum().item()
            fn += ((rain_pred_bin == 0) & (rain_true == 1)).sum().item()
            tn += ((rain_pred_bin == 0) & (rain_true == 0)).sum().item()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    csi = tp / (tp + fp + fn + 1e-9)  # Critical Success Index
    metrics = {"precision": precision, "recall": recall, "f1": f1, "csi": csi,
               "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return avg_loss, metrics


def main():
    # Загружаем настройки из центрального конфига
    settings_file = resolve_path("config", "settings.json")
    if not os.path.exists(settings_file):
        print(f"Ошибка: {settings_file} не найден.")
        exit(1)

    settings = load_settings(settings_file)

    # CLI аргументы
    parser = argparse.ArgumentParser(description="Обучение TFT модели погоды.")
    parser.add_argument("--epochs", type=int, default=settings["training"]["epochs"], help="Количество эпох")
    parser.add_argument("--lr", type=float, default=settings["training"]["lr"], help="Скорость обучения")
    parser.add_argument("--batch_size", type=int, default=settings["training"]["batch_size"], help="Размер батча")
    parser.add_argument("--patience", type=int,
                        default=settings["training"]["patience"], help="Терпение ранней остановки")
    parser.add_argument("--all-stations", action="store_true", help="Обучать на всех станциях")
    parser.add_argument("--station-id", type=int, default=None, help="ИД одной станции для обучения")
    parser.add_argument("--num-stations", type=int, default=None,
                        help="Пилотный режим: стратифицированно выбрать N станций из разных поясов высоты")
    parser.add_argument("--fast", action="store_true",
                        help="CPU-быстрый режим: hidden_size=64, num_heads=2 — для быстрой проверки архитектуры")
    parser.add_argument("--max-windows", type=int, default=None,
                        help="Ограничить число обучающих окон (для пилотных запусков)")
    args = parser.parse_args()

    if args.all_stations:
        settings["single_station"] = {"enabled": False, "station_id": None}
        settings["station_subset"] = {"enabled": False}
    elif args.station_id is not None:
        settings["single_station"] = {"enabled": True, "station_id": args.station_id}
        settings["station_subset"] = {"enabled": False}
    elif args.num_stations is not None:
        settings["single_station"] = {"enabled": False, "station_id": None}
        settings["station_subset"] = {"enabled": True, "mode": "count", "count": args.num_stations}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    # Пути из конфига
    processed_dir = resolve_path(settings["paths"]["processed_dir"])
    raw_ext_dir = resolve_path(settings["paths"]["external_dir"])
    models_dir = resolve_path(settings["paths"]["models_dir"])
    scalers_file = resolve_path(settings["paths"]["scalers_file"])
    stations_config = resolve_path(settings["paths"]["stations_config"])

    os.makedirs(models_dir, exist_ok=True)

    train_datasets = []
    val_datasets = []
    test_datasets = []

    # Сканируем Parquet файлы
    if not os.path.exists(processed_dir):
        print(f"Ошибка: Директория {processed_dir} не найдена.")
        return

    station_files = [f for f in os.listdir(processed_dir) if f.startswith(
        "station_") and f.endswith("_features.parquet")]
    station_files = filter_station_files_for_run(station_files, settings)

    if not station_files:
        print("Обработанные файлы станций не найдены.")
        return

    with open(stations_config, "r", encoding="utf-8") as f:
        stations = json.load(f)["stations"]

    stations = select_stations_for_run(stations, settings)

    # Синхронизируем список ID выбранных станций для фильтрации файлов
    settings["_subset_ids"] = [int(s["id"]) for s in stations]
    station_files = filter_station_files_for_run(station_files, settings)

    print(f"Найдено станций для обучения: {len(station_files)}")

    min_required_steps = settings["model"]["lookback_steps"] + settings["model"]["horizon_steps"]

    # Словарь, куда запишем рассчитанные скейлеры (среднее/отклонение)
    scalers_dict = {}

    for sf in station_files:
        sid = int(sf.split("_")[1])
        parquet_path = os.path.join(processed_dir, sf)
        df_features = pd.read_parquet(parquet_path)
        df_forecast = load_combined_external_forecast(raw_ext_dir, sid, settings)

        if df_forecast.empty:
            print(f"  Внимание: Внешний прогноз для станции {sid} не найден (или отключен), пропускаем.")
            continue

        df_features["timestamp"] = pd.to_datetime(df_features["timestamp"], format="mixed")
        if "timestamp" in df_forecast.columns and not pd.api.types.is_datetime64_any_dtype(df_forecast["timestamp"]):
            df_forecast["timestamp"] = pd.to_datetime(df_forecast["timestamp"], format="mixed")

        station_meta = next((s for s in stations if s["id"] == sid), None)
        if not station_meta:
            continue

        # Добавляем статические координаты
        df_features["latitude"] = float(station_meta["latitude"])
        df_features["longitude"] = float(station_meta["longitude"])
        df_features["elevation_m"] = float(station_meta["elevation_m"])

        min_ts = df_features["timestamp"].min()
        max_ts = df_features["timestamp"].max()
        days_span = (max_ts - min_ts).days

        # Умное разделение на выборки: проверяем, охватывают ли данные нужный исторический диапазон
        # Дополнительно требуем не менее 180 дней истории до 01.07.2025 для календарного сплита
        train_days = (pd.to_datetime("2025-07-01") - df_features["timestamp"].min()).days
        has_required_range = (train_days >= 180) and \
                             (df_features["timestamp"].max() > pd.to_datetime("2026-01-01"))

        # Purging buffer (288 шагов = 3 суток = 96 lookback + 192 horizon) для исключения утечек данных
        purge_steps = 288
        if not has_required_range:
            # Делим по индексам с зазором зачистки (Purged Split)
            print(f"  Станция {sid}: [Режим Теста / Purged Split 70/15/15] Зазор зачистки: {purge_steps} шагов.")
            n = len(df_features)
            idx_train_end = int(n * 0.7)
            idx_val_end = int(n * 0.85)
            df_train = df_features.iloc[:idx_train_end].copy()
            df_val = df_features.iloc[min(n, idx_train_end + purge_steps):idx_val_end].copy()
            df_test = df_features.iloc[min(n, idx_val_end + purge_steps):].copy()
        else:
            # Рабочий режим с зазором в 3 суток между датами (Purged Date Split)
            print(f"  Станция {sid}: [Рабочий режим / Purged Split] Буфер зачистки: 3 суток между выборками.")
            df_train = df_features[df_features["timestamp"] < "2025-07-01"].copy()
            df_val = df_features[(df_features["timestamp"] >= "2025-07-04") &
                                 (df_features["timestamp"] < "2026-01-01")].copy()
            df_test = df_features[df_features["timestamp"] >= "2026-01-04"].copy()

        # 2. РАССЧИТЫВАЕМ параметры масштабирования СТРОГО НА df_train
        cols_to_normalize = NORMALIZE_COLUMNS

        station_key = f"station_{sid}"
        station_scalers = build_scalers(df_train, cols_to_normalize)

        # Дополнительно считаем scaler'ы для внешних прогнозов
        ext_df_15m = df_forecast.sort_values("timestamp").set_index("timestamp").resample("15min").ffill().reset_index()
        ext_cols_for_scaling = [c for c in ["wind_speed_10m", "cloud_cover", "om_wind_speed_10m", "ms_wind_speed_10m", "om_cloud_cover", "ms_cloud_cover"] if c in ext_df_15m.columns]
        if ext_cols_for_scaling:
            ext_train = pd.merge(
                df_train[["timestamp"]],
                ext_df_15m[["timestamp"] + ext_cols_for_scaling],
                on="timestamp",
                how="left",
            )
            ext_train[ext_cols_for_scaling] = (
                ext_train[ext_cols_for_scaling]
                .replace([float("inf"), float("-inf")], pd.NA)
                .ffill()
                .bfill()
            )
            station_scalers.update(build_scalers(ext_train, ext_cols_for_scaling))

        df_train = apply_scalers(df_train, station_scalers, cols_to_normalize)
        df_val = apply_scalers(df_val, station_scalers, cols_to_normalize)
        df_test = apply_scalers(df_test, station_scalers, cols_to_normalize)

        scalers_dict[station_key] = station_scalers

        # Создаем датасеты
        if len(df_train) >= min_required_steps:
            train_datasets.append(ClimateDataset(df_train, df_forecast, scalers=station_scalers))
        if len(df_val) >= min_required_steps:
            val_datasets.append(ClimateDataset(df_val, df_forecast, scalers=station_scalers))
        if len(df_test) >= min_required_steps:
            test_datasets.append(ClimateDataset(df_test, df_forecast, scalers=station_scalers))

    if not train_datasets or not val_datasets:
        print(f"Критическая ошибка: недостаточно данных. Требуется не менее {min_required_steps} строк.")
        return

    # Сохраняем скейлеры в config/scalers.json
    with open(scalers_file, "w", encoding="utf-8") as f:
        json.dump(scalers_dict, f, indent=2)
    print(f"\nПараметры нормализации (обучающие) успешно сохранены в: {scalers_file}")

    # Объединяем
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)

    print(f"Общий объем окон для обучения: {len(train_dataset)}")
    print(f"Общий объем окон для валидации: {len(val_dataset)}")

    # Ограничиваем датасет если указан --max-windows
    if args.max_windows is not None:
        import random

        from torch.utils.data import Subset
        train_indices = random.sample(range(len(train_dataset)), min(args.max_windows, len(train_dataset)))
        val_indices = random.sample(range(len(val_dataset)),   min(args.max_windows // 4, len(val_dataset)))
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset,   val_indices)
        print(f"[Ограничение] Обучение: {len(train_dataset)} окон | Валидация: {len(val_dataset)} окон")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              drop_last=True,  pin_memory=False, num_workers=0)
    val_loader = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                            drop_last=False, pin_memory=False, num_workers=0)

    # Инициализация TFT модели (параметры берем из блока "tft" в settings.json)
    tft_cfg = settings.get("tft", {})

    # --fast: уменьшенная архитектура для быстрой проверки на CPU
    if args.fast:
        hidden_size = 64
        num_heads = 2
        num_lstm_layers = 1
        dropout = tft_cfg.get("dropout", 0.1)
        print("[FAST режим] hidden_size=64, num_heads=2, num_lstm_layers=1")
    else:
        hidden_size = tft_cfg.get("hidden_size", 128)
        num_heads = tft_cfg.get("num_heads", 4)
        num_lstm_layers = tft_cfg.get("num_lstm_layers", 2)
        dropout = tft_cfg.get("dropout", 0.1)

    model = TFTForecaster(
        num_decoder_vars=len(EXTERNAL_FORECAST_COLUMNS),
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_lstm_layers=num_lstm_layers,
        dropout=dropout,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TFTForecaster инициализирован. Параметров: {total_params:,}")
    sys.stdout.flush()

    # AdamW с раздельными группами параметров:
    # - Группа 1: веса Linear/Attention (weight_decay=1e-2)
    # - Группа 2: Bias + LayerNorm (weight_decay=0.0 — стандарт для Трансформеров)
    decay_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and p.ndim >= 2
        and not any(nd in name for nd in ["norm", "bias"])
    ]
    no_decay_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and (p.ndim < 2 or any(nd in name for nd in ["norm", "bias"]))
    ]
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": 1e-2},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=args.lr)

    # Инициализация обучаемого Softmax-нормированного лосса
    min_task_floors = torch.tensor([0.02] * 12, dtype=torch.float32)
    uncertainty_loss_fn = HomoscedasticUncertaintyLoss(num_tasks=12, min_task_weights=min_task_floors).to(device)

    # Добавляем параметры uncertainty_loss_fn в оптимизатор ДО создания планировщика
    optimizer.add_param_group({"params": uncertainty_loss_fn.parameters(), "weight_decay": 0.0, "lr": args.lr * 0.5})

    # Warmup (5 эпох: LR 1e-6 → lr) + CosineAnnealing (оставшиеся эпохи)
    warmup_epochs = min(5, max(1, args.epochs // 6))
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    training_cfg = settings.get("training", {})
    target_weights_cfg = training_cfg.get("target_weights", {})
    default_loss_weights = [0.15, 0.15, 6.0, 0.25, 0.25, 0.15, 0.15, 0.15, 6.0, 0.15, 0.15, 0.0]
    loss_weights_list = [
        float(target_weights_cfg.get(col, default_loss_weights[idx]))
        for idx, col in enumerate(MODEL_TARGET_COLUMNS)
    ]
    loss_weights = torch.tensor(loss_weights_list, dtype=torch.float32, device=device)
    rain_event_weight = float(training_cfg.get("rain_event_weight", 2.0))
    rain_binary_index = MODEL_TARGET_COLUMNS.index("rain_binary")

    # Базовый pos_weight для балансировки классов осадков (n_dry / n_rain)
    pos_weight = 8.0
    print(f"\nBCE pos_weight (n_dry/n_rain): {pos_weight:.2f}")

    best_val_loss = float("inf")
    patience_counter = 0
    model_save_path = os.path.join(models_dir, settings["paths"].get("model_filename", "tft_model.pth"))

    total_batches = len(train_loader)
    LOG_EVERY = max(10, total_batches // 20)   # ~5% прогресса
    print(f"\n--- Запуск цикла обучения (Homoscedastic Uncertainty Weighting) ---")
    print(f"    Батчей в эпохе: {total_batches} | Лог каждые {LOG_EVERY} батчей")
    sys.stdout.flush()

    for epoch in range(args.epochs):
        model.train()
        uncertainty_loss_fn.train()
        epoch_loss = 0.0
        batch_count = 0
        epoch_start = time.time()
        batch_start = time.time()

        print(f"\n{'='*70}")
        print(f"ЭПОХА {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        sys.stdout.flush()

        for batch_idx, (enc_x, dec_x, targets) in enumerate(train_loader, start=1):
            enc_x, dec_x, targets = enc_x.to(device), dec_x.to(device), targets.to(device)

            optimizer.zero_grad()
            preds = model(enc_x, dec_x)

            # Вычисляем individual losses для каждого из 12 таргетов
            task_losses = []
            for t_idx in range(12):
                if t_idx == rain_binary_index:
                    t_loss = masked_rain_event_loss(
                        preds, targets,
                        rain_binary_index=rain_binary_index,
                        weight=rain_event_weight,
                        pos_weight=pos_weight
                    )
                else:
                    # 0.5-квантильный (медианный) Pinball/Asymmetric Huber loss для строгой минимизации MAE
                    t_loss = masked_asymmetric_huber_loss(
                        preds[..., t_idx:t_idx+1],
                        targets[..., t_idx:t_idx+1],
                        tau=0.5,
                        delta=0.05
                    )
                task_losses.append(t_loss)

            task_losses_tensor = torch.stack(task_losses)
            huber_loss, _ = uncertainty_loss_fn(task_losses_tensor)

            slope_loss = masked_slope_loss(preds, targets, weights=loss_weights)
            loss = huber_loss + 0.3 * slope_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            # Частый лог: каждые LOG_EVERY батчей и в самом конце
            if batch_idx % LOG_EVERY == 0 or batch_idx == total_batches:
                elapsed = time.time() - epoch_start
                sec_per_batch = elapsed / batch_count
                remaining_batches = total_batches - batch_idx
                eta_sec = remaining_batches * sec_per_batch
                pct = 100.0 * batch_idx / total_batches
                running_avg = epoch_loss / batch_count

                # Простой ASCII прогресс-бар (40 символов)
                filled = int(40 * batch_idx / total_batches)
                bar = '█' * filled + '░' * (40 - filled)

                print(
                    f"  [{bar}] {pct:5.1f}% "
                    f"батч {batch_idx:04d}/{total_batches:04d} "
                    f"| loss: {loss.item():.4f} "
                    f"| avg: {running_avg:.4f} "
                    f"| {sec_per_batch:.2f}с/батч "
                    f"| ETA: {int(eta_sec//60)}м{int(eta_sec % 60):02d}с"
                )
                sys.stdout.flush()

        # Оценка на валидации
        val_loss, val_metrics = evaluate(
            model, val_loader, device,
            uncertainty_loss_fn=uncertainty_loss_fn,
            weights=loss_weights,
            rain_event_weight=rain_event_weight,
            pos_weight=pos_weight,
            rain_binary_index=rain_binary_index,
        )

        # Обновление LR по результатам валидации
        scheduler.step()  # SequentialLR (Warmup → CosineAnnealing) — без аргументов
        current_lr = optimizer.param_groups[0]["lr"]
        avg_train_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        epoch_time = time.time() - epoch_start

        print(f"Эпоха {epoch+1:02d}/{args.epochs:02d} | "
              f"Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f} | "
              f"LR: {current_lr:.2e} | Время: {epoch_time:.1f}с | Patience: {patience_counter}/{args.patience}")
        print(f"  Дождь (Val) | Precision: {val_metrics['precision']:.3f} | "
              f"Recall: {val_metrics['recall']:.3f} | F1: {val_metrics['f1']:.3f} | "
              f"CSI: {val_metrics['csi']:.3f} | TP={val_metrics['tp']} FP={val_metrics['fp']} FN={val_metrics['fn']}")

        if val_loss < best_val_loss:
            improvement = (best_val_loss - val_loss) if best_val_loss != float("inf") else 0.0
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  ✓ Сохранена лучшая модель с Val Loss: {best_val_loss:.6f} (улучшение: {improvement:.6f})")
            patience_counter = 0
            # Попробуем уведомить локальный API, чтобы он подгрузил новые веса автоматически
            try:
                api_url = settings.get("api_url", "http://127.0.0.1:8000")
                resp = requests.post(f"{api_url}/reload_model", timeout=5)
                print(f"  → Уведомление API о перезагрузке модели: {resp.status_code}")
            except Exception as e:
                print(f"  → Не удалось оповестить API о перезагрузке модели: {e}")
        else:
            patience_counter += 1
            print(f"  Val Loss не улучшился. Лучший результат остаётся: {best_val_loss:.6f}")
            if patience_counter >= args.patience:
                print(f"Ранняя остановка сработала на эпохе {epoch+1}. Обучение завершено.")
                break

    print(f"\nОбучение закончено. Веса модели сохранены в: {model_save_path}")


if __name__ == "__main__":
    main()
