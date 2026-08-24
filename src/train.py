"""
=============================================================================
MODULE: TFT Model Training Pipeline (train.py)
-----------------------------------------------------------------------------
PURPOSE:
Main training and validation pipeline for Temporal Fusion Transformer (TFT).

KEY FUNCTIONS:
1. Ingest processed station datasets into global ConcatDataset.
2. Training loop with AdamW optimizer and multi-task loss weighting.
3. Validation and best checkpoint persistence (`models/tft_model.pth`).
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
    Computes loss and meteorological evaluation metrics on validation split:
    Precision, Recall, F1, CSI (for rain classification).
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    # Binary rain prediction metrics
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

            # Compute Precision / Recall / F1 / CSI at 50% probability threshold
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
    # Load settings from central configuration
    settings_file = resolve_path("config", "settings.json")
    if not os.path.exists(settings_file):
        print(f"Error: {settings_file} not found.")
        exit(1)

    settings = load_settings(settings_file)

    # CLI arguments
    parser = argparse.ArgumentParser(description="Training pipeline for TFT weather model.")
    parser.add_argument("--epochs", type=int, default=settings["training"]["epochs"], help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=settings["training"]["lr"], help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=settings["training"]["batch_size"], help="Batch size")
    parser.add_argument("--patience", type=int,
                        default=settings["training"]["patience"], help="Early stopping patience")
    parser.add_argument("--all-stations", action="store_true", help="Train on all stations")
    parser.add_argument("--station-id", type=int, default=None, help="Single station ID for training")
    parser.add_argument("--num-stations", type=int, default=None,
                        help="Pilot mode: stratified selection of N stations across elevation tiers")
    parser.add_argument("--fast", action="store_true",
                        help="Fast CPU mode: hidden_size=64, num_heads=2 for rapid testing")
    parser.add_argument("--max-windows", type=int, default=None,
                        help="Limit number of training windows (for pilot runs)")
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
    print(f"Compute device: {device}")

    # Paths from configuration
    processed_dir = resolve_path(settings["paths"]["processed_dir"])
    raw_ext_dir = resolve_path(settings["paths"]["external_dir"])
    models_dir = resolve_path(settings["paths"]["models_dir"])
    scalers_file = resolve_path(settings["paths"]["scalers_file"])
    stations_config = resolve_path(settings["paths"]["stations_config"])

    os.makedirs(models_dir, exist_ok=True)

    train_datasets = []
    val_datasets = []
    test_datasets = []

    # Scan Parquet files
    if not os.path.exists(processed_dir):
        print(f"Error: Directory {processed_dir} not found.")
        return

    station_files = [f for f in os.listdir(processed_dir) if f.startswith(
        "station_") and f.endswith("_features.parquet")]
    station_files = filter_station_files_for_run(station_files, settings)

    if not station_files:
        print("Processed station files not found.")
        return

    with open(stations_config, "r", encoding="utf-8") as f:
        stations = json.load(f)["stations"]

    stations = select_stations_for_run(stations, settings)

    # Synchronize selected station IDs for file filtering
    settings["_subset_ids"] = [int(s["id"]) for s in stations]
    station_files = filter_station_files_for_run(station_files, settings)

    print(f"Found stations for training: {len(station_files)}")

    min_required_steps = settings["model"]["lookback_steps"] + settings["model"]["horizon_steps"]

    # Scalers dictionary (mean / std)
    scalers_dict = {}

    for sf in station_files:
        sid = int(sf.split("_")[1])
        parquet_path = os.path.join(processed_dir, sf)
        df_features = pd.read_parquet(parquet_path)
        df_forecast = load_combined_external_forecast(raw_ext_dir, sid, settings)

        if df_forecast.empty:
            print(f"  Warning: External forecast for station {sid} not found (or disabled), skipping.")
            continue

        df_features["timestamp"] = pd.to_datetime(df_features["timestamp"], format="mixed")
        if "timestamp" in df_forecast.columns and not pd.api.types.is_datetime64_any_dtype(df_forecast["timestamp"]):
            df_forecast["timestamp"] = pd.to_datetime(df_forecast["timestamp"], format="mixed")

        station_meta = next((s for s in stations if s["id"] == sid), None)
        if not station_meta:
            continue

        # Add static coordinates
        df_features["latitude"] = float(station_meta["latitude"])
        df_features["longitude"] = float(station_meta["longitude"])
        df_features["elevation_m"] = float(station_meta["elevation_m"])

        min_ts = df_features["timestamp"].min()
        max_ts = df_features["timestamp"].max()
        days_span = (max_ts - min_ts).days

        # Split datasets verifying historical time coverage
        # Require at least 180 days history prior to 2025-07-01 for calendar split
        train_days = (pd.to_datetime("2025-07-01") - df_features["timestamp"].min()).days
        has_required_range = (train_days >= 180) and \
                             (df_features["timestamp"].max() > pd.to_datetime("2026-01-01"))

        # Purging buffer (288 steps = 3 days = 96 lookback + 192 horizon) to prevent data leakage
        purge_steps = 288
        if not has_required_range:
            # Index-based Purged Split with buffer
            print(f"  Station {sid}: [Test Mode / Purged Split 70/15/15] Purge gap: {purge_steps} steps.")
            n = len(df_features)
            idx_train_end = int(n * 0.7)
            idx_val_end = int(n * 0.85)
            df_train = df_features.iloc[:idx_train_end].copy()
            df_val = df_features.iloc[min(n, idx_train_end + purge_steps):idx_val_end].copy()
            df_test = df_features.iloc[min(n, idx_val_end + purge_steps):].copy()
        else:
            # Operational mode with 3-day buffer between dates (Purged Date Split)
            print(f"  Station {sid}: [Operational Mode / Purged Split] Purge buffer: 3 days between splits.")
            df_train = df_features[df_features["timestamp"] < "2025-07-01"].copy()
            df_val = df_features[(df_features["timestamp"] >= "2025-07-04") &
                                 (df_features["timestamp"] < "2026-01-01")].copy()
            df_test = df_features[df_features["timestamp"] >= "2026-01-04"].copy()

        # 2. Compute scaling parameters strictly on df_train
        cols_to_normalize = NORMALIZE_COLUMNS

        station_key = f"station_{sid}"
        station_scalers = build_scalers(df_train, cols_to_normalize)

        # Compute scalers for external NWP forecasts
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

        # Create datasets
        if len(df_train) >= min_required_steps:
            train_datasets.append(ClimateDataset(df_train, df_forecast, scalers=station_scalers))
        if len(df_val) >= min_required_steps:
            val_datasets.append(ClimateDataset(df_val, df_forecast, scalers=station_scalers))
        if len(df_test) >= min_required_steps:
            test_datasets.append(ClimateDataset(df_test, df_forecast, scalers=station_scalers))

    if not train_datasets or not val_datasets:
        print(f"Critical error: insufficient data records. Requires at least {min_required_steps} rows.")
        return

    # Save scalers in config/scalers.json
    with open(scalers_file, "w", encoding="utf-8") as f:
        json.dump(scalers_dict, f, indent=2)
    print(f"\nNormalization parameters (training) successfully saved to: {scalers_file}")

    # Concatenate datasets
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)

    print(f"Total training windows: {len(train_dataset)}")
    print(f"Total validation windows: {len(val_dataset)}")

    # Limit dataset if --max-windows is specified
    if args.max_windows is not None:
        import random

        from torch.utils.data import Subset
        train_indices = random.sample(range(len(train_dataset)), min(args.max_windows, len(train_dataset)))
        val_indices = random.sample(range(len(val_dataset)),   min(args.max_windows // 4, len(val_dataset)))
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset,   val_indices)
        print(f"[Limit] Train: {len(train_dataset)} windows | Val: {len(val_dataset)} windows")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              drop_last=True,  pin_memory=False, num_workers=0)
    val_loader = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                            drop_last=False, pin_memory=False, num_workers=0)

    # Initialize TFT model from "tft" block in settings.json
    tft_cfg = settings.get("tft", {})

    # --fast: reduced architecture for rapid CPU verification
    if args.fast:
        hidden_size = 64
        num_heads = 2
        num_lstm_layers = 1
        dropout = tft_cfg.get("dropout", 0.1)
        print("[FAST Mode] hidden_size=64, num_heads=2, num_lstm_layers=1")
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
    print(f"TFTForecaster initialized. Trainable parameters: {total_params:,}")
    sys.stdout.flush()

    # AdamW with parameter group separation:
    # - Group 1: Linear/Attention weights (weight_decay=1e-2)
    # - Group 2: Bias + LayerNorm (weight_decay=0.0 - Transformer standard)
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

    # Initialize learnable Softmax-normalized loss
    min_task_floors = torch.tensor([0.02] * 12, dtype=torch.float32)
    uncertainty_loss_fn = HomoscedasticUncertaintyLoss(num_tasks=12, min_task_weights=min_task_floors).to(device)

    # Add uncertainty_loss_fn parameters to optimizer BEFORE creating scheduler
    optimizer.add_param_group({"params": uncertainty_loss_fn.parameters(), "weight_decay": 0.0, "lr": args.lr * 0.5})

    # Warmup (5 epochs: LR 1e-6 -> lr) + CosineAnnealing (remaining epochs)
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

    # Base pos_weight for rain class balancing (n_dry / n_rain)
    pos_weight = 8.0
    print(f"\nBCE pos_weight (n_dry/n_rain): {pos_weight:.2f}")

    best_val_loss = float("inf")
    patience_counter = 0
    model_save_path = os.path.join(models_dir, settings["paths"].get("model_filename", "tft_model.pth"))

    total_batches = len(train_loader)
    LOG_EVERY = max(10, total_batches // 20)   # ~5% progress
    print(f"\n--- Starting Training Loop (Homoscedastic Uncertainty Weighting) ---")
    print(f"    Batches per epoch: {total_batches} | Logging every {LOG_EVERY} batches")
    sys.stdout.flush()

    for epoch in range(args.epochs):
        model.train()
        uncertainty_loss_fn.train()
        epoch_loss = 0.0
        batch_count = 0
        epoch_start = time.time()
        batch_start = time.time()

        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        sys.stdout.flush()

        for batch_idx, (enc_x, dec_x, targets) in enumerate(train_loader, start=1):
            enc_x, dec_x, targets = enc_x.to(device), dec_x.to(device), targets.to(device)

            optimizer.zero_grad()
            preds = model(enc_x, dec_x)

            # Compute individual losses for each of the 12 targets
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
                    # 0.5-quantile (median) Pinball/Asymmetric Huber loss for strict MAE minimization
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

            # Periodic logging: every LOG_EVERY batches and epoch end
            if batch_idx % LOG_EVERY == 0 or batch_idx == total_batches:
                elapsed = time.time() - epoch_start
                sec_per_batch = elapsed / batch_count
                remaining_batches = total_batches - batch_idx
                eta_sec = remaining_batches * sec_per_batch
                pct = 100.0 * batch_idx / total_batches
                running_avg = epoch_loss / batch_count

                # ASCII progress bar (40 chars)
                filled = int(40 * batch_idx / total_batches)
                bar = '█' * filled + '░' * (40 - filled)

                print(
                    f"  [{bar}] {pct:5.1f}% "
                    f"batch {batch_idx:04d}/{total_batches:04d} "
                    f"| loss: {loss.item():.4f} "
                    f"| avg: {running_avg:.4f} "
                    f"| {sec_per_batch:.2f}s/batch "
                    f"| ETA: {int(eta_sec//60)}m{int(eta_sec % 60):02d}s"
                )
                sys.stdout.flush()

        # Validation evaluation
        val_loss, val_metrics = evaluate(
            model, val_loader, device,
            uncertainty_loss_fn=uncertainty_loss_fn,
            weights=loss_weights,
            rain_event_weight=rain_event_weight,
            pos_weight=pos_weight,
            rain_binary_index=rain_binary_index,
        )

        # Update learning rate from validation results
        scheduler.step()  # SequentialLR (Warmup -> CosineAnnealing)
        current_lr = optimizer.param_groups[0]["lr"]
        avg_train_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | "
              f"Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f} | "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s | Patience: {patience_counter}/{args.patience}")
        print(f"  Rain (Val) | Precision: {val_metrics['precision']:.3f} | "
              f"Recall: {val_metrics['recall']:.3f} | F1: {val_metrics['f1']:.3f} | "
              f"CSI: {val_metrics['csi']:.3f} | TP={val_metrics['tp']} FP={val_metrics['fp']} FN={val_metrics['fn']}")

        if val_loss < best_val_loss:
            improvement = (best_val_loss - val_loss) if best_val_loss != float("inf") else 0.0
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  ✓ Saved best model checkpoint with Val Loss: {best_val_loss:.6f} (improvement: {improvement:.6f})")
            patience_counter = 0
            # Notify local API to reload model weights
            try:
                api_url = settings.get("api_url", "http://127.0.0.1:8000")
                resp = requests.post(f"{api_url}/reload_model", timeout=5)
                print(f"  → API model reload notification: {resp.status_code}")
            except Exception as e:
                print(f"  → Could not notify API about reload: {e}")
        else:
            patience_counter += 1
            print(f"  Val Loss did not improve. Best remains: {best_val_loss:.6f}")
            if patience_counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch+1}. Training finished.")
                break

    print(f"\nTraining complete. Model weights saved to: {model_save_path}")


if __name__ == "__main__":
    main()
