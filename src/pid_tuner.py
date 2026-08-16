"""
=============================================================================
МОДУЛЬ: PID Controller Optimization & Tuning (pid_tuner.py)
-----------------------------------------------------------------------------
НАЗНАЧЕНИЕ:
Скрипт подбора и оптимизации коэффициентов PID-регулятора (Kp, Ki, Kd, alpha)
для каждой отдельной метеостанции.

ОСНОВНЫЕ ФУНКЦИИ:
1. Симуляция работы PID-регулятора на валидационных окнах исторических данных.
2. Поиск оптимальных значений коэффициентов обратной связи с помощью Nelder-Mead
   / Powell оптимизации для минимизации ошибки MAE прогнозов на первые 10 часов.
3. Сохранение откалиброванных параметров в `config/pid_params.json`.
=============================================================================
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from model import ClimateDataset, TFTForecaster, masked_huber_loss, masked_mse_loss
from data_pipeline import (MODEL_TARGET_COLUMNS, NORMALIZE_COLUMNS, apply_scalers,
                          select_stations_for_run)
from project_paths import resolve_path

FIXED_STATION_ALPHA = {
    "Maralik": 0.98, "Azatan": 0.995, "Artik": 0.98,
    "Ashotsk": 0.995, "Chambarak": 0.98, "Berd": 0.995,
    "Ijevan": 0.995, "Azatamut": 0.98, "Spitak": 0.98, "Stepanavan": 0.98,
    "Areni": 0.995, "Jermuk": 0.98, "Alaverdi": 0.995, "Dsegh": 0.995,
    "Mkhchyan": 0.995,
}

# Штраф за скачки коррекции (не самого ряда): для rain/wind почти не давим.
SMOOTH_WEIGHT_BY_VAR = {
    "temperature": 0.01,
    "pressure": 0.01,
    "humidity": 0.01,
    "rain": 0.0,
    "wind_u": 0.002,
    "wind_v": 0.002,
}

DEFAULT_INT_LIMIT = 3.0
INT_LIMIT_MIN = 0.5
INT_LIMIT_MAX = 3.0

MAX_CORRECTION_REAL = {
    "temperature": 3.0,
    "pressure": 5.0,
    "humidity": 15.0,
    "rain": 2.0,
    "wind_u": 3.0,
    "wind_v": 3.0
}

TERRAIN_DECAY_TAU = {
    "alpine": 8.0,   # Быстрое затухание (8 часов) для высокогорий (>1800м)
    "valley": 12.0,  # Умеренное затухание (12 часов) для горных котловин
    "plain": 14.0,   # Плавное затухание (14 часов) для равнин и предгорий
}


def classify_station_terrain(station_meta: dict) -> str:
    """Классифицирует метеостанцию по микроклиматическому рельефу."""
    elev = float(station_meta.get("elevation_m", 1000.0))
    cold_risk = float(station_meta.get("valley_cold_pool_risk", 0.0))

    if elev >= 1800.0:
        return "alpine"
    elif cold_risk > 0.4 or elev >= 1100.0:
        return "valley"
    else:
        return "plain"


def pad_features_df(df, min_rows):
    """Pad df at the start by repeating early rows when len(df) < min_rows."""
    if len(df) >= min_rows:
        return df
    needed = min_rows - len(df)
    import pandas as _pd
    freq = df["timestamp"].diff().median()
    if freq is None or _pd.isna(freq) or (hasattr(freq, 'total_seconds') and freq.total_seconds() <= 0):
        freq = _pd.Timedelta(minutes=15)
    donor = df.copy()
    pads = []
    rows_left = needed
    while rows_left > 0:
        take = min(rows_left, len(donor))
        pads.append(donor.iloc[:take].copy())
        rows_left -= take
    pad_df = _pd.concat(pads[::-1], ignore_index=True)
    t0 = df["timestamp"].iloc[0]
    n_pad = len(pad_df)
    pad_df["timestamp"] = [t0 - freq * (n_pad - i) for i in range(n_pad)]
    result = _pd.concat([pad_df, df], ignore_index=True)
    result = result.sort_values("timestamp").reset_index(drop=True)
    return result


def compute_int_limit(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Порог интегратора по std остатков на train; ограничен [0.5, 10]."""
    resid = preds - targets
    mask = ~torch.isnan(resid)
    if not mask.any():
        return DEFAULT_INT_LIMIT
    scale = float(torch.std(resid[mask]).item())
    if not (scale == scale) or scale <= 0.0:  # NaN / zero
        return DEFAULT_INT_LIMIT
    return float(max(INT_LIMIT_MIN, min(INT_LIMIT_MAX, scale * 2.0)))


def apply_pid(
    preds: torch.Tensor,
    targets: torch.Tensor,
    Kp_short: torch.Tensor,
    Ki_short: torch.Tensor,
    Kd_short: torch.Tensor,
    alpha_short: torch.Tensor,
    Kp_long: torch.Tensor,
    Ki_long: torch.Tensor,
    Kd_long: torch.Tensor,
    alpha_long: torch.Tensor,
    int_limit: float,
    max_corr_z: float = 4.0,
    split_step: int = 24,
    blend_window_steps: int = 4,
    decay_tau_hours: float = 12.0,
    pressure_trends: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PID-цикл с плавным (sigmoid) переходом параметров вокруг split_step
    (bumpless transfer). Ширина окна перехода — blend_window_steps шагов.
    """
    e_t0 = preds[:, 0] - targets[:, 0]
    e_t0 = torch.where(torch.isnan(e_t0), torch.zeros_like(e_t0), e_t0)

    e_prev = e_t0
    integral_sum = e_t0
    e_prev2 = e_t0

    blend_scale = max(1.0, blend_window_steps / 2.0)

    # Weather Regime PID: адаптация Kp и Kd при прохождении погодного фронта dP/dt
    if pressure_trends is not None:
        is_front = torch.abs(pressure_trends) > 1.5
        Kp_short = torch.where(is_front, Kp_short * 1.40, Kp_short)
        Kd_short = torch.where(is_front, Kd_short * 1.25, Kd_short)

    corrected = preds.clone()
    corrections = []
    for t in range(preds.shape[1]):
        w = torch.sigmoid(torch.tensor((t - split_step) / blend_scale, dtype=torch.float32))

        Kp = (1.0 - w) * Kp_short + w * Kp_long
        Ki = (1.0 - w) * Ki_short + w * Ki_long
        Kd = (1.0 - w) * Kd_short + w * Kd_long
        alpha = (1.0 - w) * alpha_short + w * alpha_long

        e_prev = e_prev * alpha
        integral_sum = torch.clamp(integral_sum * alpha + e_prev, -int_limit, int_limit)
        e_deriv = e_prev - e_prev2
        correction = Kp * e_prev + Ki * integral_sum + Kd * e_deriv
        # Адаптивное экспоненциальное затухание коррекции по горизонту (Gain Decay):
        decay_factor = torch.exp(torch.tensor(-t / max(1.0, decay_tau_hours), dtype=torch.float32))
        correction = correction * decay_factor
        correction = torch.clamp(correction, -max_corr_z, max_corr_z)
        corrected[:, t] = preds[:, t] - correction
        corrections.append(correction)
        e_prev2 = e_prev.clone()

    return corrected, torch.stack(corrections, dim=1)


def optimize_pid_parameters(
    preds: torch.Tensor,
    targets: torch.Tensor,
    epochs: int = 50,
    lr: float = 0.01,
    alpha: float | None = None,
    smooth_weight: float = 0.01,
    patience: int = 10,
    max_corr_z: float = 4.0,
    split_step: int = 24,
    blend_window_steps: int = 4,
) -> tuple[float, float, float, float, float, float, float, float, float, bool]:
    """
    Двухэтапная последовательная оптимизация параметров PID:
    - Этап 1: Оптимизирует Kp_short, Ki_short, Kd_short, alpha_short по MSE короткого горизонта (t < split_step).
    - Этап 2: Фиксирует короткие параметры, оптимизирует Kp_long, Ki_long, Kd_long, alpha_long по MSE длинного горизонта (t >= split_step).
    """
    n = int(preds.shape[0])
    if n == 0:
        return 0.0, 0.0, 0.0, 0.98, 0.0, 0.0, 0.0, 0.995, DEFAULT_INT_LIMIT, False

    split = max(1, int(n * 0.7))
    if split >= n:
        split = n

    train_preds, train_targs = preds[:split], targets[:split]
    has_val = split < n
    val_preds = preds[split:] if has_val else None
    val_targs = targets[split:] if has_val else None

    int_limit = compute_int_limit(train_preds, train_targs)

    # --- Этап 1: Оптимизация короткого горизонта ---
    Kp_short = torch.tensor(0.1, requires_grad=True)
    Ki_short = torch.tensor(0.01, requires_grad=True)
    Kd_short = torch.tensor(0.05, requires_grad=True)

    learn_alpha_short = (alpha is None)
    if learn_alpha_short:
        alpha_short_t = torch.tensor(0.98, requires_grad=True)
        params_short = [Kp_short, Ki_short, Kd_short, alpha_short_t]
    else:
        alpha_short_t = torch.tensor(float(alpha), requires_grad=False)
        params_short = [Kp_short, Ki_short, Kd_short]

    optimizer_short = torch.optim.Adam(params_short, lr=lr)
    best_val_short = float("inf")
    best_state_short = None
    no_improve_short = 0

    for _ in range(epochs):
        optimizer_short.zero_grad()
        corrected, corr_seq = apply_pid(
            train_preds, train_targs,
            Kp_short, Ki_short, Kd_short, alpha_short_t,
            Kp_short, Ki_short, Kd_short, alpha_short_t,  # dummy long
            int_limit, max_corr_z, split_step, blend_window_steps
        )
        mse = masked_huber_loss(corrected[:, :split_step], train_targs[:, :split_step], delta=0.05)
        if smooth_weight > 0.0 and corr_seq.shape[1] > 1:
            smooth = torch.diff(corr_seq[:, :split_step], dim=1).abs().mean()
            loss = mse + smooth_weight * smooth
        else:
            loss = mse

        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_short, max_norm=1.0)
            optimizer_short.step()

        with torch.no_grad():
            Kp_short.clamp_(0.0, 1.0)
            Ki_short.clamp_(0.0, 0.8)
            Kd_short.clamp_(0.0, 1.0)
            if learn_alpha_short:
                alpha_short_t.clamp_(0.90, 0.995)

            if has_val:
                val_corrected, _ = apply_pid(
                    val_preds, val_targs,
                    Kp_short, Ki_short, Kd_short, alpha_short_t,
                    Kp_short, Ki_short, Kd_short, alpha_short_t,
                    int_limit, max_corr_z, split_step, blend_window_steps
                )
                val_mse = float(masked_huber_loss(val_corrected[:, :split_step], val_targs[:, :split_step], delta=0.05).item())
            else:
                val_mse = float(mse.detach().item())

            if val_mse < best_val_short and val_mse == val_mse:
                best_val_short = val_mse
                best_state_short = {
                    "Kp_short": Kp_short.detach().clone(),
                    "Ki_short": Ki_short.detach().clone(),
                    "Kd_short": Kd_short.detach().clone(),
                    "alpha_short": alpha_short_t.detach().clone(),
                }
                no_improve_short = 0
            else:
                no_improve_short += 1
                if has_val and no_improve_short >= patience:
                    break

    if best_state_short is not None:
        Kp_short_val = best_state_short["Kp_short"]
        Ki_short_val = best_state_short["Ki_short"]
        Kd_short_val = best_state_short["Kd_short"]
        alpha_short_val = best_state_short["alpha_short"]
    else:
        Kp_short_val = Kp_short.detach().clone()
        Ki_short_val = Ki_short.detach().clone()
        Kd_short_val = Kd_short.detach().clone()
        alpha_short_val = alpha_short_t.detach().clone()

    # --- Этап 2: Оптимизация длинного горизонта ---
    Kp_long = torch.tensor(float(Kp_short_val.item()), requires_grad=True)
    Ki_long = torch.tensor(float(Ki_short_val.item()), requires_grad=True)
    Kd_long = torch.tensor(float(Kd_short_val.item()), requires_grad=True)
    alpha_long_t = torch.tensor(0.995, requires_grad=True)

    params_long = [Kp_long, Ki_long, Kd_long, alpha_long_t]
    optimizer_long = torch.optim.Adam(params_long, lr=lr)
    best_val_long = float("inf")
    best_state_long = None
    no_improve_long = 0

    for _ in range(epochs):
        optimizer_long.zero_grad()
        corrected, corr_seq = apply_pid(
            train_preds, train_targs,
            Kp_short_val, Ki_short_val, Kd_short_val, alpha_short_val,
            Kp_long, Ki_long, Kd_long, alpha_long_t,
            int_limit, max_corr_z, split_step, blend_window_steps
        )
        mse = masked_mse_loss(corrected[:, split_step:], train_targs[:, split_step:])
        if smooth_weight > 0.0 and corr_seq.shape[1] > split_step:
            smooth = torch.diff(corr_seq[:, split_step:], dim=1).abs().mean()
            loss = mse + smooth_weight * smooth
        else:
            loss = mse

        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_long, max_norm=1.0)
            optimizer_long.step()

        with torch.no_grad():
            Kp_long.clamp_(0.0, 1.0)
            Ki_long.clamp_(0.0, 0.8)
            Kd_long.clamp_(0.0, 1.0)
            # alpha_long зажата в [0.99, 0.9995] для более медленного затухания
            alpha_long_t.clamp_(0.99, 0.9995)

            if has_val:
                val_corrected, _ = apply_pid(
                    val_preds, val_targs,
                    Kp_short_val, Ki_short_val, Kd_short_val, alpha_short_val,
                    Kp_long, Ki_long, Kd_long, alpha_long_t,
                    int_limit, max_corr_z, split_step, blend_window_steps
                )
                val_mse = float(masked_mse_loss(val_corrected[:, split_step:], val_targs[:, split_step:]).item())
            else:
                val_mse = float(mse.detach().item())

            if val_mse < best_val_long and val_mse == val_mse:
                best_val_long = val_mse
                best_state_long = {
                    "Kp_long": Kp_long.detach().clone(),
                    "Ki_long": Ki_long.detach().clone(),
                    "Kd_long": Kd_long.detach().clone(),
                    "alpha_long": alpha_long_t.detach().clone(),
                }
                no_improve_long = 0
            else:
                no_improve_long += 1
                if has_val and no_improve_long >= patience:
                    break

    if best_state_long is not None:
        Kp_long_val = best_state_long["Kp_long"]
        Ki_long_val = best_state_long["Ki_long"]
        Kd_long_val = best_state_long["Kd_long"]
        alpha_long_val = best_state_long["alpha_long"]
    else:
        Kp_long_val = Kp_long.detach().clone()
        Ki_long_val = Ki_long.detach().clone()
        Kd_long_val = Kd_long.detach().clone()
        alpha_long_val = alpha_long_t.detach().clone()

    return (
        float(Kp_short_val.item()),
        float(Ki_short_val.item()),
        float(Kd_short_val.item()),
        float(alpha_short_val.item()),
        float(Kp_long_val.item()),
        float(Ki_long_val.item()),
        float(Kd_long_val.item()),
        float(alpha_long_val.item()),
        int_limit,
        has_val
    )


def main():
    device = torch.device("cpu")  # Используем CPU для тюнинга
    settings_path = resolve_path("config", "settings.json")
    if not os.path.exists(settings_path):
        print(f"Ошибка: Файл настроек {settings_path} не найден.")
        return

    with open(settings_path, "r", encoding="utf-8") as f:
        settings = json.load(f)

    stations = select_stations_for_run(
        json.load(open(resolve_path("config", "stations.json"), "r", encoding="utf-8"))["stations"],
        settings,
    )
    scalers_path = resolve_path(settings["paths"]["scalers_file"])
    if not os.path.exists(scalers_path):
        print(f"Ошибка: Файл скейлеров {scalers_path} не найден.")
        return

    with open(scalers_path, "r", encoding="utf-8") as f:
        scalers = json.load(f)

    # Инициализируем TFTForecaster с параметрами из settings.json
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
    if not os.path.exists(model_path):
        print(f"Ошибка: Файл весов модели {model_path} не найден.")
        return

    # Загружаем веса строго (strict=True), чтобы избежать нестыковок в архитектуре
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    pid_params = {}
    out_path = resolve_path("config", "pid_params.json")
    if os.path.exists(out_path):
        try:
            os.remove(out_path)
            print(f"🧹 Удален устаревший файл конфигурации PID: {out_path}")
        except Exception as e:
            print(f"⚠️  Не удалось удалить старый {out_path}: {e}")

    var_names = ["temperature", "pressure", "humidity", "rain", "wind_u", "wind_v"]

    print("--- Запуск оптимизации параметров PID-регулятора ---")

    for station in stations:
        sid, name = station["id"], station["name"]
        parquet_path = resolve_path("data", "processed", f"station_{sid}_features.parquet")
        forecast_path = resolve_path("data", "raw", "external_forecasts", f"forecast_{sid}.csv")

        if not os.path.exists(parquet_path) or not os.path.exists(forecast_path):
            continue

        print(f"Обработка станции {name} (id={sid})...")
        if f"station_{sid}" not in scalers:
            print(f"  Станция {name} ({sid}): скейлеры не найдены, пропускаем.")
            continue

        df_features = pd.read_parquet(parquet_path)
        df_forecast = pd.read_csv(forecast_path)
        df_features["timestamp"] = pd.to_datetime(df_features["timestamp"])
        df_forecast["timestamp"] = pd.to_datetime(df_forecast["timestamp"])

        df_features["latitude"] = float(station["latitude"])
        df_features["longitude"] = float(station["longitude"])
        df_features["elevation_m"] = float(station["elevation_m"])

        # --- pad if not enough rows ---
        min_rows = settings["model"]["lookback_steps"] + settings["model"]["horizon_steps"]
        if len(df_features) < min_rows:
            print(f"  Станция {name} ({sid}): только {len(df_features)} строк < {min_rows}, дополняем...")
            df_features = pad_features_df(df_features, min_rows)
            print(f"  После дополнения: {len(df_features)} строк.")

        df_f = apply_scalers(df_features, scalers[f"station_{sid}"], NORMALIZE_COLUMNS)
        dataset = ClimateDataset(df_f, df_forecast)
        if len(dataset) == 0:
            print(f"  Станция {name} ({sid}): датасет пуст даже после дополнения, пропускаем.")
            continue

        # Ограничиваем выборку последними 800 окнами для предотвращения OOM в WSL
        if len(dataset) > 800:
            dataset = Subset(dataset, range(len(dataset) - 800, len(dataset)))

        enc_x, dec_x, targets = next(iter(DataLoader(dataset, batch_size=len(dataset))))
        with torch.no_grad():
            preds_raw = model(enc_x, dec_x)
        if preds_raw.ndim == 3 and preds_raw.shape[-1] == 12 * 3:
            preds = preds_raw.view(preds_raw.size(0), preds_raw.size(1), 12, 3)[..., 1]
        else:
            preds = preds_raw

        # Read near_horizon_hours, split_step, blend_window_steps from settings
        near_horizon_hours = settings.get("residual_boost", {}).get("near_horizon_hours", 6.0)
        split_step = int(near_horizon_hours * 4.0)
        blend_window_steps = int(settings.get("residual_boost", {}).get("blend_window_steps", 4))

        # Классификация рельефа станции для подбора адаптивного затухания tau
        terrain_class = classify_station_terrain(station)
        decay_tau_hours = TERRAIN_DECAY_TAU.get(terrain_class, 12.0)
        print(f"  [Рельеф Станции: {terrain_class.upper()}] Высота: {station.get('elevation_m', 0)}м | PID Decay Tau: {decay_tau_hours}ч")

        station_pid = {}
        for var_name in var_names:
            idx = MODEL_TARGET_COLUMNS.index(var_name)
            var_preds = preds[:, :, idx]
            var_targets = targets[:, :, idx]

            std_v = float(scalers[f"station_{sid}"][var_name]["std"])
            max_corr_real = MAX_CORRECTION_REAL.get(var_name, 5.0)
            max_corr_z = max_corr_real / std_v if std_v > 0 else max_corr_real

            Kp_s, Ki_s, Kd_s, alpha_s, Kp_l, Ki_l, Kd_l, alpha_l, int_limit, has_val = optimize_pid_parameters(
                var_preds,
                var_targets,
                alpha=FIXED_STATION_ALPHA.get(name),
                smooth_weight=SMOOTH_WEIGHT_BY_VAR.get(var_name, 0.01),
                max_corr_z=max_corr_z,
                split_step=split_step,
                blend_window_steps=blend_window_steps,
            )
            station_pid[var_name] = {
                "Kp_short": Kp_s,
                "Ki_short": Ki_s,
                "Kd_short": Kd_s,
                "alpha_short": alpha_s,
                "Kp_long": Kp_l,
                "Ki_long": Ki_l,
                "Kd_long": Kd_l,
                "alpha_long": alpha_l,
                # Backward compatibility
                "Kp": Kp_s,
                "Ki": Ki_s,
                "Kd": Kd_s,
                "alpha": alpha_s,
                "int_limit": int_limit,
                "max_corr_z": max_corr_z,
                "decay_tau_hours": decay_tau_hours,
                "terrain_class": terrain_class,
            }

            # ── Верификация качества настройки: MAE на Val (или Train, если Val нет) ──
            split = max(1, int(var_preds.shape[0] * 0.7)) if has_val else var_preds.shape[0]
            eval_preds = var_preds[split:] if has_val else var_preds
            eval_targs = var_targets[split:] if has_val else var_targets

            e_raw = (eval_preds - eval_targs).abs()
            mae_raw = float(e_raw[~torch.isnan(e_raw)].mean().item()) if e_raw[~torch.isnan(e_raw)].numel() > 0 else 0.0

            eval_corrected, _ = apply_pid(
                eval_preds, eval_targs,
                torch.tensor(Kp_s), torch.tensor(Ki_s), torch.tensor(Kd_s), torch.tensor(alpha_s),
                torch.tensor(Kp_l), torch.tensor(Ki_l), torch.tensor(Kd_l), torch.tensor(alpha_l),
                int_limit, max_corr_z, split_step, blend_window_steps, decay_tau_hours
            )
            e_pid = (eval_corrected - eval_targs).abs()
            mae_pid = float(e_pid[~torch.isnan(e_pid)].mean().item()) if e_pid[~torch.isnan(e_pid)].numel() > 0 else 0.0

            improvement = ((mae_raw - mae_pid) / mae_raw * 100) if mae_raw > 0 else 0.0

            subset_name = "Val" if has_val else "Train"
            print(
                f"  {var_name:12s} | "
                f"Short Kp/Ki/Kd/alpha: {Kp_s:.3f}/{Ki_s:.3f}/{Kd_s:.3f}/{alpha_s:.3f} | "
                f"Long Kp/Ki/Kd/alpha: {Kp_l:.3f}/{Ki_l:.3f}/{Kd_l:.3f}/{alpha_l:.3f} | "
                f"MAE ({subset_name}) Raw: {mae_raw:.4f} -> PID: {mae_pid:.4f} ({improvement:+.1f}%)"
            )

        pid_params[f"station_{sid}"] = station_pid

        # Сохраняем промежуточные результаты
        out_path = resolve_path("config", "pid_params.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(pid_params, f, indent=2)
            print(f"  Параметры PID для станции {sid} сохранены.")
        except Exception as e:
            print(f"  Не удалось сохранить PID параметры для станции {sid}: {e}")

    print(f"\nОптимизация PID завершена. Итоговый файл: {out_path}")


if __name__ == "__main__":
    main()
