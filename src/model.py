"""
=============================================================================
МОДУЛЬ: TFT Neural Network Architecture (model.py)
-----------------------------------------------------------------------------
НАЗНАЧЕНИЕ:
Определение глубокой нейросетевой архитектуры Temporal Fusion Transformer (TFT)
и специализированных функций потерь.

ОСНОВНЫЕ КОМПОНЕНТЫ:
1. `TFTForecaster`: архитектура нейросети с механизмами Interpretable Multi-Head
   Attention, Variable Selection Networks и LSTM селекторами контекста.
2. `ClimateDataset`: класс подготовки скользящих окон временных рядов (96 шагов).
3. Набор лосс-функций: Pinball Loss (Quantile Loss), Huber Loss и BCE Rain Loss.
=============================================================================
"""

import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from data_pipeline import (EXTERNAL_FORECAST_COLUMNS, MODEL_SENSOR_COLUMNS,
                           MODEL_STATIC_COLUMNS, MODEL_TARGET_COLUMNS,
                           normalize_multimodel_external_df)

# Загружаем настройки из конфига
settings_file = os.path.join("config", "settings.json")
if os.path.exists(settings_file):
    with open(settings_file, "r", encoding="utf-8") as f:
        settings = json.load(f)
else:
    settings = {
        "model": {
            "lookback_steps": 96,
            "horizon_steps": 192,
        },
        "tft": {
            "hidden_size": 128,
            "num_heads": 4,
            "dropout": 0.1,
            "num_lstm_layers": 2,
        },
    }


class HomoscedasticUncertaintyLoss(nn.Module):
    """
    Стабильное мультизадачное взвешивание лоссов через Softmax-нормировку.
    Гарантирует, что сумма весов задач равна 1.0, а итоговый лосс строго положительный (>= 0).
    """

    def __init__(self, num_tasks: int = 12, min_task_weights: torch.Tensor = None):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_logits = nn.Parameter(torch.zeros(num_tasks, dtype=torch.float32))
        self.register_buffer("min_task_weights", min_task_weights if min_task_weights is not None else torch.zeros(num_tasks))

    def forward(self, task_losses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        task_losses: (num_tasks,) — скалярный лосс по каждой переменной
        returns: (total_loss, effective_weights)
        """
        softmax_weights = F.softmax(self.task_logits, dim=0)
        if self.min_task_weights is not None:
            effective_weights = torch.maximum(softmax_weights, self.min_task_weights.to(task_losses.device))
        else:
            effective_weights = softmax_weights

        weighted_loss = (task_losses * effective_weights).sum()
        return weighted_loss, effective_weights.detach()


# ---------------------------------------------------------------------------
# ClimateDataset — без изменений, интерфейс тот же самый
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Квантили для вероятностного прогноза (Pinball Loss)
# ---------------------------------------------------------------------------
QUANTILES = [0.1, 0.5, 0.9]       # P10, медиана, P90
NUM_QUANTILES = len(QUANTILES)     # 3


class ClimateDataset(Dataset):
    """
    Класс датасета для подготовки окон данных для обучения TFT модели.
    Интерфейс полностью совместим с предыдущей Seq2Seq реализацией.
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        external_df: pd.DataFrame,
        scalers: dict = None,
    ):
        self.lookback_steps = settings["model"]["lookback_steps"]
        self.horizon_steps = settings["model"]["horizon_steps"]

        # 1. Статические характеристики станции с глобальной нормализацией
        raw_static = np.array(
            [float(features_df[col].iloc[0]) for col in MODEL_STATIC_COLUMNS],
            dtype=np.float32,
        )
        self.static_features = np.array([
            (raw_static[0] - 40.2) / 0.5,      # latitude
            (raw_static[1] - 44.5) / 0.8,      # longitude
            (raw_static[2] - 1400.0) / 450.0   # elevation_m
        ], dtype=np.float32)

        # 2. Временные ряды датчиков
        sensor_cols = MODEL_SENSOR_COLUMNS
        sensor_frame = pd.DataFrame(index=features_df.index)
        for col in sensor_cols:
            if col in features_df.columns:
                sensor_frame[col] = features_df[col]
            else:
                sensor_frame[col] = 0.0
        sensor_frame = sensor_frame.replace([np.inf, -np.inf], np.nan)
        # Сохраняем маску NaN для диагностики перед заполнением нулями
        self.sensor_data = sensor_frame.fillna(0.0).values.astype(np.float32)

        # 3. Таргеты (11 погодных переменных)
        target_cols = MODEL_TARGET_COLUMNS
        target_frame = features_df[target_cols].copy()
        target_frame = target_frame.replace([np.inf, -np.inf], np.nan)
        # Сохраняем сырые данные (с NaN) для корректной фильтрации окон
        self._raw_target_data = target_frame.values.astype(np.float32)
        # Заполненная версия — для обучения (NaN → 0.0)
        self.target_data = np.nan_to_num(self._raw_target_data, nan=0.0)

        # 4. Выравниваем по времени внешние мультимодельные прогнозы (Open-Meteo + Meteostat + Spread)
        ext_cols = EXTERNAL_FORECAST_COLUMNS
        if not external_df.empty and "timestamp" in external_df.columns:
            ext_df_15m = external_df.set_index("timestamp").resample("15min").ffill().reset_index()
            merged = pd.merge(features_df[["timestamp"]], ext_df_15m, on="timestamp", how="left")
        else:
            merged = features_df[["timestamp"]].copy()

        for col in ext_cols:
            if col not in merged.columns:
                merged[col] = 0.0

        merged[ext_cols] = merged[ext_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

        # Если скейлеры не переданы явно, пробуем загрузить их из scalers.json
        station_id = int(features_df["id"].iloc[0]) if "id" in features_df.columns else None
        if scalers is None and station_id is not None:
            scalers_file = os.path.join("config", "scalers.json")
            if os.path.exists(scalers_file):
                try:
                    with open(scalers_file, "r", encoding="utf-8") as f:
                        all_scalers = json.load(f)
                    station_key = f"station_{station_id}"
                    if station_key in all_scalers:
                        scalers = all_scalers[station_key]
                except Exception as e:
                    print(f"Error loading scalers in ClimateDataset: {e}")

        # Нормализация внешних мультимодельных прогнозов
        merged_norm = normalize_multimodel_external_df(merged[ext_cols], scalers)
        self.external_data = merged_norm[ext_cols].values.astype(np.float32)

        # 5. Генерируем скользящие окна с шагом из конфига (по умолчанию 24 шага = 6 часов)
        self.valid_indices = []
        total_steps = len(features_df)
        window_step = settings.get("model", {}).get("window_step", 24)
        for i in range(0, total_steps - self.lookback_steps - self.horizon_steps + 1, window_step):
            # Проверяем NaN на СЫРЫХ данных (до fillna), иначе проверка бессмысленна
            raw_target_window = self._raw_target_data[i + self.lookback_steps: i + self.lookback_steps + self.horizon_steps]
            if np.isnan(raw_target_window).mean() <= 0.5:
                self.valid_indices.append(i)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start_idx = self.valid_indices[idx]

        # encoder_input: (lookback_steps, sensors + static)
        enc_sensors = self.sensor_data[start_idx: start_idx + self.lookback_steps]
        enc_static = np.tile(self.static_features, (self.lookback_steps, 1))
        encoder_input = np.hstack([enc_sensors, enc_static])

        # decoder_input: (horizon_steps, 8 external)
        decoder_input = self.external_data[
            start_idx + self.lookback_steps: start_idx + self.lookback_steps + self.horizon_steps
        ]

        # target: (horizon_steps, 11)
        target = self.target_data[
            start_idx + self.lookback_steps: start_idx + self.lookback_steps + self.horizon_steps
        ]

        return (
            torch.tensor(encoder_input, dtype=torch.float32),
            torch.tensor(decoder_input, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Функция потерь — с поддержкой весов
# ---------------------------------------------------------------------------

def masked_mse_loss(preds: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
    """MSE лосс с маскированием NaN значений в таргете и взвешиванием признаков."""
    mask = ~torch.isnan(targets)
    if not mask.any():
        return torch.tensor(0.0, requires_grad=True, device=preds.device)

    diff = (preds - targets) ** 2
    if weights is not None:
        diff = diff * weights.to(preds.device)

    return diff[mask].mean()


def masked_huber_loss(preds: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor = None, delta: float = 1.0) -> torch.Tensor:
    """Huber (Smooth L1) лосс с маскированием NaN значений в таргете и взвешиванием признаков."""
    mask = ~torch.isnan(targets)
    if not mask.any():
        return torch.tensor(0.0, requires_grad=True, device=preds.device)

    diff = F.smooth_l1_loss(preds, targets, reduction="none", beta=delta)
    if weights is not None:
        diff = diff * weights.to(preds.device)

    return diff[mask].mean()


def masked_asymmetric_huber_loss(preds: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor = None, tau: float = 0.65, delta: float = 1.0) -> torch.Tensor:
    """
    Асимметричный Huber (Pinball) лосс.
    Штрафует занижение таргета (preds < targets) сильнее, чем завышение, если tau > 0.5.
    """
    mask = ~torch.isnan(targets)
    if not mask.any():
        return torch.tensor(0.0, requires_grad=True, device=preds.device)

    err = targets - preds  # Положительная ошибка = занижение модели
    huber = F.smooth_l1_loss(preds, targets, reduction="none", beta=delta)
    asym_weight = torch.where(err > 0, tau, 1.0 - tau)
    loss = huber * asym_weight

    if weights is not None:
        loss = loss * weights.to(preds.device)

    return loss[mask].mean()


def masked_slope_loss(preds: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
    """
    Temporal Derivative (Slope) Loss.
    Штрафует за несоответствие первых производных по времени (фазовое запаздывание пиков).
    """
    if preds.shape[1] <= 1:
        return torch.tensor(0.0, requires_grad=True, device=preds.device)

    pred_diff = preds[:, 1:, :] - preds[:, :-1, :]
    target_diff = targets[:, 1:, :] - targets[:, :-1, :]
    mask = ~torch.isnan(target_diff)

    if not mask.any():
        return torch.tensor(0.0, requires_grad=True, device=preds.device)

    diff_err = torch.abs(pred_diff - target_diff)
    if weights is not None:
        diff_err = diff_err * weights.to(preds.device)

    return diff_err[mask].mean()


def masked_pinball_loss(
    preds_q: torch.Tensor,
    targets: torch.Tensor,
    quantiles: list = QUANTILES,
    weights: torch.Tensor = None,
) -> torch.Tensor:
    """
    Pinball Loss (Quantile Loss) по трём квантилям.

    Args:
        preds_q: (B, T, num_targets * num_quantiles) — квантильные предсказания,
                 сгруппированные как [target_0_q0, target_0_q1, target_0_q2,
                                       target_1_q0, ..., target_N_q2]
        targets:  (B, T, num_targets) — истинные значения
        quantiles: список значений квантилей [0.1, 0.5, 0.9]
        weights:  (num_targets,) — веса по целевым переменным
    Returns:
        Скалярный Loss
    """
    num_q = len(quantiles)
    B, T, total = preds_q.shape
    num_targets = targets.shape[-1]

    # Перестраиваем в (B, T, num_targets, num_quantiles)
    preds_q = preds_q.view(B, T, num_targets, num_q)

    total_loss = torch.tensor(0.0, device=preds_q.device)
    valid_count = 0

    for qi, tau in enumerate(quantiles):
        pred = preds_q[..., qi]   # (B, T, num_targets)
        mask = ~torch.isnan(targets)
        if not mask.any():
            continue
        err = targets - pred       # положительная ошибка — недооценка
        pinball = torch.where(err >= 0, tau * err, (tau - 1.0) * err)  # (B, T, targets)
        pinball = torch.where(mask, pinball, torch.zeros_like(pinball))

        if weights is not None:
            pinball = pinball * weights.unsqueeze(0).unsqueeze(0)

        total_loss = total_loss + pinball.sum() / (mask.sum().clamp(min=1) * num_q)
        valid_count += 1

    return total_loss / max(valid_count, 1)


def masked_rain_event_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    rain_binary_index: int = 11,    # Индекс rain_binary в MODEL_TARGET_COLUMNS (12-й канал)
    weight: float = 1.0,
    pos_weight: float = None,       # Вес для балансировки: n_negative / n_positive
) -> torch.Tensor:
    """
    BCE-loss для прогноза «дождь/нет дождя» по выделенному 12-му каналу rain_binary.
    Использует чистый sigmoid и pos_weight для балансировки классов.
    """
    binary_target = targets[..., rain_binary_index]    # (B, T) — уже готовый 0/1 таргет
    binary_pred = preds[..., rain_binary_index]      # (B, T) — логиты из модели
    valid_mask = ~torch.isnan(binary_target)
    if not valid_mask.any():
        return torch.tensor(0.0, requires_grad=True, device=preds.device)

    pred_prob = torch.sigmoid(binary_pred[valid_mask])
    true_labels = torch.nan_to_num(binary_target[valid_mask], nan=0.0)

    # pos_weight компенсирует дисбаланс классов (n_dry / n_rain)
    if pos_weight is not None:
        pw = torch.tensor(float(pos_weight), dtype=torch.float32, device=preds.device)
        bce = F.binary_cross_entropy(pred_prob, true_labels, reduction="none")
        bce = (bce * (true_labels * (pw - 1.0) + 1.0)).mean()
    else:
        bce = F.binary_cross_entropy(pred_prob, true_labels, reduction="mean")

    return bce * weight


# ===========================================================================
# TFT BUILDING BLOCKS
# ===========================================================================

class GRN(nn.Module):
    """
    Gated Residual Network — базовый блок TFT.
    Принимает на вход x и опциональный контекстный вектор c,
    возвращает нелинейно преобразованный выход с gate-механизмом и skip-connection.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = None,
        dropout: float = 0.1,
        context_size: int = None,
    ):
        super().__init__()
        if output_size is None:
            output_size = hidden_size

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.context_proj = nn.Linear(context_size, hidden_size, bias=False) if context_size else None
        # Выход *2 для GLU: первая половина — значения, вторая — gate
        self.fc2 = nn.Linear(hidden_size, output_size * 2)
        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
        self.norm = nn.LayerNorm(output_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        h = self.input_proj(x)
        if context is not None and self.context_proj is not None:
            h = h + self.context_proj(context)
        h = F.elu(h)
        h = self.drop(h)
        h = self.fc2(h)
        h, gate = h.chunk(2, dim=-1)
        h = h * torch.sigmoid(gate)           # Gated Linear Unit
        h = self.drop(h)
        return self.norm(h + self.skip(x))    # Residual connection + LayerNorm


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN).
    Обучаемо взвешивает важность каждого входного признака на каждом временном шаге.
    Позволяет интерпретировать, какие датчики/прогнозы наиболее значимы.
    """

    def __init__(
        self,
        num_vars: int,
        hidden_size: int,
        dropout: float = 0.1,
        context_size: int = None,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.hidden_size = hidden_size

        # Каждая переменная (скаляр) → hidden_size
        self.var_projs = nn.ModuleList([nn.Linear(1, hidden_size) for _ in range(num_vars)])
        # Индивидуальный GRN для каждой переменной
        self.var_grns = nn.ModuleList([GRN(hidden_size, hidden_size, dropout=dropout) for _ in range(num_vars)])
        # GRN для вычисления весов отбора (на входе — конкатенация всех переменных)
        self.softmax_grn = GRN(
            input_size=num_vars * hidden_size,
            hidden_size=hidden_size,
            output_size=num_vars,
            dropout=dropout,
            context_size=context_size,
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        """
        x: (B, T, num_vars) или (B, num_vars) для статических переменных
        context: (B, context_size) — статический контекст
        Возвращает: selected (B, T, hidden_size), weights (B, T, num_vars)
        """
        has_time = x.dim() == 3
        if not has_time:
            x = x.unsqueeze(1)   # (B, 1, num_vars)
        B, T, _ = x.shape

        # Проецируем и обрабатываем каждую переменную отдельно
        var_outputs = []
        for i, (proj, grn) in enumerate(zip(self.var_projs, self.var_grns)):
            v = proj(x[:, :, i: i + 1])       # (B, T, H)
            v = grn(v)
            var_outputs.append(v)

        # stacked: (B, T, num_vars, H)
        stacked = torch.stack(var_outputs, dim=2)

        # Flatten для GRN отбора: (B, T, num_vars * H)
        flat = stacked.view(B, T, -1)

        # Веса важности переменных с опциональным контекстом
        if context is not None:
            ctx = context.unsqueeze(1).expand(-1, T, -1)      # (B, T, context_size)
            weights = F.softmax(self.softmax_grn(flat, ctx), dim=-1)  # (B, T, num_vars)
        else:
            weights = F.softmax(self.softmax_grn(flat), dim=-1)

        # Взвешенная сумма: (B, T, H)
        selected = (weights.unsqueeze(-1) * stacked).sum(dim=2)

        if not has_time:
            selected = selected.squeeze(1)
            weights = weights.squeeze(1)

        return selected, weights


class StaticCovariateEncoder(nn.Module):
    """
    Кодировщик статических ковариат (координаты станции, высота).
    Генерирует 4 контекстных вектора:
      cs — контекст для Variable Selection Networks
      ce — инициализация скрытого состояния Encoder LSTM
      cd — инициализация скрытого состояния Decoder LSTM
      ch — контекст для Temporal Self-Attention
    """

    def __init__(self, num_static: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(num_static, hidden_size), nn.ELU())
        self.cs_grn = GRN(hidden_size, hidden_size, dropout=dropout)
        self.ce_grn = GRN(hidden_size, hidden_size, dropout=dropout)
        self.cd_grn = GRN(hidden_size, hidden_size, dropout=dropout)
        self.ch_grn = GRN(hidden_size, hidden_size, dropout=dropout)

    def forward(self, static: torch.Tensor):
        """static: (B, num_static) → cs, ce, cd, ch: каждый (B, hidden_size)"""
        s = self.embedding(static)
        return self.cs_grn(s), self.ce_grn(s), self.cd_grn(s), self.ch_grn(s)


class InterpretableMultiHeadAttention(nn.Module):
    """
    Интерпретируемое Multi-Head Attention из TFT.
    Использует разделённую матрицу значений (V) между всеми головами —
    это позволяет усреднять веса внимания по головам для интерпретации.
    """

    def __init__(self, num_heads: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size должен делиться на num_heads"
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.hidden_size = hidden_size

        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        # Общий проектор V (interpretable — один на все головы)
        self.W_v = nn.Linear(hidden_size, self.head_size)
        self.W_o = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        q, k, v: (B, T, H)
        Возвращает: output (B, T, H), attention_weights (B, T, T)
        """
        B, T, _ = q.shape
        H, HS = self.num_heads, self.head_size

        # Q, K: (B, H, T, head_size)
        Q = self.W_q(q).view(B, T, H, HS).transpose(1, 2)
        K = self.W_k(k).view(B, T, H, HS).transpose(1, 2)
        # V общий: (B, 1, T, head_size) → (B, H, T, head_size)
        V = self.W_v(v).unsqueeze(1).expand(-1, H, -1, -1)

        scale = HS ** 0.5
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale    # (B, H, T, T)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # (B, H, T, head_size) → (B, T, H * head_size = hidden_size)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        out = self.W_o(out)

        # Усредняем веса по головам для интерпретации
        avg_attn = attn.mean(dim=1)   # (B, T, T)
        return out, avg_attn


# ===========================================================================
# TEMPORAL FUSION TRANSFORMER
# ===========================================================================

class TFTForecaster(nn.Module):
    """
    Temporal Fusion Transformer (Lim et al., 2019) для гиперлокального
    прогнозирования погоды.

    Входные данные явно разделены на три типа:
      - Статические ковариаты: широта, долгота, высота над уровнем моря
      - Наблюдаемые временные ряды (encoder): история датчиков за 24ч (96 шагов)
      - Известное будущее (decoder): прогноз Open-Meteo на 48ч (192 шага)

        Сигнатура forward совместима с Seq2SeqForecaster:
            encoder_input: (B, 96, num_encoder_vars + 3) — конкатенация [сенсоры | 3 статических]
            decoder_input: (B, 192, 8)  — 8 внешних прогнозных признаков
            return:        (B, 192, 11) — 11 прогнозируемых метеопеременных
    """

    def __init__(
        self,
        num_encoder_vars: int = len(MODEL_SENSOR_COLUMNS),    # 16 (включая dew_point_deficit)
        num_decoder_vars: int = len(EXTERNAL_FORECAST_COLUMNS), # 8 (Open-Meteo)
        num_static_vars: int = len(MODEL_STATIC_COLUMNS),     # 3
        output_size: int = len(MODEL_TARGET_COLUMNS),          # 12 (включая rain_binary)
        hidden_size: int = 128,
        num_heads: int = 4,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_encoder_vars = num_encoder_vars
        self.num_static_vars = num_static_vars
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers

        # --- Static Covariate Encoder ---
        self.static_encoder = StaticCovariateEncoder(num_static_vars, hidden_size, dropout)

        # --- Variable Selection Networks ---
        self.enc_vsn = VariableSelectionNetwork(num_encoder_vars, hidden_size, dropout, context_size=hidden_size)
        self.dec_vsn = VariableSelectionNetwork(num_decoder_vars, hidden_size, dropout, context_size=hidden_size)

        # --- Sequence Encoder / Decoder (LSTM) ---
        lstm_dropout = dropout if num_lstm_layers > 1 else 0.0
        self.enc_lstm = nn.LSTM(hidden_size, hidden_size, num_lstm_layers, batch_first=True, dropout=lstm_dropout)
        self.dec_lstm = nn.LSTM(hidden_size, hidden_size, num_lstm_layers, batch_first=True, dropout=lstm_dropout)

        # Нормализация после LSTM (Gating + Add & Norm)
        self.enc_gate = nn.Linear(hidden_size, hidden_size * 2)   # для GLU
        self.dec_gate = nn.Linear(hidden_size, hidden_size * 2)
        self.enc_norm = nn.LayerNorm(hidden_size)
        self.dec_norm = nn.LayerNorm(hidden_size)

        # --- Static Enrichment ---
        self.static_enrichment = GRN(hidden_size, hidden_size, dropout=dropout, context_size=hidden_size)

        # --- Temporal Self-Attention ---
        self.attn = InterpretableMultiHeadAttention(num_heads, hidden_size, dropout)
        self.attn_gate = nn.Linear(hidden_size, hidden_size * 2)
        self.attn_norm = nn.LayerNorm(hidden_size)

        # --- Position-wise Feed-Forward (GRN) ---
        self.ff_grn = GRN(hidden_size, hidden_size, dropout=dropout)
        self.ff_gate = nn.Linear(hidden_size, hidden_size * 2)
        self.ff_norm = nn.LayerNorm(hidden_size)

        # --- Output projection (Прямая проекция в output_size целевых переменных) ---
        self.output_size = output_size
        self.output_proj = nn.Linear(hidden_size, output_size)


    def _glu(self, x: torch.Tensor, gate_layer: nn.Linear, residual: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        """Gate → GLU → Add Residual → LayerNorm."""
        g = gate_layer(x)
        v, gate = g.chunk(2, dim=-1)
        gated = v * torch.sigmoid(gate)
        return norm(gated + residual)

    def _init_lstm_state(self, context: torch.Tensor):
        """Инициализирует (h0, c0) для LSTM из контекстного вектора."""
        h0 = context.unsqueeze(0).repeat(self.num_lstm_layers, 1, 1)  # (L, B, H)
        c0 = torch.zeros_like(h0)
        return h0, c0

    def forward(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
        """
        encoder_input: (B, lookback, num_encoder_vars + 3) — [датчики | 3 статических] (конкатенация)
        decoder_input: (B, horizon,  8) — внешние прогнозы Open-Meteo
        return:        (B, horizon, 11) — прогноз метеопеременных
        """
        horizon = decoder_input.size(1)

        # ---------------------------------------------------------------
        # 1. Разделяем encoder_input на сенсорные ряды и статику
        # ---------------------------------------------------------------
        sensor_seq = encoder_input[:, :, :self.num_encoder_vars]   # (B, 96, 15)
        static_raw = encoder_input[:, 0, self.num_encoder_vars:]   # (B, 3) — статика одинакова на всех шагах

        # ---------------------------------------------------------------
        # 2. Кодируем статические ковариаты → 4 контекстных вектора
        # ---------------------------------------------------------------
        cs, ce, cd, ch = self.static_encoder(static_raw)
        # cs: контекст для VSN
        # ce: инициализация encoder LSTM
        # cd: инициализация decoder LSTM
        # ch: контекст для temporal attention

        # ---------------------------------------------------------------
        # 3. Variable Selection (отбор значимых признаков)
        # ---------------------------------------------------------------
        enc_selected, self.last_enc_var_weights = self.enc_vsn(sensor_seq, context=cs)    # (B, 96, H)
        dec_selected, self.last_dec_var_weights = self.dec_vsn(decoder_input, context=cs)  # (B, 192, H)

        # ---------------------------------------------------------------
        # 4. Encoder LSTM — обрабатывает историю датчиков
        # ---------------------------------------------------------------
        h0_enc, c0_enc = self._init_lstm_state(ce)
        enc_lstm_out, (h_final, c_final) = self.enc_lstm(enc_selected, (h0_enc, c0_enc))
        # Gating + Add & Norm (encoder)
        enc_out = self._glu(enc_lstm_out, self.enc_gate, enc_selected, self.enc_norm)  # (B, 96, H)

        # ---------------------------------------------------------------
        # 5. Decoder LSTM — обрабатывает будущие прогнозы Open-Meteo
        #    Инициализируется финальным состоянием encoder (теплый старт)
        # ---------------------------------------------------------------
        dec_lstm_out, _ = self.dec_lstm(dec_selected, (h_final, c_final))
        # Gating + Add & Norm (decoder)
        dec_out = self._glu(dec_lstm_out, self.dec_gate, dec_selected, self.dec_norm)  # (B, 192, H)

        # ---------------------------------------------------------------
        # 6. Объединяем encoder и decoder для Temporal Self-Attention
        # ---------------------------------------------------------------
        combined = torch.cat([enc_out, dec_out], dim=1)   # (B, 288, H)

        # ---------------------------------------------------------------
        # 7. Static Enrichment — обогащаем статическим контекстом
        # ---------------------------------------------------------------
        ch_exp = ch.unsqueeze(1).expand(-1, combined.size(1), -1)   # (B, 288, H)
        enriched = self.static_enrichment(combined, context=ch_exp)  # (B, 288, H)

        # ---------------------------------------------------------------
        # 8. Interpretable Multi-Head Temporal Self-Attention
        # ---------------------------------------------------------------
        attn_out, self.last_attn_weights = self.attn(enriched, enriched, enriched)
        # Gating + Add & Norm (attention)
        enriched = self._glu(attn_out, self.attn_gate, enriched, self.attn_norm)  # (B, 288, H)

        # ---------------------------------------------------------------
        # 9. Position-wise Feed-Forward (GRN) + Gating
        # ---------------------------------------------------------------
        ff_out = self.ff_grn(enriched)
        enriched = self._glu(ff_out, self.ff_gate, enriched, self.ff_norm)        # (B, 288, H)

        # ---------------------------------------------------------------
        # 10. Берём только шаги декодера (последние horizon шагов)
        # ---------------------------------------------------------------
        dec_final = enriched[:, -horizon:, :]    # (B, 192, H)

        # ---------------------------------------------------------------
        # 11. Проецируем в целевые переменные (B, horizon, output_size)
        # ---------------------------------------------------------------
        return self.output_proj(dec_final)



# ===========================================================================
# Точка входа для быстрой проверки архитектуры
# ===========================================================================

if __name__ == "__main__":
    print("--- Проверка архитектуры Temporal Fusion Transformer ---")

    tft_cfg = settings.get("tft", {})
    model = TFTForecaster(
        hidden_size=tft_cfg.get("hidden_size", 128),
        num_heads=tft_cfg.get("num_heads", 4),
        num_lstm_layers=tft_cfg.get("num_lstm_layers", 2),
        dropout=tft_cfg.get("dropout", 0.1),
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Всего обучаемых параметров: {total_params:,}")

    # Имитируем батч из 4 примеров
    dummy_enc_features = len(MODEL_SENSOR_COLUMNS) + len(MODEL_STATIC_COLUMNS)
    dummy_enc = torch.randn(4, 96, dummy_enc_features)
    dummy_dec = torch.randn(4, 192, 8)   # 8 внешних прогнозных признаков

    out = model(dummy_enc, dummy_dec)

    print(f"encoder_input : {dummy_enc.shape}")
    print(f"decoder_input : {dummy_dec.shape}")
    print(f"Выход TFT     : {out.shape}  (ожидается [4, 192, {len(MODEL_TARGET_COLUMNS)}])")

    assert out.shape == (4, 192, len(MODEL_TARGET_COLUMNS)), f"Ошибка размерности: {out.shape}"
    print("✓ Проверка размерностей TFT успешна!")

    # Проверка маскированного лосса
    targets = out.clone().detach()
    targets[0, 10:20, :] = float("nan")
    loss = masked_mse_loss(out, targets)
    print(f"Маскированный MSE лосс: {loss.item():.6f}")
    assert not torch.isnan(loss)
    print("✓ Masked MSE лосс работает корректно!")

    # Проверка весов внимания
    attn_w = model.last_attn_weights
    print(f"Размерность весов Attention: {attn_w.shape}  (ожидается [4, 288, 288])")
    print("✓ Все проверки пройдены успешно!")
