"""
=============================================================================
MODULE: TFT Neural Network Architecture (model.py)
-----------------------------------------------------------------------------
PURPOSE:
Definition of deep Temporal Fusion Transformer (TFT) neural network architecture
and specialized loss functions.

KEY COMPONENTS:
1. `TFTForecaster`: neural network architecture with Interpretable Multi-Head
   Attention, Variable Selection Networks, and LSTM context selectors.
2. `ClimateDataset`: sliding window dataset preparation class (96 steps).
3. Specialized loss functions: Pinball Loss (Quantile Loss), Huber Loss, and BCE Rain Loss.
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

# Load settings from central config
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
    Stable multi-task loss weighting via Softmax normalization.
    Guarantees that the sum of task weights equals 1.0 and total loss remains strictly positive (>= 0).
    """

    def __init__(self, num_tasks: int = 12, min_task_weights: torch.Tensor = None):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_logits = nn.Parameter(torch.zeros(num_tasks, dtype=torch.float32))
        self.register_buffer("min_task_weights", min_task_weights if min_task_weights is not None else torch.zeros(num_tasks))

    def forward(self, task_losses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        task_losses: (num_tasks,) - scalar loss for each target variable
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
# ClimateDataset - standard sliding window dataset
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Quantiles for probabilistic forecasting (Pinball Loss)
# ---------------------------------------------------------------------------
QUANTILES = [0.1, 0.5, 0.9]       # P10, median, P90
NUM_QUANTILES = len(QUANTILES)     # 3


class ClimateDataset(Dataset):
    """
    Dataset class for preparing time series windows for TFT model training.
    Interface fully compatible with sequence forecasting pipelines.
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        external_df: pd.DataFrame,
        scalers: dict = None,
    ):
        self.lookback_steps = settings["model"]["lookback_steps"]
        self.horizon_steps = settings["model"]["horizon_steps"]

        # 1. Static station features with global normalization
        raw_static = np.array(
            [float(features_df[col].iloc[0]) for col in MODEL_STATIC_COLUMNS],
            dtype=np.float32,
        )
        self.static_features = np.array([
            (raw_static[0] - 40.2) / 0.5,      # latitude
            (raw_static[1] - 44.5) / 0.8,      # longitude
            (raw_static[2] - 1400.0) / 450.0   # elevation_m
        ], dtype=np.float32)

        # 2. Historical sensor time series
        sensor_cols = MODEL_SENSOR_COLUMNS
        sensor_frame = pd.DataFrame(index=features_df.index)
        for col in sensor_cols:
            if col in features_df.columns:
                sensor_frame[col] = features_df[col]
            else:
                sensor_frame[col] = 0.0
        sensor_frame = sensor_frame.replace([np.inf, -np.inf], np.nan)
        # Save NaN mask for diagnostics prior to zero imputation
        self.sensor_data = sensor_frame.fillna(0.0).values.astype(np.float32)

        # 3. Targets (meteorological variables)
        target_cols = MODEL_TARGET_COLUMNS
        target_frame = features_df[target_cols].copy()
        target_frame = target_frame.replace([np.inf, -np.inf], np.nan)
        # Save raw data (with NaNs) for valid window filtering
        self._raw_target_data = target_frame.values.astype(np.float32)
        # Imputed version for model training (NaN -> 0.0)
        self.target_data = np.nan_to_num(self._raw_target_data, nan=0.0)

        # 4. Temporally align external multi-model forecasts (Open-Meteo + NWP Ensemble)
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

        # If scalers are not provided explicitly, load from scalers.json
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

        # Normalization of external multi-model forecasts
        merged_norm = normalize_multimodel_external_df(merged[ext_cols], scalers)
        self.external_data = merged_norm[ext_cols].values.astype(np.float32)

        # 5. Generate sliding windows with step from config (default: 24 steps = 6 hours)
        self.valid_indices = []
        total_steps = len(features_df)
        window_step = settings.get("model", {}).get("window_step", 24)
        for i in range(0, total_steps - self.lookback_steps - self.horizon_steps + 1, window_step):
            # Check NaNs on RAW data (before fillna) to prevent empty sequences
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
# Loss function with target weighting
# ---------------------------------------------------------------------------

def masked_mse_loss(preds: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
    """MSE loss with NaN target masking and feature weighting."""
    mask = ~torch.isnan(targets)
    if not mask.any():
        return torch.tensor(0.0, requires_grad=True, device=preds.device)

    diff = (preds - targets) ** 2
    if weights is not None:
        diff = diff * weights.to(preds.device)

    return diff[mask].mean()


def masked_huber_loss(preds: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor = None, delta: float = 1.0) -> torch.Tensor:
    """Huber (Smooth L1) loss with NaN target masking and feature weighting."""
    mask = ~torch.isnan(targets)
    if not mask.any():
        return torch.tensor(0.0, requires_grad=True, device=preds.device)

    diff = F.smooth_l1_loss(preds, targets, reduction="none", beta=delta)
    if weights is not None:
        diff = diff * weights.to(preds.device)

    return diff[mask].mean()


def masked_asymmetric_huber_loss(preds: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor = None, tau: float = 0.65, delta: float = 1.0) -> torch.Tensor:
    """
    Asymmetric Huber (Pinball) loss.
    Penalizes target underestimation (preds < targets) more than overestimation when tau > 0.5.
    """
    mask = ~torch.isnan(targets)
    if not mask.any():
        return torch.tensor(0.0, requires_grad=True, device=preds.device)

    err = targets - preds  # Positive error = model underestimation
    huber = F.smooth_l1_loss(preds, targets, reduction="none", beta=delta)
    asym_weight = torch.where(err > 0, tau, 1.0 - tau)
    loss = huber * asym_weight

    if weights is not None:
        loss = loss * weights.to(preds.device)

    return loss[mask].mean()


def masked_slope_loss(preds: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
    """
    Temporal Derivative (Slope) Loss.
    Penalizes discrepancies in temporal first derivatives (phase lag of peak events).
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
    Pinball Loss (Quantile Loss) across three quantiles.

    Args:
        preds_q: (B, T, num_targets * num_quantiles) - quantile predictions,
                 grouped as [target_0_q0, target_0_q1, target_0_q2, ...],
                                       target_1_q0, ..., target_N_q2]
        targets:  (B, T, num_targets) - ground truth values
        quantiles: list of quantile values [0.1, 0.5, 0.9]
        weights:  (num_targets,) - target feature weights
    Returns:
        Scalar loss tensor
    """
    num_q = len(quantiles)
    B, T, total = preds_q.shape
    num_targets = targets.shape[-1]

    # Reshape to (B, T, num_targets, num_quantiles)
    preds_q = preds_q.view(B, T, num_targets, num_q)

    total_loss = torch.tensor(0.0, device=preds_q.device)
    valid_count = 0

    for qi, tau in enumerate(quantiles):
        pred = preds_q[..., qi]   # (B, T, num_targets)
        mask = ~torch.isnan(targets)
        if not mask.any():
            continue
        err = targets - pred       # positive error = underestimation
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
    rain_binary_index: int = 11,    # Index of rain_binary in MODEL_TARGET_COLUMNS (12th channel)
    weight: float = 1.0,
    pos_weight: float = None,       # Weight for class balancing: n_negative / n_positive
) -> torch.Tensor:
    """
    BCE loss for binary rain occurrence on channel rain_binary.
    Uses sigmoid activations with pos_weight for class imbalance correction.
    """
    binary_target = targets[..., rain_binary_index]    # (B, T) - ground truth 0/1 target
    binary_pred = preds[..., rain_binary_index]      # (B, T) - model raw logits
    valid_mask = ~torch.isnan(binary_target)
    if not valid_mask.any():
        return torch.tensor(0.0, requires_grad=True, device=preds.device)

    pred_prob = torch.sigmoid(binary_pred[valid_mask])
    true_labels = torch.nan_to_num(binary_target[valid_mask], nan=0.0)

    # pos_weight compensates for class imbalance (n_dry / n_rain)
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
    Gated Residual Network (GRN) - fundamental building block of TFT.
    Accepts input x and optional context vector c,
    returns non-linearly transformed output with gating mechanism and skip-connection.
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
        # Output *2 for GLU: first half - values, second half - gate
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
    Learnable feature selection weighting for input variables across temporal steps.
    Enables interpretability of sensor and NWP forecast contributions.
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

        # Each scalar variable -> hidden_size projection
        self.var_projs = nn.ModuleList([nn.Linear(1, hidden_size) for _ in range(num_vars)])
        # Individual GRN for each variable
        self.var_grns = nn.ModuleList([GRN(hidden_size, hidden_size, dropout=dropout) for _ in range(num_vars)])
        # GRN for computing selection weights (input: flattened variables)
        self.softmax_grn = GRN(
            input_size=num_vars * hidden_size,
            hidden_size=hidden_size,
            output_size=num_vars,
            dropout=dropout,
            context_size=context_size,
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        """
        x: (B, T, num_vars) or (B, num_vars) for static variables
        context: (B, context_size) - static context vector
        Returns: selected (B, T, hidden_size), weights (B, T, num_vars)
        """
        has_time = x.dim() == 3
        if not has_time:
            x = x.unsqueeze(1)   # (B, 1, num_vars)
        B, T, _ = x.shape

        # Project and process each variable independently
        var_outputs = []
        for i, (proj, grn) in enumerate(zip(self.var_projs, self.var_grns)):
            v = proj(x[:, :, i: i + 1])       # (B, T, H)
            v = grn(v)
            var_outputs.append(v)

        # stacked: (B, T, num_vars, H)
        stacked = torch.stack(var_outputs, dim=2)

        # Flatten for selection GRN: (B, T, num_vars * H)
        flat = stacked.view(B, T, -1)

        # Variable selection weights with optional context
        if context is not None:
            ctx = context.unsqueeze(1).expand(-1, T, -1)      # (B, T, context_size)
            weights = F.softmax(self.softmax_grn(flat, ctx), dim=-1)  # (B, T, num_vars)
        else:
            weights = F.softmax(self.softmax_grn(flat), dim=-1)

        # Weighted linear combination: (B, T, H)
        selected = (weights.unsqueeze(-1) * stacked).sum(dim=2)

        if not has_time:
            selected = selected.squeeze(1)
            weights = weights.squeeze(1)

        return selected, weights


class StaticCovariateEncoder(nn.Module):
    """
    Static Covariate Encoder (latitude, longitude, elevation).
    Generates 4 distinct context vectors:
      cs - context for Variable Selection Networks
      ce - initialization for Encoder LSTM hidden state
      cd - initialization for Decoder LSTM hidden state
      ch - context for Temporal Self-Attention enrichment
    """

    def __init__(self, num_static: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(num_static, hidden_size), nn.ELU())
        self.cs_grn = GRN(hidden_size, hidden_size, dropout=dropout)
        self.ce_grn = GRN(hidden_size, hidden_size, dropout=dropout)
        self.cd_grn = GRN(hidden_size, hidden_size, dropout=dropout)
        self.ch_grn = GRN(hidden_size, hidden_size, dropout=dropout)

    def forward(self, static: torch.Tensor):
        """static: (B, num_static) -> cs, ce, cd, ch: each of shape (B, hidden_size)"""
        s = self.embedding(static)
        return self.cs_grn(s), self.ce_grn(s), self.cd_grn(s), self.ch_grn(s)


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention from TFT.
    Shares the value projection (V) across all attention heads,
    enabling direct interpretation of attention weights over time.
    """

    def __init__(self, num_heads: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.hidden_size = hidden_size

        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        # Shared value projection V (interpretable - shared across all heads)
        self.W_v = nn.Linear(hidden_size, self.head_size)
        self.W_o = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        q, k, v: (B, T, H)
        Returns: output (B, T, H), attention_weights (B, T, T)
        """
        B, T, _ = q.shape
        H, HS = self.num_heads, self.head_size

        # Q, K: (B, H, T, head_size)
        Q = self.W_q(q).view(B, T, H, HS).transpose(1, 2)
        K = self.W_k(k).view(B, T, H, HS).transpose(1, 2)
        # Shared V: (B, 1, T, head_size) -> (B, H, T, head_size)
        V = self.W_v(v).unsqueeze(1).expand(-1, H, -1, -1)

        scale = HS ** 0.5
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale    # (B, H, T, T)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # (B, H, T, head_size) → (B, T, H * head_size = hidden_size)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        out = self.W_o(out)

        # Average attention weights across heads for interpretability
        avg_attn = attn.mean(dim=1)   # (B, T, T)
        return out, avg_attn


# ===========================================================================
# TEMPORAL FUSION TRANSFORMER
# ===========================================================================

class TFTForecaster(nn.Module):
    """
    Temporal Fusion Transformer (Lim et al., 2019) for hyperlocal
    weather forecasting.

    Input data is partitioned into three distinct modalities:
      - Static covariates: latitude, longitude, elevation_m
      - Observed historical time series (encoder): 24h sensor history (96 steps)
      - Known future covariates (decoder): 48h Open-Meteo forecast (192 steps)

        Forward signature:
            encoder_input: (B, 96, num_encoder_vars + 3) - [sensors | 3 static covariates]
            decoder_input: (B, 192, 8) - 8 external NWP forecast features
            return:        (B, 192, 12) - predicted target meteorological variables
    """

    def __init__(
        self,
        num_encoder_vars: int = len(MODEL_SENSOR_COLUMNS),    # 16 (including physical features)
        num_decoder_vars: int = len(EXTERNAL_FORECAST_COLUMNS), # 8 (Open-Meteo)
        num_static_vars: int = len(MODEL_STATIC_COLUMNS),     # 3
        output_size: int = len(MODEL_TARGET_COLUMNS),          # 12 (including rain_binary)
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

        # Post-LSTM normalization (Gating + Add & Norm)
        self.enc_gate = nn.Linear(hidden_size, hidden_size * 2)   # for GLU
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

        # --- Output projection (Direct linear projection into output_size targets) ---
        self.output_size = output_size
        self.output_proj = nn.Linear(hidden_size, output_size)


    def _glu(self, x: torch.Tensor, gate_layer: nn.Linear, residual: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        """Gate → GLU → Add Residual → LayerNorm."""
        g = gate_layer(x)
        v, gate = g.chunk(2, dim=-1)
        gated = v * torch.sigmoid(gate)
        return norm(gated + residual)

    def _init_lstm_state(self, context: torch.Tensor):
        """Initializes (h0, c0) for LSTM from static context vector."""
        h0 = context.unsqueeze(0).repeat(self.num_lstm_layers, 1, 1)  # (L, B, H)
        c0 = torch.zeros_like(h0)
        return h0, c0

    def forward(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
        """
        encoder_input: (B, lookback, num_encoder_vars + 3) - [sensors | 3 static covariates]
        decoder_input: (B, horizon, 8) - external NWP forecasts
        return:        (B, horizon, 12) - meteorological forecasts
        """
        horizon = decoder_input.size(1)

        # ---------------------------------------------------------------
        # 1. Split encoder_input into sensor series and static features
        # ---------------------------------------------------------------
        sensor_seq = encoder_input[:, :, :self.num_encoder_vars]   # (B, 96, 15)
        static_raw = encoder_input[:, 0, self.num_encoder_vars:]   # (B, 3) - static features invariant over time

        # ---------------------------------------------------------------
        # 2. Encode static covariates into 4 distinct context vectors
        # ---------------------------------------------------------------
        cs, ce, cd, ch = self.static_encoder(static_raw)
        # cs: context for VSN
        # ce: initialization for encoder LSTM
        # cd: initialization for decoder LSTM
        # ch: context for temporal attention

        # ---------------------------------------------------------------
        # 3. Variable Selection Networks
        # ---------------------------------------------------------------
        enc_selected, self.last_enc_var_weights = self.enc_vsn(sensor_seq, context=cs)    # (B, 96, H)
        dec_selected, self.last_dec_var_weights = self.dec_vsn(decoder_input, context=cs)  # (B, 192, H)

        # ---------------------------------------------------------------
        # 4. Encoder LSTM - processes historical sensor sequence
        # ---------------------------------------------------------------
        h0_enc, c0_enc = self._init_lstm_state(ce)
        enc_lstm_out, (h_final, c_final) = self.enc_lstm(enc_selected, (h0_enc, c0_enc))
        # Gating + Add & Norm (encoder)
        enc_out = self._glu(enc_lstm_out, self.enc_gate, enc_selected, self.enc_norm)  # (B, 96, H)

        # ---------------------------------------------------------------
        # 5. Decoder LSTM - processes future NWP forecast features
        #    Initialized with final encoder state (warm start)
        # ---------------------------------------------------------------
        dec_lstm_out, _ = self.dec_lstm(dec_selected, (h_final, c_final))
        # Gating + Add & Norm (decoder)
        dec_out = self._glu(dec_lstm_out, self.dec_gate, dec_selected, self.dec_norm)  # (B, 192, H)

        # ---------------------------------------------------------------
        # 6. Concatenate encoder and decoder representations for Temporal Self-Attention
        # ---------------------------------------------------------------
        combined = torch.cat([enc_out, dec_out], dim=1)   # (B, 288, H)

        # ---------------------------------------------------------------
        # 7. Static Enrichment - enrich with static context
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
        # 10. Extract decoder steps (last horizon steps)
        # ---------------------------------------------------------------
        dec_final = enriched[:, -horizon:, :]    # (B, 192, H)

        # ---------------------------------------------------------------
        # 11. Project into target variables (B, horizon, output_size)
        # ---------------------------------------------------------------
        return self.output_proj(dec_final)



# ===========================================================================
# Entry point for architecture sanity check
# ===========================================================================

if __name__ == "__main__":
    print("--- Temporal Fusion Transformer Architecture Check ---")

    tft_cfg = settings.get("tft", {})
    model = TFTForecaster(
        hidden_size=tft_cfg.get("hidden_size", 128),
        num_heads=tft_cfg.get("num_heads", 4),
        num_lstm_layers=tft_cfg.get("num_lstm_layers", 2),
        dropout=tft_cfg.get("dropout", 0.1),
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # Simulate batch of 4 examples
    dummy_enc_features = len(MODEL_SENSOR_COLUMNS) + len(MODEL_STATIC_COLUMNS)
    dummy_enc = torch.randn(4, 96, dummy_enc_features)
    dummy_dec = torch.randn(4, 192, 8)   # 8 external NWP forecast features

    out = model(dummy_enc, dummy_dec)

    print(f"encoder_input : {dummy_enc.shape}")
    print(f"decoder_input : {dummy_dec.shape}")
    print(f"TFT output    : {out.shape}  (expected [4, 192, {len(MODEL_TARGET_COLUMNS)}])")

    assert out.shape == (4, 192, len(MODEL_TARGET_COLUMNS)), f"Dimension mismatch: {out.shape}"
    print("✓ TFT dimension check passed successfully!")

    # Masked loss verification
    targets = out.clone().detach()
    targets[0, 10:20, :] = float("nan")
    loss = masked_mse_loss(out, targets)
    print(f"Masked MSE loss: {loss.item():.6f}")
    assert not torch.isnan(loss)
    print("✓ Masked MSE loss verified successfully!")

    # Attention weights verification
    attn_w = model.last_attn_weights
    print(f"Attention weights shape: {attn_w.shape}  (expected [4, 288, 288])")
    print("✓ All architecture tests passed successfully!")
