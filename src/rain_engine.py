"""
=============================================================================
MODULE: Rain Classification & Physics Filter (rain_engine.py)
-----------------------------------------------------------------------------
PURPOSE:
Dedicated precipitation classification and rain probability forecasting engine.

KEY FUNCTIONS & ALGORITHMS:
1. Standalone CatBoost precipitation classifier with isotonic probability calibration.
2. Physics Guardrails: smooth probability attenuation based on relative humidity
   thresholds (<75%) and dew point deficit constraints (>2.5°C).
   and dew point deficit constraints (>2.5°C).
3. Precipitation feature engineering pipeline and training orchestration.
=============================================================================
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, roc_auc_score

from project_paths import resolve_path
from data_pipeline import (
    compute_dew_point, add_dew_point_deficit_trend, add_pressure_drop_rate,
    add_vapor_pressure_deficit, add_equivalent_potential_temperature,
    add_humidity_trend, add_dry_spell_counter, add_rain_accumulation
)

# =====================================================================
# 1. SMOOTH PHYSICAL CORRECTION (Physics Guardrails)
# =====================================================================

def compute_smooth_physics_adjustment(
    p_rain: np.ndarray,
    temp: np.ndarray,
    dew_point: np.ndarray,
    humidity: np.ndarray,
    cloud_cover: np.ndarray = None,
) -> np.ndarray:
    """
    Smooth physical adaptation of precipitation probability.
    Smooth attenuation of rain probability P(rain) -> P'(rain) under high dew point deficit
    and low relative humidity without hard threshold cutoffs.
    """
    p_rain = np.nan_to_num(np.asarray(p_rain, dtype=np.float32), nan=0.0)
    temp = np.nan_to_num(np.asarray(temp, dtype=np.float32), nan=20.0)
    dew_point = np.nan_to_num(np.asarray(dew_point, dtype=np.float32), nan=10.0)
    humidity = np.nan_to_num(np.asarray(humidity, dtype=np.float32), nan=50.0)

    # 1. Dew Point Deficit (T - Td)
    deficit = np.maximum(0.0, temp - dew_point)

    # 2. Dew point deficit multiplier (Sigmoid Decay)
    deficit_factor = 1.0 / (1.0 + np.exp(0.75 * (deficit - 4.5)))

    # 3. Relative humidity multiplier (Humidity Factor)
    rh_norm = np.clip(humidity, 0.0, 100.0)
    humidity_factor = 1.0 / (1.0 + np.exp(-0.15 * (rh_norm - 55.0)))

    # Cumulative physical attenuation factor [0.05, 1.0]
    physics_multiplier = np.clip(deficit_factor * humidity_factor, 0.05, 1.0)

    # Optional cloud cover factor if available
    if cloud_cover is not None:
        cc_norm = np.clip(np.nan_to_num(np.asarray(cloud_cover, dtype=np.float32), nan=50.0), 0.0, 100.0)
        cloud_factor = 0.3 + 0.7 * (cc_norm / 100.0)
        physics_multiplier *= cloud_factor

    # Smooth adjustment P'(rain)
    adjusted_p_rain = p_rain * physics_multiplier
    return np.nan_to_num(np.clip(adjusted_p_rain, 0.0, 1.0), nan=0.0).astype(np.float32)


# =====================================================================
# 2. CATBOOST PRECIPITATION CLASSIFIER
# =====================================================================

class RainCatBoostClassifier:
    """
    Standalone precipitation classifier based on CatBoost.
    Trained completely independently of primary TFT / temperature model.
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.model = None
        self.calibrator = None
        self.optimal_threshold = 0.5
        self.feature_names = []
        os.makedirs(self.model_dir, exist_ok=True)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        val_df_raw: pd.DataFrame = None,
        iterations: int = 600,
        learning_rate: float = 0.05,
    ):
        self.feature_names = list(X_train.columns)

        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        scale_pos_weight = float(n_neg / max(n_pos, 1))
        print(f"[RainCatBoost] Class balance: pos={n_pos}, neg={n_neg}, scale_pos_weight={scale_pos_weight:.2f}")

        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=6,
            scale_pos_weight=scale_pos_weight,
            eval_metric="Logloss",
            random_seed=42,
            verbose=100,
        )
        self.model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            use_best_model=True,
        )

        val_probs_raw = self.model.predict_proba(X_val)[:, 1]

        print("[RainCatBoost] Training isotonic probability calibration...")
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        val_probs_calibrated = self.calibrator.fit_transform(val_probs_raw, y_val)

        if val_df_raw is not None and "temperature" in val_df_raw.columns and "dew_point" in val_df_raw.columns and "humidity" in val_df_raw.columns:
            print("[RainCatBoost] Applying smooth physics guardrail on validation...")
            temp = val_df_raw["temperature"].values
            dew_point = val_df_raw["dew_point"].values
            humidity = val_df_raw["humidity"].values
            cloud = val_df_raw["cloud_cover"].values if "cloud_cover" in val_df_raw.columns else None
            val_probs_adjusted = compute_smooth_physics_adjustment(
                val_probs_calibrated, temp, dew_point, humidity, cloud_cover=cloud
            )
        else:
            val_probs_adjusted = val_probs_calibrated

        best_th = 0.5
        best_f1 = 0.0
        thresholds = np.linspace(0.1, 0.9, 81)
        for th in thresholds:
            preds_binary = (val_probs_adjusted >= th).astype(int)
            score = f1_score(y_val, preds_binary, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_th = th

        self.optimal_threshold = float(best_th)

        valid_auc_mask = ~np.isnan(y_val) & ~np.isnan(val_probs_adjusted)
        if valid_auc_mask.any() and len(np.unique(y_val[valid_auc_mask])) > 1:
            auc_score = roc_auc_score(y_val[valid_auc_mask], val_probs_adjusted[valid_auc_mask])
            print(f"[RainCatBoost] Optimal threshold: {self.optimal_threshold:.3f} | Best F1: {best_f1:.4f} | ROC-AUC: {auc_score:.4f}")
        else:
            print(f"[RainCatBoost] Optimal threshold: {self.optimal_threshold:.3f} | Best F1: {best_f1:.4f} | ROC-AUC: N/A")

        self.save()

    def predict_proba(self, X: pd.DataFrame, df_raw: pd.DataFrame = None) -> np.ndarray:
        if self.model is None:
            raise ValueError("CatBoost model is not trained!")

        raw_probs = self.model.predict_proba(X[self.feature_names])[:, 1]
        calibrated_probs = self.calibrator.transform(raw_probs) if self.calibrator else raw_probs

        if df_raw is not None and "temperature" in df_raw.columns and "dew_point" in df_raw.columns and "humidity" in df_raw.columns:
            temp = df_raw["temperature"].values
            dew_point = df_raw["dew_point"].values
            humidity = df_raw["humidity"].values
            cloud = df_raw["cloud_cover"].values if "cloud_cover" in df_raw.columns else None
            adjusted_probs = compute_smooth_physics_adjustment(
                calibrated_probs, temp, dew_point, humidity, cloud_cover=cloud
            )
            return adjusted_probs
        return calibrated_probs

    def predict(self, X: pd.DataFrame, df_raw: pd.DataFrame = None) -> np.ndarray:
        probs = self.predict_proba(X, df_raw=df_raw)
        return (probs >= self.optimal_threshold).astype(int)

    def save(self):
        cb_path = os.path.join(self.model_dir, "rain_catboost.cbm")
        meta_path = os.path.join(self.model_dir, "rain_catboost_meta.pkl")
        self.model.save_model(cb_path)
        joblib.dump({
            "calibrator": self.calibrator,
            "optimal_threshold": self.optimal_threshold,
            "feature_names": self.feature_names,
        }, meta_path)
        print(f"[RainCatBoost] Model and metadata saved to {self.model_dir}")

    def load(self):
        cb_path = os.path.join(self.model_dir, "rain_catboost.cbm")
        meta_path = os.path.join(self.model_dir, "rain_catboost_meta.pkl")
        if os.path.exists(cb_path) and os.path.exists(meta_path):
            self.model = CatBoostClassifier()
            self.model.load_model(cb_path)
            meta = joblib.load(meta_path)
            self.calibrator = meta["calibrator"]
            self.optimal_threshold = meta["optimal_threshold"]
            self.feature_names = meta["feature_names"]
            print(f"[RainCatBoost] Model loaded successfully (threshold={self.optimal_threshold:.3f})")
        else:
            raise FileNotFoundError(f"Model files {cb_path} or {meta_path} not found!")


# =====================================================================
# 3. FEATURE ENGINEERING & TRAINING PIPELINE (Pipeline Helpers)
# =====================================================================

def prepare_rain_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds specialized feature set for precipitation classification."""
    df_feat = df.copy()

    if "dew_point" not in df_feat.columns and "temperature" in df_feat.columns and "humidity" in df_feat.columns:
        df_feat["dew_point"] = compute_dew_point(df_feat["temperature"], df_feat["humidity"])

    df_feat = add_dew_point_deficit_trend(df_feat)
    df_feat = add_pressure_drop_rate(df_feat)
    df_feat = add_vapor_pressure_deficit(df_feat)
    df_feat = add_equivalent_potential_temperature(df_feat)
    df_feat = add_humidity_trend(df_feat)
    df_feat = add_dry_spell_counter(df_feat)
    df_feat = add_rain_accumulation(df_feat)

    feature_cols = [
        "temperature", "humidity", "pressure", "dew_point", "dew_point_deficit",
        "dew_point_deficit_trend_1h", "dew_point_deficit_trend_3h",
        "pressure_drop_1h", "pressure_drop_3h", "vpd", "theta_e",
        "humidity_trend_3h", "steps_since_last_rain", "rain_sum_24h", "rain_sum_72h"
    ]

    for ext_col in ["precipitation", "cloud_cover", "temperature_2m", "relative_humidity_2m"]:
        if ext_col in df_feat.columns:
            feature_cols.append(ext_col)

    existing_cols = [c for c in feature_cols if c in df_feat.columns]
    return df_feat[existing_cols], df_feat


def run_training_pipeline():
    print("--- [Rain Pipeline] Starting standalone precipitation classifier training ---")

    processed_dir = resolve_path("data", "processed")
    dfs = []

    for candidate in [os.path.join("weather_data", "processed_dataset.csv"), os.path.join("data", "processed_dataset.csv")]:
        if os.path.exists(candidate):
            print(f"[Rain Pipeline] Loading concatenated dataset from {candidate}...")
            dfs.append(pd.read_csv(candidate))
            break

    if not dfs and os.path.exists(processed_dir):
        parquet_files = [
            os.path.join(processed_dir, f) for f in os.listdir(processed_dir)
            if f.startswith("station_") and f.endswith("_features.parquet")
        ]
        if parquet_files:
            print(f"[Rain Pipeline] Found {len(parquet_files)} parquet station feature files in {processed_dir}. Concatenating data...")
            for pf in parquet_files:
                try:
                    df_st = pd.read_parquet(pf)
                    dfs.append(df_st)
                except Exception as e:
                    print(f" Error reading {pf}: {e}")

    if not dfs:
        print("[Rain Pipeline] Error: No suitable data (CSV or Parquet) found in data/processed!")
        return

    df = pd.concat(dfs, ignore_index=True)
    if "rain" not in df.columns:
        print("[Rain Pipeline] Error: 'rain' column not found in dataset!")
        return

    df["rain_binary"] = (pd.to_numeric(df["rain"], errors="coerce").fillna(0.0) > 0.05).astype(int)

    X_features, df_raw = prepare_rain_features(df)
    y = df["rain_binary"].values

    split_idx = int(len(X_features) * 0.8)
    X_train, X_val = X_features.iloc[:split_idx], X_features.iloc[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    val_df_raw = df_raw.iloc[split_idx:]

    print(f"[Rain Pipeline] Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    classifier = RainCatBoostClassifier()
    classifier.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        val_df_raw=val_df_raw,
    )
    print("--- [Rain Pipeline] Precipitation classifier training completed successfully! ---")


if __name__ == "__main__":
    run_training_pipeline()