"""
=============================================================================
МОДУЛЬ: Inversion Post-Processing Correction (inversion.py)
-----------------------------------------------------------------------------
НАЗНАЧЕНИЕ:
Векторизованная модель Ночной Инверсии на базе астрономического зенитного
угла Солнца (Solar Zenith Angle Inversion) и тихой ветровой обстановки (Блок 7.3).

Используется как в основном пайплайне прогнозирования (app.py), так и в
инструментах независимой офлайн-оценки качества на тестовом сплите (evaluate.py).
=============================================================================
"""

import numpy as np
import pandas as pd


def apply_inversion_correction(
    temp_raw: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    timestamps: np.ndarray,
    lat_deg: float,
    scale: float = 1.0,
    cloud_cover: np.ndarray = None,
) -> np.ndarray:
    """
    Применяет инверсионную поправку температуры на основе зенитного угла Солнца
    и скорости ветра (Блок 7.3).

    Parameters:
    -----------
    temp_raw : np.ndarray
        Массив исходных денормализованных предсказаний температуры (°C).
        Может быть 1D (horizon,) или 2D (N_windows, horizon).
    wind_u : np.ndarray
        Массив U-компоненты ветра (м/с), совпадает по форме с temp_raw.
    wind_v : np.ndarray
        Массив V-компоненты ветра (м/с), совпадает по форме с temp_raw.
    timestamps : np.ndarray / list
        Массив объектов datetime/Timestamp, совпадающий по форме с temp_raw.
    lat_deg : float
        Широта станции в градусах (например, 40.2).
    scale : float
        Коэффициент масштабирования поправки:
        0.0 = выключено (без поправки),
        0.5 = половинная поправка,
        1.0 = стандартная поправка.

    Returns:
    --------
    np.ndarray
        Скорректированный массив температуры той же формы.
    """
    if scale == 0.0:
        return temp_raw.copy()

    lat_rad = np.radians(lat_deg)
    out = temp_raw.astype(float).copy()
    w_speed = np.sqrt(wind_u**2 + wind_v**2)

    ts_array = np.asarray(timestamps)

    if ts_array.ndim == 1:
        doy = np.array([pd.Timestamp(t).dayofyear for t in ts_array])
        hour_frac = np.array([pd.Timestamp(t).hour + pd.Timestamp(t).minute / 60.0 for t in ts_array])
    else:
        doy = np.array([[pd.Timestamp(t).dayofyear for t in row] for row in ts_array])
        hour_frac = np.array([[pd.Timestamp(t).hour + pd.Timestamp(t).minute / 60.0 for t in row] for row in ts_array])

    # Астрономическое склонение солнца и часовой угол
    declination = np.radians(23.45 * np.sin(np.radians(360.0 / 365.0 * (doy - 81))))
    hour_angle = np.radians(15.0 * (hour_frac - 12.0))
    cos_zenith = np.sin(lat_rad) * np.sin(declination) + np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle)
    zenith_deg = np.degrees(np.arccos(np.clip(cos_zenith, -1.0, 1.0)))

    # Плавный фактор темноты (darkness_factor): 0.0 при дневном свете, 1.0 при абсолютной ночи
    darkness_factor = np.clip((zenith_deg - 87.0) / 9.0, 0.0, 1.0)
    # Плавный фактор ветра (Sigmoid Wind Calm Factor)
    wind_calm_factor = 1.0 / (1.0 + np.exp(4.0 * (w_speed - 1.8)))

    # Учет облачности (Clear Sky Factor): облака отражают ИК-излучение земли, отменяя инверсию
    if cloud_cover is not None:
        cc_arr = np.clip(np.asarray(cloud_cover, dtype=float), 0.0, 1.0)
        if len(cc_arr) > 0 and len(cc_arr) != len(temp_raw):
            x_old = np.linspace(0, 1, len(cc_arr))
            x_new = np.linspace(0, 1, len(temp_raw))
            cc_arr = np.interp(x_new, x_old, cc_arr)
        clear_sky_factor = (1.0 - cc_arr) ** 1.5
    else:
        clear_sky_factor = 1.0

    # Величина инверсионного снижения температуры (смягченный клип 0.0...2.0°C + удержание при облаках)
    inversion_drop = np.clip((1.8 - w_speed) * 0.7, 0.0, 2.0) * darkness_factor * wind_calm_factor * clear_sky_factor

    active_mask = (darkness_factor > 0.0) & (wind_calm_factor > 0.01)
    drop = np.where(active_mask, inversion_drop, 0.0)

    out -= drop * scale
    return out


def apply_rolling_bias_correction(
    temp_preds: np.ndarray,
    recent_observed_temp: float | np.ndarray,
    recent_forecast_temp: float | np.ndarray,
    decay_rate: float = 0.05
) -> np.ndarray:
    """
    Применяет адаптивную экспоненциальную коррекцию накопившегося смещения (Bias) за последние 24ч.
    
    \hat{y}_{t+k} = y_{TFT, t+k} + Bias_{24h} * exp(-decay_rate * k)
    """
    if temp_preds is None or len(temp_preds) == 0:
        return temp_preds

    out = temp_preds.astype(float).copy()

    if isinstance(recent_observed_temp, (list, np.ndarray)) and isinstance(recent_forecast_temp, (list, np.ndarray)):
        valid = ~np.isnan(recent_observed_temp) & ~np.isnan(recent_forecast_temp)
        if np.any(valid):
            bias = float(np.mean(np.array(recent_observed_temp)[valid] - np.array(recent_forecast_temp)[valid]))
        else:
            bias = 0.0
    else:
        try:
            bias = float(recent_observed_temp - recent_forecast_temp)
        except Exception:
            bias = 0.0

    # Ограничиваем максимальную поправку до 4°C во избежание выбросов
    bias = float(np.clip(bias, -4.0, 4.0))

    if out.ndim == 1:
        horizon = len(out)
        steps = np.arange(1, horizon + 1)
        decay = np.exp(-decay_rate * steps)
        out += bias * decay
    elif out.ndim == 2:
        horizon = out.shape[1]
        steps = np.arange(1, horizon + 1)
        decay = np.exp(-decay_rate * steps)
        out += bias * decay

    return out
