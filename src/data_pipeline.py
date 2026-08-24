"""
=============================================================================
MODULE: Feature Engineering & Preprocessing Pipeline (data_pipeline.py)
-----------------------------------------------------------------------------
PURPOSE:
Central data processing pipeline and thermodynamic feature engineering engine.

KEY FUNCTIONS & COMPUTATIONS:
1. Atmospheric thermodynamic indices: dew point, dew point deficit,
   vapor pressure deficit (VPD), equivalent potential temperature (theta_e),
   and wet-bulb temperature approximation.
2. Per-station data scaling (Standard / Robust Scalers).
3. Active station filtering and stratified subset selection.
=============================================================================
"""

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

CARDINAL_TO_DEGREES = {
    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5,
    "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
    "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
    "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5
}


def convert_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts wind direction to degrees and decomposes the wind vector
    into orthogonal components wind_u and wind_v.
    """
    df_feat = df.copy()
    if "wind direction" not in df_feat.columns or "wind speed" not in df_feat.columns:
        return df_feat

    degrees = df_feat["wind direction"].map(CARDINAL_TO_DEGREES).fillna(0.0)
    rads = np.radians(degrees)

    speed = df_feat["wind speed"].fillna(0.0)
    df_feat["wind_u"] = speed * np.sin(rads)
    df_feat["wind_v"] = speed * np.cos(rads)

    # Remove legacy wind direction columns
    df_feat = df_feat.drop(columns=["wind direction", "wind speed"])
    return df_feat


def add_wind_scalar(df: pd.DataFrame) -> pd.DataFrame:
    """Computes scalar wind speed wind_speed_scalar from orthogonal wind_u and wind_v."""
    df_feat = df.copy()
    if "wind_u" in df_feat.columns and "wind_v" in df_feat.columns:
        df_feat["wind_speed_scalar"] = np.sqrt(df_feat["wind_u"]**2 + df_feat["wind_v"]**2)
    elif "wind_speed_scalar" not in df_feat.columns:
        df_feat["wind_speed_scalar"] = 0.0
    return df_feat


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds cyclical sine and cosine harmonics for time of day, month, and day of year."""
    df_feat = df.copy()
    timestamps = df_feat["timestamp"]

    # Time of day (24-hour diurnal cycle)
    hour = timestamps.dt.hour + timestamps.dt.minute / 60.0
    df_feat["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df_feat["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    # Month of year (12-month annual cycle)
    month = timestamps.dt.month - 1 + timestamps.dt.day / 31.0
    df_feat["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df_feat["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    # Day of year (365-day seasonal cycle)
    doy = timestamps.dt.dayofyear - 1
    df_feat["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    df_feat["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)

    return df_feat


def compute_dew_point(temp: pd.Series, rh: pd.Series) -> pd.Series:
    """Magnus-Tetens formula for dew point calculation."""
    beta = 17.67
    lam = 243.5
    rh_clipped = rh.clip(1.0, 100.0)

    gamma = (beta * temp) / (lam + temp) + np.log(rh_clipped / 100.0)
    dew_point = (lam * gamma) / (beta - gamma)
    return dew_point


def add_vapor_pressure_deficit(df: pd.DataFrame) -> pd.DataFrame:
    """Computes Vapor Pressure Deficit (VPD) in kPa."""
    df_feat = df.copy()
    if "temperature" in df_feat.columns and "humidity" in df_feat.columns:
        T = df_feat["temperature"]
        RH = df_feat["humidity"].clip(1.0, 100.0)
        e_s = 0.61078 * np.exp(17.27 * T / (T + 237.3))
        df_feat["vpd"] = (e_s * (1.0 - RH / 100.0)).clip(lower=0.0)
    else:
        df_feat["vpd"] = 0.0
    return df_feat


def add_equivalent_potential_temperature(df: pd.DataFrame) -> pd.DataFrame:
    """Computes equivalent potential temperature (theta_e) in °C."""
    df_feat = df.copy()
    if "temperature" in df_feat.columns:
        T_k = df_feat["temperature"] + 273.15
        P = df_feat.get("pressure", pd.Series(1013.25, index=df_feat.index)).clip(lower=100.0)
        q = df_feat.get("specific_humidity", pd.Series(0.0, index=df_feat.index))
        theta_e_k = (T_k + 2.5 * q) * ((1000.0 / P) ** 0.286)
        df_feat["theta_e"] = theta_e_k - 273.15
    else:
        df_feat["theta_e"] = 0.0
    return df_feat


def add_pressure_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Computes barometric pressure trend over the last 3 hours (12 steps of 15 min)."""
    df_feat = df.copy()
    if "pressure" in df_feat.columns:
        df_feat["pressure_trend_3h"] = df_feat["pressure"].diff(12).fillna(0.0)
    return df_feat


def add_temperature_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    """Computes temperature derivatives: 1h, 3h trends, and 1h thermal acceleration."""
    df_feat = df.copy()
    if "temperature" in df_feat.columns:
        df_feat["temp_trend_1h"] = df_feat["temperature"].diff(4).fillna(0.0)
        df_feat["temp_trend_3h"] = df_feat["temperature"].diff(12).fillna(0.0)
        df_feat["temp_accel_1h"] = df_feat["temp_trend_1h"].diff(4).fillna(0.0)
    return df_feat


def add_inversion_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Computes nocturnal temperature inversion risk index (calm wind + nocturnal radiation cooling + clear skies)."""
    df_feat = df.copy()
    if "hour_cos" in df_feat.columns:
        wind_speed = df_feat.get("wind_speed_scalar", 0.0)
        cloud_cover = df_feat.get("cloud_cover", 0.0)
        night_factor = np.maximum(0.0, -df_feat["hour_cos"])
        wind_calm = np.exp(-wind_speed / 2.0)
        cloud_clear = np.clip(1.0 - cloud_cover, 0.0, 1.0) ** 1.5
        df_feat["inversion_risk"] = night_factor * wind_calm * cloud_clear
    else:
        df_feat["inversion_risk"] = 0.0
    return df_feat


def add_solar_zenith_angle(df: pd.DataFrame, station_lat: float = 40.2) -> pd.DataFrame:
    """Computes solar zenith angle and clear-sky potential solar radiation."""
    df_feat = df.copy()
    timestamps = df_feat["timestamp"]
    doy = timestamps.dt.dayofyear
    hour = timestamps.dt.hour + timestamps.dt.minute / 60.0

    declination = np.radians(23.45 * np.sin(np.radians(360.0 / 365.0 * (doy - 81))))
    lat_rad = np.radians(station_lat)
    hour_angle = np.radians(15.0 * (hour - 12.0))

    cos_zenith = np.sin(lat_rad) * np.sin(declination) + np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle)
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)

    df_feat["solar_zenith_angle"] = np.degrees(np.arccos(cos_zenith))
    df_feat["potential_solar_radiation"] = np.maximum(0.0, cos_zenith)
    return df_feat


def add_sky_clearness(df: pd.DataFrame) -> pd.DataFrame:
    """Computes atmospheric transparency index (sky_clearness) from lux meter readings and theoretical insolation."""
    df_feat = df.copy()
    if "lux" in df_feat.columns and "potential_solar_radiation" in df_feat.columns:
        lux = df_feat["lux"].fillna(0.0).clip(lower=0.0)
        pot_rad = df_feat["potential_solar_radiation"]
        theoretical_lux = pot_rad * 120000.0
        day_mask = pot_rad > 0.05
        clearness = np.where(day_mask, lux / (theoretical_lux + 1e-5), 0.0)
        df_feat["sky_clearness"] = np.clip(clearness, 0.0, 1.5).astype(np.float32)
    else:
        df_feat["sky_clearness"] = 0.0
    return df_feat


def add_pm_moisture_index(df: pd.DataFrame) -> pd.DataFrame:
    """Computes particulate hygroscopic index and fog nucleation potential when RH > 80%."""
    df_feat = df.copy()
    if "pm2_5" in df_feat.columns and "pm10" in df_feat.columns and "humidity" in df_feat.columns:
        pm2_5 = df_feat["pm2_5"].fillna(0.0).clip(lower=0.0)
        pm10 = df_feat["pm10"].fillna(0.0).clip(lower=0.0)
        RH = df_feat["humidity"].clip(1.0, 100.0) / 100.0
        pm_ratio = pm2_5 / (pm10 + 1e-5)
        moisture_factor = np.where(RH > 0.8, np.exp((RH - 0.8) * 10.0) - 1.0, 0.0)
        df_feat["pm_moisture_index"] = (pm_ratio * moisture_factor).clip(0.0, 10.0)
    else:
        df_feat["pm_moisture_index"] = 0.0
    return df_feat


def add_dry_spell_counter(df: pd.DataFrame) -> pd.DataFrame:
    """Computes consecutive dry step intervals (steps_since_last_rain)."""
    df_feat = df.copy()
    if "rain" in df_feat.columns:
        rain_vals = pd.to_numeric(df_feat["rain"], errors="coerce").fillna(0.0)
        is_dry = (rain_vals <= 0.05).astype(int)
        cumsum_rain = (rain_vals > 0.05).cumsum()
        steps_dry = is_dry.groupby(cumsum_rain).cumsum()
        df_feat["steps_since_last_rain"] = steps_dry.astype(np.float32)
    else:
        df_feat["steps_since_last_rain"] = 0.0
    return df_feat


def add_humidity_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Computes relative humidity trend over the last 3 hours (12 steps of 15 min)."""
    df_feat = df.copy()
    if "humidity" in df_feat.columns:
        df_feat["humidity_trend_3h"] = df_feat["humidity"].diff(12).fillna(0.0)
    return df_feat


def add_rain_accumulation(df: pd.DataFrame) -> pd.DataFrame:
    """Adds cumulative precipitation over 24h (96 steps) and 72h (288 steps)."""
    df_feat = df.copy()
    if "rain" in df_feat.columns:
        rain_series = pd.to_numeric(df_feat["rain"], errors="coerce").fillna(0.0)
        df_feat["rain_sum_24h"] = rain_series.rolling(window=96, min_periods=1).sum()
        df_feat["rain_sum_72h"] = rain_series.rolling(window=288, min_periods=1).sum()
    return df_feat


def add_lag_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Generates lag features: 1 hour ago (4 steps), 24 hours ago (96 steps), and 7 days ago (672 steps for temperature and pressure)."""
    df_feat = df.copy()
    for col in columns:
        if col in df_feat.columns:
            df_feat[f"{col}_lag_4"] = df_feat[col].shift(4)
            df_feat[f"{col}_lag_96"] = df_feat[col].shift(96)

            df_feat[f"{col}_lag_4"] = df_feat[f"{col}_lag_4"].bfill()
            df_feat[f"{col}_lag_96"] = df_feat[f"{col}_lag_96"].bfill()

            if col in ["temperature", "pressure"]:
                df_feat[f"{col}_lag_672"] = df_feat[col].shift(672)
                df_feat[f"{col}_lag_672"] = df_feat[f"{col}_lag_672"].bfill()
    return df_feat


def add_atmospheric_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """Computes atmospheric dynamic features.
    Call AFTER add_pressure_trend and add_lag_features because
    pressure_accel_3h requires pressure_trend_3h, and thermal_stratification
    requires temperature_lag_672.
    """
    df_feat = df.copy()

    # Scalar wind speed and its 1h trend
    if "wind_u" in df_feat.columns and "wind_v" in df_feat.columns:
        df_feat["wind_speed_scalar"] = np.sqrt(df_feat["wind_u"]**2 + df_feat["wind_v"]**2)
        df_feat["wind_speed_trend_1h"] = df_feat["wind_speed_scalar"].diff(4).fillna(0.0)

    # 6h pressure trend and pressure acceleration (second derivative)
    if "pressure" in df_feat.columns:
        df_feat["pressure_trend_6h"] = df_feat["pressure"].diff(24).fillna(0.0)
    if "pressure_trend_3h" in df_feat.columns:
        df_feat["pressure_accel_3h"] = df_feat["pressure_trend_3h"].diff(12).fillna(0.0)

    # 7-day temperature deviation (thermal stratification)
    if "temperature" in df_feat.columns and "temperature_lag_672" in df_feat.columns:
        df_feat["thermal_stratification"] = df_feat["temperature"] - df_feat["temperature_lag_672"]

    # 24h diurnal differences (differential relative to the same time yesterday)
    if "temperature" in df_feat.columns and "temperature_lag_96" in df_feat.columns:
        df_feat["temp_diff_24h"] = (df_feat["temperature"] - df_feat["temperature_lag_96"]).fillna(0.0)
    elif "temp_diff_24h" not in df_feat.columns:
        df_feat["temp_diff_24h"] = 0.0

    if "humidity" in df_feat.columns and "humidity_lag_96" in df_feat.columns:
        df_feat["humidity_diff_24h"] = (df_feat["humidity"] - df_feat["humidity_lag_96"]).fillna(0.0)
    elif "humidity_diff_24h" not in df_feat.columns:
        df_feat["humidity_diff_24h"] = 0.0

    if "pressure" in df_feat.columns and "pressure_lag_96" in df_feat.columns:
        df_feat["pressure_diff_24h"] = (df_feat["pressure"] - df_feat["pressure_lag_96"]).fillna(0.0)
    elif "pressure_diff_24h" not in df_feat.columns:
        df_feat["pressure_diff_24h"] = 0.0

    # Residual error between sensor reading and Open-Meteo synoptic forecast
    if "temperature" in df_feat.columns and "temperature_2m" in df_feat.columns:
        df_feat["openmeteo_temp_bias"] = (df_feat["temperature"] - df_feat["temperature_2m"]).fillna(0.0)
        df_feat["openmeteo_temp_bias_3h"] = df_feat["openmeteo_temp_bias"].diff(12).fillna(0.0)
        df_feat["openmeteo_temp_bias_24h"] = df_feat["openmeteo_temp_bias"].diff(96).fillna(0.0)
    else:
        if "openmeteo_temp_bias" not in df_feat.columns:
            df_feat["openmeteo_temp_bias"] = 0.0
        if "openmeteo_temp_bias_3h" not in df_feat.columns:
            df_feat["openmeteo_temp_bias_3h"] = 0.0
        if "openmeteo_temp_bias_24h" not in df_feat.columns:
            df_feat["openmeteo_temp_bias_24h"] = 0.0

    return df_feat


def add_phase_transition_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes thermodynamic phase transition features for water vapor.
    Call AFTER add_temperature_derivatives and add_solar_zenith_angle because
    frost_risk requires temp_trend_1h, and evaporation_index requires potential_solar_radiation.
    """
    df_feat = df.copy()
    if "temperature" not in df_feat.columns or "humidity" not in df_feat.columns:
        return df_feat

    T = df_feat["temperature"]
    RH = df_feat["humidity"].clip(1.0, 100.0)

    # Wet-bulb temperature (Stull 2011 approximation, error ~0.3°C)
    df_feat["wet_bulb_temperature"] = (
        T * np.arctan(0.151977 * (RH + 8.313659)**0.5)
        + np.arctan(T + RH)
        - np.arctan(RH - 1.676331)
        + 0.00391838 * RH**1.5 * np.arctan(0.023101 * RH)
        - 4.686035
    )

    # Specific humidity (g/kg) via Magnus formula
    if "pressure" in df_feat.columns:
        e_s = 6.112 * np.exp(17.67 * T / (T + 243.5))
        e = RH / 100.0 * e_s
        P = df_feat["pressure"].clip(lower=1.0)
        df_feat["specific_humidity"] = (0.622 * e / (P - 0.378 * e)).clip(0.0, 50.0)

    # Frost risk: T < 3°C and temperature falling
    if "temp_trend_1h" in df_feat.columns:
        df_feat["frost_risk"] = (
            (T < 3.0).astype(float) * np.maximum(0.0, -df_feat["temp_trend_1h"])
        ).clip(0.0, 1.0)
    else:
        df_feat["frost_risk"] = (T < 3.0).astype(float)

    # Evaporation index: solar radiation x saturation deficit x temperature
    if "potential_solar_radiation" in df_feat.columns:
        df_feat["evaporation_index"] = (
            df_feat["potential_solar_radiation"]
            * (1.0 - RH / 100.0)
            * np.maximum(0.0, T) / 30.0
        ).clip(0.0, 1.0)

    return df_feat


def add_station_meta_features(df: pd.DataFrame, station_meta: Dict) -> pd.DataFrame:
    """Adds static station metadata features and cold-air pool risk assessment.
    Call AFTER add_inversion_risk because valley_cold_pool_risk uses
    inversion_risk as a dynamic component.
    """
    df_feat = df.copy()
    elevation = float(station_meta.get("elevation_m", 0.0))

    # Elevation class: 0.0=valley, 0.5=slope, 1.0=alpine
    if elevation < 500.0:
        elevation_class = 0.0
    elif elevation < 1500.0:
        elevation_class = 0.5
    else:
        elevation_class = 1.0
    df_feat["elevation_class"] = np.float32(elevation_class)

    # Valley cold air pool risk:
    # low elevation + nocturnal radiation inversion conditions
    elev_norm = float(np.clip(elevation / 3000.0, 0.0, 1.0))
    cold_pool_base = 1.0 - elev_norm   # 1.0 = valley, 0.0 = alpine
    if "inversion_risk" in df_feat.columns:
        df_feat["valley_cold_pool_risk"] = (cold_pool_base * df_feat["inversion_risk"]).clip(0.0, 1.0)
    else:
        df_feat["valley_cold_pool_risk"] = np.float32(cold_pool_base)

    return df_feat


def add_station_thermal_heating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes dynamic accumulated thermal mass heating index of station chassis
    under direct solar irradiance attenuated by convective wind cooling (Thermal Mass Heating).
    """
    df_feat = df.copy()
    pot_rad = df_feat.get("potential_solar_radiation", pd.Series(0.0, index=df_feat.index)).values
    sky_clear = df_feat.get("sky_clearness", pd.Series(0.5, index=df_feat.index)).values
    wind_spd = df_feat.get("wind_speed_scalar", pd.Series(0.0, index=df_feat.index)).values

    # Solar radiation flux heating attenuated by convective wind speed
    heating_rate = (pot_rad ** 1.5) * np.clip(sky_clear, 0.2, 1.2) / (wind_spd + 1.0)
    
    # Empirical accumulation of chassis thermal mass
    thermal_heating = np.zeros(len(df_feat), dtype=np.float32)
    acc = 0.0
    decay = 0.85
    for i in range(len(df_feat)):
        acc = acc * decay + heating_rate[i] * 0.15
        thermal_heating[i] = acc

    df_feat["station_thermal_heating_index"] = thermal_heating.astype(np.float32)
    
    # Morning heating rate over 3 hours (12 steps)
    if "temperature" in df_feat.columns:
        t_series = df_feat["temperature"].astype(float)
        df_feat["temp_velocity_3h"] = (t_series - t_series.shift(12)).fillna(0.0).astype(np.float32)
    else:
        df_feat["temp_velocity_3h"] = np.float32(0.0)

    return df_feat

def add_uv_clearness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Relative UV clearness index (uv_clearness).
    Compares observed UV against theoretical peak clear-sky UV at current Solar Zenith Angle.
    """
    df_feat = df.copy()
    if "uv" in df_feat.columns and "solar_zenith_angle" in df_feat.columns:
        z_rad = np.radians(df_feat["solar_zenith_angle"].fillna(90.0))
        cos_z = np.maximum(0.0, np.cos(z_rad))
        uv_theoretical = (cos_z ** 1.2) * 12.0
        uv_actual = pd.to_numeric(df_feat["uv"], errors="coerce").fillna(0.0)
        is_day = cos_z > 0.05
        uv_clearness = np.where(is_day, uv_actual / (uv_theoretical + 0.1), 0.0)
        df_feat["uv_clearness"] = np.clip(uv_clearness, 0.0, 1.0).astype(np.float32)
    else:
        df_feat["uv_clearness"] = np.float32(0.0)
    return df_feat


def add_temp_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temperature volatility over 1 hour (4 steps of 15 minutes).
    Characterizes micro-turbulence and sensor boundary fluctuations.
    """
    df_feat = df.copy()
    if "temperature" in df_feat.columns:
        vol = df_feat["temperature"].rolling(window=4, min_periods=1).std().fillna(0.0)
        df_feat["temp_volatility_1h"] = vol.astype(np.float32)
    else:
        df_feat["temp_volatility_1h"] = np.float32(0.0)
    return df_feat


def add_foehn_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Foehn adiabatic mountain wind index.
    Computed from the combination of strong winds, rapid temperature surge,
    low relative humidity, and valley/slope orographic exposure.
    """
    df_feat = df.copy()
    wind = df_feat.get("wind_speed_scalar", pd.Series(0.0, index=df_feat.index)).fillna(0.0)
    temp_trend = df_feat.get("temp_trend_3h", pd.Series(0.0, index=df_feat.index)).fillna(0.0)
    humidity = df_feat.get("humidity", pd.Series(50.0, index=df_feat.index)).fillna(50.0)
    valley_risk = df_feat.get("valley_cold_pool_risk", pd.Series(0.0, index=df_feat.index)).fillna(0.0)

    foehn = (wind * np.maximum(0.0, temp_trend) / (humidity + 1.0)) * (1.0 + valley_risk)
    df_feat["foehn_index"] = np.clip(foehn, 0.0, 10.0).astype(np.float32)
    return df_feat


def add_pseudo_cape(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pseudo-CAPE (convective available potential energy index).
    Estimated from the combination of equivalent potential temperature (theta_e) and VPD.
    """
    df_feat = df.copy()
    theta_e = df_feat.get("theta_e", pd.Series(273.15, index=df_feat.index)).fillna(273.15)
    vpd = df_feat.get("vpd", pd.Series(0.0, index=df_feat.index)).fillna(0.0)

    pseudo_cape = np.maximum(0.0, theta_e - 273.15) * (vpd / 10.0)
    df_feat["pseudo_cape_index"] = np.clip(pseudo_cape, 0.0, 50.0).astype(np.float32)
    return df_feat


def add_dew_point_deficit_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes dew point deficit (T - Td) and its trends over 1h (4 steps) and 3h (12 steps).
    One of the strongest indicators of imminent precipitation.
    """
    df_feat = df.copy()
    if "temperature" in df_feat.columns and "dew_point" in df_feat.columns:
        deficit = df_feat["temperature"] - df_feat["dew_point"]
        df_feat["dew_point_deficit"] = np.maximum(0.0, deficit).astype(np.float32)
        df_feat["dew_point_deficit_trend_1h"] = df_feat["dew_point_deficit"].diff(4).fillna(0.0).astype(np.float32)
        df_feat["dew_point_deficit_trend_3h"] = df_feat["dew_point_deficit"].diff(12).fillna(0.0).astype(np.float32)
    return df_feat


def add_pressure_drop_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes magnitude and rate of barometric pressure drop.
    Rapid pressure drop is a primary synoptic indicator of approaching cyclone / rain fronts.
    """
    df_feat = df.copy()
    if "pressure" in df_feat.columns:
        p_diff_1h = df_feat["pressure"].diff(4).fillna(0.0)
        p_diff_3h = df_feat["pressure"].diff(12).fillna(0.0)
        df_feat["pressure_drop_1h"] = np.maximum(0.0, -p_diff_1h).astype(np.float32)
        df_feat["pressure_drop_3h"] = np.maximum(0.0, -p_diff_3h).astype(np.float32)
    return df_feat


def add_rain_decay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exponentially weighted moving rain decay sum (rain_decay_sum).
    Uses EWM with a 12-hour half-life (48 steps of 15 minutes).
    """
    df_feat = df.copy()
    if "rain" in df_feat.columns:
        rain_vals = pd.to_numeric(df_feat["rain"], errors="coerce").fillna(0.0)
        decay_sum = rain_vals.ewm(halflife=48, min_periods=1).sum()
        df_feat["rain_decay_sum"] = decay_sum.astype(np.float32)
    else:
        df_feat["rain_decay_sum"] = np.float32(0.0)
    return df_feat






MODEL_SENSOR_COLUMNS = [
    "uv", "lux", "temperature", "pressure", "humidity",
    "pm1", "pm2_5", "pm10", "rain", "wind_u", "wind_v",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "doy_sin", "doy_cos",
    "dew_point", "dew_point_deficit",
    "pressure_trend_3h", "humidity_trend_3h",
    "temp_trend_1h", "temp_trend_3h", "temp_accel_1h", "inversion_risk",
    "solar_zenith_angle", "potential_solar_radiation",
    # Group 1: Atmospheric Dynamics
    "wind_speed_scalar", "wind_speed_trend_1h",
    "pressure_trend_6h", "pressure_accel_3h", "thermal_stratification",
    # Group 2: Phase Transitions
    "wet_bulb_temperature", "specific_humidity", "frost_risk", "evaporation_index",
    # Group 3: Station Metadata
    "elevation_class", "valley_cold_pool_risk",
    # Group 4: Advanced Physics Suite
    "vpd", "theta_e", "sky_clearness", "pm_moisture_index",
    "steps_since_last_rain", "ext_temp_bias",
    # Group 5: Extended Microclimate Physics
    "foehn_index", "pseudo_cape_index", "temp_volatility_1h",
    "uv_clearness", "rain_decay_sum",
    "rain_sum_24h", "rain_sum_72h",
    "temperature_lag_4", "temperature_lag_96", "temperature_lag_672",
    "pressure_lag_4", "pressure_lag_96", "pressure_lag_672",
    "humidity_lag_4", "humidity_lag_96",
    "temp_diff_24h", "humidity_diff_24h", "pressure_diff_24h",
    "openmeteo_temp_bias", "openmeteo_temp_bias_3h", "openmeteo_temp_bias_24h",
]
# rain_binary - 12th target: binary rain occurrence flag (0/1)
# Not normalized, optimized via BCE loss with pos_weight
MODEL_TARGET_COLUMNS = ["uv", "lux", "temperature", "pressure", "humidity",
                        "pm1", "pm2_5", "pm10", "rain", "wind_u", "wind_v", "rain_binary"]
RAIN_BINARY_IDX = 11  # Index of rain_binary in MODEL_TARGET_COLUMNS
MODEL_STATIC_COLUMNS = ["latitude", "longitude", "elevation_m"]
NORMALIZE_COLUMNS = [
    "uv", "lux", "temperature", "pressure", "humidity",
    "pm1", "pm2_5", "pm10", "rain", "wind_u", "wind_v",
    "dew_point", "dew_point_deficit",
    "pressure_trend_3h", "humidity_trend_3h",
    "temp_trend_1h", "temp_trend_3h", "temp_accel_1h", "inversion_risk",
    "solar_zenith_angle", "potential_solar_radiation",
    # Group 1
    "wind_speed_scalar", "wind_speed_trend_1h",
    "pressure_trend_6h", "pressure_accel_3h", "thermal_stratification",
    # Group 2
    "wet_bulb_temperature", "specific_humidity", "frost_risk", "evaporation_index",
    # Group 3
    "elevation_class", "valley_cold_pool_risk",
    # Group 4: Advanced Physics Suite
    "vpd", "theta_e", "sky_clearness", "pm_moisture_index",
    "steps_since_last_rain", "ext_temp_bias",
    # Group 5: Extended Microclimate Physics
    "foehn_index", "pseudo_cape_index", "temp_volatility_1h",
    "uv_clearness", "rain_decay_sum",
    "rain_sum_24h", "rain_sum_72h",
    "temperature_lag_4", "temperature_lag_96", "temperature_lag_672",
    "pressure_lag_4", "pressure_lag_96", "pressure_lag_672",
    "humidity_lag_4", "humidity_lag_96",
    "temp_diff_24h", "humidity_diff_24h", "pressure_diff_24h",
    "openmeteo_temp_bias", "openmeteo_temp_bias_3h", "openmeteo_temp_bias_24h",
    # rain_binary is excluded here - remains strictly 0/1
]
EXTERNAL_FORECAST_COLUMNS = [
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "wind_speed_10m", "wind_direction_sin", "wind_direction_cos", "precipitation", "cloud_cover",
]


LOG1P_COLUMNS = [
    "rain", "rain_sum_24h", "rain_sum_72h", "rain_decay_sum",
    "steps_since_last_rain", "pm1", "pm2_5", "pm10"
]


def apply_log1p_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """Applies log1p transformation (log(1 + x)) to mitigate heavy-tailed skewness in rain and PM features."""
    df_feat = df.copy()
    for col in LOG1P_COLUMNS:
        if col in df_feat.columns:
            vals = pd.to_numeric(df_feat[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            df_feat[col] = np.log1p(np.maximum(0.0, vals)).astype(np.float32)
    return df_feat


def format_multimodel_external_df(df_om: pd.DataFrame, df_ms: pd.DataFrame = None) -> pd.DataFrame:
    """
    Formats raw Open-Meteo forecast DataFrame into standardized 8-column schema.
    """
    if df_om.empty:
        return pd.DataFrame(columns=["timestamp"] + EXTERNAL_FORECAST_COLUMNS)

    df_res = df_om.copy()
    if "wind_direction_10m" in df_res.columns and "wind_direction_sin" not in df_res.columns:
        wrad = np.deg2rad(pd.to_numeric(df_res["wind_direction_10m"], errors="coerce").fillna(0.0))
        df_res["wind_direction_sin"] = np.sin(wrad)
        df_res["wind_direction_cos"] = np.cos(wrad)

    for col in EXTERNAL_FORECAST_COLUMNS:
        if col not in df_res.columns:
            df_res[col] = 0.0

    keep_cols = ["timestamp"] + EXTERNAL_FORECAST_COLUMNS
    return df_res[keep_cols]


def load_combined_external_forecast(raw_ext_dir: str, station_id: int, settings: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Loads and prepares external Open-Meteo forecast for specified station.
    """
    om_path = os.path.join(raw_ext_dir, f"forecast_{station_id}.csv")
    if os.path.exists(om_path):
        try:
            df_om = pd.read_csv(om_path)
            if "timestamp" in df_om.columns:
                df_om["timestamp"] = pd.to_datetime(df_om["timestamp"], format="mixed")
            return format_multimodel_external_df(df_om)
        except Exception as e:
            print(f"Error loading Open-Meteo for station {station_id}: {e}")

    return pd.DataFrame()


def prepare_feature_frame(df: pd.DataFrame, station_meta: Dict[str, Any], external_df: pd.DataFrame = None) -> pd.DataFrame:
    """Prepares station DataFrame for training and inference through unified pipeline."""
    if df.empty:
        return df.copy()

    features = df.copy()
    features = clean_station_anomalies(features, station_meta)
    features = convert_wind_features(features)
    features = add_wind_scalar(features)
    features = add_cyclical_time_features(features)
    features["dew_point"] = compute_dew_point(features["temperature"], features["humidity"])
    features["dew_point_deficit"] = features["temperature"] - features["dew_point"]
    features["rain_binary"] = (pd.to_numeric(features.get("rain", pd.Series([0.0])),
                               errors="coerce").fillna(0.0) > 0.05).astype("float32")

    # Trend / derivative features
    features = add_vapor_pressure_deficit(features)
    features = add_pressure_trend(features)
    features = add_humidity_trend(features)
    features = add_temperature_derivatives(features)
    features = add_temp_volatility(features)
    features = add_inversion_risk(features)
    features = add_solar_zenith_angle(features, float(station_meta.get("latitude", 40.2)))
    features = add_sky_clearness(features)
    features = add_uv_clearness(features)
    features = add_station_thermal_heating(features)
    features = add_pm_moisture_index(features)
    features = add_dry_spell_counter(features)
    features = add_station_meta_features(features, station_meta)
    features = add_foehn_index(features)
    features = add_phase_transition_features(features)
    features = add_equivalent_potential_temperature(features)
    features = add_pseudo_cape(features)

    # Sensor minus NWP delta (current global model bias)
    if external_df is not None and not external_df.empty and "temperature_2m" in external_df.columns:
        try:
            ext_aligned = external_df.set_index("timestamp").resample("15min").ffill().reset_index()
            merged_ext = features[["timestamp"]].merge(
                ext_aligned[["timestamp", "temperature_2m"]], on="timestamp", how="left"
            )
            bias_series = pd.Series(
                (features["temperature"].values - merged_ext["temperature_2m"].ffill().values).clip(-20.0, 20.0),
                index=features.index
            )
            features["ext_temp_bias"] = bias_series.astype(np.float32)
            features["ext_temp_bias_24h"] = bias_series.rolling(window=96, min_periods=1).mean().astype(np.float32)
        except Exception:
            features["ext_temp_bias"] = np.float32(0.0)
            features["ext_temp_bias_24h"] = np.float32(0.0)
    else:
        features["ext_temp_bias"] = np.float32(0.0)
        features["ext_temp_bias_24h"] = np.float32(0.0)

    # Lag features - computed after trends
    features = add_lag_features(features, ["temperature", "humidity", "pressure", "wind_u", "wind_v"])
    features = add_rain_accumulation(features)
    features = add_rain_decay(features)
    # Atmospheric dynamics - after add_lag_features (requires temperature_lag_672)
    features = add_atmospheric_dynamics(features)

    # Log1p transformation for heavy-tailed precipitation and particulate columns
    features = apply_log1p_transformations(features)

    features["latitude"] = float(station_meta.get("latitude", 0.0))
    features["longitude"] = float(station_meta.get("longitude", 0.0))
    features["elevation_m"] = float(station_meta.get("elevation_m", 0.0))

    return features


def build_scalers(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, float]]:
    """Builds scaling parameters dictionary with variance floor guardrails."""
    # Minimum standard deviation dictionary to prevent numerical explosion
    STD_FLOORS = {
        "rain": 0.2,
        "uv": 1.0,
        "lux": 10.0,
        "pm1": 5.0,
        "pm2_5": 5.0,
        "pm10": 5.0,
        "temperature": 1.0,
        "humidity": 1.0,
        "pressure": 1.0,
    }

    scalers: Dict[str, Dict[str, float]] = {}
    for col in columns:
        if col not in df.columns:
            continue

        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().any():
            mean_val = float(values.mean(skipna=True))
            std_val = float(values.std(skipna=True))
        else:
            mean_val = 0.0
            std_val = 1.0

        # Apply variance floor to std_val
        floor_val = 0.1  # General minimum variance floor for all remaining columns
        for key, floor in STD_FLOORS.items():
            if key in col:  # Matches lagged and trend columns, e.g. rain_sum_24h, temperature_lag_4
                floor_val = floor
                break

        if std_val < floor_val:
            std_val = floor_val

        scalers[col] = {"mean": mean_val, "std": std_val}

    return scalers


def apply_scalers(df: pd.DataFrame, scalers: Dict[str, Dict[str, float]], columns: List[str]) -> pd.DataFrame:
    """Normalizes selected columns using precomputed scalers."""
    df_scaled = df.copy()
    for col in columns:
        if col not in df_scaled.columns or col not in scalers:
            continue

        mean_val = float(scalers[col]["mean"])
        std_val = float(scalers[col]["std"])
        df_scaled[col] = pd.to_numeric(df_scaled[col], errors="coerce")
        df_scaled[col] = (df_scaled[col] - mean_val) / std_val

    return df_scaled


def inverse_scalers(df: pd.DataFrame, scalers: Dict[str, Dict[str, float]], columns: List[str]) -> pd.DataFrame:
    """Inverses normalized columns back to original physical scale."""
    df_restored = df.copy()
    for col in columns:
        if col not in df_restored.columns or col not in scalers:
            continue

        mean_val = float(scalers[col]["mean"])
        std_val = float(scalers[col]["std"])
        df_restored[col] = pd.to_numeric(df_restored[col], errors="coerce")
        df_restored[col] = df_restored[col] * std_val + mean_val

    return df_restored


def convert_qnh_to_qfe(p_qnh: pd.Series | float | np.ndarray, elevation_m: float) -> pd.Series | float | np.ndarray:
    """Converts sea-level pressure QNH (hPa) to station surface pressure QFE (hPa) via hypsometric formula."""
    if elevation_m is None or elevation_m <= 0:
        return p_qnh
    factor = (1.0 - (0.0065 * float(elevation_m)) / 288.15) ** 5.255
    return p_qnh * factor


def normalize_multimodel_external_df(df: pd.DataFrame, scalers: Dict[str, Dict[str, float]] | None, elevation_m: float = None) -> pd.DataFrame:
    """Normalizes 8 external Open-Meteo forecast columns using station scalers."""
    if df.empty or scalers is None:
        return df.fillna(0.0)

    df_norm = df.copy()
    col_scaler_map = {
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "surface_pressure": "pressure",
        "precipitation": "rain",
        "wind_speed_10m": "wind_speed_10m",
        "cloud_cover": "cloud_cover"
    }

    if "surface_pressure" in df_norm.columns and elevation_m is not None and elevation_m > 0:
        df_norm["surface_pressure"] = convert_qnh_to_qfe(df_norm["surface_pressure"], elevation_m)

    for base_col, scaler_key in col_scaler_map.items():
        if base_col in df_norm.columns and scaler_key in scalers:
            mean_v = float(scalers[scaler_key]["mean"])
            std_v = float(scalers[scaler_key]["std"])
            df_norm[base_col] = (pd.to_numeric(df_norm[base_col], errors="coerce").fillna(0.0) - mean_v) / (std_v if std_v > 1e-5 else 1.0)

    return df_norm.fillna(0.0)


def prepare_model_inputs(
    sensor_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    station_meta: Dict[str, Any],
    scalers: Dict[str, Dict[str, float]],
    lookback_steps: int,
    horizon_steps: int,
    sensor_columns: List[str] | None = None,
    static_columns: List[str] | None = None,
    external_columns: List[str] | None = None,
    normalize_columns: List[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Prepares encoder/decoder inputs and temporal target grid from raw observations."""
    if sensor_columns is None:
        sensor_columns = MODEL_SENSOR_COLUMNS
    if static_columns is None:
        static_columns = MODEL_STATIC_COLUMNS
    if external_columns is None:
        external_columns = EXTERNAL_FORECAST_COLUMNS
    if normalize_columns is None:
        normalize_columns = NORMALIZE_COLUMNS

    sensor_frame = sensor_df.copy()
    sensor_frame = sensor_frame.sort_values("timestamp").tail(lookback_steps)

    last_sensor_ts = sensor_frame["timestamp"].max()
    future_timestamps = pd.date_range(start=last_sensor_ts + pd.Timedelta(minutes=15),
                                      periods=horizon_steps, freq="15min")

    forecast_frame = forecast_df.copy()
    if not forecast_frame.empty and "timestamp" in forecast_frame.columns:
        forecast_frame = forecast_frame.sort_values("timestamp")
        forecast_frame = forecast_frame.set_index("timestamp").resample("15min").ffill().reset_index()

    future_df = pd.DataFrame({"timestamp": future_timestamps})
    if not forecast_frame.empty:
        forecast_aligned = pd.merge(future_df, forecast_frame, on="timestamp", how="left").ffill().bfill()
    else:
        forecast_aligned = future_df.copy()
        for col in external_columns:
            forecast_aligned[col] = 0.0

    sensor_frame_norm = apply_scalers(sensor_frame, scalers, normalize_columns)
    enc_sensors = sensor_frame_norm[sensor_columns].values.astype(np.float32)

    # Global normalization of static features
    raw_static = np.array([float(station_meta.get(col, 0.0)) for col in static_columns], dtype=np.float32)
    if static_columns == ["latitude", "longitude", "elevation_m"]:
        norm_static = np.array([
            (raw_static[0] - 40.2) / 0.5,      # latitude
            (raw_static[1] - 44.5) / 0.8,      # longitude
            (raw_static[2] - 1400.0) / 450.0   # elevation_m
        ], dtype=np.float32)
    else:
        norm_static = raw_static

    enc_static = np.tile(norm_static, (lookback_steps, 1))
    encoder_input = np.hstack([enc_sensors, enc_static]).astype(np.float32)
    encoder_input = encoder_input.reshape(1, lookback_steps, encoder_input.shape[1])

    # Normalization of external multi-model forecasts
    forecast_aligned_norm = normalize_multimodel_external_df(forecast_aligned, scalers)

    for col in external_columns:
        if col not in forecast_aligned_norm.columns:
            forecast_aligned_norm[col] = 0.0

    decoder_input = forecast_aligned_norm[external_columns].values.astype(np.float32)
    decoder_input = decoder_input.reshape(1, horizon_steps, decoder_input.shape[1])

    return encoder_input, decoder_input, future_timestamps


def denormalize_predictions(preds: np.ndarray, target_columns: List[str], scalers: Dict[str, Dict[str, float]]) -> np.ndarray:
    """Denormalizes predictions back to physical units."""
    preds_df = pd.DataFrame(preds, columns=target_columns)
    restored = inverse_scalers(preds_df, {col: scalers[col]
                               for col in target_columns if col in scalers}, target_columns)
    return restored.to_numpy(dtype=np.float32)


# ---------------------------------------------------------------------------
# Preprocessing, regularization, and anomaly cleaning
# ---------------------------------------------------------------------------

def load_raw_station_json(filepath: str) -> pd.DataFrame:
    """Loads raw station JSON file and enforces numeric dtypes."""
    import json
    if not os.path.exists(filepath):
        return pd.DataFrame()
    with open(filepath, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    keys = raw_data.get("keys", [])
    data = raw_data.get("data", [])
    if not keys or not data:
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=keys)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    float_cols = ["uv", "lux", "temperature", "pressure", "humidity", "pm1", "pm2_5", "pm10", "wind speed", "rain"]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return df


def regularize_station_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """Rounds timestamps to 15-minute intervals and reindexes onto a regular temporal grid."""
    if df.empty:
        return df
    df_clean = df.copy()
    df_clean["timestamp"] = df_clean["timestamp"].dt.round("15min")
    df_clean = df_clean.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df_clean = df_clean.set_index("timestamp")
    full_index = pd.date_range(start=df_clean.index.min(), end=df_clean.index.max(), freq="15min")
    df_regular = df_clean.reindex(full_index)
    df_regular.index.name = "timestamp"
    return df_regular.reset_index()


def fill_gaps_tiered(df: pd.DataFrame, max_interp_gap_steps: int = 12) -> pd.DataFrame:
    """Tiered gap imputation: linear interpolation up to 3h, followed by forward/backward fill."""
    if df.empty:
        return df
    df_filled = df.copy()
    numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_filled[col] = df_filled[col].interpolate(method="linear", limit=max_interp_gap_steps)
        df_filled[col] = df_filled[col].ffill().bfill().fillna(0.0)
    return df_filled


def clean_station_anomalies(df: pd.DataFrame, station_meta: Dict[str, Any] = None) -> pd.DataFrame:
    """Physical bounds clipping and tiered gap imputation."""
    if df.empty:
        return df
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ["temperature", "humidity", "pressure"]:
            # Filter unphysical values
            if col == "humidity":
                df_clean.loc[(df_clean[col] < 0.0) | (df_clean[col] > 100.0), col] = np.nan
            elif col == "temperature":
                df_clean.loc[(df_clean[col] < -50.0) | (df_clean[col] > 60.0), col] = np.nan
    return fill_gaps_tiered(df_clean)

import os
from typing import Any, Dict, List, Optional


def get_single_station_id(settings: Dict[str, Any]) -> Optional[int]:
    """Returns station ID for single-station mode if enabled."""
    single_station_cfg = settings.get("single_station", {})
    if not isinstance(single_station_cfg, dict):
        return None

    if not single_station_cfg.get("enabled", False):
        return None

    station_id = single_station_cfg.get("station_id")
    if station_id is None:
        return None

    try:
        return int(station_id)
    except (TypeError, ValueError):
        return None


def get_station_id_from_filename(filename: str) -> Optional[int]:
    """Extracts integer station ID from filename pattern station_1_features.parquet."""
    try:
        stem = filename.split(".")[0]
        if stem.startswith("station_"):
            return int(stem.split("_")[1])
    except (IndexError, ValueError):
        return None


def select_stratified_stations(stations: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """
    Stratified selection of N stations across elevation tiers.
    Partitions stations into three tiers (lowland, mid-altitude, alpine)
    and samples evenly to cover diverse climatic zones.

    Elevation tiers:
      - lowland     : elevation_m < 1000 m
      - mid-altitude: 1000 m <= elevation_m < 2000 m
      - alpine      : elevation_m >= 2000 m
    """
    low = sorted([s for s in stations if float(s.get("elevation_m", 0)) < 1000],
                 key=lambda s: float(s.get("elevation_m", 0)))
    mid = sorted([s for s in stations if 1000 <= float(s.get("elevation_m", 0)) < 2000],
                 key=lambda s: float(s.get("elevation_m", 0)))
    high = sorted([s for s in stations if float(s.get("elevation_m", 0)) >= 2000],
                  key=lambda s: float(s.get("elevation_m", 0)))

    tiers = [low, mid, high]
    tier_names = ["lowland (<1000m)", "mid-altitude (1000-2000m)", "alpine (>=2000m)"]

    # Calculate tier quotas proportional to group sizes
    total = len(stations)
    result = []
    remaining = n

    for i, (tier, name) in enumerate(zip(tiers, tier_names)):
        if not tier:
            continue
        if i == len(tiers) - 1:
            # Allocate remainder to last non-empty tier
            quota = remaining
        else:
            quota = max(1, round(n * len(tier) / total)) if total > 0 else 0

        quota = min(quota, len(tier), remaining)
        # Sample evenly across elevation tier
        step = max(1, len(tier) // quota) if quota > 0 else 1
        chosen = tier[::step][:quota]
        result.extend(chosen)
        remaining -= len(chosen)

        print(f"  Tier '{name}': available={len(tier)}, selected={len(chosen)}: "
              + ", ".join(f"{s['name']} ({s.get('elevation_m', '?')}m)" for s in chosen))

        if remaining <= 0:
            break

    return result[:n]


def filter_station_files_for_run(station_files: List[str], settings: Dict[str, Any]) -> List[str]:
    """
    Filters station feature files according to active mode:
      - single_station.enabled -> single station
      - station_subset.enabled -> N stratified stations
      - otherwise              -> all files
    """
    # Single-station mode (backwards compatibility)
    single_station_id = get_single_station_id(settings)
    if single_station_id is not None:
        return [f for f in station_files if get_station_id_from_filename(f) == single_station_id]

    # Subset mode - target IDs passed externally via allowed_ids
    subset_cfg = settings.get("_subset_ids")
    if subset_cfg is not None:
        allowed = set(subset_cfg)
        return [f for f in station_files if get_station_id_from_filename(f) in allowed]

    return list(station_files)


def select_stations_for_run(stations: List[Dict[str, Any]], settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Filters stations list according to active mode:
      - single_station.enabled -> single station
      - station_subset.enabled -> N stratified stations
      - otherwise              -> all stations
    """
    # Single-station mode (backwards compatibility)
    single_station_id = get_single_station_id(settings)
    if single_station_id is not None:
        selected = [
            s for s in stations
            if int(s.get("id", -1)) == single_station_id
            or int(s.get("generated_id", -1)) == single_station_id
        ]
        return selected if selected else list(stations)

    # Subset mode - stratified selection of N stations
    subset_cfg = settings.get("station_subset", {})
    if isinstance(subset_cfg, dict) and subset_cfg.get("enabled", False):
        mode = subset_cfg.get("mode", "count")

        if mode == "ids":
            # Explicit IDs specified manually
            allowed = set(int(i) for i in subset_cfg.get("ids", []))
            selected = [s for s in stations if int(s.get("id", -1)) in allowed]
            return selected if selected else list(stations)

        elif mode == "count":
            count = int(subset_cfg.get("count", 15))
            print(f"\nPilot training mode: stratified selection of {count} stations from {len(stations)}")
            return select_stratified_stations(stations, count)

    return list(stations)


def update_station_elevations(stations_path: str = "config/stations.json") -> None:
    """Queries and updates ground truth station elevations via Open-Meteo Elevation API."""
    import json

    import requests

    if not os.path.exists(stations_path):
        print(f"Error: File {stations_path} not found.")
        return

    with open(stations_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    stations = config.get("stations", [])
    if not stations:
        print("Stations list is empty.")
        return

    lats = [str(s["latitude"]) for s in stations]
    lons = [str(s["longitude"]) for s in stations]

    url = "https://api.open-meteo.com/v1/elevation"
    params = {"latitude": ",".join(lats), "longitude": ",".join(lons)}

    try:
        print("Querying elevations from Open-Meteo...")
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        elevations = response.json().get("elevation", [])

        if len(elevations) != len(stations):
            print("Error: returned elevations count does not match station count.")
            return

        for station, elevation in zip(stations, elevations):
            station["elevation_m"] = round(elevation, 1)
            print(f"  {station['name']}: {station['elevation_m']} m")

        with open(stations_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        print(f"\nFile {stations_path} successfully updated with station elevations!")

    except Exception as e:
        print(f"Error fetching elevations: {e}")


def clean_processed_directory(processed_dir: str) -> None:
    """Cleans stale *.parquet files and reports from data/processed/ before generating new features."""
    if os.path.exists(processed_dir):
        count = 0
        for f in os.listdir(processed_dir):
            if f.endswith("_features.parquet") or f.endswith("_report.csv") or f.endswith("_report.json"):
                try:
                    os.remove(os.path.join(processed_dir, f))
                    count += 1
                except Exception:
                    pass
        print(f"Cleaned directory {processed_dir}: removed {count} stale files before new calculation.")


if __name__ == "__main__":
    # Load settings from central config
    settings_file = os.path.join("config", "settings.json")
    if not os.path.exists(settings_file):
        print(f"Error: {settings_file} not found.")
        exit(1)

    with open(settings_file, "r", encoding="utf-8") as f:
        settings = json.load(f)

    raw_dir = settings["paths"]["raw_dir"]
    out_dir = settings["paths"]["processed_dir"]
    stations_config_path = settings["paths"]["stations_config"]

    # Automatic cleanup of stale files before new calculation
    clean_processed_directory(out_dir)

    if os.path.exists(stations_config_path):
        with open(stations_config_path, "r", encoding="utf-8") as f:
            stations = json.load(f)["stations"]

        stations = select_stations_for_run(stations, settings)

        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        for station_meta in stations:
            sid = station_meta["id"]
            gen_id = station_meta["generated_id"]
            name = station_meta["name"]
            status = station_meta.get("Status", "online")

            if status == "offline":
                continue

            raw_file = os.path.join(raw_dir, f"station_{gen_id}.json")
            if not os.path.exists(raw_file):
                continue

            print(f"\n--- Starting preprocessing pipeline for: {name} (id={sid}) ---")
            try:
                elevation = float(station_meta["elevation_m"])

                if not os.path.exists(raw_file):
                    continue
                with open(raw_file, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                df_raw = pd.DataFrame(raw_data.get("data", []), columns=raw_data.get("keys", []))
                if df_raw.empty:
                    continue
                df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], format="mixed")
                df = prepare_feature_frame(df_raw, station_meta)

                # ext_temp_bias requires external forecasts, none here -> set to 0
                if "ext_temp_bias" not in df.columns:
                    df["ext_temp_bias"] = np.float32(0.0)

                parquet_file = os.path.join(out_dir, f"station_{sid}_features.parquet")
                df.to_parquet(parquet_file, index=False)
                print(f"  Successfully processed and saved (raw): {parquet_file} ({df.shape[0]} rows)")

            except Exception as e:
                print(f"  Error processing station {name}: {e}")

