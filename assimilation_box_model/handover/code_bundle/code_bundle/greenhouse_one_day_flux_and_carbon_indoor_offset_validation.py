#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import datetime as dt
import io
import math
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import whole_greenhouse_chamber as wgc


# -----------------------------
# Fuel / burner constants
# -----------------------------
RHO_FUEL_KG_PER_L = 0.80
LHV_J_PER_KG = 43.1e6
CO2_KG_PER_KG_FUEL = 3.106
H2O_KG_PER_KG_FUEL = 1.376
CO2_MOL_PER_L_FUEL = RHO_FUEL_KG_PER_L * CO2_KG_PER_KG_FUEL * 1000.0 / 44.01
H2O_KG_PER_L_FUEL = RHO_FUEL_KG_PER_L * H2O_KG_PER_KG_FUEL
LHV_J_PER_L_FUEL = RHO_FUEL_KG_PER_L * LHV_J_PER_KG
GCO2_PER_UMOL = 44.01e-6
GCO2_H_PER_UMOL_S = GCO2_PER_UMOL * 3600.0


# -----------------------------
# Geometry / presets
# -----------------------------
@dataclass(frozen=True)
class Geometry:
    floor_area_m2: float
    cultivated_area_m2: float
    air_volume_m3: float
    cover_area_m2: float


@dataclass(frozen=True)
class ScreenLayer:
    name: str
    direct_shade_pct: float
    diffuse_shade_pct: float
    energy_saving_pct: float
    shortwave_transmittance_closed: float
    ua_closed_ratio: float
    sky_coupling_closed_ratio: float
    source: str


@dataclass(frozen=True)
class HousePreset:
    house_key: str
    display_name: str
    geometry: Geometry
    ua_base_w_k: float
    u_ref_ms: float
    tau_store_min: float
    c_eff_j_k: float
    beta_solar: float
    r20_umol_m2_s: float
    q10: float
    fuel_rate_l_h_main: float = 3.79
    fuel_rate_l_h_secondary: float = 3.79
    has_secondary_burner: bool = False
    heater_capacity_w: float = 174000.0
    solar_heat_factor: float = 0.60
    co2_dead_time_min: float = 4.0
    co2_tau_min: float = 8.0
    main_cmd_candidates: tuple[str, ...] = ('CO2',)
    secondary_cmd_candidates: tuple[str, ...] = ()
    outdoor_co2_ppm: float = 400.0
    dewpoint_offset_c: float = 2.5
    smooth_window: str = '5min'
    daylight_threshold_w_m2: float = 20.0
    vent_closed_threshold_pct: float = 0.5
    screen_layers: tuple[ScreenLayer, ...] = ()


LUXOUS_1547 = ScreenLayer(
    name='Luxous 15 47',
    direct_shade_pct=15.0,
    diffuse_shade_pct=24.0,
    energy_saving_pct=47.0,
    shortwave_transmittance_closed=1.0 - 0.5 * (0.15 + 0.24),
    ua_closed_ratio=0.75,
    sky_coupling_closed_ratio=0.62,
    source='Ludvig Svensson Luxous 1547 D FR public product data',
)
TEMPA_5557 = ScreenLayer(
    name='Tempa 55 57',
    direct_shade_pct=55.0,
    diffuse_shade_pct=61.0,
    energy_saving_pct=57.0,
    shortwave_transmittance_closed=1.0 - 0.5 * (0.55 + 0.61),
    ua_closed_ratio=0.68,
    sky_coupling_closed_ratio=0.52,
    source='Ludvig Svensson Tempa 5557 D public product data',
)
TEMPA_9870 = ScreenLayer(
    name='Tempa 98 70',
    direct_shade_pct=98.0,
    diffuse_shade_pct=98.0,
    energy_saving_pct=70.0,
    shortwave_transmittance_closed=0.02,
    ua_closed_ratio=0.58,
    sky_coupling_closed_ratio=0.35,
    source='Model assumption from public Svensson naming key and Japanese distributor catalog',
)


TRIANGLE_GEOM = Geometry(
    floor_area_m2=45.0 * 51.0,
    cultivated_area_m2=45.0 * 46.5,
    air_volume_m3=45.0 * 51.0 * (4.2 + 0.5 * (5.6 - 4.2)),
    cover_area_m2=(
        2.0 * 51.0 * 4.2
        + 2.0 * 45.0 * 4.2
        + 2.0 * (0.5 * 45.0 * (5.6 - 4.2))
        + 2.0 * math.hypot(45.0 / 2.0, 5.6 - 4.2) * 51.0
    ),
)
TRIANGLE_PRESET = HousePreset(
    house_key='sankaku',
    display_name='Sankaku',
    geometry=TRIANGLE_GEOM,
    ua_base_w_k=13280.0,
    u_ref_ms=0.917,
    tau_store_min=140.0,
    c_eff_j_k=2.39e8,
    beta_solar=0.9,
    r20_umol_m2_s=2.3345,
    q10=1.8869,
    has_secondary_burner=False,
    main_cmd_candidates=('CO2',),
    secondary_cmd_candidates=(),
    screen_layers=(LUXOUS_1547, TEMPA_5557),
)

MARU_GEOM = Geometry(
    floor_area_m2=54.0 * 51.0,
    cultivated_area_m2=54.0 * 51.0,
    air_volume_m3=54.0 * 51.0 * 5.5,
    cover_area_m2=(54.0 * 51.0 + 2.0 * 54.0 * 5.5 + 2.0 * 51.0 * 5.5),
)
MARU_U_TRI = TRIANGLE_PRESET.ua_base_w_k / TRIANGLE_GEOM.cover_area_m2
MARU_PRESET = HousePreset(
    house_key='maru',
    display_name='Maru',
    geometry=MARU_GEOM,
    ua_base_w_k=MARU_U_TRI * MARU_GEOM.cover_area_m2,
    u_ref_ms=0.917,
    tau_store_min=440.0,
    c_eff_j_k=6.9794e8,
    beta_solar=0.285936,
    r20_umol_m2_s=3.2622,
    q10=2.5686,
    has_secondary_burner=True,
    main_cmd_candidates=('CO2',),
    secondary_cmd_candidates=('CV_H2',),
    screen_layers=(TEMPA_9870, LUXOUS_1547),
)

HOUSE_PRESETS = {
    'sankaku': TRIANGLE_PRESET,
    'maru': MARU_PRESET,
}



@dataclass(frozen=True)
class RadiationScalingMatch:
    enabled: bool
    factor: float
    csv_path: str
    matched_date: str
    match_mode: str
    source: str
    note: str = ''


@dataclass(frozen=True)
class OutdoorHumidityCo2Source:
    mode: str
    outside_pf_root: str = ''
    detail: str = ''

# -----------------------------
# Parsing helpers
# -----------------------------
def read_text_table(path: str | Path) -> pd.DataFrame:
    encodings = ('utf-8-sig', 'utf-8', 'cp932', 'shift_jis', 'latin1')
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            with open(path, 'r', encoding=enc) as f:
                txt = f.read()
            df = pd.read_csv(io.StringIO(txt), skipinitialspace=True)
            df.columns = [str(c).strip() for c in df.columns]
            df = df.loc[:, ~df.columns.str.contains(r'^Unnamed', na=False)]
            return df
        except Exception as exc:  # pragma: no cover
            last_err = exc
    raise RuntimeError(f'Could not read {path}: {last_err}')


def parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if 'timestamp' in d.columns:
        d['timestamp'] = pd.to_datetime(d['timestamp'])
    elif 'DATE' in d.columns and 'TIME' in d.columns:
        d['timestamp'] = pd.to_datetime(d['DATE'].astype(str).str.strip() + ' ' + d['TIME'].astype(str).str.strip())
    elif 'DateTime' in d.columns:
        d['timestamp'] = pd.to_datetime(d['DateTime'])
    else:
        raise ValueError('Input file must have either DATE/TIME columns or a timestamp column.')

    d = d.dropna(subset=['timestamp']).set_index('timestamp').sort_index()
    if not d.index.has_duplicates:
        return d

    agg: dict[str, pd.Series] = {}
    for col in d.columns:
        s_num = pd.to_numeric(d[col], errors='coerce')
        if s_num.notna().any():
            agg[col] = s_num.groupby(d.index).mean()
        else:
            agg[col] = d[col].groupby(d.index).last()
    out = pd.DataFrame(agg).sort_index()
    out.index.name = 'timestamp'
    return out


def first_existing(df: pd.DataFrame, candidates: Sequence[str], *, required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f'None of the required columns were found: {candidates}')
    return None


def as_float_series(df: pd.DataFrame, candidates: Sequence[str], *, required: bool = True, default: float = 0.0) -> pd.Series:
    col = first_existing(df, candidates, required=required)
    if col is None:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors='coerce').astype(float)


def align_to_index(series: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    s = series.sort_index()
    s2 = s.reindex(s.index.union(target_index)).sort_index().interpolate(method='time').reindex(target_index)
    s2 = s2.ffill().bfill()
    s2.index = target_index
    return s2.astype(float)


def elapsed_seconds(index: pd.DatetimeIndex) -> np.ndarray:
    if len(index) == 0:
        return np.array([], dtype=float)
    base = index[0]
    return np.asarray((index - base).total_seconds(), dtype=float)


def step_seconds(index: pd.DatetimeIndex) -> np.ndarray:
    if len(index) == 0:
        return np.array([], dtype=float)
    if len(index) == 1:
        return np.array([60.0], dtype=float)
    diffs = np.asarray((index[1:] - index[:-1]).total_seconds(), dtype=float)
    valid = diffs[np.isfinite(diffs) & (diffs > 0)]
    median_dt = float(np.median(valid)) if len(valid) else 60.0
    if not np.isfinite(median_dt) or median_dt <= 0:
        median_dt = 60.0
    dt = np.empty(len(index), dtype=float)
    dt[:-1] = diffs
    dt[-1] = median_dt
    dt = np.where(~np.isfinite(dt) | (dt <= 0), median_dt, dt)
    dt = np.where(dt > 1.5 * median_dt, median_dt, dt)
    return dt


def read_signal_series(df: pd.DataFrame, candidates: Sequence[str], *, default: float = 0.0) -> tuple[pd.Series, Optional[str]]:
    col = first_existing(df, candidates, required=False)
    if col is None:
        return pd.Series(default, index=df.index, dtype=float), None
    return pd.to_numeric(df[col], errors='coerce').astype(float), col


# -----------------------------
# Physical helpers
# -----------------------------
def fopdt_fraction(command_pct: pd.Series, *, dead_time_min: float, tau_min: float) -> pd.Series:
    u = command_pct.astype(float).clip(lower=0.0, upper=100.0) / 100.0
    if len(u) == 0:
        return u.astype(float)
    t = elapsed_seconds(u.index)
    td = max(float(dead_time_min) * 60.0, 0.0)
    tau = max(float(tau_min) * 60.0, 1e-9)
    u_del = np.interp(t - td, t, u.to_numpy(dtype=float), left=float(u.iloc[0]), right=float(u.iloc[-1]))
    y = np.zeros_like(u_del, dtype=float)
    y[0] = u_del[0]
    for i in range(1, len(y)):
        dt = max(float(t[i] - t[i - 1]), 0.0)
        alpha = 1.0 - math.exp(-dt / tau)
        y[i] = y[i - 1] + alpha * (u_del[i] - y[i - 1])
    return pd.Series(y, index=u.index, name='co2_effective_fraction')


def estimate_outdoor_rh_from_dewpoint(t_out_c: pd.Series, dewpoint_offset_c: float) -> pd.Series:
    dawn = t_out_c.between_time('03:00', '07:00')
    if len(dawn) == 0 or dawn.isna().all():
        tmin_early = float(t_out_c.min())
    else:
        tmin_early = float(dawn.min())
    dewpoint_c = tmin_early - float(dewpoint_offset_c)
    e_dew = float(wgc.e_s_tetens(dewpoint_c))
    rh_frac = np.clip(e_dew / wgc.e_s_tetens(t_out_c.to_numpy(dtype=float)), 0.0, 1.0)
    return pd.Series(rh_frac * 100.0, index=t_out_c.index, name='RH_out_est')


def rh_pct_from_absolute_humidity_g_m3(absolute_humidity_g_m3: pd.Series, t_c: pd.Series) -> pd.Series:
    ah = absolute_humidity_g_m3.astype(float).clip(lower=0.0)
    t = t_c.astype(float)
    vapor_pressure_pa = ah * (t + 273.15) / 216.7 * 100.0
    rh_frac = np.clip(vapor_pressure_pa / wgc.e_s_tetens(t.to_numpy(dtype=float)), 0.0, 1.0)
    return pd.Series(rh_frac * 100.0, index=t.index, name='RH_out_from_absolute_humidity')


def absolute_humidity_g_m3_from_rh_pct(rh_pct: pd.Series, t_c: pd.Series) -> pd.Series:
    rh_frac = rh_pct.astype(float).clip(lower=0.0, upper=100.0) / 100.0
    t = t_c.astype(float)
    vapor_pressure_hpa = rh_frac * pd.Series(wgc.e_s_tetens(t.to_numpy(dtype=float)), index=t.index) / 100.0
    ah = 216.7 * vapor_pressure_hpa / (t + 273.15)
    return ah.rename('absolute_humidity_g_m3')


def h_out_mc_adams(u_ms: pd.Series | np.ndarray | float) -> np.ndarray:
    u = np.maximum(np.asarray(u_ms, dtype=float), 0.0)
    return 5.8 + 4.1 * u


def calc_dynamic_ua_series(wind_ms: pd.Series, ua_base_w_k: float, cover_area_m2: float, u_ref_ms: float) -> tuple[pd.Series, float, float]:
    u_base = float(ua_base_w_k) / float(cover_area_m2)
    h_ref = h_out_mc_adams(float(u_ref_ms))
    r_fixed = 1.0 / u_base - 1.0 / h_ref
    if r_fixed <= 0:
        raise ValueError(f'Non-positive R_fixed = {r_fixed}. Check UA_base or reference wind.')
    h_now = h_out_mc_adams(wind_ms)
    u_eff = 1.0 / (r_fixed + 1.0 / h_now)
    ua_eff = u_eff * float(cover_area_m2)
    return pd.Series(ua_eff, index=wind_ms.index, name='UA_eff'), float(u_base), float(r_fixed)


def lowpass_series(x: pd.Series, tau_min: float) -> pd.Series:
    idx = x.index
    t = elapsed_seconds(idx)
    xarr = x.to_numpy(dtype=float)
    y = np.zeros_like(xarr)
    y[0] = xarr[0]
    tau_s = max(float(tau_min) * 60.0, 1e-9)
    for i in range(1, len(y)):
        dt = max(float(t[i] - t[i - 1]), 1.0)
        a = 1.0 - math.exp(-dt / tau_s)
        y[i] = y[i - 1] + a * (xarr[i] - y[i - 1])
    return pd.Series(y, index=idx)


def day_closed_ach_from_wind(wind_ms: pd.Series) -> pd.Series:
    wind = wind_ms.fillna(0.0).clip(lower=0.0).astype(float)
    return (0.10 + 0.05 * wind).clip(lower=0.10, upper=0.50).rename('Q_closed_ach_day')


def night_screen_ach(screen_pct: pd.Series, wind_ms: pd.Series) -> pd.Series:
    screen = screen_pct.fillna(0.0).astype(float)
    wind = wind_ms.fillna(0.0).clip(lower=0.0).astype(float)
    closed = screen >= 80.0
    ach = pd.Series(index=screen.index, dtype=float)
    ach_closed = (0.10 + 0.05 * wind).clip(lower=0.10, upper=0.50)
    ach_open = (0.15 + 0.05 * wind).clip(lower=0.15, upper=0.70)
    ach[closed] = ach_closed[closed]
    ach[~closed] = ach_open[~closed]
    return ach.rename('Q_night_ach_screen')


def optical_screen_closure(screen_pct: pd.Series, *, full_cover_pct: float = 90.0) -> pd.Series:
    raw = screen_pct.fillna(0.0).astype(float).clip(lower=0.0, upper=100.0)
    return (raw / float(full_cover_pct)).clip(lower=0.0, upper=1.0).rename('screen_optical_closure')


def thermal_screen_closure(screen_pct: pd.Series) -> pd.Series:
    raw = screen_pct.fillna(0.0).astype(float).clip(lower=0.0, upper=100.0)
    return (raw / 100.0).clip(lower=0.0, upper=1.0).rename('screen_thermal_closure')


def combine_screen_layers(
    *,
    screen_layers: Sequence[ScreenLayer],
    screen_pct_1: pd.Series,
    screen_pct_2: pd.Series,
) -> pd.DataFrame:
    idx = screen_pct_1.index
    screen_series = [screen_pct_1, screen_pct_2]
    tau_sw = pd.Series(1.0, index=idx, dtype=float)
    ua_ratio = pd.Series(1.0, index=idx, dtype=float)
    sky_ratio = pd.Series(1.0, index=idx, dtype=float)
    optical_closures: list[pd.Series] = []
    thermal_closures: list[pd.Series] = []

    for i, layer in enumerate(screen_layers[:2]):
        raw = screen_series[i]
        optical = optical_screen_closure(raw)
        thermal = thermal_screen_closure(raw)
        tau_sw *= 1.0 - optical * (1.0 - float(layer.shortwave_transmittance_closed))
        ua_ratio *= 1.0 - thermal * (1.0 - float(layer.ua_closed_ratio))
        sky_ratio *= 1.0 - thermal * (1.0 - float(layer.sky_coupling_closed_ratio))
        optical_closures.append(optical.rename(f'screen{i+1}_optical_closure'))
        thermal_closures.append(thermal.rename(f'screen{i+1}_thermal_closure'))

    if len(optical_closures) < 2:
        for i in range(len(optical_closures), 2):
            optical_closures.append(pd.Series(0.0, index=idx, dtype=float, name=f'screen{i+1}_optical_closure'))
            thermal_closures.append(pd.Series(0.0, index=idx, dtype=float, name=f'screen{i+1}_thermal_closure'))

    return pd.DataFrame({
        'screen1_optical_closure': optical_closures[0],
        'screen2_optical_closure': optical_closures[1],
        'screen1_thermal_closure': thermal_closures[0],
        'screen2_thermal_closure': thermal_closures[1],
        'screen_shortwave_transmission': tau_sw.clip(lower=0.0, upper=1.0),
        'screen_ua_ratio': ua_ratio.clip(lower=0.2, upper=1.0),
        'screen_sky_ratio': sky_ratio.clip(lower=0.1, upper=1.0),
    }, index=idx)


def infer_step_hours(index: pd.DatetimeIndex) -> pd.Series:
    if len(index) == 0:
        return pd.Series(dtype=float)
    dt_s = step_seconds(index)
    return pd.Series(dt_s / 3600.0, index=index, name='dt_h')



def safe_gradient(values: pd.Series | np.ndarray, index: pd.DatetimeIndex) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    if y.size == 0:
        return np.array([], dtype=float)
    if y.size == 1:
        return np.zeros(1, dtype=float)
    t = elapsed_seconds(index)
    uniq_t, inverse = np.unique(t, return_inverse=True)
    if uniq_t.size <= 1:
        return np.zeros_like(y, dtype=float)
    y_sum = np.bincount(inverse, weights=y)
    y_count = np.bincount(inverse)
    y_uniq = y_sum / np.maximum(y_count, 1)
    grad_uniq = np.gradient(y_uniq, uniq_t)
    return np.asarray(grad_uniq[inverse], dtype=float)


def resolve_default_radiation_scaling_csv(script_dir: str | Path) -> Optional[Path]:
    base = Path(script_dir)
    candidate = base / 'scaling_daily_2025.csv'
    if candidate.exists():
        return candidate
    matches = sorted(base.glob('scaling_daily_*.csv'))
    return matches[0] if matches else None


def _read_csv_flexible(path: str | Path, *, header='infer', names=None) -> pd.DataFrame:
    encodings = ('utf-8-sig', 'utf-8', 'cp932', 'shift_jis', 'latin1')
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, header=header, names=names, comment='#', skipinitialspace=True)
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f'Could not read {path}: {last_err}')


def load_radiation_scaling_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    df = _read_csv_flexible(p)
    df.columns = [str(c).strip() for c in df.columns]
    cols_lower = {str(c).strip().lower(): str(c).strip() for c in df.columns}

    if 'date' in cols_lower and 'scaling_daily' in cols_lower:
        out = pd.DataFrame({
            'date': df[cols_lower['date']],
            'scale': df[cols_lower['scaling_daily']],
        })
        if 'source' in cols_lower:
            out['row_source'] = df[cols_lower['source']]
        else:
            out['row_source'] = p.name
    elif 'date' in cols_lower and 'scaling' in cols_lower:
        out = pd.DataFrame({
            'date': df[cols_lower['date']],
            'scale': df[cols_lower['scaling']],
        })
        out['row_source'] = p.name
    else:
        df2 = _read_csv_flexible(p, header=None, names=['date', 'scale'])
        out = pd.DataFrame({
            'date': df2['date'],
            'scale': df2['scale'],
        })
        out['row_source'] = p.name

    out['date'] = pd.to_datetime(out['date'], errors='coerce')
    out['scale'] = pd.to_numeric(out['scale'], errors='coerce')
    out = out.dropna(subset=['date', 'scale']).copy()
    out = out[out['scale'] > 0].copy()
    if len(out) == 0:
        raise ValueError(f'No valid scaling rows found in {p}')
    out['date_only'] = out['date'].dt.date
    out['mmdd'] = out['date'].dt.strftime('%m-%d')
    return out.sort_values('date').reset_index(drop=True)


def lookup_radiation_scaling(path: Optional[str | Path], target_date: dt.date, *, mode: str = 'auto') -> RadiationScalingMatch:
    if path is None or mode == 'off':
        return RadiationScalingMatch(False, 1.0, str(path) if path else '', '', 'off', 'disabled', '')

    table = load_radiation_scaling_table(path)
    p = Path(path)

    if mode in {'auto', 'exact'}:
        exact = table[table['date_only'] == target_date]
        if len(exact) > 0:
            row = exact.iloc[-1]
            return RadiationScalingMatch(True, float(row['scale']), str(p), str(row['date'].date()), 'exact', str(row['row_source']), '')

    if mode in {'auto', 'monthday'}:
        mmdd = target_date.strftime('%m-%d')
        monthday = table[table['mmdd'] == mmdd].copy()
        if len(monthday) > 0:
            monthday['year_delta'] = (monthday['date'].dt.year - target_date.year).abs()
            monthday = monthday.sort_values(['year_delta', 'date'])
            row = monthday.iloc[0]
            return RadiationScalingMatch(True, float(row['scale']), str(p), str(row['date'].date()), 'monthday', str(row['row_source']), '')
        if target_date.month == 2 and target_date.day == 29:
            for mmdd_alt in ('02-28', '03-01'):
                monthday = table[table['mmdd'] == mmdd_alt].copy()
                if len(monthday) > 0:
                    monthday['year_delta'] = (monthday['date'].dt.year - target_date.year).abs()
                    monthday = monthday.sort_values(['year_delta', 'date'])
                    row = monthday.iloc[0]
                    return RadiationScalingMatch(True, float(row['scale']), str(p), str(row['date'].date()), f'monthday_fallback:{mmdd_alt}', str(row['row_source']), 'leap-day fallback')

    raise ValueError(f'No radiation scaling row found for {target_date.isoformat()} in {p} with mode={mode}')

def find_missing_intervals(index: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if len(index) == 0:
        return []
    day0 = index[0].normalize()
    day1 = day0 + pd.Timedelta(days=1)
    dt_s = step_seconds(index)
    median_dt = float(np.median(dt_s[np.isfinite(dt_s) & (dt_s > 0)])) if len(dt_s) else 60.0
    if not np.isfinite(median_dt) or median_dt <= 0:
        median_dt = 60.0
    step = pd.to_timedelta(median_dt, unit='s')
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    first_expected = day0
    if index[0] > first_expected:
        intervals.append((first_expected, index[0]))
    for prev, curr in zip(index[:-1], index[1:]):
        if (curr - prev) > 1.5 * step:
            intervals.append((prev + step, curr))
    last_expected = day1
    if index[-1] + step < last_expected:
        intervals.append((index[-1] + step, last_expected))
    return intervals


def integrate_rate_g_per_h(rate_g_h: pd.Series, dt_h: pd.Series) -> pd.Series:
    rate = rate_g_h.astype(float).fillna(0.0)
    dt = dt_h.reindex(rate.index).astype(float).fillna(0.0)
    return (rate * dt).cumsum()


# -----------------------------
# Model input preparation
# -----------------------------
def read_log(path: str | Path) -> pd.DataFrame:
    return parse_timestamp(read_text_table(path))


def resolve_default_outside_pf_root() -> Optional[Path]:
    candidates = [
        Path.home() / 'Library/CloudStorage/GoogleDrive-soi.toi.chi@gmail.com/マイドライブ/greenhouse_log/outside_pf',
        Path.home() / 'Google Drive/マイドライブ/greenhouse_log/outside_pf',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_outside_pf_log(path: str | Path) -> pd.DataFrame:
    query = """
        SELECT
            "年月日" AS ymd,
            "時間" AS time,
            "気温" AS temperature_c,
            "湿度" AS rh_pct,
            "絶対湿度" AS absolute_humidity_g_m3,
            "ＣＯ２濃度" AS co2_ppm
        FROM AGLOG
    """
    with sqlite3.connect(Path(path)) as con:
        df = pd.read_sql_query(query, con)
    if len(df) == 0:
        return pd.DataFrame(columns=['temperature_c', 'rh_pct', 'absolute_humidity_g_m3', 'co2_ppm'])
    df['timestamp'] = pd.to_datetime(df['ymd'].astype(str).str.strip() + ' ' + df['time'].astype(str).str.strip(), errors='coerce')
    df = df.dropna(subset=['timestamp']).set_index('timestamp').sort_index()
    for col in ['temperature_c', 'rh_pct', 'absolute_humidity_g_m3', 'co2_ppm']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['ymd_date'] = pd.to_datetime(df['ymd'], errors='coerce').dt.date
    return df[['temperature_c', 'rh_pct', 'absolute_humidity_g_m3', 'co2_ppm']].groupby(level=0).mean().sort_index()


def load_outside_pf_day(root: str | Path, target_date: dt.date) -> pd.DataFrame:
    root_path = Path(root)
    frames: list[pd.DataFrame] = []
    for db_path in sorted(root_path.rglob('DATABASE_*.DB')):
        if db_path.name == 'DATABASE_MASTER.DB':
            continue
        try:
            query = """
                SELECT
                    "年月日" AS ymd,
                    "時間" AS time,
                    "気温" AS temperature_c,
                    "湿度" AS rh_pct,
                    "絶対湿度" AS absolute_humidity_g_m3,
                    "ＣＯ２濃度" AS co2_ppm
                FROM AGLOG
            """
            with sqlite3.connect(db_path) as con:
                df = pd.read_sql_query(query, con)
        except sqlite3.DatabaseError:
            continue
        if len(df) == 0:
            continue
        ymd = pd.to_datetime(df['ymd'], errors='coerce').dt.date
        df = df.loc[ymd == target_date].copy()
        if len(df) == 0:
            continue
        df['timestamp'] = pd.to_datetime(df['ymd'].astype(str).str.strip() + ' ' + df['time'].astype(str).str.strip(), errors='coerce')
        df = df.dropna(subset=['timestamp']).set_index('timestamp').sort_index()
        for col in ['temperature_c', 'rh_pct', 'absolute_humidity_g_m3', 'co2_ppm']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        frames.append(df[['temperature_c', 'rh_pct', 'absolute_humidity_g_m3', 'co2_ppm']])
    if not frames:
        raise ValueError(f'No AGLOG rows found in outside_pf DB files under {root_path} for {target_date.isoformat()}')
    return pd.concat(frames).groupby(level=0).mean().sort_index()


def resolve_outdoor_humidity_and_co2(
    *,
    main_index: pd.DatetimeIndex,
    t_out: pd.Series,
    preset: HousePreset,
    humidity_co2_source: OutdoorHumidityCo2Source,
) -> tuple[pd.Series, pd.Series, pd.Series, str]:
    if humidity_co2_source.mode == 'legacy':
        rh_out = estimate_outdoor_rh_from_dewpoint(t_out.astype(float), preset.dewpoint_offset_c)
        x_out = pd.Series(float(preset.outdoor_co2_ppm), index=main_index, name='X_out')
        ah_out = absolute_humidity_g_m3_from_rh_pct(rh_out, t_out.astype(float))
        detail = f'legacy dewpoint-offset RH + fixed outdoor CO2 {preset.outdoor_co2_ppm:.1f} ppm'
        return rh_out, x_out, ah_out, detail

    if humidity_co2_source.mode != 'outside_pf':
        raise ValueError(f'Unsupported outdoor humidity/CO2 source mode: {humidity_co2_source.mode}')
    if not humidity_co2_source.outside_pf_root:
        raise ValueError('outside_pf mode requires an outside_pf root directory.')

    pf_day = load_outside_pf_day(humidity_co2_source.outside_pf_root, main_index[0].date())
    ah_out = align_to_index(pf_day['absolute_humidity_g_m3'], main_index).rename('AH_out')
    x_out = align_to_index(pf_day['co2_ppm'], main_index).rename('X_out')
    rh_out = rh_pct_from_absolute_humidity_g_m3(ah_out, t_out.astype(float))
    detail = f'outside_pf AGLOG absolute humidity + CO2 from {humidity_co2_source.outside_pf_root}'
    return rh_out, x_out, ah_out, detail


def borrow_outdoor_series(main_index: pd.DatetimeIndex, main_log: pd.DataFrame, reference_log: Optional[pd.DataFrame]) -> tuple[pd.Series, pd.Series, pd.Series, str]:
    if reference_log is None:
        sun = as_float_series(main_log, ['Sun.L', 'Sun', 'I.Sun'], required=False, default=0.0)
        t_out = as_float_series(main_log, ['Out_T', 'T_out'], required=False, default=np.nan)
        wind = as_float_series(main_log, ['Wind'], required=False, default=np.nan).ffill().bfill().fillna(0.0)
        mode = 'own greenhouse outdoor columns'
        return sun, t_out, wind, mode

    ref = reference_log
    sun_ref = as_float_series(ref, ['Sun.L', 'Sun', 'I.Sun'], required=False, default=0.0)
    t_out_ref = as_float_series(ref, ['Out_T', 'T_out'], required=False, default=np.nan)
    wind_ref = as_float_series(ref, ['Wind'], required=False, default=np.nan).ffill().bfill().fillna(0.0)
    sun = align_to_index(sun_ref, main_index)
    t_out = align_to_index(t_out_ref, main_index)
    wind = align_to_index(wind_ref, main_index)
    mode = 'borrowed Sun.L / Out_T / Wind from reference log'
    return sun, t_out, wind, mode


@dataclass
class PreparedDay:
    g: pd.DataFrame
    house: str
    preset: HousePreset
    out_wind: pd.DataFrame
    main_cmd_pct: pd.Series
    secondary_cmd_pct: pd.Series
    main_cmd_col: Optional[str]
    secondary_cmd_col: Optional[str]
    effective_fraction_main: pd.Series
    effective_fraction_secondary: pd.Series
    total_effective_fraction: pd.Series
    s_co2_main_mol_s: pd.Series
    s_co2_secondary_mol_s: pd.Series
    fuel_rate_main_l_s_effective: pd.Series
    fuel_rate_secondary_l_s_effective: pd.Series
    t_in_raw: pd.Series
    rh_in_raw: pd.Series
    x_in_raw: pd.Series
    sun_out_raw: pd.Series
    sun_out: pd.Series
    solar_scaling: RadiationScalingMatch
    t_out: pd.Series
    rh_out: pd.Series
    x_out: pd.Series
    outdoor_absolute_humidity_g_m3: pd.Series
    wind: pd.Series
    roof_vent_max_pct: pd.Series
    screen_pct: pd.Series
    screen1_pct: pd.Series
    screen2_pct: pd.Series
    screen1_optical_closure: pd.Series
    screen2_optical_closure: pd.Series
    screen1_thermal_closure: pd.Series
    screen2_thermal_closure: pd.Series
    screen_shortwave_transmission: pd.Series
    screen_ua_ratio: pd.Series
    screen_sky_ratio: pd.Series
    delta_t_sky_eff_k: pd.Series
    heater_signal_pct: pd.Series
    s_co2_mol_s: pd.Series
    w_inj_kg_s: pd.Series
    h_burn_w: pd.Series
    h_heat_w: pd.Series
    h_solar_w: pd.Series
    h_in_w: pd.Series
    ua_eff_w_k: pd.Series
    ua_base_u_w_m2_k: float
    r_fixed_m2_k_w: float
    outdoor_mode_desc: str
    outdoor_humidity_co2_desc: str


def prepare_day(
    input_log: str | Path,
    *,
    preset: HousePreset,
    borrow_outdoor_from: Optional[str | Path] = None,
    outdoor_humidity_co2_source: OutdoorHumidityCo2Source = OutdoorHumidityCo2Source(mode='legacy'),
    radiation_scaling_csv: Optional[str | Path] = None,
    radiation_scaling_mode: str = 'off',
) -> PreparedDay:
    g = read_log(input_log)
    ref = read_log(borrow_outdoor_from) if borrow_outdoor_from else None
    idx = g.index

    t_in = as_float_series(g, ['T_in', 'Temp'])
    rh_in = as_float_series(g, ['RH_in', 'Humi'])
    x_in = as_float_series(g, ['X_in', 'Co2', 'CO2'])
    sun_out_raw, t_out, wind, outdoor_mode = borrow_outdoor_series(idx, g, ref)
    if t_out.isna().all():
        raise ValueError('Outdoor temperature is required. Provide Out_T in the log or --borrow-outdoor-from.')
    if len(idx) == 0:
        raise ValueError('Input log has no valid timestamp rows.')

    target_date = idx[0].date()
    solar_scaling = lookup_radiation_scaling(radiation_scaling_csv, target_date, mode=radiation_scaling_mode)
    sun_out = (sun_out_raw.astype(float) * float(solar_scaling.factor)).rename('Sun_out_scaled')
    if solar_scaling.enabled:
        outdoor_mode = f'{outdoor_mode}; Sun.L scaling x{solar_scaling.factor:.4f} ({solar_scaling.match_mode}:{solar_scaling.matched_date})'

    rh_out, x_out, outdoor_absolute_humidity_g_m3, outdoor_humidity_co2_desc = resolve_outdoor_humidity_and_co2(
        main_index=idx,
        t_out=t_out,
        preset=preset,
        humidity_co2_source=outdoor_humidity_co2_source,
    )

    cmd1_raw, main_cmd_col = read_signal_series(g, preset.main_cmd_candidates, default=0.0)
    cmd1 = cmd1_raw.clip(lower=0.0, upper=100.0)
    if preset.has_secondary_burner and len(preset.secondary_cmd_candidates) > 0:
        cmd2_raw, secondary_cmd_col = read_signal_series(g, preset.secondary_cmd_candidates, default=0.0)
        cmd2 = cmd2_raw.clip(lower=0.0, upper=100.0)
    else:
        secondary_cmd_col = None
        cmd2 = pd.Series(0.0, index=idx, dtype=float)
    heater_signal_pct = as_float_series(g, ['CV_H1'], required=False, default=0.0).clip(lower=0.0)

    eff1 = fopdt_fraction(cmd1, dead_time_min=preset.co2_dead_time_min, tau_min=preset.co2_tau_min)
    eff2 = fopdt_fraction(cmd2, dead_time_min=preset.co2_dead_time_min, tau_min=preset.co2_tau_min)
    total_eff = eff1 + eff2

    fuel_rate_main_l_s_effective = preset.fuel_rate_l_h_main / 3600.0 * eff1
    fuel_rate_secondary_l_s_effective = preset.fuel_rate_l_h_secondary / 3600.0 * eff2
    fuel_rate_l_s_effective = fuel_rate_main_l_s_effective + fuel_rate_secondary_l_s_effective

    s_co2_main_mol_s = fuel_rate_main_l_s_effective * CO2_MOL_PER_L_FUEL
    s_co2_secondary_mol_s = fuel_rate_secondary_l_s_effective * CO2_MOL_PER_L_FUEL
    s_co2_mol_s = s_co2_main_mol_s + s_co2_secondary_mol_s

    w_inj_kg_s = fuel_rate_l_s_effective * H2O_KG_PER_L_FUEL
    h_burn_w = fuel_rate_l_s_effective * LHV_J_PER_L_FUEL
    heater_frac = heater_signal_pct.clip(lower=0.0, upper=100.0) / 100.0
    h_heat_w = preset.heater_capacity_w * heater_frac
    h_solar_w = preset.solar_heat_factor * preset.geometry.floor_area_m2 * sun_out
    h_in_w = h_burn_w + h_heat_w + h_solar_w

    ua_eff, u_base, r_fixed = calc_dynamic_ua_series(
        wind.fillna(0.0).ffill().bfill().fillna(0.0),
        preset.ua_base_w_k,
        preset.geometry.cover_area_m2,
        preset.u_ref_ms,
    )

    model = pd.DataFrame(index=idx)
    model['T_in'] = t_in
    model['RH_in'] = rh_in
    model['X_in'] = x_in
    model['T_out'] = t_out
    model['RH_out'] = rh_out
    model['X_out'] = x_out
    model['H_in'] = h_in_w
    model['S_CO2'] = s_co2_mol_s
    model['W_inj'] = w_inj_kg_s
    model['UA'] = ua_eff

    roof_cols = [c for c in ['CV_K1', 'CV_K2', 'CV_K3', 'CV_K4'] if c in g.columns]
    if roof_cols:
        roof_vent_max = g[roof_cols].apply(pd.to_numeric, errors='coerce').max(axis=1).reindex(idx)
    else:
        roof_vent_max = as_float_series(g, ['Kanki'], required=False, default=0.0)
    screen1_pct = as_float_series(g, ['CV_C1'], required=False, default=0.0).clip(lower=0.0, upper=100.0)
    screen2_pct = as_float_series(g, ['CV_C2'], required=False, default=0.0).clip(lower=0.0, upper=100.0)
    screen_pct = pd.concat([screen1_pct, screen2_pct], axis=1).max(axis=1)
    screen_effect = combine_screen_layers(
        screen_layers=preset.screen_layers,
        screen_pct_1=screen1_pct,
        screen_pct_2=screen2_pct,
    )

    h_solar_w = h_solar_w * screen_effect['screen_shortwave_transmission']
    h_in_w = h_burn_w + h_heat_w + h_solar_w
    ua_eff = ua_eff * screen_effect['screen_ua_ratio']
    delta_t_sky_eff_k = 2.0 * screen_effect['screen_sky_ratio']

    model['H_in'] = h_in_w
    model['UA'] = ua_eff
    model['deltaT_sky'] = delta_t_sky_eff_k

    out_wind = wgc.solve_whole_greenhouse_timeseries(
        model,
        V_m3=preset.geometry.air_volume_m3,
        A_cov_m2=preset.geometry.cover_area_m2,
        UA_col='UA',
        W_inj_col='W_inj',
        deltaT_sky_col='deltaT_sky',
        p_Pa=101325.0,
        h_c_W_m2_K=5.0,
        deltaT_sky_K=0.0,
        smooth_window=preset.smooth_window,
        cond_only=False,
        dh_eps_J_kg=200.0,
        enforce_Q_nonneg=False,
    )

    return PreparedDay(
        g=g,
        house=preset.display_name,
        preset=preset,
        out_wind=out_wind,
        main_cmd_pct=cmd1,
        secondary_cmd_pct=cmd2,
        main_cmd_col=main_cmd_col,
        secondary_cmd_col=secondary_cmd_col,
        effective_fraction_main=eff1,
        effective_fraction_secondary=eff2,
        total_effective_fraction=total_eff,
        s_co2_main_mol_s=s_co2_main_mol_s,
        s_co2_secondary_mol_s=s_co2_secondary_mol_s,
        fuel_rate_main_l_s_effective=fuel_rate_main_l_s_effective,
        fuel_rate_secondary_l_s_effective=fuel_rate_secondary_l_s_effective,
        t_in_raw=t_in,
        rh_in_raw=rh_in,
        x_in_raw=x_in,
        sun_out_raw=sun_out_raw,
        sun_out=sun_out,
        solar_scaling=solar_scaling,
        t_out=t_out,
        rh_out=rh_out,
        x_out=x_out,
        outdoor_absolute_humidity_g_m3=outdoor_absolute_humidity_g_m3,
        wind=wind.fillna(0.0).ffill().bfill().fillna(0.0),
        roof_vent_max_pct=roof_vent_max.fillna(0.0),
        screen_pct=screen_pct.fillna(0.0),
        screen1_pct=screen1_pct.fillna(0.0),
        screen2_pct=screen2_pct.fillna(0.0),
        screen1_optical_closure=screen_effect['screen1_optical_closure'],
        screen2_optical_closure=screen_effect['screen2_optical_closure'],
        screen1_thermal_closure=screen_effect['screen1_thermal_closure'],
        screen2_thermal_closure=screen_effect['screen2_thermal_closure'],
        screen_shortwave_transmission=screen_effect['screen_shortwave_transmission'],
        screen_ua_ratio=screen_effect['screen_ua_ratio'],
        screen_sky_ratio=screen_effect['screen_sky_ratio'],
        delta_t_sky_eff_k=delta_t_sky_eff_k,
        heater_signal_pct=heater_signal_pct,
        s_co2_mol_s=s_co2_mol_s,
        w_inj_kg_s=w_inj_kg_s,
        h_burn_w=h_burn_w,
        h_heat_w=h_heat_w,
        h_solar_w=h_solar_w,
        h_in_w=h_in_w,
        ua_eff_w_k=ua_eff,
        ua_base_u_w_m2_k=u_base,
        r_fixed_m2_k_w=r_fixed,
        outdoor_mode_desc=outdoor_mode,
        outdoor_humidity_co2_desc=outdoor_humidity_co2_desc,
    )


def apply_indoor_sensor_validation_offsets(
    prep: PreparedDay,
    *,
    temp_offset_c: float = 0.0,
    rh_scale: float = 1.0,
) -> PreparedDay:
    temp_offset_c = float(temp_offset_c)
    rh_scale = float(rh_scale)
    if abs(temp_offset_c) < 1e-12 and abs(rh_scale - 1.0) < 1e-12:
        return prep

    idx = prep.out_wind.index
    t_in_adj = (prep.t_in_raw.astype(float) + temp_offset_c).rename('T_in_validation_C')
    rh_in_adj = (prep.rh_in_raw.astype(float) * rh_scale).clip(lower=0.0, upper=110.0).rename('RH_in_validation_pct')

    model = pd.DataFrame(index=idx)
    model['T_in'] = t_in_adj
    model['RH_in'] = rh_in_adj
    model['X_in'] = prep.x_in_raw.astype(float)
    model['T_out'] = prep.t_out.astype(float)
    model['RH_out'] = prep.rh_out.astype(float)
    model['X_out'] = prep.x_out.astype(float)
    model['H_in'] = prep.h_in_w.astype(float)
    model['S_CO2'] = prep.s_co2_mol_s.astype(float)
    model['W_inj'] = prep.w_inj_kg_s.astype(float)
    model['UA'] = prep.ua_eff_w_k.astype(float)
    model['deltaT_sky'] = prep.delta_t_sky_eff_k.astype(float)

    out_wind = wgc.solve_whole_greenhouse_timeseries(
        model,
        V_m3=prep.preset.geometry.air_volume_m3,
        A_cov_m2=prep.preset.geometry.cover_area_m2,
        UA_col='UA',
        W_inj_col='W_inj',
        deltaT_sky_col='deltaT_sky',
        p_Pa=101325.0,
        h_c_W_m2_K=5.0,
        deltaT_sky_K=0.0,
        smooth_window=prep.preset.smooth_window,
        cond_only=False,
        dh_eps_J_kg=200.0,
        enforce_Q_nonneg=False,
    )
    return replace(prep, out_wind=out_wind)


# -----------------------------
# Core calculation
# -----------------------------
def apply_current_model(prep: PreparedDay) -> pd.DataFrame:
    preset = prep.preset
    geom = preset.geometry
    out = prep.out_wind
    idx = out.index

    closed = prep.roof_vent_max_pct.fillna(0.0) <= float(preset.vent_closed_threshold_pct)
    q_closed_ach = day_closed_ach_from_wind(prep.wind)

    t_lp = lowpass_series(out['T_in_C'], preset.tau_store_min)
    t_sec = elapsed_seconds(idx)
    dTlp_dt = pd.Series(safe_gradient(t_lp.to_numpy(dtype=float), idx), index=idx)
    h_struct = pd.Series(0.0, index=idx, dtype=float)
    h_struct_closed = preset.c_eff_j_k * dTlp_dt + preset.beta_solar * prep.h_solar_w
    h_struct[closed] = h_struct_closed[closed]

    w_cond_phys = pd.Series(
        wgc.condensation_flow(
            out['T_in_C'].values,
            out['RH_in_frac'].values,
            out['T_out_C'].values,
            np.full(len(out), 101325.0),
            prep.ua_eff_w_k.values,
            geom.cover_area_m2,
            h_c_W_m2_K=np.full(len(out), 5.0),
            deltaT_sky_K=prep.delta_t_sky_eff_k.values,
            L_v=2.45e6,
            cp_da=1005.0,
            cond_only=False,
        ),
        index=idx,
    )

    cp_da = 1005.0
    cp_v = 1860.0
    ls = 2.501e6 + cp_v * out['T_in_C']
    delta_w = out['omega_in'] - out['omega_out']
    delta_h = out['h_in'] - out['h_out']

    r1 = out['rho_da'] * geom.air_volume_m3 * out['domega_dt'] - prep.w_inj_kg_s + w_cond_phys
    r2 = (
        prep.h_in_w
        - prep.ua_eff_w_k * (out['T_in_C'] - (out['T_out_C'] - prep.delta_t_sky_eff_k))
        - h_struct
        - out['rho_da'] * geom.air_volume_m3 * (cp_da + cp_v * out['omega_in']) * out['dT_dt']
        - ls * (prep.w_inj_kg_s - w_cond_phys)
    )
    q_prov = (r2 - ls * r1) / (out['rho_da'] * delta_h)
    e_prov = r1 + out['rho_da'] * delta_w * q_prov

    q_final = out['Q_m3_s'].copy()
    q_final[closed] = q_closed_ach[closed] * geom.air_volume_m3 / 3600.0

    e_raw = out['E_kg_s'].copy()
    e_raw[closed] = e_prov[closed]
    e_final = e_raw.clip(lower=0.0)

    w_cond_final = (
        out['rho_da'] * q_final * (out['omega_out'] - out['omega_in'])
        + e_final
        + prep.w_inj_kg_s
        - out['rho_da'] * geom.air_volume_m3 * out['domega_dt']
    )
    p_final_mol_s = (
        out['rho_mol'] * q_final * (out['X_out_molmol'] - out['X_in_molmol'])
        + prep.s_co2_mol_s
        - out['rho_mol'] * geom.air_volume_m3 * out['dX_dt']
    )

    night_closed = (prep.sun_out < preset.daylight_threshold_w_m2) & closed
    q_night_ach = night_screen_ach(prep.screen_pct, prep.wind)
    q_final_night = q_final.copy()
    q_final_night.loc[night_closed] = q_night_ach.loc[night_closed] * geom.air_volume_m3 / 3600.0

    w_cond_night = (
        out['rho_da'] * q_final_night * (out['omega_out'] - out['omega_in'])
        + e_final
        + prep.w_inj_kg_s
        - out['rho_da'] * geom.air_volume_m3 * out['domega_dt']
    )
    p_final_night_mol_s = (
        out['rho_mol'] * q_final_night * (out['X_out_molmol'] - out['X_in_molmol'])
        + prep.s_co2_mol_s
        - out['rho_mol'] * geom.air_volume_m3 * out['dX_dt']
    )

    p_net_umol_m2_s = p_final_night_mol_s / geom.floor_area_m2 * 1e6
    r_est_umol_m2_s = preset.r20_umol_m2_s * (preset.q10 ** ((out['T_in_C'] - 20.0) / 10.0))
    day_mask = prep.sun_out > preset.daylight_threshold_w_m2
    gross_umol_m2_s = pd.Series(np.where(day_mask, np.maximum(p_net_umol_m2_s + r_est_umol_m2_s, 0.0), 0.0), index=idx)

    gross_g_h = gross_umol_m2_s * GCO2_H_PER_UMOL_S
    respiration_g_h = r_est_umol_m2_s * GCO2_H_PER_UMOL_S
    net_balance_g_h = gross_g_h - respiration_g_h
    net_p_raw_g_h = p_net_umol_m2_s * GCO2_H_PER_UMOL_S
    transpiration_g_h = e_final / geom.floor_area_m2 * 3600.0 * 1000.0
    condensation_g_h_cover = w_cond_night / geom.cover_area_m2 * 3600.0 * 1000.0
    ventilation_ach = q_final_night / geom.air_volume_m3 * 3600.0

    co2_source_main_g_h = prep.s_co2_main_mol_s / geom.floor_area_m2 * 44.01 * 3600.0
    co2_source_secondary_g_h = prep.s_co2_secondary_mol_s / geom.floor_area_m2 * 44.01 * 3600.0
    co2_source_g_h = co2_source_main_g_h + co2_source_secondary_g_h
    co2_vent_term_g_h = out['rho_mol'] * q_final_night * (out['X_out_molmol'] - out['X_in_molmol']) / geom.floor_area_m2 * 44.01 * 3600.0
    co2_storage_term_g_h = (-out['rho_mol'] * geom.air_volume_m3 * out['dX_dt']) / geom.floor_area_m2 * 44.01 * 3600.0
    co2_balance_sum_g_h = co2_source_g_h + co2_vent_term_g_h + co2_storage_term_g_h

    dt_h = infer_step_hours(idx)
    res = pd.DataFrame(index=idx)
    res.index.name = 'timestamp'
    res['house'] = prep.house
    res['DATE'] = idx.strftime('%Y-%m-%d')
    res['TIME'] = idx.strftime('%H:%M')
    res['hour_float'] = idx.hour + idx.minute / 60.0 + idx.second / 3600.0
    res['dt_h'] = dt_h
    res['day_mask'] = day_mask.astype(int)
    res['closed_vent_mask'] = closed.astype(int)
    res['night_closed_mask'] = night_closed.astype(int)

    res['Sun_out_raw_W_m2'] = prep.sun_out_raw
    res['Sun_out_W_m2'] = prep.sun_out
    res['solar_scaling_factor'] = float(prep.solar_scaling.factor)
    res['CO2_cmd1_pct'] = prep.main_cmd_pct
    res['CO2_cmd2_pct'] = prep.secondary_cmd_pct
    res['CO2_effective_fraction1'] = prep.effective_fraction_main
    res['CO2_effective_fraction2'] = prep.effective_fraction_secondary
    res['CO2_effective_fraction_total'] = prep.total_effective_fraction
    res['CO2_source_main_signal_col'] = prep.main_cmd_col if prep.main_cmd_col else ''
    res['CO2_source_secondary_signal_col'] = prep.secondary_cmd_col if prep.secondary_cmd_col else ''
    res['roof_vent_max_pct'] = prep.roof_vent_max_pct
    res['screen_pct'] = prep.screen_pct
    res['screen1_pct'] = prep.screen1_pct
    res['screen2_pct'] = prep.screen2_pct
    res['screen1_optical_closure'] = prep.screen1_optical_closure
    res['screen2_optical_closure'] = prep.screen2_optical_closure
    res['screen1_thermal_closure'] = prep.screen1_thermal_closure
    res['screen2_thermal_closure'] = prep.screen2_thermal_closure
    res['screen_shortwave_transmission'] = prep.screen_shortwave_transmission
    res['screen_ua_ratio'] = prep.screen_ua_ratio
    res['screen_sky_ratio'] = prep.screen_sky_ratio
    res['deltaT_sky_eff_K'] = prep.delta_t_sky_eff_k
    res['heater_signal_pct'] = prep.heater_signal_pct

    res['gross_assimilation_gCO2_m2_h'] = gross_g_h
    res['respiration_loss_gCO2_m2_h'] = respiration_g_h
    res['net_carbon_balance_gCO2_m2_h'] = net_balance_g_h
    res['net_photosynthesis_raw_gCO2_m2_h'] = net_p_raw_g_h
    res['transpiration_g_m2_h'] = transpiration_g_h
    res['condensation_g_m2_h_cover'] = condensation_g_h_cover
    res['ventilation_ach'] = ventilation_ach
    res['Q_night_ach_screen'] = q_night_ach

    res['T_in_C'] = out['T_in_C']
    res['RH_in_frac'] = out['RH_in_frac']
    res['T_out_C'] = prep.t_out
    res['RH_out_pct'] = prep.rh_out
    res['AH_out_g_m3'] = prep.outdoor_absolute_humidity_g_m3
    res['CO2_in_ppm'] = prep.x_in_raw
    res['CO2_out_ppm'] = prep.x_out
    res['Wind_m_s'] = prep.wind

    res['co2_source_main_gCO2_m2_h'] = co2_source_main_g_h
    res['co2_source_secondary_gCO2_m2_h'] = co2_source_secondary_g_h
    res['co2_source_gCO2_m2_h'] = co2_source_g_h
    res['co2_vent_term_gCO2_m2_h'] = co2_vent_term_g_h
    res['co2_storage_term_gCO2_m2_h'] = co2_storage_term_g_h
    res['co2_balance_sum_gCO2_m2_h'] = co2_balance_sum_g_h

    res['UA_eff_W_K'] = prep.ua_eff_w_k
    res['H_burn_W'] = prep.h_burn_w
    res['H_heat_W'] = prep.h_heat_w
    res['H_solar_W'] = prep.h_solar_w
    res['H_struct_W'] = h_struct
    res['W_inj_kg_s'] = prep.w_inj_kg_s
    res['S_CO2_main_mol_s'] = prep.s_co2_main_mol_s
    res['S_CO2_secondary_mol_s'] = prep.s_co2_secondary_mol_s
    res['S_CO2_mol_s'] = prep.s_co2_mol_s

    for col in ['omega_in', 'omega_out', 'h_in', 'h_out', 'rho_da', 'rho_mol', 'dT_dt', 'domega_dt', 'dX_dt']:
        res[col] = out[col]

    res['cumulative_gross_assimilation_gCO2_m2'] = integrate_rate_g_per_h(res['gross_assimilation_gCO2_m2_h'], dt_h)
    res['cumulative_respiration_loss_gCO2_m2'] = integrate_rate_g_per_h(res['respiration_loss_gCO2_m2_h'], dt_h)
    res['cumulative_net_carbon_balance_gCO2_m2'] = integrate_rate_g_per_h(res['net_carbon_balance_gCO2_m2_h'], dt_h)
    res['cumulative_net_photosynthesis_raw_gCO2_m2'] = integrate_rate_g_per_h(res['net_photosynthesis_raw_gCO2_m2_h'], dt_h)
    res['cumulative_co2_source_main_gCO2_m2'] = integrate_rate_g_per_h(res['co2_source_main_gCO2_m2_h'], dt_h)
    res['cumulative_co2_source_secondary_gCO2_m2'] = integrate_rate_g_per_h(res['co2_source_secondary_gCO2_m2_h'], dt_h)
    res['cumulative_co2_source_gCO2_m2'] = integrate_rate_g_per_h(res['co2_source_gCO2_m2_h'], dt_h)
    res['cumulative_co2_vent_term_gCO2_m2'] = integrate_rate_g_per_h(res['co2_vent_term_gCO2_m2_h'], dt_h)
    res['cumulative_co2_storage_term_gCO2_m2'] = integrate_rate_g_per_h(res['co2_storage_term_gCO2_m2_h'], dt_h)

    return res


# -----------------------------
# Outputs
# -----------------------------
def build_summary(
    res: pd.DataFrame,
    *,
    preset: HousePreset,
    outdoor_mode_desc: str,
    outdoor_humidity_co2_desc: str,
    input_log: str | Path,
    borrow_outdoor_from: Optional[str | Path],
    solar_scaling: RadiationScalingMatch,
) -> pd.DataFrame:
    coverage_h = float(res['dt_h'].sum())
    daytime_h = float(res.loc[res['day_mask'] > 0, 'dt_h'].sum())
    gross_day = float((res['gross_assimilation_gCO2_m2_h'] * res['dt_h']).sum())
    resp_day = float((res['respiration_loss_gCO2_m2_h'] * res['dt_h']).sum())
    net_day = float((res['net_carbon_balance_gCO2_m2_h'] * res['dt_h']).sum())
    source_main_day = float((res['co2_source_main_gCO2_m2_h'] * res['dt_h']).sum())
    source_secondary_day = float((res['co2_source_secondary_gCO2_m2_h'] * res['dt_h']).sum())
    source_day = float((res['co2_source_gCO2_m2_h'] * res['dt_h']).sum())
    vent_day = float((res['co2_vent_term_gCO2_m2_h'] * res['dt_h']).sum())
    storage_day = float((res['co2_storage_term_gCO2_m2_h'] * res['dt_h']).sum())
    net_p_day = float((res['net_photosynthesis_raw_gCO2_m2_h'] * res['dt_h']).sum())
    solar_raw_day = float((res['Sun_out_raw_W_m2'] * res['dt_h']).sum() * 3600.0 / 1e6)
    solar_scaled_day = float((res['Sun_out_W_m2'] * res['dt_h']).sum() * 3600.0 / 1e6)

    row = {
        'house': preset.display_name,
        'date': str(pd.to_datetime(res['DATE'].iloc[0]).date()),
        'coverage_h': coverage_h,
        'start_time': str(res['TIME'].iloc[0]),
        'end_time': str(res['TIME'].iloc[-1]),
        'daytime_h': daytime_h,
        'gross_assimilation_gCO2_m2_day': gross_day,
        'respiration_loss_gCO2_m2_day': resp_day,
        'net_carbon_balance_gCO2_m2_day': net_day,
        'peak_gross_assimilation_gCO2_m2_h': float(res['gross_assimilation_gCO2_m2_h'].max()),
        'peak_net_photosynthesis_gCO2_m2_h': float(res['net_photosynthesis_raw_gCO2_m2_h'].max()),
        'mean_ventilation_ach': float(res['ventilation_ach'].mean()),
        'min_ventilation_ach': float(res['ventilation_ach'].min()),
        'max_ventilation_ach': float(res['ventilation_ach'].max()),
        'co2_source_main_integral_gCO2_m2_day_equiv': source_main_day,
        'co2_source_secondary_integral_gCO2_m2_day_equiv': source_secondary_day,
        'co2_source_integral_gCO2_m2_day_equiv': source_day,
        'co2_vent_integral_gCO2_m2_day_equiv': vent_day,
        'co2_storage_integral_gCO2_m2_day_equiv': storage_day,
        'net_photosynthesis_integral_gCO2_m2_day_equiv': net_p_day,
        'solar_raw_MJ_m2_day_equiv': solar_raw_day,
        'solar_scaled_MJ_m2_day_equiv': solar_scaled_day,
        'solar_scaling_enabled': bool(solar_scaling.enabled),
        'solar_scaling_factor': float(solar_scaling.factor),
        'solar_scaling_csv': solar_scaling.csv_path,
        'solar_scaling_match_mode': solar_scaling.match_mode,
        'solar_scaling_matched_date': solar_scaling.matched_date,
        'solar_scaling_source': solar_scaling.source,
        'solar_scaling_note': solar_scaling.note,
        'outdoor_mode': outdoor_mode_desc,
        'outdoor_humidity_co2_source': outdoor_humidity_co2_desc,
        'main_source_signal_col': str(res['CO2_source_main_signal_col'].iloc[0]),
        'secondary_source_signal_col': str(res['CO2_source_secondary_signal_col'].iloc[0]),
        'input_log': str(input_log),
        'borrow_outdoor_from': str(borrow_outdoor_from) if borrow_outdoor_from else '',
    }
    return pd.DataFrame([row])


def write_notes(path: str | Path, lines: Iterable[str]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(str(line).rstrip() + '\n')


def make_notes(prep: PreparedDay, res: pd.DataFrame, summary: pd.DataFrame) -> list[str]:
    s = summary.iloc[0]
    p = prep.preset
    g = p.geometry
    return [
        f'# {p.display_name} {s.date} one-day flux and carbon plot',
        '',
        '## Input',
        f'- input log: {s.input_log}',
        f'- outdoor forcing: {prep.outdoor_mode_desc}',
        f'- outdoor humidity / CO2: {s.outdoor_humidity_co2_source}',
        f'- borrowed outdoor log: {s.borrow_outdoor_from if s.borrow_outdoor_from else "none"}',
        '',
        '## Geometry',
        f'- floor area = {g.floor_area_m2:.2f} m2',
        f'- cultivated area = {g.cultivated_area_m2:.2f} m2',
        f'- air volume = {g.air_volume_m3:.2f} m3',
        f'- cover area = {g.cover_area_m2:.2f} m2',
        '',
        '## Radiation scaling',
        f'- scaling enabled = {bool(s.solar_scaling_enabled)}',
        f'- scaling factor = {float(s.solar_scaling_factor):.6f}',
        f'- scaling csv = {s.solar_scaling_csv if s.solar_scaling_csv else "none"}',
        f'- scaling match mode = {s.solar_scaling_match_mode if s.solar_scaling_match_mode else "off"}',
        f'- scaling matched date = {s.solar_scaling_matched_date if s.solar_scaling_matched_date else ""}',
        f'- scaling source = {s.solar_scaling_source if s.solar_scaling_source else ""}',
        f'- scaling note = {s.solar_scaling_note if s.solar_scaling_note else ""}',
        f'- daily solar raw = {float(s.solar_raw_MJ_m2_day_equiv):.4f} MJ m-2 day-equiv',
        f'- daily solar scaled = {float(s.solar_scaled_MJ_m2_day_equiv):.4f} MJ m-2 day-equiv',
        '',
        '## Model settings',
        f'- UA_base = {p.ua_base_w_k:.2f} W K-1',
        f'- u_ref = {p.u_ref_ms:.3f} m s-1',
        f'- dynamic cover UA: h_out = 5.8 + 4.1 u',
        f'- CO2 FOPDT: dead time = {p.co2_dead_time_min:.1f} min, tau = {p.co2_tau_min:.1f} min',
        f'- main source signal column = {s.main_source_signal_col if s.main_source_signal_col else "not found"}',
        f'- secondary source signal column = {s.secondary_source_signal_col if s.secondary_source_signal_col else "not used"}',
        f'- fuel rate main burner = {p.fuel_rate_l_h_main:.3f} L h-1 at 100%',
        f'- fuel rate secondary burner = {p.fuel_rate_l_h_secondary:.3f} L h-1 at 100%',
        f'- screen closure model: optical closure saturates at 90% command, thermal closure reaches 100% at full command',
        f'- screen 1 = {p.screen_layers[0].name if len(p.screen_layers) > 0 else "none"}',
        f'- screen 2 = {p.screen_layers[1].name if len(p.screen_layers) > 1 else "none"}',
        f'- storage model: tau = {p.tau_store_min:.1f} min, C_eff = {p.c_eff_j_k:.4e} J K-1, beta_solar = {p.beta_solar:.6f}',
        f'- night respiration: R20 = {p.r20_umol_m2_s:.4f} umol m-2 s-1, Q10 = {p.q10:.4f}',
        f'- outdoor humidity / CO2 source = {s.outdoor_humidity_co2_source}',
        f'- legacy outdoor CO2 = {p.outdoor_co2_ppm:.1f} ppm',
        f'- legacy outdoor RH estimation: early-morning minimum Out_T - {p.dewpoint_offset_c:.1f} C',
        f'- day threshold for gross assimilation = scaled Sun.L > {p.daylight_threshold_w_m2:.1f} W m-2',
        '',
        '## Daily summary',
        f'- coverage = {float(s.coverage_h):.3f} h, start = {s.start_time}, end = {s.end_time}',
        f'- gross assimilation = {float(s.gross_assimilation_gCO2_m2_day):.4f} gCO2 m-2 day-1',
        f'- respiration loss = {float(s.respiration_loss_gCO2_m2_day):.4f} gCO2 m-2 day-1',
        f'- net carbon balance = {float(s.net_carbon_balance_gCO2_m2_day):.4f} gCO2 m-2 day-1',
        f'- CO2 source integral (main) = {float(s.co2_source_main_integral_gCO2_m2_day_equiv):.4f} gCO2 m-2 day-equiv',
        f'- CO2 source integral (secondary) = {float(s.co2_source_secondary_integral_gCO2_m2_day_equiv):.4f} gCO2 m-2 day-equiv',
        f'- CO2 source integral (total) = {float(s.co2_source_integral_gCO2_m2_day_equiv):.4f} gCO2 m-2 day-equiv',
        f'- CO2 vent term integral = {float(s.co2_vent_integral_gCO2_m2_day_equiv):.4f} gCO2 m-2 day-equiv',
        f'- CO2 storage term integral = {float(s.co2_storage_integral_gCO2_m2_day_equiv):.4f} gCO2 m-2 day-equiv',
        f'- net photosynthesis integral = {float(s.net_photosynthesis_integral_gCO2_m2_day_equiv):.4f} gCO2 m-2 day-equiv',
        '',
        '## Notes',
        '- Graph legends are English only.',
        '- Sun.L is scaled by the daily radiation correction factor before H_solar and the daylight mask are evaluated.',
        '- CO2 source is defined from the house-specific command signal(s): Sankaku uses CO2, Maru uses CO2 + CV_H2.',
        '- Each burner command is converted to effective source by FOPDT (dead time + first-order lag), then to fuel, CO2, H2O, and burner heat.',
        '- Gross assimilation is forced to zero at night, so non-physical positive nighttime photosynthesis is ignored in the cumulative carbon plot.',
        '- Duplicate timestamps are collapsed before calculation by averaging numeric columns at the same timestamp.',
        '- Maru secondary burner ignition-failure correction is not applied in this reference implementation.',
    ]


def make_offset_suffix(temp_offset_c: float, rh_scale: float) -> str:
    parts: list[str] = []
    if abs(float(temp_offset_c)) > 1e-12:
        parts.append(f'toff{float(temp_offset_c):+g}'.replace('+', 'p').replace('-', 'm').replace('.', 'p'))
    if abs(float(rh_scale) - 1.0) > 1e-12:
        parts.append(f'rhs{float(rh_scale):.2f}'.replace('.', 'p'))
    return '_'.join(parts)


# -----------------------------
# Plotting
# -----------------------------
def shade_missing(ax_list: Sequence[plt.Axes], missing_intervals: list[tuple[pd.Timestamp, pd.Timestamp]]) -> None:
    for start, end in missing_intervals:
        for ax in ax_list:
            ax.axvspan(start, end, alpha=0.12, color='0.3')


def plot_one_day(prep: PreparedDay, res: pd.DataFrame, summary: pd.DataFrame, out_png: str | Path) -> None:
    p = prep.preset
    missing_intervals = find_missing_intervals(res.index)
    day0 = res.index[0].normalize()
    day1 = day0 + pd.Timedelta(days=1)

    fig, axes = plt.subplots(6, 1, figsize=(16, 20), sharex=True, constrained_layout=True)

    # 1) drivers
    ax = axes[0]
    ax.plot(res.index, res['Sun_out_W_m2'], label='Sun (scaled)')
    if not np.allclose(res['Sun_out_W_m2'].fillna(0.0).to_numpy(), res['Sun_out_raw_W_m2'].fillna(0.0).to_numpy()):
        ax.plot(res.index, res['Sun_out_raw_W_m2'], linestyle='--', label='Sun (raw)')
    ax.set_ylabel('Sun [W m$^{-2}$]')
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(res.index, res['CO2_in_ppm'], label='Indoor CO2')
    ax2.plot(res.index, res['CO2_cmd1_pct'], linestyle='--', label='CO2 cmd 1')
    if p.has_secondary_burner:
        ax2.plot(res.index, res['CO2_cmd2_pct'], linestyle=':', label='CO2 cmd 2')
    ax2.set_ylabel('CO2 / command [ppm or %]')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', ncol=2, fontsize=9)

    # 2) carbon fluxes
    ax = axes[1]
    ax.plot(res.index, res['gross_assimilation_gCO2_m2_h'], label='Gross assimilation')
    ax.plot(res.index, res['respiration_loss_gCO2_m2_h'], label='Respiration loss')
    ax.plot(res.index, res['net_carbon_balance_gCO2_m2_h'], label='Net carbon balance')
    ax.plot(res.index, res['net_photosynthesis_raw_gCO2_m2_h'], linestyle='--', label='Net photosynthesis raw')
    ax.axhline(0.0, linewidth=1)
    ax.set_ylabel('Carbon flux [gCO2 m$^{-2}$ h$^{-1}$]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', ncol=2, fontsize=9)

    # 3) water fluxes
    ax = axes[2]
    ax.plot(res.index, res['transpiration_g_m2_h'], label='Transpiration')
    ax.plot(res.index, res['condensation_g_m2_h_cover'], label='Condensation / re-evap')
    ax.axhline(0.0, linewidth=1)
    ax.set_ylabel('Water flux [g m$^{-2}$ h$^{-1}$]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)

    # 4) ventilation and controls
    ax = axes[3]
    ax.plot(res.index, res['ventilation_ach'], label='Ventilation')
    ax.plot(res.index, res['Q_night_ach_screen'], linestyle='--', label='Night leakage model')
    ax.axhline(0.0, linewidth=1)
    ax.set_ylabel('Ventilation [ACH]')
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(res.index, res['roof_vent_max_pct'], label='Roof vent')
    ax2.plot(res.index, res['screen_pct'], label='Screen')
    ax2.set_ylabel('Control [%]')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', ncol=2, fontsize=9)

    # 5) CO2 budget stack
    ax = axes[4]
    source = res['co2_source_gCO2_m2_h'].fillna(0.0)
    vent = res['co2_vent_term_gCO2_m2_h'].fillna(0.0)
    storage = res['co2_storage_term_gCO2_m2_h'].fillna(0.0)
    vent_pos = vent.clip(lower=0.0)
    vent_neg = vent.clip(upper=0.0)
    storage_pos = storage.clip(lower=0.0)
    storage_neg = storage.clip(upper=0.0)
    y0 = np.zeros(len(res))
    y1 = source.to_numpy()
    y2 = (source + vent_pos).to_numpy()
    y3 = (source + vent_pos + storage_pos).to_numpy()
    yn1 = vent_neg.to_numpy()
    yn2 = (vent_neg + storage_neg).to_numpy()
    ax.fill_between(res.index, y0, y1, alpha=0.4, label='Source')
    ax.fill_between(res.index, y1, y2, alpha=0.4, label='Vent (+)')
    ax.fill_between(res.index, y2, y3, alpha=0.4, label='Storage (+)')
    ax.fill_between(res.index, y0, yn1, alpha=0.4, label='Vent (-)')
    ax.fill_between(res.index, yn1, yn2, alpha=0.4, label='Storage (-)')
    ax.plot(res.index, res['net_photosynthesis_raw_gCO2_m2_h'], linewidth=1.2, label='Net photosynthesis')
    ax.axhline(0.0, linewidth=1)
    ax.set_ylabel('CO2 budget [gCO2 m$^{-2}$ h$^{-1}$]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', ncol=3, fontsize=9)

    # 6) cumulative carbon
    ax = axes[5]
    ax.plot(res.index, res['cumulative_gross_assimilation_gCO2_m2'], label='Cumulative gross assimilation')
    ax.plot(res.index, res['cumulative_respiration_loss_gCO2_m2'], label='Cumulative respiration loss')
    ax.plot(res.index, res['cumulative_net_carbon_balance_gCO2_m2'], label='Cumulative net carbon balance')
    ax.set_ylabel('Cumulative carbon [gCO2 m$^{-2}$]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)

    shade_missing(axes, missing_intervals)

    for ax in axes:
        ax.set_xlim(day0, day1)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    s = summary.iloc[0]
    subtitle = prep.outdoor_mode_desc
    if prep.solar_scaling.enabled:
        subtitle = f"{subtitle} | Sun scale {prep.solar_scaling.factor:.4f} ({prep.solar_scaling.match_mode}:{prep.solar_scaling.matched_date})"
    fig.suptitle(
        f"{p.display_name} {s.date} | coverage {float(s.coverage_h):.2f} h | gross {float(s.gross_assimilation_gCO2_m2_day):.2f} | resp {float(s.respiration_loss_gCO2_m2_day):.2f} | net {float(s.net_carbon_balance_gCO2_m2_day):.2f}\n{subtitle}",
        fontsize=14,
    )
    axes[-1].set_xlabel('Time')

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


# -----------------------------
# CLI
# -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='Calculate one-day greenhouse fluxes and carbon plots from a daily Profinder log. This validation variant also supports indoor sensor offsets.')
    ap.add_argument('--house', required=True, choices=sorted(HOUSE_PRESETS), help='House preset to use.')
    ap.add_argument('--input-log', required=True, help='Daily greenhouse log (.log / .csv).')
    ap.add_argument('--borrow-outdoor-from', default=None, help='Optional reference daily log. If set, Sun.L / Out_T / Wind are borrowed from this log.')
    ap.add_argument('--outdoor-humidity-co2-source', default='legacy', choices=['legacy', 'outside_pf'], help='How to provide outdoor humidity and CO2. legacy uses fixed outdoor CO2 + dewpoint-offset RH. outside_pf reads AGLOG absolute humidity and CO2 from Profinder DBs.')
    ap.add_argument('--outside-pf-root', default=None, help='Root directory of outside_pf logs. Date subdirectories like YYYY-MM-DD are expected when --outdoor-humidity-co2-source=outside_pf.')
    ap.add_argument('--output-dir', default='one_day_flux_output', help='Output directory.')
    ap.add_argument('--radiation-scaling-csv', default=None, help='Optional daily radiation scaling CSV. If omitted, bundled scaling_daily_2025.csv is used when present.')
    ap.add_argument('--radiation-scaling-mode', default='auto', choices=['auto', 'exact', 'monthday', 'off'], help='How to match the radiation scaling row to the input-log date.')
    ap.add_argument('--indoor-temp-offset-c', type=float, default=0.0, help='Additive offset applied to indoor T_in before the inverse solver [C].')
    ap.add_argument('--indoor-rh-scale', type=float, default=1.0, help='Multiplicative scale applied to indoor RH_in before the inverse solver [-].')

    ap.add_argument('--ua-base-w-k', type=float, default=None, help='Override UA_base [W/K].')
    ap.add_argument('--u-ref-ms', type=float, default=None, help='Override reference wind for dynamic UA [m/s].')
    ap.add_argument('--tau-store-min', type=float, default=None, help='Override storage tau [min].')
    ap.add_argument('--c-eff-j-k', type=float, default=None, help='Override C_eff [J/K].')
    ap.add_argument('--beta-solar', type=float, default=None, help='Override beta_solar [-].')
    ap.add_argument('--r20-umol-m2-s', type=float, default=None, help='Override R20 [umol m-2 s-1].')
    ap.add_argument('--q10', type=float, default=None, help='Override respiration Q10 [-].')
    ap.add_argument('--fuel-rate-l-h-main', type=float, default=None, help='Override main burner fuel rate [L/h].')
    ap.add_argument('--fuel-rate-l-h-secondary', type=float, default=None, help='Override secondary burner fuel rate [L/h].')
    ap.add_argument('--heater-capacity-w', type=float, default=None, help='Override heater capacity [W].')
    ap.add_argument('--solar-heat-factor', type=float, default=None, help='Override solar heat factor [-].')
    ap.add_argument('--co2-dead-time-min', type=float, default=None, help='Override CO2 dead time [min].')
    ap.add_argument('--co2-tau-min', type=float, default=None, help='Override CO2 first-order tau [min].')
    ap.add_argument('--outdoor-co2-ppm', type=float, default=None, help='Override outdoor CO2 [ppm].')
    ap.add_argument('--dewpoint-offset-c', type=float, default=None, help='Override dewpoint offset [C].')
    ap.add_argument('--smooth-window', default=None, help='Override smoothing window. Example: 5min.')
    ap.add_argument('--daylight-threshold-w-m2', type=float, default=None, help='Override daylight threshold for gross assimilation [W/m2].')
    ap.add_argument('--vent-closed-threshold-pct', type=float, default=None, help='Override roof-vent threshold regarded as closed [%%].')
    return ap


def apply_overrides(preset: HousePreset, args: argparse.Namespace) -> HousePreset:
    kwargs = {}
    for field_name, arg_name in [
        ('ua_base_w_k', 'ua_base_w_k'),
        ('u_ref_ms', 'u_ref_ms'),
        ('tau_store_min', 'tau_store_min'),
        ('c_eff_j_k', 'c_eff_j_k'),
        ('beta_solar', 'beta_solar'),
        ('r20_umol_m2_s', 'r20_umol_m2_s'),
        ('q10', 'q10'),
        ('fuel_rate_l_h_main', 'fuel_rate_l_h_main'),
        ('fuel_rate_l_h_secondary', 'fuel_rate_l_h_secondary'),
        ('heater_capacity_w', 'heater_capacity_w'),
        ('solar_heat_factor', 'solar_heat_factor'),
        ('co2_dead_time_min', 'co2_dead_time_min'),
        ('co2_tau_min', 'co2_tau_min'),
        ('outdoor_co2_ppm', 'outdoor_co2_ppm'),
        ('dewpoint_offset_c', 'dewpoint_offset_c'),
        ('smooth_window', 'smooth_window'),
        ('daylight_threshold_w_m2', 'daylight_threshold_w_m2'),
        ('vent_closed_threshold_pct', 'vent_closed_threshold_pct'),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            kwargs[field_name] = value
    return replace(preset, **kwargs)


def main() -> None:
    args = build_argparser().parse_args()
    preset = apply_overrides(HOUSE_PRESETS[args.house], args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    default_scaling_csv = resolve_default_radiation_scaling_csv(Path(__file__).resolve().parent)
    radiation_scaling_csv = args.radiation_scaling_csv or (str(default_scaling_csv) if default_scaling_csv else None)
    default_outside_pf_root = resolve_default_outside_pf_root()
    outside_pf_root = args.outside_pf_root or (str(default_outside_pf_root) if default_outside_pf_root else '')
    outdoor_humidity_co2_source = OutdoorHumidityCo2Source(
        mode=args.outdoor_humidity_co2_source,
        outside_pf_root=outside_pf_root,
    )

    prep = prepare_day(
        args.input_log,
        preset=preset,
        borrow_outdoor_from=args.borrow_outdoor_from,
        outdoor_humidity_co2_source=outdoor_humidity_co2_source,
        radiation_scaling_csv=radiation_scaling_csv,
        radiation_scaling_mode=args.radiation_scaling_mode,
    )
    prep = apply_indoor_sensor_validation_offsets(
        prep,
        temp_offset_c=args.indoor_temp_offset_c,
        rh_scale=args.indoor_rh_scale,
    )
    res = apply_current_model(prep)
    res['T_in_sensor_raw_C'] = prep.t_in_raw
    res['RH_in_sensor_raw_pct'] = prep.rh_in_raw
    res['indoor_temp_offset_c'] = float(args.indoor_temp_offset_c)
    res['indoor_rh_scale'] = float(args.indoor_rh_scale)
    summary = build_summary(
        res,
        preset=preset,
        outdoor_mode_desc=prep.outdoor_mode_desc,
        outdoor_humidity_co2_desc=prep.outdoor_humidity_co2_desc,
        input_log=args.input_log,
        borrow_outdoor_from=args.borrow_outdoor_from,
        solar_scaling=prep.solar_scaling,
    )
    summary['indoor_temp_offset_c'] = float(args.indoor_temp_offset_c)
    summary['indoor_rh_scale'] = float(args.indoor_rh_scale)

    stem = Path(args.input_log).stem
    base_name = f'{stem}_{preset.house_key}'
    suffix = make_offset_suffix(args.indoor_temp_offset_c, args.indoor_rh_scale)
    if suffix:
        base_name = f'{base_name}_{suffix}'
    png_path = output_dir / f'{base_name}_flux_and_carbon.png'
    csv_path = output_dir / f'{base_name}_flux_timeseries.csv'
    summary_path = output_dir / f'{base_name}_summary.csv'
    notes_path = output_dir / f'{base_name}_notes.md'

    plot_one_day(prep, res, summary, png_path)
    res.to_csv(csv_path, encoding='utf-8-sig')
    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    notes = make_notes(prep, res, summary)
    notes.extend([
        '',
        '## Indoor sensor validation offsets',
        f'- indoor temp offset = {float(args.indoor_temp_offset_c):.3f} C',
        f'- indoor RH scale = {float(args.indoor_rh_scale):.4f}',
    ])
    write_notes(notes_path, notes)

    print(f'saved: {png_path}')
    print(f'saved: {csv_path}')
    print(f'saved: {summary_path}')
    print(f'saved: {notes_path}')


if __name__ == '__main__':
    main()
