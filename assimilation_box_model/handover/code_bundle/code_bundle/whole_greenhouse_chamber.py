# -*- coding: utf-8 -*-
"""
whole_greenhouse_chamber.py

「ハウス1棟＝同化箱（whole‑greenhouse chamber）」モデルで、
室内外の温湿度・CO2・入熱などから

  - 換気量 Q [m3/s]
  - 蒸散量 E [kg/s]
  - 正味同化（正味光合成）P [mol/s]
  - 結露/再蒸発 W_cond [kg/s]（被覆面温度を測らずに推定）

を時系列で推定するための最小実装。

前提:
- 室内空気は well-mixed
- 換気は単一流量（外気流入=同量流出）
- 収支は「ハウス内空気（乾き空気基準の湿り空気）」を制御体積
- d/dt は（平滑化した系列に対する）差分で近似する

注意:
- 元の引き継ぎメモの「水蒸気収支の線形化（R1）」は符号が不整合だったため、
  ここでは整合する形に修正して実装している。
  具体的には

    ρ_da V dω/dt = ρ_da Q(ω_out - ω_in) + E + W_inj - W_cond

  から

    -ρ_da Δω Q + E = ρ_da V dω/dt - W_inj + W_cond

  を採用している（W_cond>0 は結露＝水蒸気が減る、という符号規約と整合）。

依存:
- numpy
- pandas
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd


ArrayLike = Union[np.ndarray, float]


# -----------------------------
# 1) 基本（温湿度→派生量）
# -----------------------------
def e_s_tetens(T_C: ArrayLike) -> np.ndarray:
    """
    飽和水蒸気圧 e_s(T) [Pa]（Tetens近似）
    T_C: ℃
    """
    T = np.asarray(T_C, dtype=float)
    return 611.2 * np.exp(17.67 * T / (T + 243.5))


def vapor_pressure(RH_frac: ArrayLike, T_C: ArrayLike) -> np.ndarray:
    """水蒸気分圧 e [Pa] = RH * e_s(T)"""
    RH = np.asarray(RH_frac, dtype=float)
    return RH * e_s_tetens(T_C)


def humidity_ratio(RH_frac: ArrayLike, T_C: ArrayLike, p_Pa: ArrayLike = 101325.0) -> np.ndarray:
    """
    混合比 ω [kg_v / kg_da]
    ω = 0.62198 * e / (p - e)
    """
    e = vapor_pressure(RH_frac, T_C)
    p = np.asarray(p_Pa, dtype=float)
    denom = np.maximum(p - e, 1e-6)  # 안전: e→p で発散しないように
    return 0.62198 * e / denom


def omega_sat(T_C: ArrayLike, p_Pa: ArrayLike = 101325.0) -> np.ndarray:
    """飽和混合比 ω_sat(T)"""
    return humidity_ratio(1.0, T_C, p_Pa)


def d_omega_sat_dT(T_C: ArrayLike, p_Pa: ArrayLike = 101325.0, delta: float = 0.1) -> np.ndarray:
    """
    s = dω_sat/dT を数値微分で近似
    delta: [K]（℃でも同じ幅）
    """
    T = np.asarray(T_C, dtype=float)
    return (omega_sat(T + delta, p_Pa) - omega_sat(T - delta, p_Pa)) / (2.0 * delta)


def moist_air_enthalpy(T_C: ArrayLike, omega: ArrayLike, cp_da: float = 1005.0, cp_v: float = 1860.0) -> np.ndarray:
    """
    湿り空気の比エンタルピー h [J/kg_da]
    h = cp_da*T + ω*(2.501e6 + cp_v*T)
    """
    T = np.asarray(T_C, dtype=float)
    w = np.asarray(omega, dtype=float)
    return cp_da * T + w * (2.501e6 + cp_v * T)


def rho_dry_air(T_C: ArrayLike, RH_frac: ArrayLike, p_Pa: ArrayLike = 101325.0, R_da: float = 287.05) -> np.ndarray:
    """
    乾き空気密度 ρ_da [kg_da/m3]
    ρ_da = (p - e_in) / (R_da * T_K)
    """
    T_K = np.asarray(T_C, dtype=float) + 273.15
    e = vapor_pressure(RH_frac, T_C)
    p = np.asarray(p_Pa, dtype=float)
    return (p - e) / (R_da * T_K)


def mol_density_air(T_C: ArrayLike, p_Pa: ArrayLike = 101325.0, R: float = 8.314462618) -> np.ndarray:
    """
    空気のモル密度 ρ_mol [mol/m3]
    ρ_mol = p / (R * T_K)
    """
    T_K = np.asarray(T_C, dtype=float) + 273.15
    p = np.asarray(p_Pa, dtype=float)
    return p / (R * T_K)


def L_star(T_C: ArrayLike, cp_v: float = 1860.0) -> np.ndarray:
    """L*(T) = 2.501e6 + cp_v*T [J/kg]"""
    T = np.asarray(T_C, dtype=float)
    return 2.501e6 + cp_v * T


# -----------------------------
# 2) 結露モデル（被覆面温度なし）
# -----------------------------
def condensation_flow(
    T_in_C: ArrayLike,
    RH_in_frac: ArrayLike,
    T_out_C: ArrayLike,
    p_Pa: ArrayLike,
    UA_W_K: ArrayLike,
    A_cov_m2: float,
    *,
    h_c_W_m2_K: ArrayLike = 5.0,
    deltaT_sky_K: ArrayLike = 0.0,
    L_v: float = 2.45e6,
    cp_da: float = 1005.0,
    cond_only: bool = False,
    s_delta: float = 0.1,
) -> np.ndarray:
    """
    被覆面温度を測らない近似で W_cond [kg/s] を計算（符号付き）。

    W_cond > 0 : 結露（空気中の水蒸気が減る）
    W_cond < 0 : 再蒸発（空気中の水蒸気が増える）

    cond_only=True のときは、再蒸発（負値）を 0 にクリップする。
    """
    T_in = np.asarray(T_in_C, dtype=float)
    RH_in = np.asarray(RH_in_frac, dtype=float)
    T_out = np.asarray(T_out_C, dtype=float)
    p = np.asarray(p_Pa, dtype=float)
    UA = np.asarray(UA_W_K, dtype=float)
    h_c = np.asarray(h_c_W_m2_K, dtype=float)
    dT_sky = np.asarray(deltaT_sky_K, dtype=float)

    T_out_eff = T_out - dT_sky
    dT_eff = T_in - T_out_eff  # [K]

    w_in = humidity_ratio(RH_in, T_in, p)
    w_sat_in = omega_sat(T_in, p)
    d_w_sat = w_in - w_sat_in  # ω_in - ω_sat(T_in)

    s = d_omega_sat_dT(T_in, p, delta=s_delta)
    U = UA / A_cov_m2  # [W/m2/K]
    gamma = L_v / cp_da  # [K]

    j = (h_c / cp_da) * (d_w_sat + s * (U / h_c) * dT_eff) / (1.0 + gamma * s)  # [kg/m2/s]

    if cond_only:
        j = np.maximum(j, 0.0)

    return A_cov_m2 * j  # [kg/s]


# -----------------------------
# 3) 数値微分・平滑化ユーティリティ
# -----------------------------
def _ensure_datetime_index(df: pd.DataFrame, time_col: Optional[str] = None) -> pd.DataFrame:
    d = df.copy()
    if time_col is not None:
        d[time_col] = pd.to_datetime(d[time_col])
        d = d.set_index(time_col)
    if not isinstance(d.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex, or provide time_col.")
    return d.sort_index()


def _rolling_mean(series: pd.Series, window) -> pd.Series:
    if window is None or window == 0 or window == 1:
        return series.astype(float)
    if isinstance(window, (str, pd.Timedelta)):
        return series.astype(float).rolling(window=window, center=True, min_periods=1).mean()
    return series.astype(float).rolling(window=int(window), center=True, min_periods=1).mean()


def _time_seconds(index: pd.DatetimeIndex) -> np.ndarray:
    if len(index) == 0:
        return np.array([], dtype=float)
    base = index[0]
    return np.asarray((index - base).total_seconds(), dtype=float)


def _gradient(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    if y.size == 0:
        return np.array([], dtype=float)
    if y.size == 1:
        return np.zeros(1, dtype=float)
    uniq_t, inverse = np.unique(t, return_inverse=True)
    if uniq_t.size <= 1:
        return np.zeros_like(y, dtype=float)
    y_sum = np.bincount(inverse, weights=y)
    y_count = np.bincount(inverse)
    y_uniq = y_sum / np.maximum(y_count, 1)
    grad_uniq = np.gradient(y_uniq, uniq_t)
    return np.asarray(grad_uniq[inverse], dtype=float)


# -----------------------------
# 4) 1時刻分の解（微分は外から与える）
# -----------------------------
def solve_whole_greenhouse_step(
    *,
    T_in_C: float,
    RH_in: float,
    X_in: float,
    T_out_C: float,
    RH_out: float,
    X_out: float,
    dT_dt: float,
    domega_dt: float,
    dX_dt: float,
    H_in_W: float,
    S_CO2_mol_s: float,
    W_inj_kg_s: float = 0.0,
    V_m3: float = 1.0,
    A_cov_m2: float = 1.0,
    UA_W_K: float = 0.0,
    h_c_W_m2_K: float = 5.0,
    deltaT_sky_K: float = 0.0,
    p_Pa: float = 101325.0,
    cp_da: float = 1005.0,
    cp_v: float = 1860.0,
    L_v: float = 2.45e6,
    R: float = 8.314462618,
    R_da: float = 287.05,
    rh_unit: str = "percent",
    co2_unit: str = "ppm",
    cond_only: bool = False,
    dh_eps_J_kg: float = 200.0,
    enforce_Q_nonneg: bool = False,
) -> dict:
    """
    1時刻分の (Q,E,P) を計算。

    dT_dt, domega_dt, dX_dt は、できれば
      - 平滑化（1〜5分移動平均など）
      - 中央差分
    で作ってから渡すこと。
    """
    # 単位変換
    RH_in_f = RH_in / 100.0 if rh_unit.lower() in ["percent", "%", "pct"] else RH_in
    RH_out_f = RH_out / 100.0 if rh_unit.lower() in ["percent", "%", "pct"] else RH_out
    RH_in_f = float(np.clip(RH_in_f, 0.0, 1.2))
    RH_out_f = float(np.clip(RH_out_f, 0.0, 1.2))

    X_in_m = X_in * 1e-6 if co2_unit.lower() in ["ppm", "ppmv"] else X_in
    X_out_m = X_out * 1e-6 if co2_unit.lower() in ["ppm", "ppmv"] else X_out

    # 派生量
    omega_in = float(humidity_ratio(RH_in_f, T_in_C, p_Pa))
    omega_out = float(humidity_ratio(RH_out_f, T_out_C, p_Pa))
    h_in = float(moist_air_enthalpy(T_in_C, omega_in, cp_da, cp_v))
    h_out = float(moist_air_enthalpy(T_out_C, omega_out, cp_da, cp_v))
    rho_da_val = float(rho_dry_air(T_in_C, RH_in_f, p_Pa, R_da))
    rho_mol_val = float(mol_density_air(T_in_C, p_Pa, R))
    delta_h = h_in - h_out
    delta_w = omega_in - omega_out
    Ls = float(L_star(T_in_C, cp_v))

    # 結露
    W_cond = float(
        condensation_flow(
            T_in_C,
            RH_in_f,
            T_out_C,
            p_Pa,
            UA_W_K,
            A_cov_m2,
            h_c_W_m2_K=h_c_W_m2_K,
            deltaT_sky_K=deltaT_sky_K,
            L_v=L_v,
            cp_da=cp_da,
            cond_only=cond_only,
        )
    )

    # RHS（符号は修正済み）
    R1 = rho_da_val * V_m3 * domega_dt - W_inj_kg_s + W_cond  # [kg/s]
    T_out_eff = T_out_C - deltaT_sky_K
    R2 = (
        H_in_W
        - UA_W_K * (T_in_C - T_out_eff)
        - rho_da_val * V_m3 * (cp_da + cp_v * omega_in) * dT_dt
        - Ls * (W_inj_kg_s - W_cond)
    )  # [W]

    if abs(delta_h) < dh_eps_J_kg:
        return {
            "omega_in": omega_in,
            "omega_out": omega_out,
            "h_in": h_in,
            "h_out": h_out,
            "rho_da": rho_da_val,
            "rho_mol": rho_mol_val,
            "W_cond": W_cond,
            "R1": R1,
            "R2": R2,
            "delta_h": delta_h,
            "Q_m3_s": np.nan,
            "E_kg_s": np.nan,
            "P_mol_s": np.nan,
            "flag_dh_small": True,
        }

    Q = (R2 - Ls * R1) / (rho_da_val * delta_h)
    if enforce_Q_nonneg:
        Q = max(0.0, Q)

    E = R1 + rho_da_val * delta_w * Q
    P = rho_mol_val * Q * (X_out_m - X_in_m) + S_CO2_mol_s - rho_mol_val * V_m3 * dX_dt

    return {
        "omega_in": omega_in,
        "omega_out": omega_out,
        "h_in": h_in,
        "h_out": h_out,
        "rho_da": rho_da_val,
        "rho_mol": rho_mol_val,
        "W_cond": W_cond,
        "R1": R1,
        "R2": R2,
        "delta_h": delta_h,
        "Q_m3_s": Q,
        "E_kg_s": E,
        "P_mol_s": P,
        "flag_dh_small": False,
    }


# -----------------------------
# 5) 時系列の一括推定（微分も内部で計算）
# -----------------------------
def solve_whole_greenhouse_timeseries(
    df: pd.DataFrame,
    *,
    time_col: Optional[str] = None,
    # 列名（必要に応じて変更）
    T_in_col: str = "T_in",
    RH_in_col: str = "RH_in",
    X_in_col: str = "X_in",
    T_out_col: str = "T_out",
    RH_out_col: str = "RH_out",
    X_out_col: str = "X_out",
    H_in_col: str = "H_in",
    S_CO2_col: str = "S_CO2",
    W_inj_col: Optional[str] = None,
    p_col: Optional[str] = None,
    UA_col: Optional[str] = None,
    h_c_col: Optional[str] = None,
    deltaT_sky_col: Optional[str] = None,
    # パラメータ（列が無い場合のスカラー値）
    V_m3: float = 1.0,
    A_cov_m2: float = 1.0,
    UA_W_K: float = 0.0,
    h_c_W_m2_K: float = 5.0,
    deltaT_sky_K: float = 0.0,
    p_Pa: float = 101325.0,
    # 物性
    cp_da: float = 1005.0,
    cp_v: float = 1860.0,
    L_v: float = 2.45e6,
    R: float = 8.314462618,
    R_da: float = 287.05,
    # 入力の単位
    rh_unit: str = "percent",  # 'percent' or 'fraction'
    co2_unit: str = "ppm",  # 'ppm' or 'mol/mol'
    # 平滑化（例: '5min' あるいは 5 (サンプル数)）
    smooth_window: Union[str, int, None] = "5min",
    # 結露モデル
    cond_only: bool = False,
    # 数値・QC
    dh_eps_J_kg: float = 200.0,
    enforce_Q_nonneg: bool = False,
) -> pd.DataFrame:
    """
    df から Q,E,P を時系列で推定し、派生量も含めた DataFrame を返す。

    必須列（デフォルト名）:
      - 室内: T_in [℃], RH_in [%], X_in [ppm]
      - 外気: T_out [℃], RH_out [%], X_out [ppm]
      - 入力: H_in [W], S_CO2 [mol/s]
    任意列:
      - W_inj [kg/s], p [Pa], UA [W/K], h_c [W/m2/K], deltaT_sky [K]

    返り値:
      - Q_m3_s, E_kg_s, P_mol_s, W_cond など
      - dT_dt, domega_dt, dX_dt（差分）
      - flag_dh_small（Δh が小さく Q が不安定）
    """
    d = _ensure_datetime_index(df, time_col)

    # RH
    RH_in = d[RH_in_col].astype(float)
    RH_out = d[RH_out_col].astype(float)
    if rh_unit.lower() in ["percent", "%", "pct"]:
        RH_in = RH_in / 100.0
        RH_out = RH_out / 100.0
    RH_in = RH_in.clip(lower=0.0, upper=1.2)  # センサー誤差で 1 を少し超える場合に対応
    RH_out = RH_out.clip(lower=0.0, upper=1.2)

    # CO2
    X_in = d[X_in_col].astype(float)
    X_out = d[X_out_col].astype(float)
    if co2_unit.lower() in ["ppm", "ppmv"]:
        X_in = X_in * 1e-6
        X_out = X_out * 1e-6

    # パラメータ（時系列 or スカラー）
    p = d[p_col].astype(float) if (p_col is not None and p_col in d.columns) else pd.Series(p_Pa, index=d.index, dtype=float)
    UA = d[UA_col].astype(float) if (UA_col is not None and UA_col in d.columns) else pd.Series(UA_W_K, index=d.index, dtype=float)
    h_c = d[h_c_col].astype(float) if (h_c_col is not None and h_c_col in d.columns) else pd.Series(h_c_W_m2_K, index=d.index, dtype=float)
    dT_sky = (
        d[deltaT_sky_col].astype(float)
        if (deltaT_sky_col is not None and deltaT_sky_col in d.columns)
        else pd.Series(deltaT_sky_K, index=d.index, dtype=float)
    )
    W_inj = d[W_inj_col].astype(float) if (W_inj_col is not None and W_inj_col in d.columns) else pd.Series(0.0, index=d.index, dtype=float)

    H_in = d[H_in_col].astype(float)
    S_CO2 = d[S_CO2_col].astype(float)

    # 平滑化（微分ノイズ低減）
    T_in_s = _rolling_mean(d[T_in_col], smooth_window)
    T_out_s = _rolling_mean(d[T_out_col], smooth_window)
    RH_in_s = _rolling_mean(RH_in, smooth_window)
    RH_out_s = _rolling_mean(RH_out, smooth_window)
    X_in_s = _rolling_mean(X_in, smooth_window)

    # 派生量
    w_in = pd.Series(humidity_ratio(RH_in_s.values, T_in_s.values, p.values), index=d.index)
    w_out = pd.Series(humidity_ratio(RH_out_s.values, T_out_s.values, p.values), index=d.index)
    h_in = pd.Series(moist_air_enthalpy(T_in_s.values, w_in.values, cp_da, cp_v), index=d.index)
    h_out = pd.Series(moist_air_enthalpy(T_out_s.values, w_out.values, cp_da, cp_v), index=d.index)
    rho_da = pd.Series(rho_dry_air(T_in_s.values, RH_in_s.values, p.values, R_da), index=d.index)
    rho_mol = pd.Series(mol_density_air(T_in_s.values, p.values, R), index=d.index)

    # 微分（中心差分）
    t_sec = _time_seconds(d.index)
    dT_dt = pd.Series(_gradient(T_in_s.values, t_sec), index=d.index)
    dw_dt = pd.Series(_gradient(w_in.values, t_sec), index=d.index)
    dX_dt = pd.Series(_gradient(X_in_s.values, t_sec), index=d.index)

    # 結露
    W_cond = pd.Series(
        condensation_flow(
            T_in_s.values,
            RH_in_s.values,
            T_out_s.values,
            p.values,
            UA.values,
            A_cov_m2,
            h_c_W_m2_K=h_c.values,
            deltaT_sky_K=dT_sky.values,
            L_v=L_v,
            cp_da=cp_da,
            cond_only=cond_only,
        ),
        index=d.index,
    )

    # コア計算
    delta_w = w_in - w_out
    delta_h = h_in - h_out
    Ls = pd.Series(L_star(T_in_s.values, cp_v=cp_v), index=d.index)

    # R1（修正済み）
    R1 = rho_da * V_m3 * dw_dt - W_inj + W_cond  # [kg/s]
    R2 = (
        H_in
        - UA * (T_in_s - (T_out_s - dT_sky))  # UA*(T_in - T_out_eff)
        - rho_da * V_m3 * (cp_da + cp_v * w_in) * dT_dt
        - Ls * (W_inj - W_cond)
    )  # [W]

    # Δhが小さいと Q が不安定
    dh_small = delta_h.abs() < dh_eps_J_kg

    Q = (R2 - Ls * R1) / (rho_da * delta_h)  # [m3/s]
    Q = Q.where(~dh_small, np.nan)
    if enforce_Q_nonneg:
        Q = Q.clip(lower=0.0)

    E = R1 + rho_da * delta_w * Q  # [kg/s]
    P = rho_mol * Q * (X_out - X_in_s) + S_CO2 - rho_mol * V_m3 * dX_dt  # [mol/s]

    out = pd.DataFrame(index=d.index)
    out["T_in_C"] = T_in_s
    out["RH_in_frac"] = RH_in_s
    out["X_in_molmol"] = X_in_s
    out["T_out_C"] = T_out_s
    out["RH_out_frac"] = RH_out_s
    out["X_out_molmol"] = X_out

    out["omega_in"] = w_in
    out["omega_out"] = w_out
    out["h_in"] = h_in
    out["h_out"] = h_out
    out["rho_da"] = rho_da
    out["rho_mol"] = rho_mol

    out["dT_dt"] = dT_dt
    out["domega_dt"] = dw_dt
    out["dX_dt"] = dX_dt

    out["UA"] = UA
    out["h_c"] = h_c
    out["deltaT_sky"] = dT_sky
    out["W_inj"] = W_inj
    out["W_cond"] = W_cond

    out["R1"] = R1
    out["R2"] = R2
    out["delta_h"] = delta_h

    out["Q_m3_s"] = Q
    out["E_kg_s"] = E
    out["P_mol_s"] = P

    out["flag_dh_small"] = dh_small.astype(int)
    out["flag_Q_negative"] = ((Q < 0).astype(int)).where(Q.notna(), np.nan)
    out["flag_E_negative"] = ((E < 0).astype(int)).where(E.notna(), np.nan)
    out["flag_P_negative"] = ((P < 0).astype(int)).where(P.notna(), np.nan)

    return out
