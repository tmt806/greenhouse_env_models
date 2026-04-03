from __future__ import annotations

import argparse
import io
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from greenhouse_assimilation_model import GreenhouseParams, StateInputs, solve_q_e_p

RHO_FUEL_KG_PER_L = 0.80
LHV_J_PER_KG = 43.1e6
CO2_KG_PER_KG_FUEL = 3.106
H2O_KG_PER_KG_FUEL = 1.376
CO2_MOL_PER_L_FUEL = RHO_FUEL_KG_PER_L * CO2_KG_PER_KG_FUEL * 1000.0 / 44.01
H2O_KG_PER_L_FUEL = RHO_FUEL_KG_PER_L * H2O_KG_PER_KG_FUEL
LHV_J_PER_L_FUEL = RHO_FUEL_KG_PER_L * LHV_J_PER_KG
GCO2_PER_UMOL = 44.01e-6
GCO2_H_PER_UMOL_S = GCO2_PER_UMOL * 3600.0


@dataclass(frozen=True)
class Geometry:
    floor_area_m2: float
    air_volume_m3: float
    cover_area_m2: float


@dataclass(frozen=True)
class HousePreset:
    house_key: str
    display_name: str
    geometry: Geometry
    ua_base_w_k: float
    r20_umol_m2_s: float
    q10: float
    fuel_rate_l_h_main: float = 3.79
    fuel_rate_l_h_secondary: float = 3.79
    heater_capacity_w: float = 174000.0
    solar_heat_factor: float = 0.60
    co2_dead_time_min: float = 4.0
    co2_tau_min: float = 8.0
    outdoor_co2_ppm: float = 400.0
    screen_solar_transmittance_closed: float = 0.35
    screen_ua_closed_ratio: float = 0.70
    screen_sky_coupling_closed_ratio: float = 0.30
    main_cmd_candidates: tuple[str, ...] = ("CO2",)
    secondary_cmd_candidates: tuple[str, ...] = ()


TRIANGLE_GEOM = Geometry(
    floor_area_m2=45.0 * 51.0,
    air_volume_m3=45.0 * 51.0 * (4.2 + 0.5 * (5.6 - 4.2)),
    cover_area_m2=(
        2.0 * 51.0 * 4.2
        + 2.0 * 45.0 * 4.2
        + 2.0 * (0.5 * 45.0 * (5.6 - 4.2))
        + 2.0 * math.hypot(45.0 / 2.0, 5.6 - 4.2) * 51.0
    ),
)
MARU_GEOM = Geometry(
    floor_area_m2=54.0 * 51.0,
    air_volume_m3=54.0 * 51.0 * 5.5,
    cover_area_m2=(54.0 * 51.0 + 2.0 * 54.0 * 5.5 + 2.0 * 51.0 * 5.5),
)
MARU_U_TRI = 13280.0 / TRIANGLE_GEOM.cover_area_m2
HOUSE_PRESETS = {
    "sankaku": HousePreset(
        house_key="sankaku",
        display_name="Sankaku",
        geometry=TRIANGLE_GEOM,
        ua_base_w_k=13280.0,
        r20_umol_m2_s=2.3345,
        q10=1.8869,
        main_cmd_candidates=("CO2",),
        secondary_cmd_candidates=(),
    ),
    "maru": HousePreset(
        house_key="maru",
        display_name="Maru",
        geometry=MARU_GEOM,
        ua_base_w_k=MARU_U_TRI * MARU_GEOM.cover_area_m2,
        r20_umol_m2_s=3.2622,
        q10=2.5686,
        solar_heat_factor=0.285936,
        main_cmd_candidates=("CO2",),
        secondary_cmd_candidates=("CV_H2",),
    ),
}


def read_text_table(path: Path) -> pd.DataFrame:
    encodings = ("utf-8-sig", "utf-8", "cp932", "shift_jis", "latin1")
    last_err: Exception | None = None
    for enc in encodings:
        try:
            with path.open("r", encoding=enc) as f:
                txt = f.read()
            df = pd.read_csv(io.StringIO(txt), skipinitialspace=True)
            df.columns = [str(c).strip() for c in df.columns]
            return df.loc[:, ~df.columns.str.contains(r"^Unnamed", na=False)]
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Could not read {path}: {last_err}")


def parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["DATE"].astype(str).str.strip() + " " + d["TIME"].astype(str).str.strip())
    d = d.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    if not d.index.has_duplicates:
        return d
    agg: dict[str, pd.Series] = {}
    for col in d.columns:
        s_num = pd.to_numeric(d[col], errors="coerce")
        if s_num.notna().any():
            agg[col] = s_num.groupby(d.index).mean()
        else:
            agg[col] = d[col].groupby(d.index).last()
    out = pd.DataFrame(agg).sort_index()
    out.index.name = "timestamp"
    return out


def as_float_series(df: pd.DataFrame, candidates: tuple[str, ...], default: float = 0.0) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").astype(float)
    return pd.Series(default, index=df.index, dtype=float)


def elapsed_seconds(index: pd.DatetimeIndex) -> np.ndarray:
    if len(index) == 0:
        return np.array([], dtype=float)
    return np.asarray((index - index[0]).total_seconds(), dtype=float)


def safe_gradient(values: np.ndarray, index: pd.DatetimeIndex) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros(len(values), dtype=float)
    t = elapsed_seconds(index)
    uniq_t, inverse = np.unique(t, return_inverse=True)
    if len(uniq_t) <= 1:
        return np.zeros(len(values), dtype=float)
    y_sum = np.bincount(inverse, weights=values)
    y_count = np.bincount(inverse)
    grad_uniq = np.gradient(y_sum / np.maximum(y_count, 1), uniq_t)
    return np.asarray(grad_uniq[inverse], dtype=float)


def rolling_mean(series: pd.Series, window: str = "5min") -> pd.Series:
    return series.astype(float).rolling(window=window, center=True, min_periods=1).mean()


def fopdt_fraction(command_pct: pd.Series, dead_time_min: float, tau_min: float) -> pd.Series:
    u = command_pct.astype(float).clip(lower=0.0, upper=100.0) / 100.0
    if len(u) == 0:
        return u
    t = elapsed_seconds(u.index)
    td = max(dead_time_min * 60.0, 0.0)
    tau = max(tau_min * 60.0, 1e-9)
    u_del = np.interp(t - td, t, u.to_numpy(), left=float(u.iloc[0]), right=float(u.iloc[-1]))
    y = np.zeros_like(u_del)
    y[0] = u_del[0]
    for i in range(1, len(y)):
        dt = max(float(t[i] - t[i - 1]), 0.0)
        alpha = 1.0 - math.exp(-dt / tau)
        y[i] = y[i - 1] + alpha * (u_del[i] - y[i - 1])
    return pd.Series(y, index=u.index)


def h_out_mc_adams(u_ms: pd.Series) -> np.ndarray:
    u = np.maximum(u_ms.to_numpy(dtype=float), 0.0)
    return 5.8 + 4.1 * u


def calc_dynamic_ua_series(wind_ms: pd.Series, preset: HousePreset) -> pd.Series:
    u_base = preset.ua_base_w_k / preset.geometry.cover_area_m2
    h_ref = 5.8 + 4.1 * 0.917
    r_fixed = 1.0 / u_base - 1.0 / h_ref
    h_now = h_out_mc_adams(wind_ms)
    u_eff = 1.0 / (r_fixed + 1.0 / h_now)
    ua_eff = u_eff * preset.geometry.cover_area_m2
    return pd.Series(ua_eff, index=wind_ms.index)


def estimate_outdoor_rh_from_dewpoint(t_out_c: pd.Series, dewpoint_offset_c: float = 2.5) -> pd.Series:
    dawn = t_out_c.between_time("03:00", "07:00")
    tmin_early = float(dawn.min()) if len(dawn) and not dawn.isna().all() else float(t_out_c.min())
    dewpoint_c = tmin_early - dewpoint_offset_c
    e_dew = 611.2 * np.exp(17.67 * dewpoint_c / (dewpoint_c + 243.5))
    e_sat = 611.2 * np.exp(17.67 * t_out_c / (t_out_c + 243.5))
    return (100.0 * np.clip(e_dew / e_sat, 0.0, 1.0)).rename("RH_out")


def integrate_rate(rate: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    if len(index) == 0:
        return pd.Series(dtype=float)
    dt_h = np.diff(elapsed_seconds(index), append=np.nan)
    valid = dt_h[np.isfinite(dt_h) & (dt_h > 0)]
    median_h = (float(np.median(valid)) / 3600.0) if len(valid) else 1.0 / 60.0
    dt_h = np.where(np.isfinite(dt_h) & (dt_h > 0), dt_h / 3600.0, median_h)
    return (rate.fillna(0.0) * dt_h).cumsum()


def prepare_and_solve(log_path: Path, house_key: str) -> pd.DataFrame:
    preset = HOUSE_PRESETS[house_key]
    raw = parse_timestamp(read_text_table(log_path))
    idx = raw.index

    t_in = rolling_mean(as_float_series(raw, ("Temp", "T_in")))
    rh_in_pct = rolling_mean(as_float_series(raw, ("Humi", "RH_in")))
    rh_in = rh_in_pct / 100.0
    x_in_ppm = rolling_mean(as_float_series(raw, ("Co2", "CO2", "X_in")))
    x_in = x_in_ppm * 1e-6
    t_out = rolling_mean(as_float_series(raw, ("Out_T", "T_out")))
    rh_out_pct = estimate_outdoor_rh_from_dewpoint(t_out)
    rh_out = rh_out_pct / 100.0
    wind = as_float_series(raw, ("Wind",), default=0.0).ffill().bfill().fillna(0.0)
    sun = as_float_series(raw, ("Sun.L", "Sun", "I.Sun"), default=0.0).clip(lower=0.0)
    screen_pct = pd.concat([
        as_float_series(raw, ("CV_C1",), 0.0),
        as_float_series(raw, ("CV_C2",), 0.0),
    ], axis=1).max(axis=1).clip(lower=0.0, upper=100.0)
    heater_pct = as_float_series(raw, ("CV_H1",), 0.0).clip(lower=0.0, upper=100.0)
    cmd1 = as_float_series(raw, preset.main_cmd_candidates, 0.0).clip(lower=0.0, upper=100.0)
    cmd2 = as_float_series(raw, preset.secondary_cmd_candidates, 0.0).clip(lower=0.0, upper=100.0)

    eff1 = fopdt_fraction(cmd1, preset.co2_dead_time_min, preset.co2_tau_min)
    eff2 = fopdt_fraction(cmd2, preset.co2_dead_time_min, preset.co2_tau_min)
    fuel_l_s = preset.fuel_rate_l_h_main / 3600.0 * eff1 + preset.fuel_rate_l_h_secondary / 3600.0 * eff2
    s_co2 = fuel_l_s * CO2_MOL_PER_L_FUEL
    w_inj = fuel_l_s * H2O_KG_PER_L_FUEL
    h_burn = fuel_l_s * LHV_J_PER_L_FUEL
    h_heat = preset.heater_capacity_w * (heater_pct / 100.0)
    h_solar = preset.solar_heat_factor * preset.geometry.floor_area_m2 * sun
    ua = calc_dynamic_ua_series(wind, preset)

    dT_dt = safe_gradient(t_in.to_numpy(dtype=float), idx)

    def mixing_ratio(T_c: pd.Series, rh_frac: pd.Series) -> np.ndarray:
        t = T_c.to_numpy(dtype=float)
        rh = rh_frac.to_numpy(dtype=float)
        e = rh * (611.2 * np.exp(17.67 * t / (t + 243.5)))
        return 0.62198 * e / (101325.0 - e)

    domega_dt = safe_gradient(mixing_ratio(t_in, rh_in), idx)
    dX_dt = safe_gradient(x_in.to_numpy(dtype=float), idx)

    rows: list[dict[str, float | str]] = []
    for i, ts in enumerate(idx):
        params = GreenhouseParams(
            V=preset.geometry.air_volume_m3,
            A_cov=preset.geometry.cover_area_m2,
            UA=float(ua.iloc[i]),
            h_c=5.0,
            delta_T_sky=2.0,
            p=101325.0,
            clip_condensation_to_positive=False,
            screen_solar_transmittance_closed=preset.screen_solar_transmittance_closed,
            screen_ua_closed_ratio=preset.screen_ua_closed_ratio,
            screen_sky_coupling_closed_ratio=preset.screen_sky_coupling_closed_ratio,
        )
        state = StateInputs(
            T_in=float(t_in.iloc[i]),
            RH_in=float(rh_in.iloc[i]),
            X_in=float(x_in.iloc[i]),
            T_out=float(t_out.iloc[i]),
            RH_out=float(rh_out.iloc[i]),
            X_out=preset.outdoor_co2_ppm * 1e-6,
            dT_in_dt=float(dT_dt[i]),
            d_omega_in_dt=float(domega_dt[i]),
            dX_in_dt=float(dX_dt[i]),
            H_air=float(h_burn.iloc[i] + h_heat.iloc[i]),
            H_solar=float(h_solar.iloc[i]),
            S_CO2=float(s_co2.iloc[i]),
            W_inj=float(w_inj.iloc[i]),
            screen_closure=float(screen_pct.iloc[i] / 100.0),
        )
        out = solve_q_e_p(params, state)
        respiration_umol = preset.r20_umol_m2_s * (preset.q10 ** ((t_in.iloc[i] - 20.0) / 10.0))
        p_umol = out["P_mol_s"] / preset.geometry.floor_area_m2 * 1e6
        rows.append({
            "timestamp": ts,
            "hour": ts.hour + ts.minute / 60.0 + ts.second / 3600.0,
            "house": preset.display_name,
            "gross_assimilation_gCO2_m2_h": max(p_umol + respiration_umol, 0.0) * GCO2_H_PER_UMOL_S,
            "net_photosynthesis_raw_gCO2_m2_h": p_umol * GCO2_H_PER_UMOL_S,
            "transpiration_g_m2_h": out["E_kg_s"] / preset.geometry.floor_area_m2 * 3600.0 * 1000.0,
            "condensation_g_m2_h_cover": out["W_cond_kg_s"] / preset.geometry.cover_area_m2 * 3600.0 * 1000.0,
            "ventilation_ach": out["Q_m3_s"] / preset.geometry.air_volume_m3 * 3600.0,
            "CO2_in_ppm": float(x_in_ppm.iloc[i]),
            "Sun_W_m2": float(sun.iloc[i]),
            "screen_pct": float(screen_pct.iloc[i]),
        })

    res = pd.DataFrame(rows).set_index("timestamp")
    res["cumulative_net_photosynthesis_gCO2_m2"] = integrate_rate(res["net_photosynthesis_raw_gCO2_m2_h"], res.index)
    return res


def combined_ylim(series_list: list[pd.Series], q_low: float = 0.02, q_high: float = 0.98) -> tuple[float, float]:
    arrays = []
    for s in series_list:
        vals = s.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if len(vals):
            arrays.append(vals)
    if not arrays:
        return (-1.0, 1.0)
    vals = np.concatenate(arrays)
    lo = float(np.quantile(vals, q_low))
    hi = float(np.quantile(vals, q_high))
    if math.isclose(lo, hi):
        pad = 1.0 if hi == 0 else abs(hi) * 0.2
        return lo - pad, hi + pad
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def plot_compare(sankaku: pd.DataFrame, maru: pd.DataFrame, out_png: Path) -> None:
    metrics = [
        ("gross_assimilation_gCO2_m2_h", "Gross assimilation [gCO2 m-2 h-1]"),
        ("net_photosynthesis_raw_gCO2_m2_h", "Net photosynthesis [gCO2 m-2 h-1]"),
        ("transpiration_g_m2_h", "Transpiration [g m-2 h-1]"),
        ("condensation_g_m2_h_cover", "Condensation / re-evap [g m-2 h-1]"),
        ("ventilation_ach", "Ventilation [ACH]"),
    ]
    fig, axes = plt.subplots(len(metrics), 2, figsize=(16, 18), sharex="col", constrained_layout=True)
    houses = [("Sankaku", sankaku), ("Maru", maru)]

    for row, (col, ylabel) in enumerate(metrics):
        ylim = combined_ylim([sankaku[col], maru[col]])
        for col_idx, (title, df) in enumerate(houses):
            ax = axes[row, col_idx]
            ax.plot(df["hour"], df[col], color="#124c5c", linewidth=1.3)
            ax.set_xlim(0.0, 24.0)
            ax.set_ylim(*ylim)
            ax.set_ylabel(ylabel)
            ax.set_xticks(np.arange(0, 25, 2))
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(title)
            if row == len(metrics) - 1:
                ax.set_xlabel("Hour")
            if col in {"net_photosynthesis_raw_gCO2_m2_h", "condensation_g_m2_h_cover"}:
                ax.axhline(0.0, color="0.4", linewidth=0.8)

    fig.suptitle("2026-04-03 One-day Flux Comparison", fontsize=15)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sankaku-log", required=True)
    ap.add_argument("--maru-log", required=True)
    ap.add_argument("--output-dir", default="out")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sankaku = prepare_and_solve(Path(args.sankaku_log), "sankaku")
    maru = prepare_and_solve(Path(args.maru_log), "maru")

    sankaku.to_csv(output_dir / "20260403_sankaku_flux.csv", encoding="utf-8-sig")
    maru.to_csv(output_dir / "20260403_maru_flux.csv", encoding="utf-8-sig")
    plot_compare(sankaku, maru, output_dir / "20260403_flux_compare.png")

    print(output_dir / "20260403_sankaku_flux.csv")
    print(output_dir / "20260403_maru_flux.csv")
    print(output_dir / "20260403_flux_compare.png")


if __name__ == "__main__":
    main()
