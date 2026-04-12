#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import fields, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CODE_BUNDLE_DIR = Path(__file__).resolve().parent / "code_bundle" / "code_bundle"
if str(CODE_BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_BUNDLE_DIR))

import greenhouse_one_day_flux_and_carbon as gh


def parse_csv_list(text: str, *, cast=str) -> list:
    values = []
    for item in text.split(","):
        s = item.strip()
        if not s:
            continue
        values.append(cast(s))
    if not values:
        raise ValueError("Expected at least one value.")
    return values


def dataclass_field_names(cls: type) -> set[str]:
    return {f.name for f in fields(cls)}


def build_preset_for_validation() -> object:
    preset = gh.MARU_PRESET
    kwargs: dict[str, object] = {}
    names = dataclass_field_names(type(preset))
    if "heater_leak_gco2_m2_h_per_pct" in names:
        kwargs["heater_leak_gco2_m2_h_per_pct"] = 0.0
    if "storage_model" in names:
        kwargs["storage_model"] = "legacy_lpf"
    if "sensor_lag_model" in names:
        kwargs["sensor_lag_model"] = "off"
    return replace(preset, **kwargs) if kwargs else preset


def build_outdoor_source() -> object:
    kwargs: dict[str, object] = {
        "mode": "outside_pf",
        "outside_pf_root": str(gh.resolve_default_outside_pf_root()),
    }
    names = dataclass_field_names(gh.OutdoorHumidityCo2Source)
    if "outside_pf_co2_offset_ppm" in names:
        kwargs["outside_pf_co2_offset_ppm"] = 0.0
    if "outside_pf_ah_mode" in names:
        kwargs["outside_pf_ah_mode"] = "raw"
    if "outside_pf_dewpoint_bias_c" in names:
        kwargs["outside_pf_dewpoint_bias_c"] = 0.0
    return gh.OutdoorHumidityCo2Source(**kwargs)


def prep_series(prep: object, preferred: str, fallback: str) -> pd.Series:
    if hasattr(prep, preferred):
        return getattr(prep, preferred).astype(float)
    return getattr(prep, fallback).astype(float)


def dewpoint_c_from_vapor_pressure_pa(vapor_pressure_pa: pd.Series | np.ndarray | float) -> np.ndarray:
    if hasattr(gh, "dewpoint_c_from_vapor_pressure_pa"):
        return gh.dewpoint_c_from_vapor_pressure_pa(vapor_pressure_pa)
    e = np.maximum(np.asarray(vapor_pressure_pa, dtype=float), 1e-6)
    alpha = np.log(e / 611.2)
    return 243.5 * alpha / (17.67 - alpha)


def build_base_prep(date_yyyymmdd: str) -> gh.PreparedDay:
    input_log = (
        Path.home()
        / "Library/CloudStorage/GoogleDrive-soi.toi.chi@gmail.com/マイドライブ/greenhouse_log"
        / "maru"
        / f"{date_yyyymmdd}.log"
    )
    preset = build_preset_for_validation()
    humidity = build_outdoor_source()
    scaling_csv = gh.resolve_default_radiation_scaling_csv(CODE_BUNDLE_DIR)
    return gh.prepare_day(
        input_log,
        preset=preset,
        outdoor_humidity_co2_source=humidity,
        radiation_scaling_csv=str(scaling_csv) if scaling_csv else None,
        radiation_scaling_mode="monthday",
    )


def apply_indoor_offsets(
    prep: gh.PreparedDay,
    *,
    temp_offset_c: float,
    dewpoint_offset_c: float,
) -> gh.PreparedDay:
    idx = prep.out_wind.index
    t_in_base = prep_series(prep, "t_in_model", "t_in_raw")
    rh_in_base = prep_series(prep, "rh_in_model", "rh_in_raw").clip(lower=0.0, upper=110.0)

    vapor_pressure_pa = rh_in_base / 100.0 * pd.Series(
        gh.wgc.e_s_tetens(t_in_base.to_numpy(dtype=float)),
        index=idx,
    )
    dewpoint_c = dewpoint_c_from_vapor_pressure_pa(vapor_pressure_pa)

    t_in_adj = (t_in_base + float(temp_offset_c)).rename("T_in_model_bulk_C")
    vapor_pressure_adj_pa = pd.Series(
        gh.wgc.e_s_tetens(dewpoint_c + float(dewpoint_offset_c)),
        index=idx,
    )
    rh_in_adj_frac = np.clip(
        vapor_pressure_adj_pa.to_numpy(dtype=float) / gh.wgc.e_s_tetens(t_in_adj.to_numpy(dtype=float)),
        0.0,
        1.1,
    )
    rh_in_adj_pct = pd.Series(rh_in_adj_frac * 100.0, index=idx, name="RH_in_model_bulk_pct")
    omega_in_adj = pd.Series(
        gh.wgc.humidity_ratio(rh_in_adj_frac, t_in_adj.to_numpy(dtype=float), 101325.0),
        index=idx,
        name="omega_in_model_bulk_kgkg",
    )
    x_in_base = prep_series(prep, "x_in_model", "x_in_raw")

    model = pd.DataFrame(index=idx)
    model["T_in"] = t_in_adj
    model["RH_in"] = rh_in_adj_pct
    model["X_in"] = x_in_base
    model["T_out"] = prep.t_out.astype(float)
    model["RH_out"] = prep.rh_out.astype(float)
    model["X_out"] = prep.x_out.astype(float)
    model["H_in"] = prep.h_in_w.astype(float)
    model["S_CO2"] = prep.s_co2_mol_s.astype(float)
    model["W_inj"] = prep.w_inj_kg_s.astype(float)
    model["UA"] = prep.ua_eff_w_k.astype(float)
    model["day_smooth_mask"] = (prep.sun_out > prep.preset.daylight_threshold_w_m2).astype(int)
    model["deltaT_sky"] = prep.delta_t_sky_eff_k.astype(float)

    out_wind = gh.wgc.solve_whole_greenhouse_timeseries(
        model,
        V_m3=prep.preset.geometry.air_volume_m3,
        A_cov_m2=prep.preset.geometry.cover_area_m2,
        UA_col="UA",
        W_inj_col="W_inj",
        deltaT_sky_col="deltaT_sky",
        p_Pa=101325.0,
        h_c_W_m2_K=5.0,
        deltaT_sky_K=0.0,
        smooth_window=prep.preset.smooth_window,
        smooth_window_day=prep.preset.smooth_window_day,
        day_mask_col="day_smooth_mask",
        savgol_window_points=prep.preset.savgol_window_points,
        savgol_polyorder=prep.preset.savgol_polyorder,
        cond_only=False,
        dh_eps_J_kg=200.0,
        enforce_Q_nonneg=False,
    )

    return replace(
        prep,
        out_wind=out_wind,
        **{
            key: value
            for key, value in {
                "t_in_model": t_in_adj,
                "rh_in_model": rh_in_adj_pct,
                "omega_in_model": omega_in_adj,
                "x_in_model": x_in_base,
            }.items()
            if key in dataclass_field_names(type(prep))
        },
    )


def summarize_case(date_yyyymmdd: str, temp_offset_c: float, dewpoint_offset_c: float, res: pd.DataFrame) -> dict[str, float | str]:
    dt_h = res["dt_h"].astype(float)
    vent = res["ventilation_ach"].astype(float)
    net = res["net_photosynthesis_raw_gCO2_m2_h"].astype(float)
    hour = res["hour_float"].astype(float)
    day = res["day_mask"].astype(int) > 0
    morning = hour < 9.0
    morning_open = morning & (res["roof_vent_max_pct"].astype(float) > 20.0)
    return {
        "date_yyyymmdd": date_yyyymmdd,
        "temp_offset_c": float(temp_offset_c),
        "dewpoint_offset_c": float(dewpoint_offset_c),
        "min_vent_ach_24h": float(vent.min()),
        "morning_min_vent_ach": float(vent[morning].min()),
        "morning_neg_vent_h": float(dt_h[morning & (vent < 0.0)].sum()),
        "morning_open_neg_vent_h": float(dt_h[morning_open & (vent < 0.0)].sum()),
        "day_min_net_photo_gco2_m2_h": float(net[day].min()),
        "day_neg_net_photo_h": float(dt_h[day & (net < 0.0)].sum()),
        "day_neg_net_photo_integral": float((net[day].clip(upper=0.0) * dt_h[day]).sum()),
    }


def heatmap_matrix(summary: pd.DataFrame, value_col: str, temp_offsets: list[float], dewpoint_offsets: list[float]) -> np.ndarray:
    piv = summary.pivot(index="temp_offset_c", columns="dewpoint_offset_c", values=value_col)
    piv = piv.reindex(index=temp_offsets, columns=dewpoint_offsets)
    return piv.to_numpy(dtype=float)


def plot_date_heatmaps(
    summary: pd.DataFrame,
    date_yyyymmdd: str,
    temp_offsets: list[float],
    dewpoint_offsets: list[float],
    out_png: Path,
) -> None:
    date_label = pd.Timestamp(date_yyyymmdd).strftime("%Y-%m-%d")
    panels = [
        ("morning_min_vent_ach", "Morning min vent [ACH]"),
        ("morning_neg_vent_h", "Morning negative vent [h]"),
        ("day_min_net_photo_gco2_m2_h", "Day min net photo [gCO2 m-2 h-1]"),
        ("day_neg_net_photo_h", "Day negative net photo [h]"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes = axes.ravel()

    for ax, (col, title) in zip(axes, panels):
        matrix = heatmap_matrix(summary, col, temp_offsets, dewpoint_offsets)
        finite = matrix[np.isfinite(matrix)]
        if finite.size == 0:
            vmin, vmax = -1.0, 1.0
        else:
            vmax = float(np.nanmax(np.abs(finite)))
            if vmax <= 0.0:
                vmax = 1.0
            if "vent" in col or "photo" in col and "min" in col:
                vmin = -vmax
            else:
                vmin = 0.0
        im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(dewpoint_offsets)))
        ax.set_xticklabels([f"{v:+.0f}" for v in dewpoint_offsets])
        ax.set_yticks(np.arange(len(temp_offsets)))
        ax.set_yticklabels([f"{v:+.0f}" for v in temp_offsets])
        ax.set_xlabel("Indoor dewpoint offset [C]")
        ax.set_ylabel("Indoor T offset [C]")
        for i in range(len(temp_offsets)):
            for j in range(len(dewpoint_offsets)):
                val = matrix[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8, color="black")
        fig.colorbar(im, ax=ax, shrink=0.82)

    fig.suptitle(f"Maru {date_label} indoor bulk-air offset sweep", fontsize=14)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep indoor sensor offsets to mimic canopy-near sensor bias versus bulk air.")
    ap.add_argument("--dates", default="20260405,20260407,20260411")
    ap.add_argument("--temp-offsets", default="0,1,2,3")
    ap.add_argument("--dewpoint-offsets", default="0,-1,-2,-3,-4")
    ap.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "indoor_bulk_offset_sweep"))
    args = ap.parse_args()

    dates = parse_csv_list(args.dates, cast=str)
    temp_offsets = parse_csv_list(args.temp_offsets, cast=float)
    dewpoint_offsets = parse_csv_list(args.dewpoint_offsets, cast=float)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, float | str]] = []
    best_rows: list[dict[str, float | str]] = []

    for date in dates:
        prep_base = build_base_prep(date)
        date_rows: list[dict[str, float | str]] = []
        for temp_offset_c in temp_offsets:
            for dewpoint_offset_c in dewpoint_offsets:
                prep_mod = apply_indoor_offsets(
                    prep_base,
                    temp_offset_c=temp_offset_c,
                    dewpoint_offset_c=dewpoint_offset_c,
                )
                res = gh.apply_current_model(prep_mod)
                row = summarize_case(date, temp_offset_c, dewpoint_offset_c, res)
                all_rows.append(row)
                date_rows.append(row)

        date_summary = pd.DataFrame(date_rows).sort_values(["temp_offset_c", "dewpoint_offset_c"])
        date_dir = output_dir / date
        date_dir.mkdir(parents=True, exist_ok=True)
        csv_path = date_dir / f"{date}_indoor_bulk_offset_summary.csv"
        png_path = date_dir / f"{date}_indoor_bulk_offset_heatmaps.png"
        date_summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
        plot_date_heatmaps(date_summary, date, temp_offsets, dewpoint_offsets, png_path)

        score = (
            date_summary["morning_open_neg_vent_h"].astype(float)
            + date_summary["day_neg_net_photo_h"].astype(float)
            + 0.02 * np.abs(date_summary["morning_min_vent_ach"].astype(float))
            + 0.01 * np.abs(date_summary["day_min_net_photo_gco2_m2_h"].astype(float))
        )
        best = date_summary.iloc[int(score.argmin())].to_dict()
        best_rows.append(best)

    all_summary = pd.DataFrame(all_rows).sort_values(["date_yyyymmdd", "temp_offset_c", "dewpoint_offset_c"])
    all_summary_path = output_dir / "indoor_bulk_offset_summary_all.csv"
    best_path = output_dir / "indoor_bulk_offset_best_by_date.csv"
    all_summary.to_csv(all_summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(best_rows).sort_values("date_yyyymmdd").to_csv(best_path, index=False, encoding="utf-8-sig")
    print(output_dir)
    print(all_summary_path)
    print(best_path)


if __name__ == "__main__":
    main()
