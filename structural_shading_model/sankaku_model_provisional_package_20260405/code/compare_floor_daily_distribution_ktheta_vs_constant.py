import importlib.util
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_base_module():
    script_path = Path(__file__).with_name(
        "generate_topview_photosynthesis_snapshots_0321_triangularRoof5sun_angleDep_blankMargins_southwall50.py"
    )
    spec = importlib.util.spec_from_file_location("triangular_roof_model", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_geometry(house):
    geom = {
        "unit_width": 7.5,
        "gutter_width": 0.60,
        "gutter_depth": 0.10,
        "gutter_ztop": 4.20,
        "curtain1_width": 0.70,
        "curtain1_thickness": 0.05,
        "curtain1_ztop": 3.90,
        "curtain2_width": 0.70,
        "curtain2_thickness": 0.05,
        "curtain2_ztop": 3.60,
        "frame_member_width_y": 0.10,
        "frame_member_thickness_x": 0.05,
        "pillar_pitch_x": 3.0,
        "rafter_width_y": 0.033,
        "rafter_thickness_x": 0.033,
        "rafter_pitch_x": 0.50,
        "purlin_width_y": 0.060,
        "purlin_depth_z": 0.030,
        "purlin_rows_per_slope": 4,
        "ridge_z": 4.20 + 7.5 / 2.0 * 0.5,
    }
    geom["gutter_y_positions"] = np.arange(geom["unit_width"], house["cultivation_width_m"] - 1e-9, geom["unit_width"])
    geom["support_y_positions"] = np.arange(0.0, house["cultivation_width_m"] + 1e-9, geom["unit_width"])
    geom["pillar_x_positions"] = np.arange(0.0, house["depth_m"] + 1e-9, geom["pillar_pitch_x"])
    geom["rafter_x_positions"] = np.arange(0.0, house["depth_m"] + 1e-9, geom["rafter_pitch_x"])
    return geom


def extrapolated_k_theta(base, angles_deg):
    angles_deg = np.asarray(angles_deg, dtype=float)
    k = np.empty_like(angles_deg)
    mask_core = angles_deg <= 60.0
    k[mask_core] = base.angle_correction_etfe_gr_series(angles_deg[mask_core])
    mask_ext = ~mask_core
    if np.any(mask_ext):
        theta = angles_deg[mask_ext]
        k60 = float(base.angle_correction_etfe_gr_series(np.array([60.0]))[0])
        sec60 = 1.0 / np.cos(np.radians(60.0))
        sec85 = 1.0 / np.cos(np.radians(85.0))
        target85 = 0.20
        alpha = np.log(k60 / target85) / (sec85 - sec60)
        sec_theta = 1.0 / np.cos(np.radians(theta))
        k[mask_ext] = k60 * np.exp(-alpha * (sec_theta - sec60))
    return k


def roof_total_tau_extrapolated(base, incidence_deg, par_total_normal):
    return par_total_normal * extrapolated_k_theta(base, incidence_deg)


def roof_total_tau_constant(incidence_deg, par_total_normal):
    incidence_deg = np.asarray(incidence_deg, dtype=float)
    return np.full_like(incidence_deg, par_total_normal, dtype=float)


def total_entry_tau(base, xf, yf, sx, sy, sz, house, geom, roof_mode):
    if roof_mode == "extrapolated":
        roof_tau_fn = lambda inc: roof_total_tau_extrapolated(base, inc, house["roof_par_total_normal"])
    elif roof_mode == "constant":
        roof_tau_fn = lambda inc: roof_total_tau_constant(inc, house["roof_par_total_normal"])
    else:
        raise ValueError(f"Unknown roof_mode: {roof_mode}")

    tol = 1e-9
    house_depth = house["depth_m"]
    house_width = house["cultivation_width_m"]
    unit_width = geom["unit_width"]
    gutter_z = geom["gutter_ztop"]
    ridge_z = geom["ridge_z"]
    south_tau = house["south_wall_tau_direct"]
    east_west_tau = house["roof_par_total_normal"]

    n = xf.size
    t_best = np.full(n, np.inf, dtype=float)
    total_tau = np.full(n, house["roof_par_total_normal"], dtype=float)

    if sx < -tol:
        t = (0.0 - xf) / sx
        y_int = yf + t * sy
        z_int = t * sz
        roof_z = base.span_roof_height(y_int, unit_width=unit_width, gutter_z=gutter_z, ridge_z=ridge_z)
        cond = (t > 0.0) & (y_int >= 0.0) & (y_int <= house_width) & (z_int >= 0.0) & (z_int <= roof_z)
        upd = cond & (t < t_best)
        t_best[upd] = t[upd]
        total_tau[upd] = south_tau

    if sy < -tol:
        t = (0.0 - yf) / sy
        x_int = xf + t * sx
        z_int = t * sz
        cond = (t > 0.0) & (x_int >= 0.0) & (x_int <= house_depth) & (z_int >= 0.0) & (z_int <= gutter_z)
        upd = cond & (t < t_best)
        t_best[upd] = t[upd]
        total_tau[upd] = east_west_tau
    if sy > tol:
        t = (house_width - yf) / sy
        x_int = xf + t * sx
        z_int = t * sz
        cond = (t > 0.0) & (x_int >= 0.0) & (x_int <= house_depth) & (z_int >= 0.0) & (z_int <= gutter_z)
        upd = cond & (t < t_best)
        t_best[upd] = t[upd]
        total_tau[upd] = east_west_tau

    m = (ridge_z - gutter_z) / (unit_width / 2.0)
    n_units = int(round(house_width / unit_width))
    norm = math.sqrt(1.0 + m * m)
    n_west = np.array([0.0, -m / norm, 1.0 / norm])
    n_east = np.array([0.0, m / norm, 1.0 / norm])
    svec = np.array([sx, sy, sz])
    inc_west = math.degrees(math.acos(min(1.0, max(0.0, float(np.dot(n_west, svec))))))
    inc_east = math.degrees(math.acos(min(1.0, max(0.0, float(np.dot(n_east, svec))))))
    roof_tau_west = float(roof_tau_fn(np.array([inc_west]))[0])
    roof_tau_east = float(roof_tau_fn(np.array([inc_east]))[0])

    for j in range(n_units):
        y0 = j * unit_width
        yr = y0 + unit_width / 2.0
        y1 = y0 + unit_width
        denom_w = sz - m * sy
        if abs(denom_w) > tol:
            t = (gutter_z + m * (yf - y0)) / denom_w
            y_int = yf + t * sy
            x_int = xf + t * sx
            cond = (t > 0.0) & (x_int >= 0.0) & (x_int <= house_depth) & (y_int >= y0 - 1e-9) & (y_int <= yr + 1e-9)
            upd = cond & (t < t_best)
            t_best[upd] = t[upd]
            total_tau[upd] = roof_tau_west
        denom_e = sz + m * sy
        if abs(denom_e) > tol:
            t = (gutter_z + m * (y1 - yf)) / denom_e
            y_int = yf + t * sy
            x_int = xf + t * sx
            cond = (t > 0.0) & (x_int >= 0.0) & (x_int <= house_depth) & (y_int >= yr - 1e-9) & (y_int <= y1 + 1e-9)
            upd = cond & (t < t_best)
            t_best[upd] = t[upd]
            total_tau[upd] = roof_tau_east

    return total_tau


def structure_lit_fraction(base, xf, yf, sx, sy, sz, house, geom):
    z_floor = 0.0
    shadow = np.zeros_like(xf, dtype=bool)
    shadow |= base.shadow_by_linear_objects(yf, z_floor, sy, sz, geom["gutter_y_positions"], geom["gutter_width"], geom["gutter_ztop"], geom["gutter_ztop"] - geom["gutter_depth"])
    shadow |= base.shadow_by_linear_objects(yf, z_floor, sy, sz, geom["gutter_y_positions"], geom["curtain1_width"], geom["curtain1_ztop"], geom["curtain1_ztop"] - geom["curtain1_thickness"])
    shadow |= base.shadow_by_linear_objects(yf, z_floor, sy, sz, geom["gutter_y_positions"], geom["curtain2_width"], geom["curtain2_ztop"], geom["curtain2_ztop"] - geom["curtain2_thickness"])
    shadow |= base.shadow_by_transverse_frame_members(xf, yf, z_floor, sx, sy, sz, geom["pillar_x_positions"], geom["frame_member_thickness_x"], geom["support_y_positions"], geom["frame_member_width_y"], geom["frame_member_width_y"], house["cultivation_width_m"], geom["unit_width"], geom["gutter_ztop"], geom["ridge_z"])
    shadow |= base.shadow_by_transverse_frame_members(xf, yf, z_floor, sx, sy, sz, geom["rafter_x_positions"], geom["rafter_thickness_x"], geom["support_y_positions"], geom["rafter_width_y"], geom["rafter_width_y"], house["cultivation_width_m"], geom["unit_width"], geom["gutter_ztop"], geom["ridge_z"])
    shadow |= base.shadow_by_roof_longitudinal_members(yf, z_floor, sy, sz, geom["purlin_y_positions"], geom["purlin_width_y"], geom["purlin_depth_z"], geom["unit_width"], geom["gutter_ztop"], geom["ridge_z"])
    shadow |= base.shadow_by_pillars(xf, yf, z_floor, sx, sy, sz, geom["pillar_x_positions"], geom["gutter_y_positions"], geom["frame_member_thickness_x"], geom["frame_member_width_y"], geom["gutter_ztop"])
    return (~shadow).astype(float)


def run(output_dir="output_floor_daily_direct_compare_ktheta"):
    base = load_base_module()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    house = {
        "latitude_deg": 35.0725,
        "longitude_deg": 136.73388888888888,
        "day_of_year": 80,
        "cultivation_width_m": 45.0,
        "depth_m": 51.0,
        "roof_par_total_normal": 0.903,
        "south_wall_tau_direct": 0.50,
    }
    geom = build_geometry(house)
    geom["purlin_y_positions"] = base.roof_member_y_positions(
        house["cultivation_width_m"], geom["unit_width"], geom["purlin_rows_per_slope"]
    )

    nx, ny = 120, 100
    x = np.linspace(0.0, house["depth_m"], nx)
    y = np.linspace(0.0, house["cultivation_width_m"], ny)
    X, Y = np.meshgrid(x, y)
    xf = X.ravel()
    yf = Y.ravel()

    cumulative = {
        "constant": np.zeros_like(X, dtype=float),
        "extrapolated": np.zeros_like(X, dtype=float),
    }
    records = []
    outdoor_integral = 0.0

    for hour in range(4, 21):
        for minute in range(60):
            local_time = hour + minute / 60.0
            beta_deg, az_deg = base.solar_position_simple(
                house["latitude_deg"], house["day_of_year"], local_time, lon_deg=house["longitude_deg"]
            )
            if beta_deg <= 0.0:
                continue
            sx, sy, sz = base.sun_vector(beta_deg, az_deg)
            if sz <= 0.0:
                continue
            intensity = math.sin(math.radians(beta_deg))
            outdoor_integral += intensity
            struct_lit = structure_lit_fraction(base, xf, yf, sx, sy, sz, house, geom)

            row = {
                "time_h": local_time,
                "solar_elev_deg": beta_deg,
                "solar_az_deg": az_deg,
                "outdoor_relative_direct": intensity,
            }
            for mode in ("constant", "extrapolated"):
                tau = total_entry_tau(base, xf, yf, sx, sy, sz, house, geom, mode)
                floor_map = (tau * struct_lit).reshape(X.shape)
                cumulative[mode] += floor_map * intensity
                row[f"mean_tau_{mode}"] = float(np.mean(floor_map))
            records.append(row)

    normalized = {mode: arr / outdoor_integral for mode, arr in cumulative.items()}
    diff = normalized["extrapolated"] - normalized["constant"]
    ratio = np.divide(
        normalized["extrapolated"],
        normalized["constant"],
        out=np.full_like(diff, np.nan),
        where=normalized["constant"] > 1e-9,
    )

    summary = pd.DataFrame(
        [
            {
                "day_of_year": house["day_of_year"],
                "mean_transmittance_constant": float(np.mean(normalized["constant"])),
                "mean_transmittance_extrapolated": float(np.mean(normalized["extrapolated"])),
                "mean_difference_extrapolated_minus_constant": float(np.mean(diff)),
                "mean_ratio_extrapolated_over_constant": float(np.nanmean(ratio)),
                "min_difference": float(np.nanmin(diff)),
                "max_difference": float(np.nanmax(diff)),
            }
        ]
    )
    summary.to_csv(out / "summary_ktheta_vs_constant_0321.csv", index=False)
    pd.DataFrame(records).to_csv(out / "minutely_ktheta_vs_constant_0321.csv", index=False)
    pd.DataFrame(
        {
            "x_m": X.ravel(),
            "y_m": Y.ravel(),
            "transmittance_constant": normalized["constant"].ravel(),
            "transmittance_extrapolated": normalized["extrapolated"].ravel(),
            "difference_extrapolated_minus_constant": diff.ravel(),
            "ratio_extrapolated_over_constant": ratio.ravel(),
        }
    ).to_csv(out / "distribution_ktheta_vs_constant_0321.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    panels = [
        (normalized["constant"], "K = 1 constant", "viridis"),
        (normalized["extrapolated"], "Extrapolated K(theta)", "viridis"),
        (diff, "Difference (extrapolated - constant)", "coolwarm"),
        (ratio, "Ratio (extrapolated / constant)", "magma"),
    ]
    for ax, (arr, title, cmap) in zip(axes.ravel(), panels):
        im = ax.imshow(
            arr,
            origin="lower",
            extent=[0, house["depth_m"], 0, house["cultivation_width_m"]],
            aspect="auto",
            cmap=cmap,
        )
        for gy in geom["gutter_y_positions"]:
            ax.axhline(gy, color="white", lw=0.5, alpha=0.35)
        ax.set_title(title)
        ax.set_xlabel("x [m] (north positive)")
        ax.set_ylabel("y [m] (east positive)")
        cb = fig.colorbar(im, ax=ax, shrink=0.88)
        cb.set_label("transmittance")
    fig.savefig(out / "distribution_ktheta_vs_constant_0321.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
