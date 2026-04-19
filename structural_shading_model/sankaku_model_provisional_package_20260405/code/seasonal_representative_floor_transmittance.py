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


def envelope_taus_with_extrapolated_k(base, xf, yf, sx, sy, sz, house, geom):
    tol = 1e-9
    house_depth = house["depth_m"]
    house_width = house["cultivation_width_m"]
    unit_width = geom["unit_width"]
    gutter_z = geom["gutter_ztop"]
    ridge_z = geom["ridge_z"]

    n = xf.size
    t_best = np.full(n, np.inf, dtype=float)
    direct_tau = np.full(n, house["roof_par_total_normal"] * house["roof_direct_fraction"], dtype=float)
    beam_diffuse_tau = np.full(n, house["roof_par_total_normal"] * house["roof_diffuse_fraction"], dtype=float)
    total_tau = np.full(n, house["roof_par_total_normal"], dtype=float)

    if sx < -tol:
        t = (0.0 - xf) / sx
        y_int = yf + t * sy
        z_int = t * sz
        roof_z = base.span_roof_height(y_int, unit_width=unit_width, gutter_z=gutter_z, ridge_z=ridge_z)
        cond = (t > 0.0) & (y_int >= 0.0) & (y_int <= house_width) & (z_int >= 0.0) & (z_int <= roof_z)
        upd = cond & (t < t_best)
        t_best[upd] = t[upd]
        direct_tau[upd] = house["south_wall_tau_direct"]
        beam_diffuse_tau[upd] = 0.0
        total_tau[upd] = house["south_wall_tau_direct"]

    if sy < -tol:
        t = (0.0 - yf) / sy
        x_int = xf + t * sx
        z_int = t * sz
        cond = (t > 0.0) & (x_int >= 0.0) & (x_int <= house_depth) & (z_int >= 0.0) & (z_int <= gutter_z)
        upd = cond & (t < t_best)
        t_best[upd] = t[upd]
        direct_tau[upd] = house["roof_par_total_normal"]
        beam_diffuse_tau[upd] = 0.0
        total_tau[upd] = house["roof_par_total_normal"]
    if sy > tol:
        t = (house_width - yf) / sy
        x_int = xf + t * sx
        z_int = t * sz
        cond = (t > 0.0) & (x_int >= 0.0) & (x_int <= house_depth) & (z_int >= 0.0) & (z_int <= gutter_z)
        upd = cond & (t < t_best)
        t_best[upd] = t[upd]
        direct_tau[upd] = house["roof_par_total_normal"]
        beam_diffuse_tau[upd] = 0.0
        total_tau[upd] = house["roof_par_total_normal"]

    m = (ridge_z - gutter_z) / (unit_width / 2.0)
    n_units = int(round(house_width / unit_width))
    norm = math.sqrt(1.0 + m * m)
    n_west = np.array([0.0, -m / norm, 1.0 / norm])
    n_east = np.array([0.0, m / norm, 1.0 / norm])
    svec = np.array([sx, sy, sz])
    inc_west = math.degrees(math.acos(min(1.0, max(0.0, float(np.dot(n_west, svec))))))
    inc_east = math.degrees(math.acos(min(1.0, max(0.0, float(np.dot(n_east, svec))))))
    total_west = house["roof_par_total_normal"] * extrapolated_k_theta(base, np.array([inc_west]))[0]
    total_east = house["roof_par_total_normal"] * extrapolated_k_theta(base, np.array([inc_east]))[0]

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
            total_tau[upd] = total_west
            direct_tau[upd] = total_west * house["roof_direct_fraction"]
            beam_diffuse_tau[upd] = total_west * house["roof_diffuse_fraction"]
        denom_e = sz + m * sy
        if abs(denom_e) > tol:
            t = (gutter_z + m * (y1 - yf)) / denom_e
            y_int = yf + t * sy
            x_int = xf + t * sx
            cond = (t > 0.0) & (x_int >= 0.0) & (x_int <= house_depth) & (y_int >= yr - 1e-9) & (y_int <= y1 + 1e-9)
            upd = cond & (t < t_best)
            t_best[upd] = t[upd]
            total_tau[upd] = total_east
            direct_tau[upd] = total_east * house["roof_direct_fraction"]
            beam_diffuse_tau[upd] = total_east * house["roof_diffuse_fraction"]

    return direct_tau, beam_diffuse_tau, total_tau


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


def seasonal_transmittance_for_zone(base, day_of_year, zone_mask, X, Y, xf, yf, house, geom):
    outdoor_integral = 0.0
    indoor_zone_integral = 0.0
    daylight_minutes = 0
    minute_rows = []
    zone_map_sum = np.zeros_like(X, dtype=float)

    for hour in range(4, 21):
        for minute in range(60):
            local_time = hour + minute / 60.0
            beta_deg, az_deg = base.solar_position_simple(
                house["latitude_deg"], day_of_year, local_time, lon_deg=house["longitude_deg"]
            )
            if beta_deg <= 0.0:
                continue
            sx, sy, sz = base.sun_vector(beta_deg, az_deg)
            if sz <= 0.0:
                continue

            direct_tau, beam_diffuse_tau, total_tau = envelope_taus_with_extrapolated_k(
                base, xf, yf, sx, sy, sz, house, geom
            )
            struct_lit = structure_lit_fraction(base, xf, yf, sx, sy, sz, house, geom)
            floor_transmittance = (direct_tau * struct_lit + beam_diffuse_tau).reshape(X.shape)

            outdoor = math.sin(math.radians(beta_deg))
            outdoor_integral += outdoor
            zone_mean = float(np.mean(floor_transmittance[zone_mask]))
            indoor_zone_integral += zone_mean * outdoor
            zone_map_sum += floor_transmittance * outdoor
            daylight_minutes += 1
            minute_rows.append(
                {
                    "day_of_year": day_of_year,
                    "time_h": local_time,
                    "solar_elev_deg": beta_deg,
                    "solar_az_deg": az_deg,
                    "outdoor_relative_direct": outdoor,
                    "zone_mean_transmittance": zone_mean,
                }
            )

    daily_map = zone_map_sum / outdoor_integral if outdoor_integral > 0.0 else zone_map_sum
    seasonal_mean = indoor_zone_integral / outdoor_integral if outdoor_integral > 0.0 else 0.0
    return seasonal_mean, daylight_minutes, daily_map, minute_rows


def run(output_dir="output_floor_seasonal_zone"):
    base = load_base_module()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    house = {
        "latitude_deg": 35.0725,
        "longitude_deg": 136.73388888888888,
        "depth_m": 51.0,
        "cultivation_width_m": 45.0,
        "roof_par_total_normal": 0.903,
        "roof_direct_fraction": 0.55,
        "roof_diffuse_fraction": 0.45,
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

    zone_mask = (X >= 21.0) & (X <= 24.0) & (Y >= 15.0) & (Y <= 22.5)
    seasons = [
        ("winter_solstice", 355),
        ("spring_equinox", 80),
        ("summer_solstice", 172),
    ]

    summary_rows = []
    minute_tables = []
    daily_maps = {}

    for label, day_of_year in seasons:
        mean_trans, daylight_minutes, daily_map, minute_rows = seasonal_transmittance_for_zone(
            base, day_of_year, zone_mask, X, Y, xf, yf, house, geom
        )
        summary_rows.append(
            {
                "season": label,
                "day_of_year": day_of_year,
                "daylight_minutes": daylight_minutes,
                "zone_mean_daily_transmittance": mean_trans,
            }
        )
        minute_tables.extend(minute_rows)
        daily_maps[label] = daily_map

    pd.DataFrame(summary_rows).to_csv(out / "seasonal_zone_transmittance_summary.csv", index=False)
    pd.DataFrame(minute_tables).to_csv(out / "seasonal_zone_transmittance_minutely.csv", index=False)

    dist_rows = []
    for label, _ in seasons:
        dist_rows.append(
            pd.DataFrame(
                {
                    "season": label,
                    "x_m": X.ravel(),
                    "y_m": Y.ravel(),
                    "daily_floor_transmittance": daily_maps[label].ravel(),
                    "in_zone": zone_mask.ravel().astype(int),
                }
            )
        )
    pd.concat(dist_rows, ignore_index=True).to_csv(out / "seasonal_floor_transmittance_distribution.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    for ax, (label, _) in zip(axes, seasons):
        arr = daily_maps[label]
        im = ax.imshow(
            arr,
            origin="lower",
            extent=[0, house["depth_m"], 0, house["cultivation_width_m"]],
            aspect="auto",
            cmap="viridis",
            vmin=min(np.min(m) for m in daily_maps.values()),
            vmax=max(np.max(m) for m in daily_maps.values()),
        )
        for gy in geom["gutter_y_positions"]:
            ax.axhline(gy, color="white", lw=0.5, alpha=0.35)
        ax.add_patch(plt.Rectangle((21.0, 15.0), 3.0, 7.5, fill=False, ec="cyan", lw=1.4))
        ax.set_title(label.replace("_", " "))
        ax.set_xlabel("x [m] (north positive)")
        ax.set_ylabel("y [m] (east positive)")
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88)
    cb.set_label("daily floor transmittance vs outdoor")
    fig.savefig(out / "seasonal_floor_transmittance_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
