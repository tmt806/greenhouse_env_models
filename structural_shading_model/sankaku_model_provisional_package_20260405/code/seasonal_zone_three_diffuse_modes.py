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


def envelope_taus_with_entry(base, xf, yf, sx, sy, sz, house, geom):
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
    entry_x = np.full(n, np.nan, dtype=float)
    entry_y = np.full(n, np.nan, dtype=float)
    entry_z = np.full(n, np.nan, dtype=float)
    boundary = np.full(n, "roof", dtype=object)

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
        entry_x[upd] = 0.0
        entry_y[upd] = y_int[upd]
        entry_z[upd] = z_int[upd]
        boundary[upd] = "south_wall"

    if sy < -tol:
        t = (0.0 - yf) / sy
        x_int = xf + t * sx
        z_int = t * sz
        cond = (t > 0.0) & (x_int >= 0.0) & (x_int <= house_depth) & (z_int >= 0.0) & (z_int <= gutter_z)
        upd = cond & (t < t_best)
        t_best[upd] = t[upd]
        direct_tau[upd] = house["roof_par_total_normal"]
        beam_diffuse_tau[upd] = 0.0
        entry_x[upd] = x_int[upd]
        entry_y[upd] = 0.0
        entry_z[upd] = z_int[upd]
        boundary[upd] = "west_wall"

    if sy > tol:
        t = (house_width - yf) / sy
        x_int = xf + t * sx
        z_int = t * sz
        cond = (t > 0.0) & (x_int >= 0.0) & (x_int <= house_depth) & (z_int >= 0.0) & (z_int <= gutter_z)
        upd = cond & (t < t_best)
        t_best[upd] = t[upd]
        direct_tau[upd] = house["roof_par_total_normal"]
        beam_diffuse_tau[upd] = 0.0
        entry_x[upd] = x_int[upd]
        entry_y[upd] = house_width
        entry_z[upd] = z_int[upd]
        boundary[upd] = "east_wall"

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
            z_int = t * sz
            cond = (t > 0.0) & (x_int >= 0.0) & (x_int <= house_depth) & (y_int >= y0 - 1e-9) & (y_int <= yr + 1e-9)
            upd = cond & (t < t_best)
            t_best[upd] = t[upd]
            direct_tau[upd] = total_west * house["roof_direct_fraction"]
            beam_diffuse_tau[upd] = total_west * house["roof_diffuse_fraction"]
            entry_x[upd] = x_int[upd]
            entry_y[upd] = y_int[upd]
            entry_z[upd] = z_int[upd]
            boundary[upd] = "roof_west"

        denom_e = sz + m * sy
        if abs(denom_e) > tol:
            t = (gutter_z + m * (y1 - yf)) / denom_e
            y_int = yf + t * sy
            x_int = xf + t * sx
            z_int = t * sz
            cond = (t > 0.0) & (x_int >= 0.0) & (x_int <= house_depth) & (y_int >= yr - 1e-9) & (y_int <= y1 + 1e-9)
            upd = cond & (t < t_best)
            t_best[upd] = t[upd]
            direct_tau[upd] = total_east * house["roof_direct_fraction"]
            beam_diffuse_tau[upd] = total_east * house["roof_diffuse_fraction"]
            entry_x[upd] = x_int[upd]
            entry_y[upd] = y_int[upd]
            entry_z[upd] = z_int[upd]
            boundary[upd] = "roof_east"

    return direct_tau, beam_diffuse_tau, entry_x, entry_y, entry_z, boundary


def make_fft_kernel(nx, ny, dx, dy, sigma_m):
    x = (np.arange(nx) - nx // 2) * dx
    y = (np.arange(ny) - ny // 2) * dy
    Xk, Yk = np.meshgrid(x, y)
    kernel = np.exp(-(Xk * Xk + Yk * Yk) / (2.0 * sigma_m * sigma_m))
    kernel /= np.sum(kernel)
    kernel = np.fft.ifftshift(kernel)
    return np.fft.rfftn(kernel)


def fft_convolve_same(arr, kernel_fft):
    return np.fft.irfftn(np.fft.rfftn(arr) * kernel_fft, s=arr.shape)


def spread_diffuse_map(source_map, kernel_fft, loss_factor):
    spread = fft_convolve_same(source_map, kernel_fft)
    spread = np.clip(spread, 0.0, None)
    total_in = np.sum(source_map)
    total_out = np.sum(spread)
    if total_out > 0.0:
        spread *= total_in / total_out
    return spread * loss_factor


def seasonal_metrics(base, day_of_year, X, Y, xf, yf, zone_mask, house, geom, kernel_fft):
    outdoor_integral = 0.0
    daylight_minutes = 0
    minute_rows = []
    maps = {
        "direct_only": np.zeros_like(X, dtype=float),
        "uniform_diffuse": np.zeros_like(X, dtype=float),
        "spread_diffuse": np.zeros_like(X, dtype=float),
    }
    zone_weighted = {key: 0.0 for key in maps}

    nx = X.shape[1]
    ny = X.shape[0]
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    x_coords = X[0, :]
    y_coords = Y[:, 0]
    diffuse_loss_factor = 0.88

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

            direct_tau, beam_diffuse_tau, entry_x, entry_y, entry_z, boundary = envelope_taus_with_entry(
                base, xf, yf, sx, sy, sz, house, geom
            )
            struct_lit = structure_lit_fraction(base, xf, yf, sx, sy, sz, house, geom)
            direct_map = (direct_tau * struct_lit).reshape(X.shape)
            uniform_diffuse_map = beam_diffuse_tau.reshape(X.shape)

            source_map = np.zeros_like(X, dtype=float)
            roof_mask = np.char.startswith(boundary.astype(str), "roof")
            if np.any(roof_mask):
                x_idx = np.clip(np.rint((entry_x[roof_mask] - x_coords[0]) / dx).astype(int), 0, nx - 1)
                y_idx = np.clip(np.rint((entry_y[roof_mask] - y_coords[0]) / dy).astype(int), 0, ny - 1)
                np.add.at(source_map, (y_idx, x_idx), beam_diffuse_tau[roof_mask])
            wall_mask = ~roof_mask
            if np.any(wall_mask):
                wall_map = beam_diffuse_tau[wall_mask]
                x_idx = np.clip(np.rint((xf[wall_mask] - x_coords[0]) / dx).astype(int), 0, nx - 1)
                y_idx = np.clip(np.rint((yf[wall_mask] - y_coords[0]) / dy).astype(int), 0, ny - 1)
                np.add.at(source_map, (y_idx, x_idx), wall_map)

            spread_map = spread_diffuse_map(source_map, kernel_fft, diffuse_loss_factor)

            total_maps = {
                "direct_only": direct_map,
                "uniform_diffuse": direct_map + uniform_diffuse_map,
                "spread_diffuse": direct_map + spread_map,
            }

            outdoor = math.sin(math.radians(beta_deg))
            outdoor_integral += outdoor
            daylight_minutes += 1

            row = {
                "day_of_year": day_of_year,
                "time_h": local_time,
                "solar_elev_deg": beta_deg,
                "solar_az_deg": az_deg,
                "outdoor_relative_direct": outdoor,
            }
            for key, arr in total_maps.items():
                maps[key] += arr * outdoor
                zone_mean = float(np.mean(arr[zone_mask]))
                zone_weighted[key] += zone_mean * outdoor
                row[f"zone_mean_{key}"] = zone_mean
            minute_rows.append(row)

    zone_means = {key: zone_weighted[key] / outdoor_integral for key in zone_weighted}
    daily_maps = {key: maps[key] / outdoor_integral for key in maps}
    return zone_means, daily_maps, minute_rows, daylight_minutes


def run(output_dir="output_floor_seasonal_three_modes"):
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

    nx, ny = 100, 84
    x = np.linspace(0.0, house["depth_m"], nx)
    y = np.linspace(0.0, house["cultivation_width_m"], ny)
    X, Y = np.meshgrid(x, y)
    xf = X.ravel()
    yf = Y.ravel()
    zone_mask = (X >= 21.0) & (X <= 24.0) & (Y >= 15.0) & (Y <= 22.5)
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    kernel_fft = make_fft_kernel(nx, ny, dx, dy, sigma_m=2.5)

    seasons = [
        ("winter_solstice", 355),
        ("spring_equinox", 80),
        ("summer_solstice", 172),
    ]

    summary_rows = []
    minute_rows = []
    seasonal_maps = {}

    for season_label, day_of_year in seasons:
        zone_means, daily_maps, rows, daylight_minutes = seasonal_metrics(
            base, day_of_year, X, Y, xf, yf, zone_mask, house, geom, kernel_fft
        )
        summary_rows.append(
            {
                "season": season_label,
                "day_of_year": day_of_year,
                "daylight_minutes": daylight_minutes,
                "zone_mean_direct_only": zone_means["direct_only"],
                "zone_mean_uniform_diffuse": zone_means["uniform_diffuse"],
                "zone_mean_spread_diffuse": zone_means["spread_diffuse"],
            }
        )
        minute_rows.extend([{**row, "season": season_label} for row in rows])
        seasonal_maps[season_label] = daily_maps

    pd.DataFrame(summary_rows).to_csv(out / "seasonal_three_modes_summary.csv", index=False)
    pd.DataFrame(minute_rows).to_csv(out / "seasonal_three_modes_minutely.csv", index=False)

    dist_rows = []
    for season_label, _ in seasons:
        daily_maps = seasonal_maps[season_label]
        for mode_name, arr in daily_maps.items():
            dist_rows.append(
                pd.DataFrame(
                    {
                        "season": season_label,
                        "mode": mode_name,
                        "x_m": X.ravel(),
                        "y_m": Y.ravel(),
                        "daily_floor_transmittance": arr.ravel(),
                        "in_zone": zone_mask.ravel().astype(int),
                    }
                )
            )
    pd.concat(dist_rows, ignore_index=True).to_csv(out / "seasonal_three_modes_distribution.csv", index=False)

    mode_titles = {
        "direct_only": "Direct only",
        "uniform_diffuse": "Direct + diffuse uniform",
        "spread_diffuse": "Direct + diffuse spread/loss",
    }
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    vmin = min(np.min(seasonal_maps[s][m]) for s, _ in seasons for m in mode_titles)
    vmax = max(np.max(seasonal_maps[s][m]) for s, _ in seasons for m in mode_titles)
    for row_idx, (season_label, _) in enumerate(seasons):
        for col_idx, mode_name in enumerate(mode_titles):
            ax = axes[row_idx, col_idx]
            arr = seasonal_maps[season_label][mode_name]
            im = ax.imshow(
                arr,
                origin="lower",
                extent=[0, house["depth_m"], 0, house["cultivation_width_m"]],
                aspect="auto",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            for gy in geom["gutter_y_positions"]:
                ax.axhline(gy, color="white", lw=0.5, alpha=0.35)
            ax.add_patch(plt.Rectangle((21.0, 15.0), 3.0, 7.5, fill=False, ec="cyan", lw=1.3))
            ax.set_title(f"{season_label.replace('_', ' ')} | {mode_titles[mode_name]}")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88)
    cb.set_label("daily floor transmittance vs outdoor")
    fig.savefig(out / "seasonal_three_modes_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
