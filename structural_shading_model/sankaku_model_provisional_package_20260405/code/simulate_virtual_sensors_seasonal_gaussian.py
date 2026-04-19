import importlib.util
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from diffuse_transport_3d import build_roof_diffuse_sources, diffuse_irradiance_at_points


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


def shadow_from_point_by_linear_objects(y, z, ys, zs, y_centers, width, z_top, z_bottom):
    tol = 1.0e-9
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    ys = np.asarray(ys, dtype=float)
    zs = np.asarray(zs, dtype=float)
    y_left = y_centers - width / 2.0
    y_right = y_centers + width / 2.0
    z_top_arr = np.atleast_1d(np.asarray(z_top, dtype=float))
    z_bottom_arr = np.atleast_1d(np.asarray(z_bottom, dtype=float))
    shadow = np.zeros_like(y, dtype=bool)
    if np.any(np.abs(ys) > tol):
        t_left = (y_left[None, :] - y[:, None]) / ys[:, None]
        z_left = z[:, None] + t_left * zs[:, None]
        shadow |= ((t_left > 0.0) & (z_left >= z_bottom_arr[None, :]) & (z_left <= z_top_arr[None, :])).any(axis=1)
        t_right = (y_right[None, :] - y[:, None]) / ys[:, None]
        z_right = z[:, None] + t_right * zs[:, None]
        shadow |= ((t_right > 0.0) & (z_right >= z_bottom_arr[None, :]) & (z_right <= z_top_arr[None, :])).any(axis=1)
    if np.max(np.abs(zs)) > tol:
        t_bottom = (z_bottom_arr[None, :] - z[:, None]) / zs[:, None]
        y_bottom = y[:, None] + t_bottom * ys[:, None]
        shadow |= ((t_bottom > 0.0) & (y_bottom >= y_left[None, :]) & (y_bottom <= y_right[None, :])).any(axis=1)
    return shadow


def shadow_from_point_by_transverse_members(x, y, z, sx, sy, sz, x_centers, x_thickness,
                                            column_y_centers, column_width_y, frame_depth_z,
                                            house_width, unit_width, gutter_z, ridge_z):
    tol = 1e-9
    shadow = np.zeros_like(x, dtype=bool)
    half_x = x_thickness / 2.0
    y_expand = np.where(np.abs(sx) > tol, half_x * np.abs(sy / np.maximum(np.abs(sx), tol)), 0.0)
    z_expand = np.where(np.abs(sx) > tol, half_x * np.abs(sz / np.maximum(np.abs(sx), tol)), 0.0)
    roof_slope = (ridge_z - gutter_z) / (unit_width / 2.0)

    for x_center in np.asarray(x_centers, dtype=float):
        valid = np.abs(sx) > tol
        t = np.zeros_like(x)
        t[valid] = (x_center - x[valid]) / sx[valid]
        valid &= t > 0.0
        y_hit = y + t * sy
        z_hit = z + t * sz
        col_cond = (
            valid[:, None]
            & (z_hit[:, None] >= 0.0)
            & (z_hit[:, None] <= gutter_z + z_expand[:, None])
            & (np.abs(y_hit[:, None] - column_y_centers[None, :]) <= (column_width_y / 2.0 + y_expand[:, None]))
        )
        shadow |= col_cond.any(axis=1)
        y_mod = np.mod(y_hit, unit_width)
        west = y_mod <= unit_width / 2.0
        roof_z = np.where(west, gutter_z + roof_slope * y_mod, gutter_z + roof_slope * (unit_width - y_mod))
        roof_tol = frame_depth_z / 2.0 + z_expand + roof_slope * y_expand
        roof_cond = (
            valid
            & (y_hit >= -y_expand)
            & (y_hit <= house_width + y_expand)
            & (z_hit >= gutter_z - roof_tol)
            & (z_hit <= ridge_z + roof_tol)
            & (np.abs(z_hit - roof_z) <= roof_tol)
        )
        shadow |= roof_cond
    return shadow


def shadow_from_point_by_pillars(x, y, z, sx, sy, sz, x_centers, y_centers, pillar_thickness_x, pillar_width_y, z_top):
    tol = 1e-9
    Xc, Yc = np.meshgrid(x_centers, y_centers, indexing="xy")
    Xc = Xc.ravel()
    Yc = Yc.ravel()
    x_left = Xc - pillar_thickness_x / 2.0
    x_right = Xc + pillar_thickness_x / 2.0
    y_low = Yc - pillar_width_y / 2.0
    y_high = Yc + pillar_width_y / 2.0
    shadow = np.zeros_like(x, dtype=bool)

    valid_x = np.abs(sx) > tol
    if np.any(valid_x):
        for x_face in (x_left, x_right):
            t = np.zeros((x.size, x_face.size), dtype=float)
            t[valid_x, :] = (x_face[None, :] - x[valid_x, None]) / sx[valid_x, None]
            y_int = y[:, None] + t * sy[:, None]
            z_int = z[:, None] + t * sz[:, None]
            cond = (t > 0.0) & (y_int >= y_low[None, :]) & (y_int <= y_high[None, :]) & (z_int >= 0.0) & (z_int <= z_top)
            shadow |= cond.any(axis=1)

    valid_y = np.abs(sy) > tol
    if np.any(valid_y):
        for y_face in (y_low, y_high):
            t = np.zeros((x.size, y_face.size), dtype=float)
            t[valid_y, :] = (y_face[None, :] - y[valid_y, None]) / sy[valid_y, None]
            x_int = x[:, None] + t * sx[:, None]
            z_int = z[:, None] + t * sz[:, None]
            cond = (t > 0.0) & (x_int >= x_left[None, :]) & (x_int <= x_right[None, :]) & (z_int >= 0.0) & (z_int <= z_top)
            shadow |= cond.any(axis=1)
    return shadow


def visibility_from_source_to_points(base, xp, yp, zp, xs, ys, zs, house, geom):
    dx = xs - xp
    dy = ys - yp
    dz = zs - zp
    norm = np.sqrt(dx * dx + dy * dy + dz * dz)
    sx = dx / np.maximum(norm, 1e-9)
    sy = dy / np.maximum(norm, 1e-9)
    sz = dz / np.maximum(norm, 1e-9)
    shadow = np.zeros_like(xp, dtype=bool)
    shadow |= shadow_from_point_by_linear_objects(yp, zp, sy, sz, geom["gutter_y_positions"], geom["gutter_width"], geom["gutter_ztop"], geom["gutter_ztop"] - geom["gutter_depth"])
    shadow |= shadow_from_point_by_linear_objects(yp, zp, sy, sz, geom["gutter_y_positions"], geom["curtain1_width"], geom["curtain1_ztop"], geom["curtain1_ztop"] - geom["curtain1_thickness"])
    shadow |= shadow_from_point_by_linear_objects(yp, zp, sy, sz, geom["gutter_y_positions"], geom["curtain2_width"], geom["curtain2_ztop"], geom["curtain2_ztop"] - geom["curtain2_thickness"])
    shadow |= shadow_from_point_by_transverse_members(xp, yp, zp, sx, sy, sz, geom["pillar_x_positions"], geom["frame_member_thickness_x"], geom["support_y_positions"], geom["frame_member_width_y"], geom["frame_member_width_y"], house["cultivation_width_m"], geom["unit_width"], geom["gutter_ztop"], geom["ridge_z"])
    shadow |= shadow_from_point_by_transverse_members(xp, yp, zp, sx, sy, sz, geom["rafter_x_positions"], geom["rafter_thickness_x"], geom["support_y_positions"], geom["rafter_width_y"], geom["rafter_width_y"], house["cultivation_width_m"], geom["unit_width"], geom["gutter_ztop"], geom["ridge_z"])
    purlin_z = base.span_roof_height(geom["purlin_y_positions"], unit_width=geom["unit_width"], gutter_z=geom["gutter_ztop"], ridge_z=geom["ridge_z"])
    shadow |= shadow_from_point_by_linear_objects(yp, zp, sy, sz, geom["purlin_y_positions"], geom["purlin_width_y"], purlin_z + geom["purlin_depth_z"] / 2.0, purlin_z - geom["purlin_depth_z"] / 2.0)
    shadow |= shadow_from_point_by_pillars(xp, yp, zp, sx, sy, sz, geom["pillar_x_positions"], geom["gutter_y_positions"], geom["frame_member_thickness_x"], geom["frame_member_width_y"], geom["gutter_ztop"])
    return (~shadow).astype(float)


def direct_open_at_points(base, x, y, z, sx, sy, sz, house, geom):
    shadow = np.zeros_like(x, dtype=bool)
    shadow |= shadow_from_point_by_linear_objects(y, z, np.full_like(x, sy), np.full_like(x, sz), geom["gutter_y_positions"], geom["gutter_width"], geom["gutter_ztop"], geom["gutter_ztop"] - geom["gutter_depth"])
    shadow |= shadow_from_point_by_linear_objects(y, z, np.full_like(x, sy), np.full_like(x, sz), geom["gutter_y_positions"], geom["curtain1_width"], geom["curtain1_ztop"], geom["curtain1_ztop"] - geom["curtain1_thickness"])
    shadow |= shadow_from_point_by_linear_objects(y, z, np.full_like(x, sy), np.full_like(x, sz), geom["gutter_y_positions"], geom["curtain2_width"], geom["curtain2_ztop"], geom["curtain2_ztop"] - geom["curtain2_thickness"])
    shadow |= shadow_from_point_by_transverse_members(x, y, z, np.full_like(x, sx), np.full_like(x, sy), np.full_like(x, sz), geom["pillar_x_positions"], geom["frame_member_thickness_x"], geom["support_y_positions"], geom["frame_member_width_y"], geom["frame_member_width_y"], house["cultivation_width_m"], geom["unit_width"], geom["gutter_ztop"], geom["ridge_z"])
    shadow |= shadow_from_point_by_transverse_members(x, y, z, np.full_like(x, sx), np.full_like(x, sy), np.full_like(x, sz), geom["rafter_x_positions"], geom["rafter_thickness_x"], geom["support_y_positions"], geom["rafter_width_y"], geom["rafter_width_y"], house["cultivation_width_m"], geom["unit_width"], geom["gutter_ztop"], geom["ridge_z"])
    purlin_z = base.span_roof_height(geom["purlin_y_positions"], unit_width=geom["unit_width"], gutter_z=geom["gutter_ztop"], ridge_z=geom["ridge_z"])
    shadow |= shadow_from_point_by_linear_objects(y, z, np.full_like(x, sy), np.full_like(x, sz), geom["purlin_y_positions"], geom["purlin_width_y"], purlin_z + geom["purlin_depth_z"] / 2.0, purlin_z - geom["purlin_depth_z"] / 2.0)
    shadow |= shadow_from_point_by_pillars(x, y, z, np.full_like(x, sx), np.full_like(x, sy), np.full_like(x, sz), geom["pillar_x_positions"], geom["gutter_y_positions"], geom["frame_member_thickness_x"], geom["frame_member_width_y"], geom["gutter_ztop"])
    return (~shadow).astype(float)


def envelope_taus_with_entry(base, x, y, z, sx, sy, sz, house, geom):
    tol = 1e-9
    house_depth = house["depth_m"]
    house_width = house["cultivation_width_m"]
    unit_width = geom["unit_width"]
    gutter_z = geom["gutter_ztop"]
    ridge_z = geom["ridge_z"]

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    n = x.size
    t_best = np.full(n, np.inf, dtype=float)
    direct_tau = np.full(n, house["roof_par_total_normal"] * house["roof_direct_fraction"], dtype=float)
    beam_diffuse_tau = np.zeros(n, dtype=float)
    entry_x = np.full(n, np.nan, dtype=float)
    entry_y = np.full(n, np.nan, dtype=float)
    entry_z = np.full(n, np.nan, dtype=float)
    boundary = np.full(n, "roof", dtype=object)

    if sx < -tol:
        t = (0.0 - x) / sx
        y_int = y + t * sy
        z_int = z + t * sz
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
        t = (0.0 - y) / sy
        x_int = x + t * sx
        z_int = z + t * sz
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
        t = (house_width - y) / sy
        x_int = x + t * sx
        z_int = z + t * sz
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
            t = (gutter_z + m * (y - y0) - z) / denom_w
            y_int = y + t * sy
            x_int = x + t * sx
            z_int = z + t * sz
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
            t = (gutter_z + m * (y1 - y) - z) / denom_e
            y_int = y + t * sy
            x_int = x + t * sx
            z_int = z + t * sz
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


def build_diffuse_sources(base, sample_x, sample_y, sx, sy, sz, house, geom):
    xx, yy = np.meshgrid(sample_x, sample_y)
    zz = np.zeros_like(xx, dtype=float)
    direct_tau, beam_diffuse_tau, entry_x, entry_y, entry_z, boundary = envelope_taus_with_entry(
        base, xx.ravel(), yy.ravel(), zz.ravel(), sx, sy, sz, house, geom
    )
    roof_mask = np.char.startswith(boundary.astype(str), "roof")
    valid = roof_mask & np.isfinite(entry_x) & np.isfinite(entry_y) & np.isfinite(entry_z) & (beam_diffuse_tau > 0.0)
    if not np.any(valid):
        return pd.DataFrame(columns=["x", "y", "z", "w"])
    bin_dx = 1.5
    bin_dy = 1.5
    bx = np.floor(entry_x[valid] / bin_dx).astype(int)
    by = np.floor(entry_y[valid] / bin_dy).astype(int)
    df = pd.DataFrame(
        {
            "bx": bx,
            "by": by,
            "x": entry_x[valid],
            "y": entry_y[valid],
            "z": entry_z[valid],
            "w": beam_diffuse_tau[valid],
        }
    )
    grouped = df.groupby(["bx", "by"], as_index=False).apply(
        lambda g: pd.Series(
            {
                "x": np.average(g["x"], weights=g["w"]),
                "y": np.average(g["y"], weights=g["w"]),
                "z": np.average(g["z"], weights=g["w"]),
                "w": g["w"].sum(),
            }
        ),
        include_groups=False,
    ).reset_index(drop=True)
    return grouped


def gaussian_diffuse_at_sensors(base, sensors_df, sources_df, house, geom, sigma_m=2.5, sensor_area_m2=1.0, cutoff_sigma=4.0):
    if sources_df.empty:
        return np.zeros(len(sensors_df), dtype=float)
    xs = sensors_df["x_m"].to_numpy()
    ys = sensors_df["y_m"].to_numpy()
    zs = sensors_df["z_m"].to_numpy()
    diffuse = np.zeros(len(sensors_df), dtype=float)
    norm_factor = 1.0 / (2.0 * math.pi * sigma_m * sigma_m)
    cutoff_dist2 = (cutoff_sigma * sigma_m) ** 2
    for sensor_idx, sensor in sensors_df.iterrows():
        dx = sources_df["x"].to_numpy() - sensor["x_m"]
        dy = sources_df["y"].to_numpy() - sensor["y_m"]
        dist2 = dx * dx + dy * dy
        candidate_mask = dist2 <= cutoff_dist2
        if not np.any(candidate_mask):
            continue
        nearby = sources_df.loc[candidate_mask].copy()
        nearby_dist2 = dist2[candidate_mask]
        vis = visibility_from_source_to_points(
            base,
            np.full(len(nearby), sensor["x_m"], dtype=float),
            np.full(len(nearby), sensor["y_m"], dtype=float),
            np.full(len(nearby), sensor["z_m"], dtype=float),
            nearby["x"].to_numpy(),
            nearby["y"].to_numpy(),
            nearby["z"].to_numpy(),
            house,
            geom,
        )
        kernel = np.exp(-nearby_dist2 / (2.0 * sigma_m * sigma_m)) * norm_factor
        diffuse[sensor_idx] = float(np.sum(nearby["w"].to_numpy() * kernel * sensor_area_m2 * vis))
    return diffuse


def run(output_dir="output_virtual_sensors_gaussian"):
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

    sensor_x = [12.0, 24.0, 36.0]
    sensor_y = [3.75, 18.75, 33.75]
    sensor_z = [3.0, 1.5]
    sensors = []
    sid = 1
    for z in sensor_z:
        for x in sensor_x:
            for y in sensor_y:
                sensors.append({"sensor_id": f"S{sid:02d}", "x_m": x, "y_m": y, "z_m": z})
                sid += 1
    sensors_df = pd.DataFrame(sensors)

    sample_x = np.linspace(0.0, house["depth_m"], 16)
    sample_y = np.linspace(0.0, house["cultivation_width_m"], 12)
    seasons = [
        ("winter_solstice", 355),
        ("spring_equinox", 80),
        ("summer_solstice", 172),
    ]

    rows = []
    summary_rows = []

    for season_label, day_of_year in seasons:
        season_accum = {sid: 0.0 for sid in sensors_df["sensor_id"]}
        outdoor_accum = 0.0
        daylight_minutes = 0

        for hour in range(4, 21):
            for minute in range(0, 60, 5):
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
                    base,
                    sensors_df["x_m"].to_numpy(),
                    sensors_df["y_m"].to_numpy(),
                    sensors_df["z_m"].to_numpy(),
                    sx,
                    sy,
                    sz,
                    house,
                    geom,
                )
                direct_open = direct_open_at_points(
                    base,
                    sensors_df["x_m"].to_numpy(),
                    sensors_df["y_m"].to_numpy(),
                    sensors_df["z_m"].to_numpy(),
                    sx,
                    sy,
                    sz,
                    house,
                    geom,
                )
                direct_component = direct_tau * direct_open
                sources_df = build_roof_diffuse_sources(
                    base, house, geom, sx, sy, sz, source_x_step_m=6.0, source_y_step_m=2.5
                )
                diffuse_component = diffuse_irradiance_at_points(
                    base,
                    sources_df,
                    sensors_df["x_m"].to_numpy(),
                    sensors_df["y_m"].to_numpy(),
                    sensors_df["z_m"].to_numpy(),
                    house,
                    geom,
                )
                indoor_signal = direct_component + diffuse_component

                outdoor_relative = math.sin(math.radians(beta_deg))
                outdoor_accum += outdoor_relative
                daylight_minutes += 5

                for i, sensor in sensors_df.iterrows():
                    rows.append(
                        {
                            "season": season_label,
                            "day_of_year": day_of_year,
                            "time_h": local_time,
                            "sensor_id": sensor["sensor_id"],
                            "x_m": sensor["x_m"],
                            "y_m": sensor["y_m"],
                            "z_m": sensor["z_m"],
                            "solar_elev_deg": beta_deg,
                            "solar_az_deg": az_deg,
                            "outdoor_relative_direct": outdoor_relative,
                            "direct_component": float(direct_component[i]),
                            "diffuse_component": float(diffuse_component[i]),
                            "indoor_relative_signal": float(indoor_signal[i]),
                        }
                    )
                    season_accum[sensor["sensor_id"]] += float(indoor_signal[i]) * outdoor_relative

        for sensor in sensors_df.to_dict("records"):
            summary_rows.append(
                {
                    "season": season_label,
                    "day_of_year": day_of_year,
                    "sensor_id": sensor["sensor_id"],
                    "x_m": sensor["x_m"],
                    "y_m": sensor["y_m"],
                    "z_m": sensor["z_m"],
                    "daylight_minutes": daylight_minutes,
                    "daily_mean_transmittance": season_accum[sensor["sensor_id"]] / outdoor_accum if outdoor_accum > 0 else 0.0,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out / "virtual_sensor_timeseries.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out / "virtual_sensor_daily_summary.csv", index=False)

    colors = plt.cm.tab10(np.linspace(0, 1, 9))
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True, sharex=True, sharey=True)
    for row_idx, (season_label, _) in enumerate(seasons):
        for col_idx, z_target in enumerate([3.0, 1.5]):
            ax = axes[row_idx, col_idx]
            sub = df[(df["season"] == season_label) & (df["z_m"] == z_target)]
            for color_idx, sensor_id in enumerate(sorted(sub["sensor_id"].unique())):
                ss = sub[sub["sensor_id"] == sensor_id]
                ax.plot(ss["time_h"], ss["indoor_relative_signal"], lw=1.1, color=colors[color_idx % len(colors)], label=sensor_id)
            ax.set_title(f"{season_label.replace('_', ' ')} | z={z_target:.1f} m")
            ax.set_xlabel("time [h]")
            ax.set_ylabel("relative light")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=9, fontsize=8)
    fig.savefig(out / "virtual_sensor_timeseries.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
