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

from diffuse_transport_3d import build_roof_diffuse_sources, diffuse_irradiance_to_horizontal_plane


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
        "roof_strip_width_y": 0.10,
        "roof_strip_depth_z": 0.010,
        "ridge_z": 4.20 + 7.5 / 2.0 * 0.5,
    }
    geom["gutter_y_positions"] = np.arange(geom["unit_width"], house["cultivation_width_m"] - 1e-9, geom["unit_width"])
    geom["support_y_positions"] = np.arange(0.0, house["cultivation_width_m"] + 1e-9, geom["unit_width"])
    geom["pillar_x_positions"] = np.arange(0.0, house["depth_m"] + 1e-9, geom["pillar_pitch_x"])
    geom["rafter_x_positions"] = np.arange(0.0, house["depth_m"] + 1e-9, geom["rafter_pitch_x"])
    geom["roof_strip_y_positions"] = np.arange(geom["unit_width"] * 0.5, house["cultivation_width_m"], geom["unit_width"])
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


def structure_shadow_direct(base, xf, yf, sx, sy, sz, house, geom):
    z_floor = 0.0
    shadow = np.zeros_like(xf, dtype=bool)
    shadow |= base.shadow_by_linear_objects(yf, z_floor, sy, sz, geom["gutter_y_positions"], geom["gutter_width"], geom["gutter_ztop"], geom["gutter_ztop"] - geom["gutter_depth"])
    shadow |= base.shadow_by_linear_objects(yf, z_floor, sy, sz, geom["gutter_y_positions"], geom["curtain1_width"], geom["curtain1_ztop"], geom["curtain1_ztop"] - geom["curtain1_thickness"])
    shadow |= base.shadow_by_linear_objects(yf, z_floor, sy, sz, geom["gutter_y_positions"], geom["curtain2_width"], geom["curtain2_ztop"], geom["curtain2_ztop"] - geom["curtain2_thickness"])
    shadow |= base.shadow_by_transverse_frame_members(xf, yf, z_floor, sx, sy, sz, geom["pillar_x_positions"], geom["frame_member_thickness_x"], geom["support_y_positions"], geom["frame_member_width_y"], geom["frame_member_width_y"], house["cultivation_width_m"], geom["unit_width"], geom["gutter_ztop"], geom["ridge_z"])
    shadow |= base.shadow_by_transverse_frame_members(xf, yf, z_floor, sx, sy, sz, geom["rafter_x_positions"], geom["rafter_thickness_x"], geom["support_y_positions"], geom["rafter_width_y"], geom["rafter_width_y"], house["cultivation_width_m"], geom["unit_width"], geom["gutter_ztop"], geom["ridge_z"])
    shadow |= base.shadow_by_roof_longitudinal_members(yf, z_floor, sy, sz, geom["purlin_y_positions"], geom["purlin_width_y"], geom["purlin_depth_z"], geom["unit_width"], geom["gutter_ztop"], geom["ridge_z"])
    shadow |= base.shadow_by_roof_longitudinal_members(yf, z_floor, sy, sz, geom["roof_strip_y_positions"], geom["roof_strip_width_y"], geom["roof_strip_depth_z"], geom["unit_width"], geom["gutter_ztop"], geom["ridge_z"])
    shadow |= base.shadow_by_pillars(xf, yf, z_floor, sx, sy, sz, geom["pillar_x_positions"], geom["gutter_y_positions"], geom["frame_member_thickness_x"], geom["frame_member_width_y"], geom["gutter_ztop"])
    return (~shadow).astype(float)


def vertical_open_mask(base, xf, yf, house, geom):
    z_floor = np.zeros_like(xf, dtype=float)
    sx = 0.0
    sy = 0.0
    sz = 1.0
    blocked = np.zeros_like(xf, dtype=bool)
    for centers, width in (
        (geom["gutter_y_positions"], geom["gutter_width"]),
        (geom["gutter_y_positions"], geom["curtain1_width"]),
        (geom["gutter_y_positions"], geom["curtain2_width"]),
    ):
        blocked |= (np.abs(yf[:, None] - centers[None, :]) <= width / 2.0).any(axis=1)
    blocked |= base.shadow_by_transverse_frame_members(xf, yf, z_floor, sx, sy, sz, geom["pillar_x_positions"], geom["frame_member_thickness_x"], geom["support_y_positions"], geom["frame_member_width_y"], geom["frame_member_width_y"], house["cultivation_width_m"], geom["unit_width"], geom["gutter_ztop"], geom["ridge_z"])
    blocked |= base.shadow_by_transverse_frame_members(xf, yf, z_floor, sx, sy, sz, geom["rafter_x_positions"], geom["rafter_thickness_x"], geom["support_y_positions"], geom["rafter_width_y"], geom["rafter_width_y"], house["cultivation_width_m"], geom["unit_width"], geom["gutter_ztop"], geom["ridge_z"])
    blocked |= base.shadow_by_roof_longitudinal_members(yf, z_floor, sy, sz, geom["purlin_y_positions"], geom["purlin_width_y"], geom["purlin_depth_z"], geom["unit_width"], geom["gutter_ztop"], geom["ridge_z"])
    blocked |= base.shadow_by_roof_longitudinal_members(yf, z_floor, sy, sz, geom["roof_strip_y_positions"], geom["roof_strip_width_y"], geom["roof_strip_depth_z"], geom["unit_width"], geom["gutter_ztop"], geom["ridge_z"])

    Xc, Yc = np.meshgrid(geom["pillar_x_positions"], geom["gutter_y_positions"], indexing="xy")
    Xc = Xc.ravel()
    Yc = Yc.ravel()
    blocked |= (
        (np.abs(xf[:, None] - Xc[None, :]) <= geom["frame_member_thickness_x"] / 2.0)
        & (np.abs(yf[:, None] - Yc[None, :]) <= geom["frame_member_width_y"] / 2.0)
    ).any(axis=1)
    return (~blocked).astype(float)


def envelope_taus_with_entry(base, xf, yf, sx, sy, sz, house, geom):
    tol = 1e-9
    house_depth = house["depth_m"]
    house_width = house["cultivation_width_m"]
    unit_width = geom["unit_width"]
    gutter_z = geom["gutter_ztop"]
    ridge_z = geom["ridge_z"]

    n = xf.size
    t_best = np.full(n, np.inf, dtype=float)
    total_tau = np.full(n, house["roof_par_total_normal"], dtype=float)
    beam_diffuse_tau = np.zeros(n, dtype=float)
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
        total_tau[upd] = house["south_wall_tau_direct"]
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
        total_tau[upd] = house["roof_par_total_normal"]
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
        total_tau[upd] = house["roof_par_total_normal"]
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
            cond = (t > 0.0) & (x_int >= 0.0) & (x_int <= house_depth) & (y_int >= y0 - 1e-9) & (y_int <= yr + 1e-9)
            upd = cond & (t < t_best)
            t_best[upd] = t[upd]
            total_tau[upd] = total_west
            beam_diffuse_tau[upd] = total_west * house["roof_diffuse_fraction"]
            entry_x[upd] = x_int[upd]
            entry_y[upd] = y_int[upd]
            entry_z[upd] = t[upd] * sz
            boundary[upd] = "roof_west"
        denom_e = sz + m * sy
        if abs(denom_e) > tol:
            t = (gutter_z + m * (y1 - yf)) / denom_e
            y_int = yf + t * sy
            x_int = xf + t * sx
            cond = (t > 0.0) & (x_int >= 0.0) & (x_int <= house_depth) & (y_int >= yr - 1e-9) & (y_int <= y1 + 1e-9)
            upd = cond & (t < t_best)
            t_best[upd] = t[upd]
            total_tau[upd] = total_east
            beam_diffuse_tau[upd] = total_east * house["roof_diffuse_fraction"]
            entry_x[upd] = x_int[upd]
            entry_y[upd] = y_int[upd]
            entry_z[upd] = t[upd] * sz
            boundary[upd] = "roof_east"

    return total_tau, beam_diffuse_tau, entry_x, entry_y, entry_z, boundary


def shadow_from_point_by_linear_objects(y, z, ys, zs, y_centers, width, z_top, z_bottom):
    tol = 1.0e-9
    if np.all(np.abs(ys) <= tol) and np.all(np.abs(zs) <= tol):
        return np.zeros_like(y, dtype=bool)
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
                                            column_y_centers, column_width_y,
                                            frame_depth_z, house_width, unit_width,
                                            gutter_z, ridge_z):
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


def visibility_from_source(base, xf, yf, xs, ys, zs, house, geom):
    dx = xs - xf
    dy = ys - yf
    dz = zs
    norm = np.sqrt(dx * dx + dy * dy + dz * dz)
    sx = dx / np.maximum(norm, 1e-9)
    sy = dy / np.maximum(norm, 1e-9)
    sz = dz / np.maximum(norm, 1e-9)
    z_floor = np.zeros_like(xf, dtype=float)
    shadow = np.zeros_like(xf, dtype=bool)
    z_arr = np.zeros_like(xf, dtype=float)
    shadow |= shadow_from_point_by_linear_objects(yf, z_arr, sy, sz, geom["gutter_y_positions"], geom["gutter_width"], geom["gutter_ztop"], geom["gutter_ztop"] - geom["gutter_depth"])
    shadow |= shadow_from_point_by_linear_objects(yf, z_arr, sy, sz, geom["gutter_y_positions"], geom["curtain1_width"], geom["curtain1_ztop"], geom["curtain1_ztop"] - geom["curtain1_thickness"])
    shadow |= shadow_from_point_by_linear_objects(yf, z_arr, sy, sz, geom["gutter_y_positions"], geom["curtain2_width"], geom["curtain2_ztop"], geom["curtain2_ztop"] - geom["curtain2_thickness"])
    shadow |= shadow_from_point_by_transverse_members(xf, yf, z_floor, sx, sy, sz, geom["pillar_x_positions"], geom["frame_member_thickness_x"], geom["support_y_positions"], geom["frame_member_width_y"], geom["frame_member_width_y"], house["cultivation_width_m"], geom["unit_width"], geom["gutter_ztop"], geom["ridge_z"])
    shadow |= shadow_from_point_by_transverse_members(xf, yf, z_floor, sx, sy, sz, geom["rafter_x_positions"], geom["rafter_thickness_x"], geom["support_y_positions"], geom["rafter_width_y"], geom["rafter_width_y"], house["cultivation_width_m"], geom["unit_width"], geom["gutter_ztop"], geom["ridge_z"])
    shadow |= shadow_from_point_by_linear_objects(yf, z_arr, sy, sz, geom["purlin_y_positions"], geom["purlin_width_y"], geom["purlin_depth_z"] / 2.0 + base.span_roof_height(geom["purlin_y_positions"], unit_width=geom["unit_width"], gutter_z=geom["gutter_ztop"], ridge_z=geom["ridge_z"]), base.span_roof_height(geom["purlin_y_positions"], unit_width=geom["unit_width"], gutter_z=geom["gutter_ztop"], ridge_z=geom["ridge_z"]) - geom["purlin_depth_z"] / 2.0)
    shadow |= shadow_from_point_by_linear_objects(yf, z_arr, sy, sz, geom["roof_strip_y_positions"], geom["roof_strip_width_y"], geom["roof_strip_depth_z"] / 2.0 + base.span_roof_height(geom["roof_strip_y_positions"], unit_width=geom["unit_width"], gutter_z=geom["gutter_ztop"], ridge_z=geom["ridge_z"]), base.span_roof_height(geom["roof_strip_y_positions"], unit_width=geom["unit_width"], gutter_z=geom["gutter_ztop"], ridge_z=geom["ridge_z"]) - geom["roof_strip_depth_z"] / 2.0)
    shadow |= shadow_from_point_by_pillars(xf, yf, z_floor, sx, sy, sz, geom["pillar_x_positions"], geom["gutter_y_positions"], geom["frame_member_thickness_x"], geom["frame_member_width_y"], geom["gutter_ztop"])
    return (~shadow).astype(float)


def gaussian_blocked_diffuse_map(base, X, Y, xf, yf, entry_x, entry_y, entry_z, boundary, beam_diffuse_tau, house, geom, sigma_m=2.5):
    roof_mask = np.char.startswith(boundary.astype(str), "roof")
    valid = roof_mask & np.isfinite(entry_x) & np.isfinite(entry_y) & np.isfinite(entry_z) & (beam_diffuse_tau > 0.0)
    if not np.any(valid):
        return np.zeros_like(X, dtype=float)

    bin_dx = 1.5
    bin_dy = 1.5
    bx = np.floor(entry_x[valid] / bin_dx).astype(int)
    by = np.floor(entry_y[valid] / bin_dy).astype(int)
    source_df = pd.DataFrame({
        "bx": bx,
        "by": by,
        "x": entry_x[valid],
        "y": entry_y[valid],
        "z": entry_z[valid],
        "w": beam_diffuse_tau[valid],
    })
    grouped = source_df.groupby(["bx", "by"], as_index=False).apply(
        lambda g: pd.Series({
            "x": np.average(g["x"], weights=g["w"]),
            "y": np.average(g["y"], weights=g["w"]),
            "z": np.average(g["z"], weights=g["w"]),
            "w": g["w"].sum(),
        }),
        include_groups=False,
    ).reset_index(drop=True)

    diffuse = np.zeros_like(X, dtype=float)
    for _, row in grouped.iterrows():
        vis = visibility_from_source(base, xf, yf, row["x"], row["y"], row["z"], house, geom).reshape(X.shape)
        dist2 = (X - row["x"]) ** 2 + (Y - row["y"]) ** 2
        kernel = np.exp(-dist2 / (2.0 * sigma_m * sigma_m))
        s = kernel.sum()
        if s <= 0.0:
            continue
        kernel /= s
        diffuse += row["w"] * kernel * vis
    return diffuse


def run(output_dir="output_floor_seasonal_nonscatter_vs_gaussian"):
    base = load_base_module()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    house = {
        "latitude_deg": 35.0725,
        "longitude_deg": 136.73388888888888,
        "depth_m": 51.0,
        "cultivation_width_m": 45.0,
        "roof_par_total_normal": 0.903,
        "roof_diffuse_fraction": 0.45,
        "south_wall_tau_direct": 0.50,
    }
    geom = build_geometry(house)
    geom["purlin_y_positions"] = base.roof_member_y_positions(
        house["cultivation_width_m"], geom["unit_width"], geom["purlin_rows_per_slope"]
    )

    nx, ny = 60, 50
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
    minute_rows = []
    seasonal_maps = {}

    for season_label, day_of_year in seasons:
        outdoor_integral = 0.0
        daylight_minutes = 0
        accum = {
            "nonscatter": np.zeros_like(X, dtype=float),
            "diffuse_3d_blocked": np.zeros_like(X, dtype=float),
        }
        zone_weighted = {k: 0.0 for k in accum}

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

                total_tau, beam_diffuse_tau, entry_x, entry_y, entry_z, boundary = envelope_taus_with_entry(
                    base, xf, yf, sx, sy, sz, house, geom
                )
                direct_open = structure_shadow_direct(base, xf, yf, sx, sy, sz, house, geom).reshape(X.shape)
                nonscatter_map = total_tau.reshape(X.shape) * direct_open
                direct_component = (total_tau - beam_diffuse_tau).reshape(X.shape) * direct_open
                diffuse_sources = build_roof_diffuse_sources(
                    base, house, geom, sx, sy, sz, source_x_step_m=6.0, source_y_step_m=2.5
                )
                diffuse_spread = diffuse_irradiance_to_horizontal_plane(
                    base,
                    diffuse_sources,
                    x,
                    y,
                    0.0,
                    house,
                    geom,
                )
                diffuse_3d_blocked_map = direct_component + diffuse_spread

                outdoor = math.sin(math.radians(beta_deg))
                outdoor_integral += outdoor
                daylight_minutes += 5

                minute_row = {
                    "season": season_label,
                    "day_of_year": day_of_year,
                    "time_h": local_time,
                    "solar_elev_deg": beta_deg,
                    "solar_az_deg": az_deg,
                    "outdoor_relative_direct": outdoor,
                }
                for key, arr in {
                    "nonscatter": nonscatter_map,
                    "diffuse_3d_blocked": diffuse_3d_blocked_map,
                }.items():
                    accum[key] += arr * outdoor
                    zone_mean = float(np.mean(arr[zone_mask]))
                    zone_weighted[key] += zone_mean * outdoor
                    minute_row[f"zone_mean_{key}"] = zone_mean
                minute_rows.append(minute_row)

        daily_maps = {key: accum[key] / outdoor_integral for key in accum}
        seasonal_maps[season_label] = daily_maps
        summary_rows.append(
            {
                "season": season_label,
                "day_of_year": day_of_year,
                "daylight_minutes": daylight_minutes,
                "house_mean_nonscatter": float(np.mean(daily_maps["nonscatter"])),
                "house_mean_diffuse_3d_blocked": float(np.mean(daily_maps["diffuse_3d_blocked"])),
                "zone_mean_nonscatter": zone_weighted["nonscatter"] / outdoor_integral,
                "zone_mean_diffuse_3d_blocked": zone_weighted["diffuse_3d_blocked"] / outdoor_integral,
            }
        )

    pd.DataFrame(summary_rows).to_csv(out / "seasonal_nonscatter_vs_gaussian_summary.csv", index=False)
    pd.DataFrame(minute_rows).to_csv(out / "seasonal_nonscatter_vs_gaussian_minutely.csv", index=False)

    dist_rows = []
    for season_label, _ in seasons:
        for mode_name, arr in seasonal_maps[season_label].items():
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
    pd.concat(dist_rows, ignore_index=True).to_csv(out / "seasonal_nonscatter_vs_gaussian_distribution.csv", index=False)

    fig, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    mode_titles = {
        "nonscatter": "No scattering",
        "diffuse_3d_blocked": "3D diffuse + blocked",
    }
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
    fig.savefig(out / "seasonal_nonscatter_vs_gaussian_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
