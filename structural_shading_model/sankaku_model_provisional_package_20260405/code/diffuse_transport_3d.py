import math
from functools import lru_cache

import numpy as np
import pandas as pd


DEFAULT_SOURCE_X_STEP_M = 3.0
DEFAULT_SOURCE_Y_STEP_M = 1.25
DEFAULT_SIGMA_DEG = 30.0
DEFAULT_CUTOFF_SIGMA = 2.5


def opaque_interval_overlap(center, width, opaque_centers, opaque_width):
    if opaque_centers is None or len(opaque_centers) == 0 or opaque_width <= 0.0:
        return 0.0
    left = center - width / 2.0
    right = center + width / 2.0
    opaque_centers = np.asarray(opaque_centers, dtype=float)
    opaque_left = opaque_centers - opaque_width / 2.0
    opaque_right = opaque_centers + opaque_width / 2.0
    overlap = np.maximum(0.0, np.minimum(right, opaque_right) - np.maximum(left, opaque_left))
    return float(np.sum(overlap))


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


def interval_centers_and_widths(length_m, step_m):
    edges = np.arange(0.0, length_m + 1e-9, step_m, dtype=float)
    if edges.size == 0 or edges[-1] < length_m - 1e-9:
        edges = np.append(edges, length_m)
    else:
        edges[-1] = length_m
    widths = np.diff(edges)
    centers = edges[:-1] + widths / 2.0
    valid = widths > 1.0e-9
    return centers[valid], widths[valid]


def offset_centers_and_widths(length_m, step_m, offset_m):
    centers = np.arange(offset_m, length_m, step_m, dtype=float)
    if centers.size == 0:
        centers = np.array([min(max(offset_m, length_m / 2.0), length_m)], dtype=float)
    edges = np.empty(centers.size + 1, dtype=float)
    edges[0] = 0.0
    edges[-1] = length_m
    if centers.size > 1:
        edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    widths = np.diff(edges)
    valid = widths > 1.0e-9
    return centers[valid], widths[valid]


def roof_normals(geom):
    slope = (geom["ridge_z"] - geom["gutter_ztop"]) / (geom["unit_width"] / 2.0)
    norm = math.sqrt(1.0 + slope * slope)
    outward_west = np.array([0.0, -slope / norm, 1.0 / norm], dtype=float)
    outward_east = np.array([0.0, slope / norm, 1.0 / norm], dtype=float)
    return {
        "west": {"outward": outward_west, "inward": -outward_west},
        "east": {"outward": outward_east, "inward": -outward_east},
    }


def build_roof_diffuse_sources(
    base,
    house,
    geom,
    sx,
    sy,
    sz,
    source_x_step_m=DEFAULT_SOURCE_X_STEP_M,
    source_y_step_m=DEFAULT_SOURCE_Y_STEP_M,
):
    if sz <= 0.0:
        return pd.DataFrame(columns=["x", "y", "z", "power", "nx", "ny", "nz"])

    normals = roof_normals(geom)
    unit_width = geom["unit_width"]
    half_span = unit_width / 2.0
    slope_factor = math.sqrt(1.0 + ((geom["ridge_z"] - geom["gutter_ztop"]) / half_span) ** 2)
    n_units = int(round(house["cultivation_width_m"] / unit_width))
    x_centers, x_widths = offset_centers_and_widths(house["depth_m"], source_x_step_m, 0.25)
    u_centers, u_widths = interval_centers_and_widths(half_span, source_y_step_m)
    sun_vec = np.array([sx, sy, sz], dtype=float)
    rows = []

    for side_name, normal_set in normals.items():
        outward = normal_set["outward"]
        inward = normal_set["inward"]
        cos_inc = max(0.0, float(np.dot(outward, sun_vec)))
        if cos_inc <= 0.0:
            continue
        inc_deg = math.degrees(math.acos(min(1.0, max(0.0, cos_inc))))
        total_tau = house["roof_par_total_normal"] * extrapolated_k_theta(base, np.array([inc_deg], dtype=float))[0]
        beam_diffuse_tau = total_tau * house["roof_diffuse_fraction"]
        if beam_diffuse_tau <= 0.0:
            continue
        power_density = beam_diffuse_tau * cos_inc / max(sz, 1.0e-6)

        for unit_idx in range(n_units):
            y0 = unit_idx * unit_width
            if side_name == "west":
                y_centers = y0 + u_centers
            else:
                y_centers = y0 + half_span + u_centers

            z_centers = base.span_roof_height(
                y_centers,
                unit_width=unit_width,
                gutter_z=geom["gutter_ztop"],
                ridge_z=geom["ridge_z"],
            )

            for y_center, y_width, z_center in zip(y_centers, u_widths, z_centers):
                opaque_overlap = opaque_interval_overlap(
                    y_center,
                    y_width,
                    geom.get("roof_strip_y_positions"),
                    geom.get("roof_strip_width_y", 0.0),
                )
                open_width = max(0.0, y_width - opaque_overlap)
                slope_strip_width = open_width * slope_factor
                if slope_strip_width <= 0.0:
                    continue
                for x_center, x_width in zip(x_centers, x_widths):
                    patch_area = x_width * slope_strip_width
                    rows.append(
                        {
                            "x": x_center,
                            "y": y_center,
                            "z": float(z_center),
                            "power": float(power_density * patch_area),
                            "nx": float(inward[0]),
                            "ny": float(inward[1]),
                            "nz": float(inward[2]),
                        }
                    )

    if not rows:
        return pd.DataFrame(columns=["x", "y", "z", "power", "nx", "ny", "nz"])
    return pd.DataFrame(rows)


@lru_cache(maxsize=None)
def phase_normalization(sigma_rad):
    angles = np.linspace(0.0, math.pi / 2.0, 4097)
    raw = np.cos(angles) * np.exp(-0.5 * (angles / sigma_rad) ** 2)
    return 2.0 * math.pi * np.trapezoid(raw * np.sin(angles), angles)


def phase_density_from_cos(cos_alpha, sigma_rad):
    cos_alpha = np.asarray(cos_alpha, dtype=float)
    density = np.zeros_like(cos_alpha, dtype=float)
    positive = cos_alpha > 0.0
    if not np.any(positive):
        return density
    alpha = np.arccos(np.clip(cos_alpha[positive], 0.0, 1.0))
    raw = cos_alpha[positive] * np.exp(-0.5 * (alpha / sigma_rad) ** 2)
    density[positive] = raw / phase_normalization(sigma_rad)
    return density


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


def shadow_from_point_by_transverse_members(
    x,
    y,
    z,
    sx,
    sy,
    sz,
    x_centers,
    x_thickness,
    column_y_centers,
    column_width_y,
    frame_depth_z,
    house_width,
    unit_width,
    gutter_z,
    ridge_z,
):
    tol = 1.0e-9
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
    tol = 1.0e-9
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    sx = np.asarray(sx, dtype=float)
    sy = np.asarray(sy, dtype=float)
    sz = np.asarray(sz, dtype=float)
    xc, yc = np.meshgrid(x_centers, y_centers, indexing="xy")
    xc = xc.ravel()
    yc = yc.ravel()
    x_left = xc - pillar_thickness_x / 2.0
    x_right = xc + pillar_thickness_x / 2.0
    y_low = yc - pillar_width_y / 2.0
    y_high = yc + pillar_width_y / 2.0
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
    sx = dx / np.maximum(norm, 1.0e-9)
    sy = dy / np.maximum(norm, 1.0e-9)
    sz = dz / np.maximum(norm, 1.0e-9)
    shadow = np.zeros_like(xp, dtype=bool)
    shadow |= shadow_from_point_by_linear_objects(
        yp,
        zp,
        sy,
        sz,
        geom["gutter_y_positions"],
        geom["gutter_width"],
        geom["gutter_ztop"],
        geom["gutter_ztop"] - geom["gutter_depth"],
    )
    shadow |= shadow_from_point_by_linear_objects(
        yp,
        zp,
        sy,
        sz,
        geom["gutter_y_positions"],
        geom["curtain1_width"],
        geom["curtain1_ztop"],
        geom["curtain1_ztop"] - geom["curtain1_thickness"],
    )
    shadow |= shadow_from_point_by_linear_objects(
        yp,
        zp,
        sy,
        sz,
        geom["gutter_y_positions"],
        geom["curtain2_width"],
        geom["curtain2_ztop"],
        geom["curtain2_ztop"] - geom["curtain2_thickness"],
    )
    shadow |= shadow_from_point_by_transverse_members(
        xp,
        yp,
        zp,
        sx,
        sy,
        sz,
        geom["pillar_x_positions"],
        geom["frame_member_thickness_x"],
        geom["support_y_positions"],
        geom["frame_member_width_y"],
        geom["frame_member_width_y"],
        house["cultivation_width_m"],
        geom["unit_width"],
        geom["gutter_ztop"],
        geom["ridge_z"],
    )
    shadow |= shadow_from_point_by_transverse_members(
        xp,
        yp,
        zp,
        sx,
        sy,
        sz,
        geom["rafter_x_positions"],
        geom["rafter_thickness_x"],
        geom["support_y_positions"],
        geom["rafter_width_y"],
        geom["rafter_width_y"],
        house["cultivation_width_m"],
        geom["unit_width"],
        geom["gutter_ztop"],
        geom["ridge_z"],
    )
    purlin_z = base.span_roof_height(
        geom["purlin_y_positions"],
        unit_width=geom["unit_width"],
        gutter_z=geom["gutter_ztop"],
        ridge_z=geom["ridge_z"],
    )
    shadow |= shadow_from_point_by_linear_objects(
        yp,
        zp,
        sy,
        sz,
        geom["purlin_y_positions"],
        geom["purlin_width_y"],
        purlin_z + geom["purlin_depth_z"] / 2.0,
        purlin_z - geom["purlin_depth_z"] / 2.0,
    )
    roof_strip_y = np.asarray(geom.get("roof_strip_y_positions", []), dtype=float)
    if roof_strip_y.size > 0 and geom.get("roof_strip_width_y", 0.0) > 0.0:
        roof_strip_z = base.span_roof_height(
            roof_strip_y,
            unit_width=geom["unit_width"],
            gutter_z=geom["gutter_ztop"],
            ridge_z=geom["ridge_z"],
        )
        shadow |= shadow_from_point_by_linear_objects(
            yp,
            zp,
            sy,
            sz,
            roof_strip_y,
            geom["roof_strip_width_y"],
            roof_strip_z + geom.get("roof_strip_depth_z", 0.01) / 2.0,
            roof_strip_z - geom.get("roof_strip_depth_z", 0.01) / 2.0,
        )
    shadow |= shadow_from_point_by_pillars(
        xp,
        yp,
        zp,
        sx,
        sy,
        sz,
        geom["pillar_x_positions"],
        geom["gutter_y_positions"],
        geom["frame_member_thickness_x"],
        geom["frame_member_width_y"],
        geom["gutter_ztop"],
    )
    return (~shadow).astype(float)


def first_exit_t_from_inside(xs, ys, zs, vx, vy, vz, house, geom):
    tol = 1.0e-9
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    vz = np.asarray(vz, dtype=float)
    t_best = np.full_like(vx, np.inf, dtype=float)
    house_depth = house["depth_m"]
    house_width = house["cultivation_width_m"]
    unit_width = geom["unit_width"]
    gutter_z = geom["gutter_ztop"]
    ridge_z = geom["ridge_z"]

    if xs > tol and np.any(vx < -tol):
        t = (0.0 - xs) / vx
        y_int = ys + t * vy
        z_int = zs + t * vz
        roof_z = geom["gutter_ztop"] + (ridge_z - gutter_z) * (
            1.0 - np.abs(np.mod(y_int, unit_width) - unit_width / 2.0) / (unit_width / 2.0)
        )
        cond = (t > tol) & (y_int >= 0.0) & (y_int <= house_width) & (z_int >= 0.0) & (z_int <= roof_z)
        t_best = np.where(cond & (t < t_best), t, t_best)

    if xs < house_depth - tol and np.any(vx > tol):
        t = (house_depth - xs) / vx
        y_int = ys + t * vy
        z_int = zs + t * vz
        roof_z = geom["gutter_ztop"] + (ridge_z - gutter_z) * (
            1.0 - np.abs(np.mod(y_int, unit_width) - unit_width / 2.0) / (unit_width / 2.0)
        )
        cond = (t > tol) & (y_int >= 0.0) & (y_int <= house_width) & (z_int >= 0.0) & (z_int <= roof_z)
        t_best = np.where(cond & (t < t_best), t, t_best)

    if ys > tol and np.any(vy < -tol):
        t = (0.0 - ys) / vy
        x_int = xs + t * vx
        z_int = zs + t * vz
        cond = (t > tol) & (x_int >= 0.0) & (x_int <= house_depth) & (z_int >= 0.0) & (z_int <= gutter_z)
        t_best = np.where(cond & (t < t_best), t, t_best)

    if ys < house_width - tol and np.any(vy > tol):
        t = (house_width - ys) / vy
        x_int = xs + t * vx
        z_int = zs + t * vz
        cond = (t > tol) & (x_int >= 0.0) & (x_int <= house_depth) & (z_int >= 0.0) & (z_int <= gutter_z)
        t_best = np.where(cond & (t < t_best), t, t_best)

    slope = (ridge_z - gutter_z) / (unit_width / 2.0)
    n_units = int(round(house_width / unit_width))
    for unit_idx in range(n_units):
        y0 = unit_idx * unit_width
        yr = y0 + unit_width / 2.0
        y1 = y0 + unit_width

        denom_w = vz - slope * vy
        valid_w = np.abs(denom_w) > tol
        if np.any(valid_w):
            t = np.full_like(vx, np.inf, dtype=float)
            t[valid_w] = (gutter_z + slope * (ys - y0) - zs) / denom_w[valid_w]
            y_int = ys + t * vy
            x_int = xs + t * vx
            cond = (t > tol) & (x_int >= 0.0) & (x_int <= house_depth) & (y_int >= y0 - 1.0e-9) & (y_int <= yr + 1.0e-9)
            t_best = np.where(cond & (t < t_best), t, t_best)

        denom_e = vz + slope * vy
        valid_e = np.abs(denom_e) > tol
        if np.any(valid_e):
            t = np.full_like(vx, np.inf, dtype=float)
            t[valid_e] = (gutter_z + slope * (y1 - ys) - zs) / denom_e[valid_e]
            y_int = ys + t * vy
            x_int = xs + t * vx
            cond = (t > tol) & (x_int >= 0.0) & (x_int <= house_depth) & (y_int >= yr - 1.0e-9) & (y_int <= y1 + 1.0e-9)
            t_best = np.where(cond & (t < t_best), t, t_best)

    return t_best


def diffuse_irradiance_to_horizontal_plane(
    base,
    sources_df,
    x_coords,
    y_coords,
    plane_z,
    house,
    geom,
    sigma_deg=DEFAULT_SIGMA_DEG,
    cutoff_sigma=DEFAULT_CUTOFF_SIGMA,
):
    x_coords = np.asarray(x_coords, dtype=float)
    y_coords = np.asarray(y_coords, dtype=float)
    plane = np.zeros((y_coords.size, x_coords.size), dtype=float)
    if sources_df.empty:
        return plane

    sigma_rad = math.radians(sigma_deg)
    alpha_max = min(math.radians(85.0), cutoff_sigma * sigma_rad)
    target_normal_z = 1.0

    for row in sources_df.itertuples(index=False):
        if plane_z >= row.z - 1.0e-9:
            continue
        t_center = (plane_z - row.z) / row.nz
        if t_center <= 0.0:
            continue
        x_center = row.x + t_center * row.nx
        y_center = row.y + t_center * row.ny
        radius = t_center * math.tan(alpha_max)
        x_sel = np.where(np.abs(x_coords - x_center) <= radius)[0]
        y_sel = np.where(np.abs(y_coords - y_center) <= radius)[0]
        if x_sel.size == 0 or y_sel.size == 0:
            continue

        xx, yy = np.meshgrid(x_coords[x_sel], y_coords[y_sel])
        dx = xx - row.x
        dy = yy - row.y
        dz = plane_z - row.z
        r2 = dx * dx + dy * dy + dz * dz
        r = np.sqrt(r2)
        ux = dx / np.maximum(r, 1.0e-9)
        uy = dy / np.maximum(r, 1.0e-9)
        uz = dz / np.maximum(r, 1.0e-9)
        cos_alpha = ux * row.nx + uy * row.ny + uz * row.nz
        phase = phase_density_from_cos(cos_alpha, sigma_rad)
        if not np.any(phase > 0.0):
            continue
        cos_target = -uz * target_normal_z
        vis_boundary = first_exit_t_from_inside(row.x, row.y, row.z, ux, uy, uz, house, geom) >= (r - 1.0e-6)
        vis_struct = visibility_from_source_to_points(
            base,
            xx.ravel(),
            yy.ravel(),
            np.full(xx.size, plane_z, dtype=float),
            row.x,
            row.y,
            row.z,
            house,
            geom,
        ).reshape(xx.shape)
        contrib = row.power * phase * np.clip(cos_target, 0.0, None) / np.maximum(r2, 1.0e-9)
        contrib *= vis_boundary.astype(float) * vis_struct
        plane[np.ix_(y_sel, x_sel)] += contrib

    return plane


def diffuse_irradiance_at_points(
    base,
    sources_df,
    x_targets,
    y_targets,
    z_targets,
    house,
    geom,
    sigma_deg=DEFAULT_SIGMA_DEG,
    cutoff_sigma=DEFAULT_CUTOFF_SIGMA,
):
    x_targets = np.asarray(x_targets, dtype=float)
    y_targets = np.asarray(y_targets, dtype=float)
    z_targets = np.asarray(z_targets, dtype=float)
    values = np.zeros_like(x_targets, dtype=float)
    if sources_df.empty:
        return values

    sigma_rad = math.radians(sigma_deg)
    alpha_max = min(math.radians(85.0), cutoff_sigma * sigma_rad)
    unique_z = np.unique(z_targets)

    for row in sources_df.itertuples(index=False):
        for z_target in unique_z:
            if z_target >= row.z - 1.0e-9:
                continue
            t_center = (z_target - row.z) / row.nz
            if t_center <= 0.0:
                continue
            x_center = row.x + t_center * row.nx
            y_center = row.y + t_center * row.ny
            radius = t_center * math.tan(alpha_max)
            mask = (
                (np.abs(z_targets - z_target) <= 1.0e-9)
                & (np.abs(x_targets - x_center) <= radius)
                & (np.abs(y_targets - y_center) <= radius)
            )
            if not np.any(mask):
                continue

            dx = x_targets[mask] - row.x
            dy = y_targets[mask] - row.y
            dz = z_targets[mask] - row.z
            r2 = dx * dx + dy * dy + dz * dz
            r = np.sqrt(r2)
            ux = dx / np.maximum(r, 1.0e-9)
            uy = dy / np.maximum(r, 1.0e-9)
            uz = dz / np.maximum(r, 1.0e-9)
            cos_alpha = ux * row.nx + uy * row.ny + uz * row.nz
            phase = phase_density_from_cos(cos_alpha, sigma_rad)
            if not np.any(phase > 0.0):
                continue
            cos_target = -uz
            vis_boundary = first_exit_t_from_inside(row.x, row.y, row.z, ux, uy, uz, house, geom) >= (r - 1.0e-6)
            vis_struct = visibility_from_source_to_points(
                base,
                x_targets[mask],
                y_targets[mask],
                z_targets[mask],
                row.x,
                row.y,
                row.z,
                house,
                geom,
            )
            values[mask] += row.power * phase * np.clip(cos_target, 0.0, None) / np.maximum(r2, 1.0e-9) * vis_boundary.astype(float) * vis_struct

    return values
