
import math
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from diffuse_transport_3d import build_roof_diffuse_sources, diffuse_irradiance_to_horizontal_plane

@dataclass
class LayerParam:
    name: str
    lai: float
    z_mid_m: float
    mean_leaf_angle_deg: float
    leaf_angle_sd_deg: float
    kd: float
    amax: float
    phi: float
    theta: float
    rd: float

def solar_position_simple(lat_deg: float, day_of_year: int, hour_local: float, lon_deg: float = 136.73388888888888):
    deg2rad = math.pi / 180.0
    rad2deg = 180.0 / math.pi
    delta_deg = 23.45 * math.sin((360.0 / 365.0 * (284.0 + day_of_year)) * deg2rad)
    delta_rad = delta_deg * deg2rad
    b = (360.0 / 365.0 * (day_of_year - 81.0)) * deg2rad
    e = 9.87 * math.sin(2.0 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)
    l_std = 15.0 * round(lon_deg / 15.0)
    t_solar = hour_local + (4.0 * (lon_deg - l_std) + e) / 60.0
    omega_rad = (15.0 * (t_solar - 12.0)) * deg2rad
    lat_rad = lat_deg * deg2rad

    alt_rad = math.asin(
        math.sin(delta_rad) * math.sin(lat_rad)
        + math.cos(delta_rad) * math.cos(lat_rad) * math.cos(omega_rad)
    )
    altitude_deg = alt_rad * rad2deg
    if altitude_deg <= 0.0:
        return 0.0, 180.0

    az_rad = math.atan2(
        -math.cos(delta_rad) * math.sin(omega_rad),
        math.sin(delta_rad) * math.cos(lat_rad) - math.cos(delta_rad) * math.sin(lat_rad) * math.cos(omega_rad),
    )
    azimuth_deg = (az_rad * rad2deg + 360.0) % 360.0
    return altitude_deg, azimuth_deg

def sun_vector(beta_deg: float, az_deg: float):
    beta = math.radians(beta_deg)
    az = math.radians(az_deg)
    # x positive = north, x negative direction = south
    # y positive = east,  y negative direction = west
    sx = math.cos(beta) * math.cos(az)
    sy = math.cos(beta) * math.sin(az)
    sz = math.sin(beta)
    return sx, sy, sz

def sample_leaf_normals(mean_inc_deg_from_horizontal: float, sd_deg: float, n: int, rng):
    inc = rng.normal(mean_inc_deg_from_horizontal, sd_deg, n)
    inc = np.clip(inc, 2.0, 88.0)
    zen = np.radians(inc)
    az = rng.uniform(0.0, 2.0 * np.pi, n)
    x = np.sin(zen) * np.cos(az)
    y = np.sin(zen) * np.sin(az)
    z = np.cos(zen)
    return np.column_stack([x, y, z])

def leaf_angle_stats(normals: np.ndarray, beta_deg: float, az_deg: float):
    sx, sy, sz = sun_vector(beta_deg, az_deg)
    s = np.array([sx, sy, sz])
    cosi = normals @ s
    pos = cosi > 0.0
    g_all = float(np.mean(np.clip(cosi, 0.0, None)))
    mean_pos = float(np.mean(cosi[pos])) if np.any(pos) else 0.0
    return g_all, mean_pos

def leaf_photosynthesis_nrh(i_abs, amax, phi, theta, rd):
    x = phi * i_abs + amax
    disc = np.maximum(x * x - 4.0 * theta * phi * i_abs * amax, 0.0)
    gross = (x - np.sqrt(disc)) / (2.0 * theta)
    return gross - rd

def compute_stem_density(area_m2, bed_count, edge_bed_count, mean_bed_len_m, interior_heads_per_3m, edge_heads_per_3m):
    interior_beds = bed_count - edge_bed_count
    total_heads = (interior_beds * mean_bed_len_m * interior_heads_per_3m / 3.0 +
                   edge_bed_count * mean_bed_len_m * edge_heads_per_3m / 3.0)
    return total_heads / area_m2

def shadow_by_linear_objects(y, z, sy, sz, y_centers, width, z_top, z_bottom):
    tol = 1.0e-9
    if abs(sy) <= tol and abs(sz) <= tol:
        return np.zeros_like(y, dtype=bool)
    y_left = y_centers - width / 2.0
    y_right = y_centers + width / 2.0
    shadow = np.zeros_like(y, dtype=bool)
    if abs(sy) > tol:
        t_left = (y_left[None, :] - y[:, None]) / sy
        z_left = z + t_left * sz
        cond_left = (t_left > 0.0) & (z_left >= z_bottom) & (z_left <= z_top)
        shadow |= cond_left.any(axis=1)
        t_right = (y_right[None, :] - y[:, None]) / sy
        z_right = z + t_right * sz
        cond_right = (t_right > 0.0) & (z_right >= z_bottom) & (z_right <= z_top)
        shadow |= cond_right.any(axis=1)
    if abs(sz) > tol:
        t_bottom = (z_bottom - z) / sz
        y_bottom = y[:, None] + t_bottom * sy
        cond_bottom = (t_bottom > 0.0) & (y_bottom >= y_left[None, :]) & (y_bottom <= y_right[None, :])
        shadow |= cond_bottom.any(axis=1)
    return shadow

def shadow_by_pillars(x, y, z, sx, sy, sz, x_centers, y_centers, pillar_thickness_x, pillar_width_y, z_top):
    tol = 1e-9
    if abs(sx) <= tol and abs(sy) <= tol:
        return np.zeros_like(x, dtype=bool)
    Xc, Yc = np.meshgrid(x_centers, y_centers, indexing='xy')
    Xc = Xc.ravel()
    Yc = Yc.ravel()
    x_left = Xc - pillar_thickness_x / 2.0
    x_right = Xc + pillar_thickness_x / 2.0
    y_low = Yc - pillar_width_y / 2.0
    y_high = Yc + pillar_width_y / 2.0
    shadow = np.zeros_like(x, dtype=bool)

    if abs(sx) > tol:
        for x_face in (x_left, x_right):
            t = (x_face[None, :] - x[:, None]) / sx
            y_int = y[:, None] + t * sy
            z_int = z + t * sz
            cond = (t > 0.0) & (y_int >= y_low[None, :]) & (y_int <= y_high[None, :]) & (z_int >= 0.0) & (z_int <= z_top)
            shadow |= cond.any(axis=1)

    if abs(sy) > tol:
        for y_face in (y_low, y_high):
            t = (y_face[None, :] - y[:, None]) / sy
            x_int = x[:, None] + t * sx
            z_int = z + t * sz
            cond = (t > 0.0) & (x_int >= x_left[None, :]) & (x_int <= x_right[None, :]) & (z_int >= 0.0) & (z_int <= z_top)
            shadow |= cond.any(axis=1)

    if abs(sz) > tol:
        t = (0.0 - z) / sz
        x_int = x[:, None] + t * sx
        y_int = y[:, None] + t * sy
        cond = (t > 0.0) & (x_int >= x_left[None, :]) & (x_int <= x_right[None, :]) & (y_int >= y_low[None, :]) & (y_int <= y_high[None, :])
        shadow |= cond.any(axis=1)

    return shadow

def roof_member_y_positions(house_width, unit_width, rows_per_slope):
    if rows_per_slope <= 0:
        return np.array([], dtype=float)
    positions = []
    half = unit_width / 2.0
    offsets = half * np.arange(1, rows_per_slope + 1, dtype=float) / (rows_per_slope + 1.0)
    n_units = int(round(house_width / unit_width))
    for j in range(n_units):
        y0 = j * unit_width
        y1 = y0 + unit_width
        positions.extend(y0 + offsets)
        positions.extend(y1 - offsets)
    return np.array(sorted(positions), dtype=float)

def span_roof_height(y, unit_width=7.5, gutter_z=4.2, ridge_z=6.075):
    y_local = np.mod(y, unit_width)
    half = unit_width / 2.0
    frac = 1.0 - np.abs(y_local - half) / half
    frac = np.clip(frac, 0.0, 1.0)
    return gutter_z + (ridge_z - gutter_z) * frac

def shadow_by_transverse_frame_members(x, y, z, sx, sy, sz, x_centers, x_thickness,
                                       column_y_centers, column_width_y,
                                       frame_depth_z, house_width, unit_width,
                                       gutter_z, ridge_z):
    tol = 1e-9
    shadow = np.zeros_like(x, dtype=bool)
    if x_centers.size == 0:
        return shadow

    half_x = x_thickness / 2.0
    y_expand = half_x * abs(sy / sx) if abs(sx) > tol else 0.0
    z_expand = half_x * abs(sz / sx) if abs(sx) > tol else 0.0
    roof_slope = (ridge_z - gutter_z) / (unit_width / 2.0)

    for x_center in np.asarray(x_centers, dtype=float):
        if abs(sx) > tol:
            t = (x_center - x) / sx
            valid = t > 0.0
            y_hit = y + t * sy
            z_hit = z + t * sz
        else:
            valid = np.abs(x - x_center) <= half_x
            y_hit = y
            z_hit = z

        if not np.any(valid):
            continue

        col_cond = (
            valid[:, None]
            & (z_hit[:, None] >= 0.0)
            & (z_hit[:, None] <= gutter_z + z_expand)
            & (np.abs(y_hit[:, None] - column_y_centers[None, :]) <= (column_width_y / 2.0 + y_expand))
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

def shadow_by_roof_longitudinal_members(y, z, sy, sz, y_centers, member_width_y,
                                        member_depth_z, unit_width, gutter_z, ridge_z):
    tol = 1e-9
    shadow = np.zeros_like(y, dtype=bool)
    if y_centers.size == 0:
        return shadow

    half_y = member_width_y / 2.0
    if abs(sy) > tol:
        z_expand = half_y * abs(sz / sy)
        for y_center in np.asarray(y_centers, dtype=float):
            t = (y_center - y) / sy
            z_hit = z + t * sz
            z_center = span_roof_height(np.array([y_center]), unit_width=unit_width, gutter_z=gutter_z, ridge_z=ridge_z)[0]
            cond = (t > 0.0) & (np.abs(z_hit - z_center) <= (member_depth_z / 2.0 + z_expand))
            shadow |= cond
    elif abs(sz) > tol:
        for y_center in np.asarray(y_centers, dtype=float):
            z_center = span_roof_height(np.array([y_center]), unit_width=unit_width, gutter_z=gutter_z, ridge_z=ridge_z)[0]
            t = (z_center - z) / sz
            cond = (t > 0.0) & (np.abs(y - y_center) <= half_y)
            shadow |= cond

    return shadow
def monotone_cubic_interp(xp: np.ndarray, fp: np.ndarray, x: np.ndarray):
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    x = np.asarray(x, dtype=float)
    h = np.diff(xp)
    delta = np.diff(fp) / h
    m = np.zeros_like(fp)
    m[0] = delta[0]
    m[-1] = delta[-1]

    for i in range(1, xp.size - 1):
        if delta[i - 1] * delta[i] <= 0.0:
            m[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    x_clip = np.clip(x, xp[0], xp[-1])
    idx = np.searchsorted(xp, x_clip, side="right") - 1
    idx = np.clip(idx, 0, xp.size - 2)
    x0 = xp[idx]
    x1 = xp[idx + 1]
    y0 = fp[idx]
    y1 = fp[idx + 1]
    m0 = m[idx]
    m1 = m[idx + 1]
    t = (x_clip - x0) / (x1 - x0)
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return h00 * y0 + h10 * (x1 - x0) * m0 + h01 * y1 + h11 * (x1 - x0) * m1


def angle_correction_etfe_gr_series(incidence_deg: np.ndarray):
    angles = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=float)
    factors = np.array([1.000, 1.000, 0.999, 0.996, 0.986, 0.960, 0.898], dtype=float)
    inc = np.clip(np.asarray(incidence_deg, dtype=float), angles[0], angles[-1])
    return monotone_cubic_interp(angles, factors, inc)


def roof_optical_taus_from_incidence(incidence_deg: np.ndarray, par_total_normal=0.903,
                                     direct_fraction=0.55, diffuse_fraction=0.45):
    k = angle_correction_etfe_gr_series(incidence_deg)
    total_tau = par_total_normal * k
    direct_tau = total_tau * direct_fraction
    beam_to_diffuse_tau = total_tau * diffuse_fraction
    return direct_tau, beam_to_diffuse_tau, total_tau

def envelope_entry_tau_triangular(x, y, z, sx, sy, sz, house_depth, house_width,
                                  unit_width=7.5, gutter_z=4.2, ridge_z=6.075,
                                  roof_tau=0.80, south_tau=0.50,
                                  east_west_tau=None,
                                  roof_par_total_normal=0.903,
                                  roof_direct_fraction=0.55,
                                  roof_diffuse_fraction=0.45):
    tol = 1e-9
    if east_west_tau is None:
        east_west_tau = roof_tau
    n = x.size
    t_best = np.full(n, np.inf, dtype=float)
    direct_tau = np.full(n, roof_tau, dtype=float)
    beam_diffuse_tau = np.zeros(n, dtype=float)
    total_tau = np.full(n, roof_tau, dtype=float)
    boundary = np.full(n, 'roof', dtype=object)
    incidence_deg = np.full(n, np.nan, dtype=float)

    # South wall (x=0), transparent at specified transmissivity.
    if sx < -tol:
        t = (0.0 - x) / sx
        y_int = y + t * sy
        z_int = z + t * sz
        roof_z = span_roof_height(y_int, unit_width=unit_width, gutter_z=gutter_z, ridge_z=ridge_z)
        cond = (t > 0.0) & (y_int >= 0.0) & (y_int <= house_width) & (z_int >= 0.0) & (z_int <= roof_z)
        upd = cond & (t < t_best)
        t_best[upd] = t[upd]
        direct_tau[upd] = south_tau
        beam_diffuse_tau[upd] = 0.0
        total_tau[upd] = south_tau
        incidence_deg[upd] = 0.0
        boundary[upd] = 'south_wall'

    # East / west walls are transparent like the roof, but only below eave height.
    if sy < -tol:
        t = (0.0 - y) / sy
        x_int = x + t * sx
        z_int = z + t * sz
        cond = (t > 0.0) & (x_int >= 0.0) & (x_int <= house_depth) & (z_int >= 0.0) & (z_int <= gutter_z)
        upd = cond & (t < t_best)
        t_best[upd] = t[upd]
        direct_tau[upd] = east_west_tau
        beam_diffuse_tau[upd] = 0.0
        total_tau[upd] = east_west_tau
        incidence_deg[upd] = 0.0
        boundary[upd] = 'west_wall'
    if sy > tol:
        t = (house_width - y) / sy
        x_int = x + t * sx
        z_int = z + t * sz
        cond = (t > 0.0) & (x_int >= 0.0) & (x_int <= house_depth) & (z_int >= 0.0) & (z_int <= gutter_z)
        upd = cond & (t < t_best)
        t_best[upd] = t[upd]
        direct_tau[upd] = east_west_tau
        beam_diffuse_tau[upd] = 0.0
        total_tau[upd] = east_west_tau
        incidence_deg[upd] = 0.0
        boundary[upd] = 'east_wall'

    # Roof planes for each span. The first hit among all spans is the envelope entry surface.
    m = (ridge_z - gutter_z) / (unit_width / 2.0)
    n_units = int(round(house_width / unit_width))
    norm = math.sqrt(1.0 + m * m)
    n_west = np.array([0.0, -m / norm, 1.0 / norm])
    n_east = np.array([0.0,  m / norm, 1.0 / norm])
    svec = np.array([sx, sy, sz])
    cos_inc_west = max(0.0, float(np.dot(n_west, svec)))
    cos_inc_east = max(0.0, float(np.dot(n_east, svec)))
    inc_west = math.degrees(math.acos(min(1.0, max(0.0, cos_inc_west))))
    inc_east = math.degrees(math.acos(min(1.0, max(0.0, cos_inc_east))))
    roof_direct_tau_west, roof_beam_diff_west, roof_total_west = roof_optical_taus_from_incidence(
        np.array([inc_west]), par_total_normal=roof_par_total_normal,
        direct_fraction=roof_direct_fraction, diffuse_fraction=roof_diffuse_fraction
    )
    roof_direct_tau_east, roof_beam_diff_east, roof_total_east = roof_optical_taus_from_incidence(
        np.array([inc_east]), par_total_normal=roof_par_total_normal,
        direct_fraction=roof_direct_fraction, diffuse_fraction=roof_diffuse_fraction
    )

    for j in range(n_units):
        y0 = j * unit_width
        yr = y0 + unit_width / 2.0
        y1 = y0 + unit_width

        denom_w = sz - m * sy
        if abs(denom_w) > tol:
            t = (gutter_z + m * (y - y0) - z) / denom_w
            y_int = y + t * sy
            x_int = x + t * sx
            cond = (t > 0.0) & (x_int >= 0.0) & (x_int <= house_depth) & (y_int >= y0 - 1e-9) & (y_int <= yr + 1e-9)
            upd = cond & (t < t_best)
            t_best[upd] = t[upd]
            direct_tau[upd] = roof_direct_tau_west[0]
            beam_diffuse_tau[upd] = roof_beam_diff_west[0]
            total_tau[upd] = roof_total_west[0]
            incidence_deg[upd] = inc_west
            boundary[upd] = 'roof_west_slope'

        denom_e = sz + m * sy
        if abs(denom_e) > tol:
            t = (gutter_z + m * (y1 - y) - z) / denom_e
            y_int = y + t * sy
            x_int = x + t * sx
            cond = (t > 0.0) & (x_int >= 0.0) & (x_int <= house_depth) & (y_int >= yr - 1e-9) & (y_int <= y1 + 1e-9)
            upd = cond & (t < t_best)
            t_best[upd] = t[upd]
            direct_tau[upd] = roof_direct_tau_east[0]
            beam_diffuse_tau[upd] = roof_beam_diff_east[0]
            total_tau[upd] = roof_total_east[0]
            incidence_deg[upd] = inc_east
            boundary[upd] = 'roof_east_slope'

    return direct_tau, beam_diffuse_tau, total_tau, incidence_deg, boundary

def run(output_dir="output"):
    base_module = sys.modules[__name__]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    house = {
        "latitude_deg": 35.0725,
        "longitude_deg": 136.73388888888888,
        "day_of_year": 80,  # 3/21
        "cultivation_width_m": 45.0,
        "depth_m": 51.0,
        "floor_area_m2": 45.0 * 51.0,
        "bed_count_assumed": 27,
        "edge_bed_count": 2,
        "mean_bed_length_m": 46.13,
        "interior_heads_per_3m": 14.0,
        "edge_heads_per_3m": 10.0,
        "roof_par_total_normal": 0.903,
        "roof_direct_fraction": 0.55,
        "roof_diffuse_fraction": 0.45,
        "cover_tau_diffuse": 0.90,
        "south_wall_tau_direct": 0.50,
    }
    stem_density = compute_stem_density(
        house["floor_area_m2"], house["bed_count_assumed"], house["edge_bed_count"],
        house["mean_bed_length_m"], house["interior_heads_per_3m"], house["edge_heads_per_3m"]
    )
    layer_lai = {"top": stem_density * 0.30, "mid": stem_density * 0.56, "bottom": stem_density * 0.33}
    layers = [
        LayerParam("top", layer_lai["top"], 3.40, 35.0, 12.0, 0.60, 31.0, 0.055, 0.78, 1.20),
        LayerParam("mid", layer_lai["mid"], 2.45, 50.0, 10.0, 0.72, 27.0, 0.052, 0.76, 1.00),
        LayerParam("bottom", layer_lai["bottom"], 1.50, 68.0, 12.0, 0.82, 20.0, 0.048, 0.74, 0.80),
    ]
    geom = {
        "unit_width": 7.5,
        "n_units": 6,
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
    geom["purlin_y_positions"] = roof_member_y_positions(
        house["cultivation_width_m"], geom["unit_width"], geom["purlin_rows_per_slope"]
    )

    nx, ny = 220, 190
    x = np.linspace(0.0, house["depth_m"], nx)
    y = np.linspace(0.0, house["cultivation_width_m"], ny)
    X, Y = np.meshgrid(x, y)
    xf = X.ravel()
    yf = Y.ravel()

    # User-specified non-cultivation zones at north/south ends.
    crop_mask = (X >= 1.5) & (X <= 48.0)
    # Keep the two service rectangles as blank non-cultivation area.
    service = (((X >= 2.0) & (X <= 3.0) & (Y >= 15.0) & (Y <= 17.5)) |
               ((X >= 2.0) & (X <= 3.0) & (Y >= 29.0) & (Y <= 31.5)))
    crop_mask &= ~service

    diffuse_tau_layer = {"top": 0.876, "mid": 0.878, "bottom": 0.881}
    rng_leaf = np.random.default_rng(42)
    normals = {layer.name: sample_leaf_normals(layer.mean_leaf_angle_deg, layer.leaf_angle_sd_deg, 1600, rng_leaf)
               for layer in layers}
    times = [8.0, 10.0, 12.0, 14.0, 16.0]
    summary_records = []
    snapshot_maps = {}

    for t in times:
        beta_deg, az_deg = solar_position_simple(
            house["latitude_deg"], house["day_of_year"], t, lon_deg=house["longitude_deg"]
        )
        sx, sy, sz = sun_vector(beta_deg, az_deg)
        sinb = math.sin(math.radians(beta_deg))
        outside_ppfd = 2000.0 * (sinb ** 1.12) if beta_deg > 0 else 0.0
        fdir = min(0.82, max(0.40, 0.45 + 0.42 * sinb)) if beta_deg > 0 else 0.0
        dir_h = outside_ppfd * fdir
        dif_h = outside_ppfd - dir_h
        ibeam_normal = dir_h / max(sinb, 1e-3) if beta_deg > 0 else 0.0
        diffuse_sources = (
            build_roof_diffuse_sources(
                base=base_module,
                house=house,
                geom=geom,
                sx=sx,
                sy=sy,
                sz=sz,
                source_x_step_m=6.0,
                source_y_step_m=2.5,
            )
            if beta_deg > 0 and sz > 0
            else pd.DataFrame(columns=["x", "y", "z", "power", "nx", "ny", "nz"])
        )

        lai_above = 0.0
        total_map = np.zeros_like(X, dtype=float)
        layer_maps = {}
        for layer in layers:
            g_all, mean_pos = leaf_angle_stats(normals[layer.name], beta_deg, az_deg) if beta_deg > 0 else (0.0, 0.0)
            kb = max(0.15, min(4.5, g_all / max(sinb, 0.05))) if beta_deg > 0 else 0.5
            lai_mid = lai_above + 0.5 * layer.lai

            if beta_deg > 0 and sz > 0:
                shadow = np.zeros_like(xf, dtype=bool)
                shadow |= shadow_by_linear_objects(yf, layer.z_mid_m, sy, sz, geom["gutter_y_positions"],
                                                   geom["gutter_width"], geom["gutter_ztop"],
                                                   geom["gutter_ztop"] - geom["gutter_depth"])
                shadow |= shadow_by_linear_objects(yf, layer.z_mid_m, sy, sz, geom["gutter_y_positions"],
                                                   geom["curtain1_width"], geom["curtain1_ztop"],
                                                   geom["curtain1_ztop"] - geom["curtain1_thickness"])
                shadow |= shadow_by_linear_objects(yf, layer.z_mid_m, sy, sz, geom["gutter_y_positions"],
                                                   geom["curtain2_width"], geom["curtain2_ztop"],
                                                   geom["curtain2_ztop"] - geom["curtain2_thickness"])
                shadow |= shadow_by_transverse_frame_members(
                    xf, yf, layer.z_mid_m, sx, sy, sz,
                    geom["pillar_x_positions"], geom["frame_member_thickness_x"],
                    geom["support_y_positions"], geom["frame_member_width_y"],
                    geom["frame_member_width_y"], house["cultivation_width_m"],
                    geom["unit_width"], geom["gutter_ztop"], geom["ridge_z"]
                )
                shadow |= shadow_by_transverse_frame_members(
                    xf, yf, layer.z_mid_m, sx, sy, sz,
                    geom["rafter_x_positions"], geom["rafter_thickness_x"],
                    geom["support_y_positions"], geom["rafter_width_y"],
                    geom["rafter_width_y"], house["cultivation_width_m"],
                    geom["unit_width"], geom["gutter_ztop"], geom["ridge_z"]
                )
                shadow |= shadow_by_roof_longitudinal_members(
                    yf, layer.z_mid_m, sy, sz,
                    geom["purlin_y_positions"], geom["purlin_width_y"], geom["purlin_depth_z"],
                    geom["unit_width"], geom["gutter_ztop"], geom["ridge_z"]
                )
                shadow |= shadow_by_pillars(xf, yf, layer.z_mid_m, sx, sy, sz,
                                            geom["pillar_x_positions"], geom["gutter_y_positions"],
                                            geom["frame_member_thickness_x"], geom["frame_member_width_y"], geom["gutter_ztop"])
                dir_tau_struct = (~shadow).astype(float).reshape(Y.shape)
                entry_direct_tau_flat, entry_beam_diffuse_tau_flat, entry_total_tau_flat, entry_inc_deg_flat, boundary_kind = envelope_entry_tau_triangular(
                    xf, yf, layer.z_mid_m, sx, sy, sz,
                    house_depth=house["depth_m"], house_width=house["cultivation_width_m"],
                    unit_width=geom["unit_width"], gutter_z=geom["gutter_ztop"], ridge_z=geom["ridge_z"],
                    south_tau=house["south_wall_tau_direct"],
                    east_west_tau=house["roof_par_total_normal"],
                    roof_par_total_normal=house["roof_par_total_normal"],
                    roof_direct_fraction=house["roof_direct_fraction"],
                    roof_diffuse_fraction=house["roof_diffuse_fraction"]
                )
                entry_direct_tau = entry_direct_tau_flat.reshape(Y.shape)
                entry_beam_diffuse_tau = entry_beam_diffuse_tau_flat.reshape(Y.shape)
                entry_total_tau = entry_total_tau_flat.reshape(Y.shape)
                entry_inc_deg = entry_inc_deg_flat.reshape(Y.shape)
                boundary_kind = boundary_kind.reshape(Y.shape)
            else:
                dir_tau_struct = np.zeros_like(X, dtype=float)
                entry_direct_tau = np.zeros_like(X, dtype=float)
                entry_beam_diffuse_tau = np.zeros_like(X, dtype=float)
                entry_total_tau = np.zeros_like(X, dtype=float)
                entry_inc_deg = np.full_like(X, np.nan, dtype=float)
                boundary_kind = np.full_like(X, "roof", dtype=object)

            sunlit_lai_open = math.exp(-kb * lai_above) * (1.0 - math.exp(-kb * layer.lai)) / kb if beta_deg > 0 else 0.0
            sunlit_lai_open = min(layer.lai, max(0.0, sunlit_lai_open))
            sunlit_lai = dir_tau_struct * sunlit_lai_open
            shaded_lai = layer.lai - sunlit_lai

            diffuse_beam_map = (
                diffuse_irradiance_to_horizontal_plane(
                    base=base_module,
                    sources_df=diffuse_sources,
                    x_coords=x,
                    y_coords=y,
                    plane_z=layer.z_mid_m,
                    house=house,
                    geom=geom,
                )
                if beta_deg > 0 and sz > 0 and not diffuse_sources.empty
                else np.zeros_like(X, dtype=float)
            )

            dir_mid = dir_h * entry_direct_tau * dir_tau_struct * math.exp(-kb * lai_mid)
            dif_mid = dif_h * house["cover_tau_diffuse"] * diffuse_tau_layer[layer.name] * math.exp(-layer.kd * lai_mid)
            beam_scattered_mid = dir_h * diffuse_beam_map * math.exp(-layer.kd * lai_mid)
            scattered = dir_mid * 0.18
            ppfd_shaded = dif_mid + beam_scattered_mid + scattered
            direct_on_leaf = ibeam_normal * entry_direct_tau * dir_tau_struct * mean_pos * math.exp(-kb * lai_mid) * 0.90
            ppfd_sunlit = ppfd_shaded + direct_on_leaf

            a_sun = leaf_photosynthesis_nrh(ppfd_sunlit, layer.amax, layer.phi, layer.theta, layer.rd)
            a_sh = leaf_photosynthesis_nrh(ppfd_shaded, layer.amax, layer.phi, layer.theta, layer.rd)
            a_layer = a_sun * sunlit_lai + a_sh * shaded_lai
            a_layer[~crop_mask] = np.nan

            layer_maps[layer.name] = a_layer
            total_map += np.nan_to_num(a_layer, nan=0.0)
            lai_above += layer.lai

            summary_records.append({
                "time_h": t,
                "layer": layer.name,
                "solar_elev_deg": beta_deg,
                "solar_az_deg": az_deg,
                "outside_ppfd": outside_ppfd,
                "mean_assim": float(np.nanmean(a_layer)),
                "min_assim": float(np.nanmin(a_layer)),
                "max_assim": float(np.nanmax(a_layer)),
                "sunlit_frac_area": float(np.nanmean(dir_tau_struct[crop_mask])),
                "southwall_entry_frac": float(np.nanmean((boundary_kind[crop_mask] == "south_wall").astype(float))),
                "roof_entry_frac": float(np.nanmean(np.char.startswith(boundary_kind[crop_mask].astype(str), "roof").astype(float))),
                "mean_roof_incidence_deg": float(np.nanmean(entry_inc_deg[np.char.startswith(boundary_kind.astype(str), "roof") & crop_mask])) if np.any(np.char.startswith(boundary_kind.astype(str), "roof") & crop_mask) else float("nan"),
                "mean_entry_total_tau": float(np.nanmean(entry_total_tau[crop_mask])),
                "mean_entry_direct_tau": float(np.nanmean(entry_direct_tau[crop_mask])),
            })

        total_map[~crop_mask] = np.nan
        snapshot_maps[t] = {"total": total_map, **layer_maps, "solar_elev_deg": beta_deg, "solar_az_deg": az_deg}

    df = pd.DataFrame(summary_records)
    df.to_csv(out / "snapshot_summary.csv", index=False)

    total_summary = (
        df.groupby("time_h")["mean_assim"].sum().rename("total_mean_assim").reset_index()
    )
    total_summary.to_csv(out / "snapshot_total_summary.csv", index=False)

    vmax = max(np.nanmax(snapshot_maps[t]["total"]) for t in times)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    axes = axes.ravel()
    for ax, t in zip(axes, times):
        arr = snapshot_maps[t]["total"]
        im = ax.imshow(arr, origin="lower", extent=[0, house["depth_m"], 0, house["cultivation_width_m"]],
                       vmin=0, vmax=vmax, aspect="auto")
        for gy in geom["gutter_y_positions"]:
            ax.axhline(gy, color="white", lw=0.6, alpha=0.45)
        ax.axvline(1.5, color="cyan", lw=0.8, alpha=0.6)
        ax.axvline(48.0, color="cyan", lw=0.8, alpha=0.6)
        ax.set_title(f"{t:02.0f}:00 total canopy")
        ax.set_xlabel("x [m] (positive = north, negative direction = south)")
        ax.set_ylabel("y [m] (positive = east)")
    axes[-1].axis("off")
    cb = fig.colorbar(im, ax=axes[:-1], shrink=0.90)
    cb.set_label("net photosynthesis [μmol CO2 m^-2 ground s^-1]")
    fig.savefig(out / "topview_photosynthesis_snapshots.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    t = 12.0
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, name in zip(axes.ravel(), ["top", "mid", "bottom", "total"]):
        arr = snapshot_maps[t][name]
        vmax_name = np.nanmax(snapshot_maps[t]["total"]) if name == "total" else np.nanmax(snapshot_maps[t][name])
        im = ax.imshow(arr, origin="lower", extent=[0, house["depth_m"], 0, house["cultivation_width_m"]],
                       vmin=0, vmax=vmax_name, aspect="auto")
        for gy in geom["gutter_y_positions"]:
            ax.axhline(gy, color="white", lw=0.6, alpha=0.45)
        ax.axvline(1.5, color="cyan", lw=0.8, alpha=0.6)
        ax.axvline(48.0, color="cyan", lw=0.8, alpha=0.6)
        ax.set_title(f"12:00 {name}")
        ax.set_xlabel("x [m] (positive = north, negative direction = south)")
        ax.set_ylabel("y [m] (positive = east)")
        cb = fig.colorbar(im, ax=ax, shrink=0.88)
        cb.set_label("μmol CO2 m^-2 s^-1")
    fig.savefig(out / "topview_layer_breakdown_1200.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    notes = f"""3/21, oriented layout, east/west walls transparent
triangular roof envelope with 5-sun slope (5/10)
cultivation mask: x in [1.50, 48.00] m only, service rectangles blank
south wall direct transmittance = {house['south_wall_tau_direct']:.2f}
roof PAR total normal transmittance = {house['roof_par_total_normal']:.3f}
roof direct fraction = {house['roof_direct_fraction']:.2f}
roof beam-to-diffuse fraction = {house['roof_diffuse_fraction']:.2f}
ridge height from 5-sun slope = {geom['ridge_z']:.3f} m
curtain1 z-range = {geom['curtain1_ztop']-geom['curtain1_thickness']:.2f} to {geom['curtain1_ztop']:.2f} m
curtain2 z-range = {geom['curtain2_ztop']-geom['curtain2_thickness']:.2f} to {geom['curtain2_ztop']:.2f} m
gutter z-range = {geom['gutter_ztop']-geom['gutter_depth']:.2f} to {geom['gutter_ztop']:.2f} m
frame shadow members = 100x50 mm at 3.0 m pitch
rafter shadow members = 60 mm at 0.50 m pitch
purlin shadow members = C60x30x10, modelled as 4 rows per roof slope
pillar y-width = {geom['frame_member_width_y']:.2f} m
pillar x-thickness = {geom['frame_member_thickness_x']:.2f} m
"""
    (out / 'notes.txt').write_text(notes, encoding='utf-8')

if __name__ == "__main__":
    run("output")
