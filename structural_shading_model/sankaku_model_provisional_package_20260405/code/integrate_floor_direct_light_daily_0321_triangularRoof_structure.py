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
    return geom


def direct_lit_floor(base, house, geom, xf, yf, z_floor, sx, sy, sz):
    shadow = np.zeros_like(xf, dtype=bool)
    shadow |= base.shadow_by_linear_objects(
        yf, z_floor, sy, sz, geom["gutter_y_positions"],
        geom["gutter_width"], geom["gutter_ztop"], geom["gutter_ztop"] - geom["gutter_depth"]
    )
    shadow |= base.shadow_by_linear_objects(
        yf, z_floor, sy, sz, geom["gutter_y_positions"],
        geom["curtain1_width"], geom["curtain1_ztop"], geom["curtain1_ztop"] - geom["curtain1_thickness"]
    )
    shadow |= base.shadow_by_linear_objects(
        yf, z_floor, sy, sz, geom["gutter_y_positions"],
        geom["curtain2_width"], geom["curtain2_ztop"], geom["curtain2_ztop"] - geom["curtain2_thickness"]
    )
    shadow |= base.shadow_by_transverse_frame_members(
        xf, yf, z_floor, sx, sy, sz,
        geom["pillar_x_positions"], geom["frame_member_thickness_x"],
        geom["support_y_positions"], geom["frame_member_width_y"],
        geom["frame_member_width_y"], house["cultivation_width_m"],
        geom["unit_width"], geom["gutter_ztop"], geom["ridge_z"]
    )
    shadow |= base.shadow_by_transverse_frame_members(
        xf, yf, z_floor, sx, sy, sz,
        geom["rafter_x_positions"], geom["rafter_thickness_x"],
        geom["support_y_positions"], geom["rafter_width_y"],
        geom["rafter_width_y"], house["cultivation_width_m"],
        geom["unit_width"], geom["gutter_ztop"], geom["ridge_z"]
    )
    shadow |= base.shadow_by_roof_longitudinal_members(
        yf, z_floor, sy, sz,
        geom["purlin_y_positions"], geom["purlin_width_y"], geom["purlin_depth_z"],
        geom["unit_width"], geom["gutter_ztop"], geom["ridge_z"]
    )
    shadow |= base.shadow_by_pillars(
        xf, yf, z_floor, sx, sy, sz,
        geom["pillar_x_positions"], geom["gutter_y_positions"],
        geom["frame_member_thickness_x"], geom["frame_member_width_y"], geom["gutter_ztop"]
    )
    return (~shadow).astype(float)


def run(output_dir="output_floor_daily_direct"):
    base = load_base_module()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    house = {
        "latitude_deg": 35.0725,
        "longitude_deg": 136.73388888888888,
        "day_of_year": 80,
        "cultivation_width_m": 45.0,
        "depth_m": 51.0,
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
    z_floor = 0.0

    cumulative = np.zeros_like(X, dtype=float)
    records = []
    daylight_minutes = 0

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

            lit = direct_lit_floor(base, house, geom, xf, yf, z_floor, sx, sy, sz).reshape(X.shape)
            intensity = max(0.0, math.sin(math.radians(beta_deg)))
            cumulative += lit * intensity
            daylight_minutes += 1
            records.append(
                {
                    "time_h": local_time,
                    "solar_elev_deg": beta_deg,
                    "solar_az_deg": az_deg,
                    "relative_direct_intensity": intensity,
                    "lit_fraction_floor": float(np.mean(lit)),
                }
            )

    normalized = cumulative / np.max(cumulative) if np.max(cumulative) > 0.0 else cumulative

    pd.DataFrame(records).to_csv(out / "floor_direct_minutely_timeseries_0321.csv", index=False)
    pd.DataFrame(
        {
            "x_m": X.ravel(),
            "y_m": Y.ravel(),
            "daily_relative_direct_light": cumulative.ravel(),
            "daily_relative_direct_light_normalized": normalized.ravel(),
        }
    ).to_csv(out / "floor_direct_daily_distribution_0321.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "day_of_year": house["day_of_year"],
                "daylight_minutes": daylight_minutes,
                "mean_daily_relative_direct_light": float(np.mean(cumulative)),
                "max_daily_relative_direct_light": float(np.max(cumulative)),
                "mean_daily_relative_direct_light_normalized": float(np.mean(normalized)),
            }
        ]
    )
    summary.to_csv(out / "floor_direct_daily_summary_0321.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    maps = [
        (cumulative, "Accumulated relative direct light"),
        (normalized, "Normalized accumulated direct light"),
    ]
    for ax, (arr, title) in zip(axes, maps):
        im = ax.imshow(
            arr,
            origin="lower",
            extent=[0, house["depth_m"], 0, house["cultivation_width_m"]],
            aspect="auto",
            cmap="inferno",
        )
        for gy in geom["gutter_y_positions"]:
            ax.axhline(gy, color="white", lw=0.5, alpha=0.35)
        ax.set_title(title)
        ax.set_xlabel("x [m] (north positive)")
        ax.set_ylabel("y [m] (east positive)")
        cb = fig.colorbar(im, ax=ax, shrink=0.88)
        cb.set_label("relative light")
    fig.savefig(out / "floor_direct_daily_distribution_0321.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
