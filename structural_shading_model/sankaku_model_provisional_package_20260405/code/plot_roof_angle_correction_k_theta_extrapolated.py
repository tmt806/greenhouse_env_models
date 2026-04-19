import importlib.util
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


def run(output_dir="output_floor_daily_direct"):
    base = load_base_module()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    angles_anchor = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=float)
    k_anchor = np.array([1.000, 1.000, 0.999, 0.996, 0.986, 0.960, 0.898], dtype=float)
    angles = np.linspace(0.0, 85.0, 341)
    k_ext = extrapolated_k_theta(base, angles)

    pd.DataFrame({"incidence_deg": angles, "k_theta_extrapolated": k_ext}).to_csv(
        out / "roof_angle_correction_k_theta_extrapolated.csv", index=False
    )

    plt.figure(figsize=(7.6, 4.8))
    plt.plot(angles, k_ext, color="tab:blue", lw=2.2, label="Extrapolated K(theta)")
    plt.plot(
        angles[angles <= 60.0],
        base.angle_correction_etfe_gr_series(angles[angles <= 60.0]),
        color="tab:green",
        lw=1.8,
        alpha=0.85,
        label="Base curve (0-60 deg)",
    )
    plt.scatter(angles_anchor, k_anchor, color="tab:red", s=36, zorder=3, label="Anchor values")
    plt.axvline(60.0, color="gray", lw=1.0, ls="--", alpha=0.7)
    plt.text(61.0, 0.97, "extrapolated", color="gray", fontsize=9)
    plt.xlabel("Incidence angle theta [deg]")
    plt.ylabel("K(theta)")
    plt.title("Roof film angle correction with extrapolation")
    plt.grid(True, alpha=0.3)
    plt.xlim(0.0, 85.0)
    plt.ylim(0.0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "roof_angle_correction_k_theta_extrapolated.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    run()
