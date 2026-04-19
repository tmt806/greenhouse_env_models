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


def run(output_dir="output_floor_daily_direct"):
    base = load_base_module()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    angles_anchor = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=float)
    k_anchor = np.array([1.000, 1.000, 0.999, 0.996, 0.986, 0.960, 0.898], dtype=float)
    angles = np.linspace(0.0, 60.0, 241)
    k = base.angle_correction_etfe_gr_series(angles)

    df = pd.DataFrame({"incidence_deg": angles, "k_theta": k})
    df.to_csv(out / "roof_angle_correction_k_theta.csv", index=False)

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(angles, k, color="tab:blue", lw=2.0, label="Monotone cubic K(theta)")
    plt.scatter(angles_anchor, k_anchor, color="tab:red", s=36, zorder=3, label="Anchor values")
    plt.xlabel("Incidence angle theta [deg]")
    plt.ylabel("K(theta)")
    plt.title("Roof film angle correction")
    plt.grid(True, alpha=0.3)
    plt.xlim(0.0, 60.0)
    plt.ylim(min(k) - 0.01, 1.01)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "roof_angle_correction_k_theta.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    run()
