from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run(
    csv_path="output_virtual_sensors_gaussian/virtual_sensor_timeseries.csv",
    output_path="output_virtual_sensors_gaussian/virtual_sensor_timeseries_with_outdoor.png",
):
    df = pd.read_csv(csv_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    seasons = ["winter_solstice", "spring_equinox", "summer_solstice"]
    z_levels = [3.0, 1.5]
    colors = plt.cm.tab10(np.linspace(0, 1, 9))

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True, sharex=True, sharey=True)
    for row_idx, season in enumerate(seasons):
        season_df = df[df["season"] == season]
        outdoor = season_df[["time_h", "outdoor_relative_direct"]].drop_duplicates().sort_values("time_h")
        for col_idx, z in enumerate(z_levels):
            ax = axes[row_idx, col_idx]
            sub = season_df[season_df["z_m"] == z]
            for color_idx, sensor_id in enumerate(sorted(sub["sensor_id"].unique())):
                ss = sub[sub["sensor_id"] == sensor_id]
                ax.plot(ss["time_h"], ss["indoor_relative_signal"], lw=1.0, color=colors[color_idx % len(colors)], label=sensor_id)
            ax.plot(
                outdoor["time_h"],
                outdoor["outdoor_relative_direct"],
                color="black",
                lw=1.6,
                ls="--",
                label="outdoor" if row_idx == 0 and col_idx == 0 else None,
            )
            ax.set_title(f"{season.replace('_', ' ')} | z={z:.1f} m")
            ax.set_xlabel("time [h]")
            ax.set_ylabel("relative light")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=10, fontsize=8)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
