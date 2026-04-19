from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def roof_height(y, unit_width=7.5, gutter_z=4.2, ridge_z=6.075):
    y_local = np.mod(y, unit_width)
    half = unit_width / 2.0
    frac = 1.0 - np.abs(y_local - half) / half
    frac = np.clip(frac, 0.0, 1.0)
    return gutter_z + (ridge_z - gutter_z) * frac


def roof_member_y_positions(house_width, unit_width, rows_per_slope):
    half = unit_width / 2.0
    offsets = half * np.arange(1, rows_per_slope + 1, dtype=float) / (rows_per_slope + 1.0)
    positions = []
    n_units = int(round(house_width / unit_width))
    for unit_idx in range(n_units):
        y0 = unit_idx * unit_width
        y1 = y0 + unit_width
        positions.extend(y0 + offsets)
        positions.extend(y1 - offsets)
    return np.array(sorted(positions), dtype=float)


def draw_dimension(ax, start, end, offset, text, orientation="horizontal", text_offset=0.25):
    if orientation == "horizontal":
        x0, x1 = start, end
        y = offset
        ax.annotate("", xy=(x0, y), xytext=(x1, y), arrowprops=dict(arrowstyle="<->", lw=0.8, color="0.2"))
        ax.text((x0 + x1) / 2.0, y + text_offset, text, ha="center", va="bottom", fontsize=8)
    else:
        y0, y1 = start, end
        x = offset
        ax.annotate("", xy=(x, y0), xytext=(x, y1), arrowprops=dict(arrowstyle="<->", lw=0.8, color="0.2"))
        ax.text(x + text_offset, (y0 + y1) / 2.0, text, ha="left", va="center", fontsize=8, rotation=90)


def run(output_dir="output_structure_drawings"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    house_depth = 51.0
    house_width = 45.0
    unit_width = 7.5
    gutter_z = 4.2
    ridge_z = 6.075
    gutter_y = np.arange(unit_width, house_width - 1e-9, unit_width)
    support_y = np.arange(0.0, house_width + 1e-9, unit_width)
    frame_x = np.arange(0.0, house_depth + 1e-9, 3.0)
    rafter_x = np.arange(0.0, house_depth + 1e-9, 0.5)
    purlin_y = roof_member_y_positions(house_width, unit_width, 4)
    roof_strip_y = np.arange(unit_width * 0.5, house_width, unit_width)
    roof_strip_width = 0.10

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])
    ax_plan = fig.add_subplot(gs[0, :])
    ax_cross = fig.add_subplot(gs[1, 0])
    ax_long = fig.add_subplot(gs[1, 1])

    # Plan view
    ax_plan.add_patch(Rectangle((0, 0), house_depth, house_width, fill=False, lw=2.0, ec="0.1"))
    for x in frame_x:
        ax_plan.plot([x, x], [0, house_width], color="0.65", lw=0.6)
    for y in gutter_y:
        ax_plan.plot([0, house_depth], [y, y], color="0.2", lw=1.2)
    for idx, y in enumerate(purlin_y):
        label = "Purlin (C steel)" if idx == 0 else None
        ax_plan.plot([0, house_depth], [y, y], color="tab:blue", lw=0.7, ls="--", alpha=0.65, label=label)
    for idx, y in enumerate(roof_strip_y):
        label = "Roof strip 100 mm" if idx == 0 else None
        ax_plan.add_patch(
            Rectangle(
                (0.0, y - roof_strip_width / 2.0),
                house_depth,
                roof_strip_width,
                fill=False,
                lw=1.2,
                ec="tab:orange",
                label=label,
            )
        )
    for y in support_y:
        ax_plan.scatter(frame_x, np.full_like(frame_x, y), s=10, color="0.1")
    ax_plan.set_title("Plan View")
    ax_plan.set_xlim(-2.5, house_depth + 2.5)
    ax_plan.set_ylim(-4.0, house_width + 4.0)
    ax_plan.set_aspect("equal")
    ax_plan.set_xlabel("x [m]")
    ax_plan.set_ylabel("y [m]")
    ax_plan.text(1.0, house_width + 2.2, "Triangular-roof greenhouse frame layout", fontsize=11, weight="bold")
    ax_plan.text(1.0, -2.8, "Frame pitch: 3.0 m | Span width: 7.5 m | 6 spans | Depth: 51.0 m", fontsize=8)
    ax_plan.text(30.0, -2.8, "Blue dashed lines = purlin (C60 x 30 x 10)", fontsize=8, color="tab:blue")
    ax_plan.text(30.0, -4.1, "Orange bands = roof strip 100 mm @ y=3.75, 11.25, ...", fontsize=8, color="tab:orange")
    draw_dimension(ax_plan, 0.0, house_depth, house_width + 1.0, "51.0 m", "horizontal")
    draw_dimension(ax_plan, 0.0, house_width, house_depth + 1.0, "45.0 m", "vertical")
    draw_dimension(ax_plan, 0.0, unit_width, -1.0, "7.5 m", "horizontal", text_offset=-1.2)

    # Cross section
    y_plot = np.linspace(0.0, house_width, 1200)
    z_plot = roof_height(y_plot, unit_width=unit_width, gutter_z=gutter_z, ridge_z=ridge_z)
    ax_cross.plot(y_plot, z_plot, color="0.1", lw=2.0)
    ax_cross.plot([0, house_width], [0, 0], color="0.1", lw=1.2)
    for y in support_y:
        ax_cross.plot([y, y], [0, gutter_z], color="0.35", lw=1.0)
    for y in gutter_y:
        ax_cross.add_patch(Rectangle((y - 0.30, gutter_z - 0.10), 0.60, 0.10, fill=False, lw=1.0, ec="0.25"))
        ax_cross.add_patch(Rectangle((y - 0.35, 3.85), 0.70, 0.05, fill=False, lw=0.8, ec="0.45"))
        ax_cross.add_patch(Rectangle((y - 0.35, 3.55), 0.70, 0.05, fill=False, lw=0.8, ec="0.45"))
    purlin_z = roof_height(purlin_y, unit_width=unit_width, gutter_z=gutter_z, ridge_z=ridge_z)
    ax_cross.scatter(purlin_y, purlin_z, s=10, color="tab:blue", zorder=3)
    roof_strip_z = roof_height(roof_strip_y, unit_width=unit_width, gutter_z=gutter_z, ridge_z=ridge_z)
    for y, z in zip(roof_strip_y, roof_strip_z):
        ax_cross.add_patch(
            Rectangle(
                (y - roof_strip_width / 2.0, z - 0.06),
                roof_strip_width,
                0.12,
                fill=False,
                lw=1.2,
                ec="tab:orange",
                zorder=4,
            )
        )
    ax_cross.set_title("Transverse Section")
    ax_cross.set_xlim(-1.5, house_width + 1.5)
    ax_cross.set_ylim(-0.5, ridge_z + 1.2)
    ax_cross.set_aspect("equal")
    ax_cross.set_xlabel("y [m]")
    ax_cross.set_ylabel("z [m]")
    draw_dimension(ax_cross, 0.0, house_width, -0.3, "45.0 m", "horizontal", text_offset=-0.55)
    draw_dimension(ax_cross, 0.0, gutter_z, house_width + 0.5, "4.2 m eave", "vertical")
    draw_dimension(ax_cross, 0.0, ridge_z, house_width + 2.0, "6.075 m ridge", "vertical")
    draw_dimension(ax_cross, 0.0, unit_width, ridge_z + 0.35, "7.5 m span", "horizontal")
    ax_cross.text(1.0, ridge_z + 0.8, "Purlin rows: 4 / slope", fontsize=8)
    ax_cross.text(25.0, ridge_z + 0.8, "Gutter + curtain storage shown", fontsize=8)
    ax_cross.text(1.0, ridge_z + 0.45, "Roof strip 100 mm at y=3.75, 11.25, ...", fontsize=8, color="tab:orange")
    left_idx = 2
    right_idx = -3
    ax_cross.annotate(
        "Purlin (C steel)\nC60 x 30 x 10",
        xy=(purlin_y[left_idx], purlin_z[left_idx]),
        xytext=(2.0, ridge_z - 0.15),
        fontsize=8,
        color="tab:blue",
        arrowprops=dict(arrowstyle="->", lw=0.9, color="tab:blue"),
    )
    ax_cross.annotate(
        "Purlin (C steel)",
        xy=(purlin_y[right_idx], purlin_z[right_idx]),
        xytext=(33.5, ridge_z - 0.35),
        fontsize=8,
        color="tab:blue",
        arrowprops=dict(arrowstyle="->", lw=0.9, color="tab:blue"),
    )

    # Longitudinal section
    ax_long.plot([0, house_depth], [0, 0], color="0.1", lw=1.2)
    for x in rafter_x:
        ax_long.plot([x, x], [gutter_z - 0.12, ridge_z + 0.02], color="0.82", lw=0.25)
    for x in frame_x:
        ax_long.plot([x, x], [0.0, gutter_z], color="0.25", lw=1.1)
    ax_long.plot([0, house_depth], [gutter_z, gutter_z], color="0.35", lw=1.0)
    ax_long.plot([0, house_depth], [ridge_z, ridge_z], color="0.2", lw=0.9, ls="--")
    ax_long.set_title("Longitudinal Section")
    ax_long.set_xlim(-2.5, house_depth + 2.5)
    ax_long.set_ylim(-0.5, ridge_z + 1.2)
    ax_long.set_aspect("auto")
    ax_long.set_xlabel("x [m]")
    ax_long.set_ylabel("z [m]")
    draw_dimension(ax_long, 0.0, 3.0, -0.15, "3.0 m frame pitch", "horizontal", text_offset=-0.5)
    draw_dimension(ax_long, 0.0, 0.5, ridge_z + 0.35, "0.5 m rafter pitch", "horizontal")
    draw_dimension(ax_long, 0.0, house_depth, ridge_z + 0.9, "51.0 m depth", "horizontal")
    ax_long.text(1.0, ridge_z + 0.55, "Frame member: 100 x 50 mm", fontsize=8)
    ax_long.text(1.0, ridge_z + 0.2, "Rafter: 33 mm vinipet @ 0.5 m", fontsize=8)
    ax_long.text(1.0, gutter_z - 0.45, "Purlin (C steel): C60 x 30 x 10", fontsize=8, color="tab:blue")
    ax_long.text(1.0, gutter_z - 0.8, "Roof strip: 100 mm, running along x", fontsize=8, color="tab:orange")

    fig.suptitle("Greenhouse Structural Schematic", fontsize=14, weight="bold")
    png_path = out / "greenhouse_structure_schematic.png"
    pdf_path = out / "greenhouse_structure_schematic.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
