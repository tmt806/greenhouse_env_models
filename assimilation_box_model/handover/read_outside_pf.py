#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_OUTSIDE_PF_ROOT = Path.home() / 'Library/CloudStorage/GoogleDrive-soi.toi.chi@gmail.com/マイドライブ/greenhouse_log/outside_pf'


def read_outside_pf_db(db_path: str | Path) -> pd.DataFrame:
    query = """
        SELECT
            "年月日" AS ymd,
            "時間" AS time,
            "気温" AS temperature_c,
            "絶対湿度" AS absolute_humidity_g_m3,
            "ＣＯ２濃度" AS co2_ppm
        FROM AGLOG
    """
    with sqlite3.connect(Path(db_path)) as con:
        df = pd.read_sql_query(query, con)
    if len(df) == 0:
        return pd.DataFrame(columns=['temperature_c', 'absolute_humidity_g_m3', 'co2_ppm'])

    df['timestamp'] = pd.to_datetime(
        df['ymd'].astype(str).str.strip() + ' ' + df['time'].astype(str).str.strip(),
        errors='coerce',
    )
    df = df.dropna(subset=['timestamp']).set_index('timestamp').sort_index()
    for col in ['temperature_c', 'absolute_humidity_g_m3', 'co2_ppm']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df[['temperature_c', 'absolute_humidity_g_m3', 'co2_ppm']].groupby(level=0).mean().sort_index()


def load_outside_pf_range(root: str | Path, start_date: str, end_date: str) -> pd.DataFrame:
    root_path = Path(root)
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    frames: list[pd.DataFrame] = []

    for db_path in sorted(root_path.rglob('DATABASE_*.DB')):
        if db_path.name == 'DATABASE_MASTER.DB':
            continue
        try:
            df = read_outside_pf_db(db_path)
        except sqlite3.DatabaseError:
            continue
        if len(df) == 0:
            continue
        day = pd.Series(df.index.date, index=df.index)
        df = df.loc[(day >= start) & (day <= end)]
        if len(df) > 0:
            frames.append(df)

    if not frames:
        raise ValueError(f'No outside_pf rows found in {root_path} between {start_date} and {end_date}')

    out = pd.concat(frames).groupby(level=0).mean().sort_index()
    out['date'] = out.index.strftime('%Y-%m-%d')
    return out


def plot_outside_pf(df: pd.DataFrame, output_png: str | Path, *, title: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True, constrained_layout=True)
    series_info = [
        ('temperature_c', 'Temperature [C]'),
        ('co2_ppm', 'CO2 [ppm]'),
        ('absolute_humidity_g_m3', 'Absolute humidity [g m$^{-3}$]'),
    ]
    colors = {'2026-04-05': '#1f77b4', '2026-04-06': '#d62728', '2026-04-07': '#2ca02c'}

    for ax, (col, ylabel) in zip(axes, series_info):
        for day, g in df.groupby('date'):
            ax.plot(g.index, g[col], label=day, linewidth=1.2, color=colors.get(day))
        ax.set_ylabel(ylabel)
        ax.grid(True, which='major', alpha=0.35)
        ax.legend(loc='upper right')

    day0 = df.index.min().normalize()
    day1 = df.index.max().normalize() + pd.Timedelta(days=1)
    axes[-1].set_xlim(day0, day1)
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    axes[-1].set_xlabel('Time')
    fig.suptitle(title)

    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='Read outside_pf SQLite DB files and plot temperature, CO2, and absolute humidity.')
    ap.add_argument('--outside-pf-root', default=str(DEFAULT_OUTSIDE_PF_ROOT), help='Root directory of outside_pf DB folders.')
    ap.add_argument('--start-date', required=True, help='Start date, e.g. 2026-04-05.')
    ap.add_argument('--end-date', required=True, help='End date, e.g. 2026-04-07.')
    ap.add_argument('--output-dir', default='outside_pf_plots', help='Output directory.')
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    output_dir = Path(args.output_dir)
    df = load_outside_pf_range(args.outside_pf_root, args.start_date, args.end_date)

    csv_path = output_dir / f'outside_pf_{args.start_date.replace("-", "")}_{args.end_date.replace("-", "")}.csv'
    png_path = output_dir / f'outside_pf_{args.start_date.replace("-", "")}_{args.end_date.replace("-", "")}_temperature_co2_ah.png'

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, encoding='utf-8-sig')
    plot_outside_pf(df, png_path, title=f'outside_pf {args.start_date} to {args.end_date}')

    print(f'saved: {csv_path}')
    print(f'saved: {png_path}')


if __name__ == '__main__':
    main()
