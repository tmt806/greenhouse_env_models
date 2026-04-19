from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='snapshot_summary.csv から、各層の面積平均光合成速度の時間変化を描画する。'
    )
    parser.add_argument('csv', help='入力CSV (snapshot_summary.csv)')
    parser.add_argument('-o', '--output', default='layer_time_series_area_average.png', help='出力PNG')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    required = {'time_h', 'layer', 'mean_assim'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'CSVに必要列がありません: {sorted(missing)}')

    layer_map = {
        'top': '上層',
        'mid': '中層',
        'bottom': '下層',
    }

    pivot = df.pivot(index='time_h', columns='layer', values='mean_assim').sort_index()

    plt.figure(figsize=(8, 5))
    for layer in ['top', 'mid', 'bottom']:
        if layer in pivot.columns:
            plt.plot(pivot.index, pivot[layer], marker='o', label=layer_map.get(layer, layer))

    plt.xlabel('Time [h]')
    plt.ylabel('Area-mean canopy net photosynthesis [μmol CO2 m$^{-2}$ ground s$^{-1}$]')
    plt.title('Layer-wise area-mean photosynthesis')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
