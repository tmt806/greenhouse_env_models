# sankaku_model_provisional_package_20260405

## 内容
- `sankaku_model_handoff_20260405.md`
  - 次チャットへの引継ぎ資料
- `code/`
  - `generate_topview_photosynthesis_snapshots_0321_triangularRoof5sun_angleDep_blankMargins_southwall50.py`
    - 現在の暫定本体コード
  - `plot_layer_time_series_from_snapshot_summary.py`
    - `snapshot_summary.csv` から各層の面積平均光合成速度の時間変化を描画するコード
- `inputs/`
  - `提出レイアウト20210507.pdf`
  - `light_distribution_v7.zip`
  - `efclean_gr_nashiji_100um_optical_properties.md`
  - `leaf_photos/`
    - 葉群写真6枚
- `example_output/`
  - 現在条件の出力例

## 本体コードの概要
- 日付: 3/21 条件
- 屋根: 三角屋根、五寸勾配
- 屋根フィルム: エフクリーン GRナシジ 100um の暫定角度依存透過率
- 南面壁: 透過率 0.50
- 東西壁: 透明
- 栽培空白部: x=0-1.5 m, 48-51 m
- カーテン収納: 上層 3.90 m, 下層 3.60 m
- 柱: y方向幅 0.20 m, x方向厚み 0.05 m

## グラフ化コードの使い方
```bash
python code/plot_layer_time_series_from_snapshot_summary.py   example_output/snapshot_summary.csv   -o example_output/layer_time_series_area_average.png
```
