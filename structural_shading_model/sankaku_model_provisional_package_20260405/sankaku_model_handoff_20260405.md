# 三角ハウス 光遮蔽・群落光合成モデル 引継ぎ資料

## 1. 背景
三角ハウスについて、温室構造による遮蔽、屋根フィルム透過、群落の層別受光と層別光合成を結合した簡易モデルを段階的に作成した。

流れは以下。
1. 日射スケーリングモデルで「温室に実効的に入る日射量」を整える
2. 構造体遮蔽モデルで「温室内のどこに光が落ちるか」を時刻ごとに計算する
3. 3層群落モデルで「上・中・下層の受光と光合成」を計算する
4. 平面2次元マップとして「ある時刻にハウス内のどこがどれくらい光合成しているか」を可視化する
5. 屋根を平板近似から三角屋根へ変更し、屋根フィルムの角度依存透過率も入れた

## 2. 目的
目的は次の3点。
- 三角ハウス内で、構造体と屋根フィルムによる光の空間分布を計算する
- 層別群落モデルと結合し、時刻別・位置別の群落純光合成速度を出す
- カーテン位置、柱寸法、屋根形状、屋根フィルム光学特性などの変更が、光合成分布へどう効くかを比較できるようにする

## 3. 主要入力ファイル
- レイアウト図: `/mnt/data/提出レイアウト20210507.pdf`
- 構造体直達光モデルの元コード: `/mnt/data/light_distribution_v7.zip`
- 屋根フィルム暫定光学特性: `/mnt/data/efclean_gr_nashiji_100um_optical_properties.md`
- 葉群写真: `/mnt/data/0E37378D-D5A5-4595-9DC1-5B11C65E6964.jpeg` ほか

## 4. 作成したコードと役割

### 4.1 日射スケーリング関連
- `/mnt/data/_inspect/scaling_annual_v9/skycheck_daily.py`
  - 快晴判定と快晴日の基準化
- `/mnt/data/_inspect/scaling_annual_v9/scaling_period.py`
  - 年周期でのスケーリング補間
- `/mnt/data/_inspect/scaling_annual_v9/temp_target_daily.py`
  - scaled MJ を温度目標へ変換
- 補足資料: `/mnt/data/_inspect/scaling_annual_v9/handoff.md`

役割は「温室に実際に入った日積算日射の補正」であり、群落光合成そのものではない。

### 4.2 3層群落の最初のデモ
- `/mnt/data/sankaku_multilayer_demo/sankaku_multilayer_canopy_demo.py`

内容:
- 3層群落
- 直達光と散乱光の分離
- 各層LAI
- 各層 sunlit / shaded 分離
- 単葉光合成応答

出力:
- `/mnt/data/sankaku_multilayer_demo/output/`

### 4.3 構造影を加えた版
- `/mnt/data/sankaku_multilayer_structure_demo/sankaku_multilayer_structure_canopy_demo.py`

内容:
- 雨樋、カーテン収納、柱を遮蔽物として入れた
- 3層群落へ結合

出力:
- `/mnt/data/sankaku_multilayer_structure_demo/output/`

### 4.4 平面2次元光合成マップ系列
初期版:
- `/mnt/data/sankaku_spatial_photosynthesis_snapshots/generate_topview_photosynthesis_snapshots.py`

その後、方角修正、東西壁透明、カーテン高さ変更、柱寸法変更、端部空白化、南面壁50%透過を段階的に入れた版を作成した。

主な途中版:
- `/mnt/data/sankaku_spatial_photosynthesis_snapshots_1221_oriented_transparentEW/`
- `/mnt/data/sankaku_spatial_photosynthesis_snapshots_0321_oriented_transparentEW/`
- `/mnt/data/sankaku_spatial_photosynthesis_snapshots_0321_oriented_transparentEW_curtain390_360/`
- `/mnt/data/sankaku_spatial_photosynthesis_snapshots_0321_oriented_transparentEW_curtain390_360_pillar020_005/`
- `/mnt/data/sankaku_spatial_photosynthesis_snapshots_0321_oriented_transparentEW_curtain390_360_pillar020_005_blankMargins_southwall50/`

### 4.5 現在の主モデル
- `/mnt/data/sankaku_spatial_photosynthesis_snapshots_0321_triangularRoof5sun_angleDep_blankMargins_southwall50/generate_topview_photosynthesis_snapshots_0321_triangularRoof5sun_angleDep_blankMargins_southwall50.py`

内容:
- 三角屋根外皮
- 五寸勾配
- 屋根面入射角の計算
- エフクリーンGRナシジ100umの暫定角度依存透過率
- 南面壁50%透過
- 東西壁透明
- x=0–1.5 m, 48–51 m は非栽培空白
- 雨樋、カーテン収納、柱影
- 3層群落
- 時刻別平面2次元群落純光合成マップ

出力:
- `/mnt/data/sankaku_spatial_photosynthesis_snapshots_0321_triangularRoof5sun_angleDep_blankMargins_southwall50/output/topview_photosynthesis_snapshots.png`
- `/mnt/data/sankaku_spatial_photosynthesis_snapshots_0321_triangularRoof5sun_angleDep_blankMargins_southwall50/output/topview_layer_breakdown_1200.png`
- `/mnt/data/sankaku_spatial_photosynthesis_snapshots_0321_triangularRoof5sun_angleDep_blankMargins_southwall50/output/snapshot_summary.csv`
- `/mnt/data/sankaku_spatial_photosynthesis_snapshots_0321_triangularRoof5sun_angleDep_blankMargins_southwall50/output/snapshot_total_summary.csv`
- `/mnt/data/sankaku_spatial_photosynthesis_snapshots_0321_triangularRoof5sun_angleDep_blankMargins_southwall50/output/notes.txt`

## 5. 現在採用中の幾何設定

### 5.1 ハウスと栽培領域
- 栽培面積: `2295.0 m2`
- 仮定ベッド本数: `27本`
- 内部ベッド成長点: `14本 / 3m`
- 壁際ベッド成長点: `10本 / 3m`
- 平均ベッド長: `46.13 m`
- 成長点密度: `2.479 本 / m2`
- 全長方向 x: `0–51 m`
- 栽培領域: `x = 1.5–48.0 m` のみ
- 空白領域: `x = 0–1.5 m`, `48–51 m`
- 横方向 y: `0–45 m`
- 棟数: `6連棟`
- 1棟幅: `7.5 m`

### 5.2 方角
- 方角修正済み
- 南面壁を x=0 側として扱う
- 東西壁は透明

### 5.3 屋根
- 平板屋根ではなく三角屋根
- 勾配: `五寸勾配 = 0.5`
- 半棟スパン: `3.75 m`
- 雨樋上端: `4.20 m`
- 棟上昇量: `1.875 m`
- 棟高: `6.075 m`

### 5.4 構造体
- 雨樋幅: `0.60 m`
- 雨樋 z: `4.10–4.20 m`
- カーテン収納1 z: `3.85–3.90 m`
- カーテン収納2 z: `3.55–3.60 m`
- 柱の幅 y方向: `0.20 m`
- 柱の厚み x方向: `0.05 m`
- 柱ピッチ: `3.00 m`

## 6. 屋根フィルム暫定光学特性
詳細は別紙:
- `/mnt/data/efclean_gr_nashiji_100um_optical_properties.md`

現行実装の代表値:
- フィルム: `エフクリーン GRナシジ 100um`
- 乾燥時 PAR全透過率: `0.903`
- 乾燥時 直達比: `0.55`
- 乾燥時 散乱化比: `0.45`
- 天空散乱光透過率: `0.90`
- 南面壁直達透過率: `0.50`

角度依存補正 `K(theta)`:
- 0°: `1.000`
- 10°: `1.000`
- 20°: `0.999`
- 30°: `0.996`
- 40°: `0.986`
- 50°: `0.960`
- 60°: `0.898`

モデル投入式:
- `T_par_total(theta) = 0.903 * K(theta)`
- `T_par_direct(theta) = 0.55 * T_par_total(theta)`
- `T_par_beam_to_diffuse(theta) = 0.45 * T_par_total(theta)`

## 7. 群落モデルの内容
- 上・中・下の3層
- 各層ごとの平均受光を計算
- 各層で sunlit / shaded を分ける簡易扱い
- 単葉光合成応答を層別に積分
- 出力単位は基本的に `μmol CO2 m^-2 ground s^-1`

### 7.1 葉群パラメータの現状
葉群写真を見て、層別の実効葉面傾斜角を粗く与える方針を採った。
議論の要点:
- 上層はやや立ち葉
- 中層は横に張る
- 下層は下垂成分が大きい
- 実効葉面傾斜角だけでは、上向き/下垂や通路側/内側の違いを十分表せない
- 将来的には `a(z)` と `p(theta, phi | z)` の連続分布モデルへ拡張可能

## 8. 現在モデルの主要結果
使用したモデル:
- `/mnt/data/sankaku_spatial_photosynthesis_snapshots_0321_triangularRoof5sun_angleDep_blankMargins_southwall50/`

### 8.1 3/21 のハウス全体面積平均総群落純光合成速度
`snapshot_total_summary.csv` より。

| 時刻 | 総群落純光合成速度 |
|---|---:|
| 08:00 | 20.4915 |
| 10:00 | 34.2744 |
| 12:00 | 39.5763 |
| 14:00 | 34.2894 |
| 16:00 | 20.6033 |

単位: `μmol CO2 m^-2 ground s^-1`

### 8.2 層別の面積平均値
`snapshot_summary.csv` より。

#### 08:00
- 上層: `11.2170`
- 中層: `8.1208`
- 下層: `1.1536`

#### 10:00
- 上層: `15.6515`
- 中層: `15.1614`
- 下層: `3.4615`

#### 12:00
- 上層: `17.1364`
- 中層: `17.9738`
- 下層: `4.4661`

#### 14:00
- 上層: `15.6601`
- 中層: `15.2595`
- 下層: `3.3698`

#### 16:00
- 上層: `11.2221`
- 中層: `8.2699`
- 下層: `1.1113`

単位: `μmol CO2 m^-2 ground s^-1`

### 8.3 参照図
- 現在モデルの平面スナップショット: `/mnt/data/sankaku_spatial_photosynthesis_snapshots_0321_triangularRoof5sun_angleDep_blankMargins_southwall50/output/topview_photosynthesis_snapshots.png`
- 12:00 の層別内訳: `/mnt/data/sankaku_spatial_photosynthesis_snapshots_0321_triangularRoof5sun_angleDep_blankMargins_southwall50/output/topview_layer_breakdown_1200.png`
- 層別時系列グラフ: `/mnt/data/layer_time_series_area_average_0321_current.png`

## 9. 散乱化比較
ユーザー要望で、同じ角度依存全透過率を使い、
- 全部直射光
- 直達55% + 散乱化45%
の比較も行った。

この比較では、45%散乱化の方が群落全体の光合成が大きくなる傾向が出た。理由は、上層偏在だった光が中層へ回り、光合成の非線形性のために全体同化が増えたため。

この比較の説明は会話中で整理済みだが、比較用ファイル群は現時点で /mnt/data には残っていない可能性がある。必要なら再計算が必要。

## 10. 現在モデルの限界
- 葉群は3層の簡易モデルであり、連続高さ分布ではない
- 葉向きは実効角で近似しており、通路側/内側の方位分布を厳密には持たない
- 屋根フィルムの角度依存は、GRナシジ固有値ではなく ETFE 系代用
- 結露時の角度依存や散乱比変化は未実装
- 構造体の形状は簡略化されている
- 散乱光場は簡易扱いで、完全なBTDF/半球分布ではない
- 温度、VPD、葉温、気孔応答まで結んだ完全な蒸散モデルにはなっていない

## 11. 次チャットで最低限伝えるべき要点
- 現在の主コードは `...0321_triangularRoof5sun_angleDep_blankMargins_southwall50/...py`
- 三角屋根・五寸勾配・屋根角度依存透過率を入れた版が現在の基準
- 3/21 の面積平均総群落純光合成速度は 08,10,12,14,16時で `20.49, 34.27, 39.58, 34.29, 20.60`
- カーテン収納高さは `3.85–3.90 m` と `3.55–3.60 m`
- 雨樋は `4.10–4.20 m`
- 柱は `幅0.20 m, 厚み0.05 m`
- 栽培領域は `x=1.5–48.0 m` のみ、南面壁は50%透過、東西壁透明
- 屋根フィルム暫定パラメータは別紙 `efclean_gr_nashiji_100um_optical_properties.md`

