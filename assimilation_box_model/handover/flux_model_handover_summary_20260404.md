# 温室フラックス計算 引継ぎまとめ 2026-04-04

## 1. 現在のリポジトリ構成

- 親リポジトリ: `/Users/soichi/Codex/greenhouse_env_models`
- handover 完成版:
  - `assimilation_box_model/handover/code_bundle/code_bundle/greenhouse_one_day_flux_and_carbon.py`
  - `assimilation_box_model/handover/code_bundle/code_bundle/whole_greenhouse_chamber.py`
- 簡易版:
  - `assimilation_box_model/simple_model/greenhouse_assimilation_model.py`
  - `assimilation_box_model/simple_model/day_flux_compare.py`

## 2. モデルの位置づけ

- handover 版が主系統。
- `whole_greenhouse_chamber.py` が 1-box の熱・水蒸気・CO2 収支のコア。
- `greenhouse_one_day_flux_and_carbon.py` が 1 日ログの前処理、source 推定、出力作成、運用補正を担当。
- simple_model は Google Drive 上のログをすぐ流して比較するための簡易実行版で、handover 版の完全再現ではない。

## 3. source / 基本前提

- Sankaku:
  - CO2 source は `CO2` 1 本
- Maru:
  - CO2 source は `CO2` と `CV_H2`
- source 信号は FOPDT で有効 source に変換
  - dead time = 4 min
  - tau = 8 min
- duplicate timestamp は平均化
- 日射は `scaling_daily_2025.csv` で補正
- 2025 年スケーリングを month-day 対応で適用

## 4. 今回追加したスクリーンモデル

### 4.1 対象設備

- Maru
  - カーテン1: Tempa 98 70
  - カーテン2: Luxous 15 47
- Sankaku
  - カーテン1: Luxous 15 47
  - カーテン2: Tempa 55 57

### 4.2 物性の扱い

- `Luxous 15 47`
  - 公開値ベース
  - direct shade 15%
  - diffuse shade 24%
  - energy saving 47%
- `Tempa 55 57`
  - 公開値ベース
  - direct shade 55%
  - diffuse shade 61%
  - energy saving 57%
- `Tempa 98 70`
  - 誠和掲載型番と Svensson 命名規則に整合
  - direct shade 98%
  - energy saving 70%
  - diffuse shade は個票未確認のため暫定的に 98% 仮定

### 4.3 閉度の扱い

- ログの `CV_C1`, `CV_C2` を各スクリーンの指令値として使用
- 光学閉度:
  - 0-90% を 0-1 に線形変換
  - 90% で「温室内からの上方向視界はスクリーンで埋まる」とみなして飽和
- 熱的閉度:
  - 0-100% を 0-1 に使用
  - 90-100% は重なりによる全閉化を表現

### 4.4 モデルに入れた効果

- 短波:
  - `H_solar` にスクリーン透過率を乗算
- 熱損失:
  - 動的 `UA` にスクリーン由来の `screen_ua_ratio` を乗算
- 夜空放射結合:
  - `deltaT_sky` に `screen_sky_ratio` を反映

### 4.5 まだ入れていない効果

- スクリーンによる昼間換気抵抗の直接モデル
- スクリーンによる上下混合低下
- 群落エネルギーバランス
- 葉温・気孔開度の明示モデル

## 5. 実装ファイル

- 変更済み:
  - `assimilation_box_model/handover/code_bundle/code_bundle/greenhouse_one_day_flux_and_carbon.py`
- 追加した主な要素:
  - `ScreenLayer`
  - house ごとの `screen_layers`
  - `optical_screen_closure`
  - `thermal_screen_closure`
  - `combine_screen_layers`
  - 出力列:
    - `screen1_pct`
    - `screen2_pct`
    - `screen1_optical_closure`
    - `screen2_optical_closure`
    - `screen1_thermal_closure`
    - `screen2_thermal_closure`
    - `screen_shortwave_transmission`
    - `screen_ua_ratio`
    - `screen_sky_ratio`
    - `deltaT_sky_eff_K`

## 6. 動作確認

- 改訂版 handover は以下で実行確認済み
  - Sankaku `2026-04-03`
  - Maru `2026-04-03`
- 出力生成:
  - `*_flux_and_carbon.png`
  - `*_flux_timeseries.csv`
  - `*_summary.csv`
  - `*_notes.md`

## 7. スクリーン動作日の比較

### 7.1 2026-04-02 は日中スクリーン動作日

- 以前こちらが「高閉度は夜間中心」と見たのは抽出条件が厳しすぎたため。
- 実際には日中もスクリーン使用あり。
- `screen >= 70%` かつ `Sun.L > 100` の日中区間で比較。

### 7.2 2026-04-02 比較結果

比較条件:

- baseline: スクリーン未考慮の元 handover 版
- screened: 今回のスクリーン考慮版

三角:

- 日中スクリーン区間: 10:17-14:02
- 平均閉度: 84.8%
- 平均透過率: 0.816
- gross assimilation: 8.369 -> 7.634 gCO2 m-2 h-1
- net photosynthesis: 7.888 -> 7.153
- transpiration: 487.8 -> 394.2 g m-2 h-1
- ventilation: 15.15 -> 12.22 ACH
- gross 日積算: 70.38 -> 67.57 gCO2 m-2 day-1

丸:

- 日中スクリーン区間: 09:49-14:02
- 平均閉度: 86.7%
- 平均透過率: 0.812
- gross assimilation: 11.877 -> 11.081 gCO2 m-2 h-1
- net photosynthesis: 11.060 -> 10.264
- transpiration: 544.6 -> 437.8 g m-2 h-1
- ventilation: 25.06 -> 20.40 ACH
- gross 日積算: 95.67 -> 92.27 gCO2 m-2 day-1

### 7.3 解釈

- スクリーン導入で
  - 同化は少し低下
  - 蒸散は明瞭に低下
  - 換気も低下
  という向きになった。
- 方向としては妥当。
- ただし定量値はまだ暫定。
- 現モデルは昼間換気抵抗や混合低下を直接は解いていないため、蒸散差や同化差をそのまま真値とはみなさない。

## 8. 2026-01-22 の補助比較

- `screen >= 90%` かつ `Sun.L > 100` が十分ある日として 2026-01-22 も比較。
- この日は 4/2 より差が小さかった。
- 主には蒸散低下が出て、gross / net は大きくは動かなかった。

## 9. Git 履歴

- handover スクリーン実装コミット:
  - `c032a96 Add screen optics and thermal effects to handover model`
- simple_model 分離コミット:
  - `0fce188 Move simple model into dedicated directory`
- handover bundle と簡易ツール追加:
  - `113f0a7 Add handover bundle and day flux comparison tools`

## 10. 次にやる候補

1. `Tempa 98 70` の拡散光特性の公開資料確認
2. before / after 比較図を 1 枚に並べる専用スクリプト作成
3. スクリーンによる昼間換気抵抗の直接効果を追加
4. スクリーン時の混合低下を 2 層化または残差診断で評価
5. 蒸散差の妥当性を別データで照合
