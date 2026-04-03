# 温室フラックス計算 引継ぎまとめ

## 1. 目的

丸ハウス・三角ハウスを 1 棟 1 ボックスの同化箱として扱い、Profinder ログから次を推定することを目的とした。

- CO2 収支からの正味光合成 `P_net`
- gross assimilation
- respiration loss
- net carbon balance
- transpiration
- condensation / re-evap
- ventilation
- 日積算・累積炭素収支

解析対象は主として 2025/11/01 以降のログで、比較の中心は 2026 年 1–4 月である。

## 2. 現行モデルの骨格

炭素収支の中心式は次である。

\[
P = \rho_{mol} Q (X_{out}-X_{in}) + S_{CO2} - \rho_{mol} V \frac{dX_{in}}{dt}
\]

ここで

- `Q`: 換気流量
- `X_out`, `X_in`: 外気・室内 CO2
- `S_CO2`: CO2 source
- `V dX/dt`: CO2 storage term

である。出力ではさらに

- `gross assimilation = max(P_net + R(T), 0)`
- `respiration loss = R(T)`
- `net carbon balance = gross - respiration`

としている。

水蒸気・熱収支も同時に用いて `Q` と `E` を解いている。現行は 1-box であり、CO2・温湿度の空間不均一は直接は解いていない。

## 3. source の扱い

### Sankaku

- source 信号は `CO2` 1 本

### Maru

- source 信号は `CO2`
- 2 台目は `CV_H2`

両者とも、信号はそのまま source にせず、FOPDT で有効 source に変換している。

- dead time = 4 min
- tau = 8 min

Maru の `CV_H2` については、失火補正は未実装である。したがって、Maru の source 項はなお不確かさが大きい。

## 4. 日射の扱い

日射は生の `Sun.L` をそのまま使わず、2025 年のスケーリング CSV を用いて補正する運用に変更した。

- 使用ファイル: `scaling_daily_2025.csv`
- 入力ログの日付が 2026 年などであっても、month-day 対応で 2025 年値を適用する

重要事項として、2022 年給排液データと 2026 年 Profinder ログでは日射センサーが異なるため、**2022 側で得た日射式をそのまま日射補正に使わない**。日射補正は 2025 年センサー基準の `scaling_daily_2025.csv` を用いる。

## 5. ここまでの主な結論

### 5.1 長期比較

2025/11/01–2026/03/11 の比較では、Maru は gross はやや大きい一方、respiration loss も大きく、net carbon balance は大差ないか、やや不利となる場面があった。

ただし、この差のかなりの部分は物理差よりも

- CO2 代表性
- Maru 2 台目 source の不確かさ
- 日中 `Q` 推定

の影響を強く受けていると判断した。

### 5.2 気孔開度 proxy

蒸散と VPD から作った canopy conductance / stomatal-opening proxy では、Maru が常時低いというより、**1–3 月の午後低下が大きい**という形が見えた。

ただし、これは葉そのものの気孔開度ではなく、群落スケール proxy である。

### 5.3 3/18 のハウス差

3/18 は Maru の測定 CO2 が高いのに gross assimilation が低く、換気損失が大きかった。これは

- 施用 CO2 の効率悪化
- CO2 の非代表性
- 混合不良

を含む可能性が高く、単純な生理差とは読まない方がよいと判断した。

### 5.4 3/23 Maru のメンテ日

3/23 は 10:30–16:00 に発生機メンテナンスで、実際には停止していた。信号は後で入っていても、解析では 10:30–16:00 の `CO2` と `CV_H2` を 0 扱いにする必要があった。

この補正を入れると、3/23 の gross / net は大きく低下し、近い日射日の関係にも整合するようになった。

### 5.5 4/2 の雨上がり晴天・スクリーン高閉鎖

4/2 は Sankaku でラクソス 1547 を 90% 近く閉めた時間帯があった。短時間比較では

- gross / net は少し低下
- transpiration はむしろ増加
- ventilation もやや増加

となった。

ただし、これは日射増加と同時に起きており、現行モデルには昼間スクリーンの光学・長波・混合・昼間換気抵抗の直接表現がない。したがって、**スクリーン応答の向きそのものは参考にはなるが、スクリーン因果効果を自然に表現できているとまでは言えない**、という判断にした。

## 6. 日蒸散制約の検討

2022 年の給排液データから得た

\[
T = 0.089230 R + 0.299268
\]

を用い、2026 年 3 月の Sankaku で日蒸散量制約を入れる補助解析を行った。

ただし、これは

- 2022 年データ由来
- センサーが異なる
- 年が異なる

ため、**そのまま最終モデルには採用しない**。

また、日蒸散制約を入れる際は `solar_heat_factor` を 1 日一定の自由度として調整した。この値が 0.3 近傍になることがあったが、これは「透過率 30%」ではなく、現行 1-box における**有効熱入力の見かけ係数**である。

## 7. 現行コードの使い方

実行コードは別 zip にまとめてある。

### 同梱物

- `greenhouse_one_day_flux_and_carbon.py`
- `whole_greenhouse_chamber.py`
- `scaling_daily_2025.csv`
- `requirements.txt`
- 実行用シェルスクリプト

### 基本実行

```bash
python greenhouse_one_day_flux_and_carbon.py \
  --house sankaku \
  --input-log /path/to/20260317.log \
  --output-dir out_sankaku_20260317
```

```bash
python greenhouse_one_day_flux_and_carbon.py \
  --house maru \
  --input-log /path/to/20260317.log \
  --output-dir out_maru_20260317
```

### Maru で Sankaku outdoor を借りる

```bash
python greenhouse_one_day_flux_and_carbon.py \
  --house maru \
  --input-log /path/to/maru_20260317.log \
  --borrow-outdoor-from /path/to/sankaku_20260317.log \
  --output-dir out_maru_20260317_borrowed_outdoor
```

### インストール

```bash
pip install -r requirements.txt
```

### 便利なシェルスクリプト

コード zip には次を入れてある。

- `run_sankaku_day.sh`
- `run_maru_day.sh`
- `run_maru_borrow_outdoor.sh`

環境変数でログパスを渡す形式で、パスの書き換えだけで使えるようにしてある。

## 8. 現行コードの既知の制限

1. 1-box モデルである。上下層や source plume を直接表現していない。
2. 昼間スクリーンの直接効果が入っていない。
3. CO2 代表点の非代表性をまだ強く受ける。
4. Maru の `CV_H2` 失火補正が未実装である。
5. `solar_heat_factor` は光学定数ではなく、熱・放射・蓄熱の不足物理を吸った lumped parameter である。

## 9. 今後の発展

### 最優先

1. **CO2 多点化**
   - canopy 層
   - 上層 / vent 側
   - source 近傍は代表値ではなく plume 監視用

2. **日中 `Q` の拘束改善**
   - 小開口・弱光・曇天条件での `Q` を安定化
   - leak モデルから日中推定への遷移改善

3. **Maru 2 台目 source の失火補正**
   - 信号と実燃焼のずれを補正

### 中期

4. **2 層 CO2 モデル**
   - `k` のような固定混合係数を極力避け、storage 用と vent 用の代表 CO2 を分ける
   - もしくは 2 層の内部フラックスを残差として診断する

5. **日射・熱の扱いの高度化**
   - canopy-top 入射短波
   - frame / cover / canopy / floor への分配
   - 長波放射
   - cover / frame / floor / canopy の蓄熱分離

6. **昼間スクリーンモデル**
   - 光学透過率
   - 長波影響
   - 混合・実効換気抵抗

### 長期

7. **canopy energy balance**
   - 葉温
   - stomatal conductance
   - LAI
   - 必要なら IR 葉温計、内日射、追加 T/RH/CO2

## 10. 現時点の判断

炭素収支推定で最も重要なのは、依然として **CO2 の代表性** である。追加デバイスを 1 種類だけ優先するなら、CO2 センサー多点化が最優先である。

一方で、新規デバイスなしでも

- 2025 日射スケーリングの適用
- source の信号ベース補正
- 特定日の運転条件反映
- 1 日ごとのフラックス可視化

までは実施できる。
