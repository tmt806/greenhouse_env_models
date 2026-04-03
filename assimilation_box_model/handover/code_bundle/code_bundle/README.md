# 1日フラックス計算コード bundle

この bundle は、Profinder の 1 日ログから次を計算して図と CSV を出力します。

- gross assimilation
- respiration loss
- net carbon balance
- net photosynthesis raw
- transpiration
- condensation / re-evap
- ventilation
- CO2 budget breakdown
- cumulative carbon

## 同梱ファイル

- `greenhouse_one_day_flux_and_carbon.py`
- `whole_greenhouse_chamber.py`
- `scaling_daily_2025.csv`
- `requirements.txt`
- `run_sankaku_day.sh`
- `run_maru_day.sh`
- `run_maru_borrow_outdoor.sh`

## 前提

- 日射は `scaling_daily_2025.csv` の `scaling_daily` を使って補正します。
- Sankaku の CO2 source は `CO2` 信号 1 本です。
- Maru の CO2 source は `CO2` と `CV_H2` の 2 本です。
- source 信号は FOPDT で有効 source に変換します。
  - dead time = 4 min
  - tau = 8 min
- duplicate timestamp は平均化してから計算します。

## インストール

```bash
pip install -r requirements.txt
```

## 基本実行

### Sankaku

```bash
python greenhouse_one_day_flux_and_carbon.py \
  --house sankaku \
  --input-log /path/to/20260317.log \
  --output-dir out_sankaku_20260317
```

### Maru

```bash
python greenhouse_one_day_flux_and_carbon.py \
  --house maru \
  --input-log /path/to/20260317.log \
  --output-dir out_maru_20260317
```

### Maru で Sankaku の outdoor を借りる

```bash
python greenhouse_one_day_flux_and_carbon.py \
  --house maru \
  --input-log /path/to/maru_20260317.log \
  --borrow-outdoor-from /path/to/sankaku_20260317.log \
  --output-dir out_maru_20260317_borrowed_outdoor
```

## シェルスクリプト

### `run_sankaku_day.sh`

環境変数 `SAN_LOG` と `OUT_DIR` を設定して実行します。

```bash
export SAN_LOG="$HOME/Google Drive/マイドライブ/greenhouse_log/sankaku/20260317.log"
export OUT_DIR="out_sankaku_20260317"
./run_sankaku_day.sh
```

### `run_maru_day.sh`

```bash
export MARU_LOG="$HOME/Google Drive/マイドライブ/greenhouse_log/maru/20260317.log"
export OUT_DIR="out_maru_20260317"
./run_maru_day.sh
```

### `run_maru_borrow_outdoor.sh`

```bash
export MARU_LOG="$HOME/Google Drive/マイドライブ/greenhouse_log/maru/20260317.log"
export SAN_LOG="$HOME/Google Drive/マイドライブ/greenhouse_log/sankaku/20260317.log"
export OUT_DIR="out_maru_20260317_borrowed_outdoor"
./run_maru_borrow_outdoor.sh
```

## 主な出力

- `*_flux_and_carbon.png`
- `*_flux_timeseries.csv`
- `*_summary.csv`
- `*_notes.md`

## 既知の制限

- 1-box モデルです。
- 昼間スクリーンの光学・長波・混合への直接効果は未実装です。
- Maru の `CV_H2` 失火補正は未実装です。
- 日積算蒸散制約などの特殊解析は、この bundle の外で実施した補助解析です。
