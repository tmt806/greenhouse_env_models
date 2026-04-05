# 温室フラックス計算 引継ぎまとめ 2026-04-05

## 1. 今回の更新内容

- 外気 CO2 と外気水蒸気を `outside_pf` の Profinder から与えるモードを handover 版に追加した。
- 従来方式も残してあり、切替可能。
  - `legacy`
    - 温室ログの `Out_T` と仮定露点差から外気 RH を推定
    - 外気 CO2 は固定値
  - `outside_pf`
    - `outside_pf/YYYY-MM-DD/DATABASE_*.DB` の `AGLOG` から `絶対湿度` と `ＣＯ２濃度` を取得
    - 外気温は引き続き温室ログ側の `Out_T` を使用
    - 取得した絶対湿度と `Out_T` から RH を再計算してモデルへ投入

## 2. 変更したファイル

- `assimilation_box_model/handover/code_bundle/code_bundle/greenhouse_one_day_flux_and_carbon.py`
- `assimilation_box_model/handover/code_bundle/code_bundle/README.md`

## 3. 実装概要

- 追加した CLI 引数
  - `--outdoor-humidity-co2-source {legacy,outside_pf}`
  - `--outside-pf-root /path/to/outside_pf`
- `outside_pf` ルートは未指定時に既知の Google Drive パスを自動探索
- `DATABASE_MASTER.DB` ではなく、日付ディレクトリ下の `DATABASE_*.DB` を走査
- `AGLOG` から以下を読む
  - `気温`
  - `湿度`
  - `絶対湿度`
  - `ＣＯ２濃度`
- モデル入力に追加 / 明示したもの
  - 外気 RH
  - 外気 CO2
  - 出力列 `AH_out_g_m3`
- summary / notes にも外気入力モードを記録

## 4. 前提と解釈

- `outside_pf` の `気温` は引き込み配管の影響を受けるため、外気温そのものには使わない。
- 水蒸気について欲しい量は絶対湿度なので、`outside_pf` の `絶対湿度` を優先して使う。
- その上で、`whole_greenhouse_chamber.py` が RH ベースで外気状態量を組むため、`Out_T` と絶対湿度から RH を再計算している。

## 5. 4/5 の確認

- `outside_pf` 実データは次に存在
  - `/Users/soichi/Library/CloudStorage/GoogleDrive-soi.toi.chi@gmail.com/マイドライブ/greenhouse_log/outside_pf/2026-04-05`
- 実測が入っていたのは `DATABASE_1.DB`
- 確認できた観測時間帯
  - `2026-04-05 17:14-21:19`
- この時間帯で Sankaku / Maru とも試算実行済み
- 生成物はローカルに
  - `assimilation_box_model/handover/outside_pf_20260405/`
  に出してある

## 6. 4/5 試算の注意

- 4/5 の `outside_pf` 観測時間帯だけで切り出して計算すると、特に Maru で `ventilation_ach` が大きく負になる区間がある。
- これは現行モデルがその時間帯の `Q` を常に非負拘束していないためで、外気入力追加それ自体とは分けて解釈する必要がある。
- したがって今回の主目的は
  - `outside_pf` データを読めること
  - 既存モデルをそのまま回せること
  の確認と考える。

## 7. Git 履歴

- 今回の外気 Profinder 対応コミット:
  - `c20a50b Add outside PF outdoor humidity and CO2 input mode`
- 既存の 2026-04-04 まとめ追加:
  - `f5474ce Add 20260404 handover summary`
- handover スクリーン実装:
  - `c032a96 Add screen optics and thermal effects to handover model`

## 8. 次にやる候補

1. `legacy` と `outside_pf` の before / after 比較を同日・同時間帯で自動出力
2. `Q` の非負拘束や診断列追加で、Maru の負換気区間を点検
3. `outside_pf` を使う日次バッチ用シェルを追加
