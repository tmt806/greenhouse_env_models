#!/usr/bin/env bash
set -euo pipefail
: "${SAN_LOG:?Set SAN_LOG to input log path}"
OUT_DIR="${OUT_DIR:-out_sankaku_day}"
python greenhouse_one_day_flux_and_carbon.py \
  --house sankaku \
  --input-log "$SAN_LOG" \
  --output-dir "$OUT_DIR"
