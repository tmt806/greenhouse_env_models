#!/usr/bin/env bash
set -euo pipefail
: "${MARU_LOG:?Set MARU_LOG to input log path}"
: "${SAN_LOG:?Set SAN_LOG to Sankaku outdoor donor log path}"
OUT_DIR="${OUT_DIR:-out_maru_borrowed_outdoor}"
python greenhouse_one_day_flux_and_carbon.py \
  --house maru \
  --input-log "$MARU_LOG" \
  --borrow-outdoor-from "$SAN_LOG" \
  --output-dir "$OUT_DIR"
