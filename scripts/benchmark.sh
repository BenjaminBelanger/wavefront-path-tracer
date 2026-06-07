#!/usr/bin/env bash
# Benchmark the headless path tracer across resolutions and write a CSV.
# Uses the renderer's own timing output (ms/sample, samples/s) — no Nsight needed.
#
# Usage: bash scripts/cloud/benchmark.sh [FRAMES]
set -euo pipefail

FRAMES="${1:-256}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="$ROOT/build/bin/wavefront-path-tracer"
OUT_DIR="$ROOT/docs/demo/profiling/benchmarks"
CSV="$OUT_DIR/benchmarks.csv"

if [[ ! -x "$BIN" ]]; then
  echo "Build first: bash scripts/cloud/setup-linux.sh  (missing $BIN)" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo unknown)"

RESOLUTIONS=("1280 720" "1920 1080" "2560 1440" "3840 2160")

echo "gpu,resolution,frames,ms_total,ms_per_sample,samples_per_s" > "$CSV"
echo "==> GPU: $GPU_NAME    frames/sample-count: $FRAMES"

for res in "${RESOLUTIONS[@]}"; do
  read -r W H <<< "$res"
  echo "=== ${W}x${H} ==="
  OUT="$("$BIN" --headless --width "$W" --height "$H" --frames "$FRAMES")"
  echo "$OUT" | grep -E 'ms total|samples/s' || true

  LINE="$(echo "$OUT" | grep 'ms total' | tail -n1)"
  MS_TOTAL="$(echo "$LINE"   | sed -E 's/.* ([0-9.]+) ms total.*/\1/')"
  MS_SAMPLE="$(echo "$LINE"  | sed -E 's/.*, ([0-9.]+) ms\/sample.*/\1/')"
  SPS="$(echo "$LINE"        | sed -E 's/.*, ([0-9.]+) samples\/s.*/\1/')"
  echo "$GPU_NAME,${W}x${H},$FRAMES,$MS_TOTAL,$MS_SAMPLE,$SPS" >> "$CSV"
done

echo "==> Wrote $CSV"
column -t -s, "$CSV" 2>/dev/null || cat "$CSV"