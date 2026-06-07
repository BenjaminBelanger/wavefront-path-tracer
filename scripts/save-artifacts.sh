#!/usr/bin/env bash
# Collect all demo artifacts produced on a cloud pod (hero render, benchmark CSV,
# Nsight report + per-kernel CSV) into docs/demo/ and bundle them into a single
# tarball for easy transfer back to your laptop. Run this on the pod after
# rendering / profiling.
#
# Usage: bash scripts/cloud/save-artifacts.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MEDIA_DIR="docs/demo/media"
PROF_DIR="docs/demo/profiling"
NSYS_DIR="$PROF_DIR/nsight-systems"
mkdir -p "$MEDIA_DIR" "$NSYS_DIR" "$PROF_DIR/benchmarks"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | tr ' ' '_' || echo gpu)"
STAMP="$(date +%Y%m%d_%H%M%S)"

shopt -s nullglob
echo "==> Collecting renders into $MEDIA_DIR/"
while IFS= read -r png; do
  rel="${png#./}"
  dest="$MEDIA_DIR/$rel"
  mkdir -p "$(dirname "$dest")"
  cp -f "$png" "$dest"
  echo "  + $dest"
done < <(find . -type f -iname '*.png' \
  -not -path './.git/*' \
  -not -path './build/*' \
  -not -path './external/*' \
  -not -path './assets/*' \
  -not -path './docs/*' \
  -not -path './out/*' 2>/dev/null)

REP="$NSYS_DIR/frame.nsys-rep"
if [[ -f "$REP" && ! -f "$NSYS_DIR/frame_gpukernsum.csv" ]]; then
  NSYS="$(command -v nsys || true)"
  [[ -z "$NSYS" ]] && NSYS="$(ls /opt/nvidia/nsight-systems/*/target-linux-x64/nsys 2>/dev/null | head -n1 || true)"
  [[ -n "$NSYS" ]] && "$NSYS" stats --report gpukernsum --format csv --output "$NSYS_DIR/frame" "$REP" >/dev/null 2>&1 || true
fi

TARBALL="demo-artifacts_${GPU_NAME}_${STAMP}.tar.gz"
tar czf "$TARBALL" \
  $( [[ -d docs/demo/media ]] && echo docs/demo/media ) \
  $( [[ -d docs/demo/profiling ]] && echo docs/demo/profiling )

echo "==> Bundled artifacts into: $ROOT/$TARBALL"