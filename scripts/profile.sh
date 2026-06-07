#!/usr/bin/env bash
# Profile the headless path tracer with Nsight Systems on a cloud GPU.
# Locates nsys even when it isn't on PATH (common in CUDA devel images), and
# offers to install it if missing. No display / GL context required.
#
# Usage: bash scripts/cloud/profile.sh [WIDTH] [HEIGHT] [FRAMES]
set -euo pipefail

WIDTH="${1:-1920}"
HEIGHT="${2:-1080}"
FRAMES="${3:-1024}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="$ROOT/build/bin/wavefront-path-tracer"
OUT_DIR="$ROOT/docs/demo/profiling/nsight-systems"
REP="$OUT_DIR/frame"

if [[ ! -x "$BIN" ]]; then
  echo "Build first: bash scripts/cloud/setup-linux.sh  (missing $BIN)" >&2
  exit 1
fi

find_tool() {
  local tool="$1"
  if command -v "$tool" >/dev/null 2>&1; then command -v "$tool"; return; fi
  local c
  for c in \
    "/usr/local/cuda/bin/$tool" \
    /opt/nvidia/nsight-systems/*/target-linux-x64/"$tool" \
    /opt/nvidia/nsight-systems-cli/*/target-linux-x64/"$tool" \
    /opt/nvidia/nsight-compute/*/"$tool"; do
    [[ -x "$c" ]] && { echo "$c"; return; }
  done
}

detect_cuda_ver() {
  local nvcc ver
  nvcc="$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)"
  if [[ -x "$nvcc" ]]; then
    ver="$("$nvcc" --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')"
  fi
  if [[ -z "${ver:-}" && -f /usr/local/cuda/version.json ]]; then
    ver="$(grep -oE '"version" *: *"[0-9]+\.[0-9]+' /usr/local/cuda/version.json | head -n1 | grep -oE '[0-9]+\.[0-9]+')"
  fi
  [[ -n "${ver:-}" ]] && echo "${ver/./-}"
}

matched_nsys_path() {
  local cm="$1" dep
  command -v apt-get >/dev/null 2>&1 || return 1
  apt-get install -y "cuda-nsight-systems-$cm" >/dev/null 2>&1 || return 1
  dep="$(apt-cache depends "cuda-nsight-systems-$cm" 2>/dev/null \
          | awk '/nsight-systems-[0-9]/{print $NF}' | head -n1)"
  [[ -z "$dep" ]] && return 1
  dpkg -L "$dep" 2>/dev/null | grep -E '/target-linux-x64/nsys$' | head -n1
}

NSYS="${NSYS:-}"
if [[ -z "$NSYS" ]]; then
  command -v apt-get >/dev/null 2>&1 && apt-get update -qq || true
  CUDA_MM="$(detect_cuda_ver || true)"
  if [[ -n "${CUDA_MM:-}" ]]; then
    echo "==> Detected CUDA $CUDA_MM; selecting matching Nsight Systems..."
    NSYS="$(matched_nsys_path "$CUDA_MM" || true)"
    [[ -n "$NSYS" ]] && echo "==> Using CUDA-matched nsys: $NSYS"
  fi
fi
[[ -z "$NSYS" ]] && NSYS="$(find_tool nsys || true)"
if [[ -z "${NSYS:-}" ]]; then
  echo "==> No nsys found; installing newest available..."
  if ! apt-get install -y nsight-systems-cli 2>/dev/null; then
    PKG="$(apt-cache pkgnames nsight-systems 2>/dev/null \
            | grep -E '^nsight-systems-[0-9]' | sort -V | tail -n1)"
    [[ -n "${PKG:-}" ]] && { echo "==> Installing $PKG"; apt-get install -y "$PKG" || true; }
  fi
  NSYS="$(find_tool nsys || true)"
fi
if [[ -z "${NSYS:-}" ]]; then
  echo "nsys not found. Install a CUDA-matched Nsight Systems manually, e.g.:" >&2
  echo "  apt-get update && apt-get install -y cuda-nsight-systems-12-6   # match your CUDA version" >&2
  echo "  or download from https://developer.nvidia.com/nsight-systems" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
echo "==> nsys: $NSYS"
echo "==> Profiling headless render ${WIDTH}x${HEIGHT}, ${FRAMES} samples"

"$NSYS" profile \
  --trace=cuda,nvtx,osrt \
  --force-overwrite true \
  --output "$REP" \
  "$BIN" --headless --width "$WIDTH" --height "$HEIGHT" --frames "$FRAMES"

"$NSYS" stats --report gpukernsum --format csv --output "$REP" "$REP.nsys-rep" >/dev/null 2>&1 || true

echo "==> Wrote $REP.nsys-rep"
echo "    Per-kernel CSV: ${REP}_gpukernsum.csv (if stats succeeded)"