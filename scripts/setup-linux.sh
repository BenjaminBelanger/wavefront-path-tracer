#!/usr/bin/env bash
# Set up and build the wavefront path tracer on a Linux cloud GPU instance
# (Ubuntu 22.04, NVIDIA driver + CUDA toolkit already installed, e.g. RunPod/vast.ai).
# Installs build + OpenGL/X11 deps, builds Release, and sets up VirtualGL + TurboVNC
# so the interactive GLFW window gets a GPU-backed OpenGL context.
#
# Usage:  bash scripts/cloud/setup-linux.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "==> Checking GPU + CUDA"
nvidia-smi || { echo "No NVIDIA driver visible. Pick a GPU instance."; exit 1; }
command -v nvcc >/dev/null || echo "WARN: nvcc not on PATH; ensure CUDA toolkit is installed."

echo "==> Installing build, OpenGL and X11 dependencies"
export DEBIAN_FRONTEND=noninteractive
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  if command -v sudo >/dev/null; then SUDO="sudo"; else
    echo "ERROR: not root and sudo not found; re-run as root."; exit 1; fi
fi
$SUDO apt-get update
$SUDO apt-get install -y \
  build-essential git cmake ninja-build pkg-config \
  libgl1-mesa-dev libglu1-mesa-dev mesa-utils \
  xorg-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
$SUDO apt-get install -y ffmpeg || \
  echo "NOTE: ffmpeg install failed; headless animation still writes PNG frames you can encode later."
$SUDO apt-get install -y virtualgl turbovnc || \
  echo "NOTE: virtualgl/turbovnc not available via apt; install their .deb for interactive use (headless profiling still works)."

need_new_cmake=1
if command -v cmake >/dev/null; then
  CMAKE_VER="$(cmake --version | head -n1 | awk '{print $3}')"
  MAJOR="${CMAKE_VER%%.*}"; MINOR="$(echo "$CMAKE_VER" | cut -d. -f2)"
  if [ "$MAJOR" -gt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 24 ]; }; then
    need_new_cmake=0
  fi
fi
if [ "$need_new_cmake" -eq 1 ]; then
  echo "==> apt cmake ${CMAKE_VER:-none} is too old (<3.24); installing a current cmake via pip."
  $SUDO apt-get install -y python3-pip >/dev/null
  $SUDO pip3 install --upgrade --break-system-packages cmake 2>/dev/null || \
    $SUDO pip3 install --upgrade cmake
  hash -r
  echo "==> Using cmake: $(command -v cmake) ($(cmake --version | head -n1))"
fi

echo "==> Configuring + building (Release)"
rm -rf build
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' .')"
if [ -n "$CC" ]; then
  echo "==> Detected compute capability sm_$CC; building natively for it."
  ARCH_FLAG="-DCMAKE_CUDA_ARCHITECTURES=$CC"
else
  echo "WARN: could not detect compute capability; using CMake defaults."
  ARCH_FLAG=""
fi
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release $ARCH_FLAG
cmake --build build --config Release --target wavefront-path-tracer -j"$(nproc)"

BIN="build/bin/wavefront-path-tracer"
[ -x "$BIN" ] || BIN="$(find build -name wavefront-path-tracer -type f | head -n1)"
echo "==> Built: $BIN"

echo "==> Done."