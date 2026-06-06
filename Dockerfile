# Reproducible CUDA + OpenGL build for the wavefront path tracer.
# Default base is CUDA 13 (supports Blackwell sm_120). For older GPUs you can use a
# 12.x tag, but sm_120 (RTX 50-series) requires CUDA >= 12.8.
ARG CUDA_TAG=13.0.3-devel-ubuntu24.04
FROM nvidia/cuda:${CUDA_TAG}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git cmake ninja-build pkg-config ca-certificates \
    libgl1-mesa-dev libglu1-mesa-dev \
    xorg-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Build for a broad set of architectures (override with --build-arg CUDA_ARCH=120).
ARG CUDA_ARCH="75;80;86;89;90;120"
RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
 && cmake --build build --config Release --target wavefront-path-tracer -j

# Interactive use needs a display (mount an X socket or run VNC/VirtualGL).
# Headless profiling works once a GL context exists. See docs/CLOUD.md.
CMD ["bash", "-lc", "echo 'Built. Run with --gpus all and an X/VNC display, e.g. vglrun build/bin/wavefront-path-tracer'; ls build/bin"]
