# Performance

Numbers captured on an **NVIDIA GeForce RTX 4090 (Ada, SM 8.9, 24 GB)**, CUDA 13.3,
Release build (`-O3 --use_fast_math`), interactive demo scene, 8 bounces, unless noted.
Source data: [`demo/profiling/benchmarks/benchmarks.csv`](demo/profiling/benchmarks/benchmarks.csv).

## Throughput vs resolution

| Resolution | ms / frame | FPS | Mrays/s (primary) |
|---|---|---|---|
| 1280×720 | 4.67 | 214 | 197 |
| 1920×1080 | 10.01 | 99.9 | 207 |
| 2560×1440 | 17.62 | 56.7 | 209 |
| 3840×2160 | 38.71 | 25.8 | 214 |

Primary-ray throughput holds at ~200 Mrays/s as pixel count grows 9×, so the wavefront
pipeline stays GPU-bound rather than kernel-launch-bound.

## Per-kernel story (what to profile)

The two hot kernels and the claims they back:

| Kernel | Role | What to measure | Why it matters |
|---|---|---|---|
| `intersect` | SAH-BVH traversal | Achieved occupancy, L1/L2 hit rate, divergence | Latency-bound; occupancy hides memory latency |
| `shade_surface` | BSDF eval + sampling | Global load efficiency, mem throughput, registers | SoA layout → coalesced loads vs AoS |
| `accumulate` / `tonemap` | Streaming write-out | DRAM throughput | Memory-bound, near bandwidth limit |

Small per-kernel register footprints (wavefront, not megakernel) keep **achieved
occupancy high**; the SoA path-state layout makes per-path global loads **coalesce into
single transactions** per warp. Capture both kernels in Nsight Compute to quantify these.

## Method
- Build: `powershell -File run.ps1`
- Benchmarks: `bash scripts/cloud/benchmark.sh` (renders 720p→4K headless, writes the CSV)
- System trace: `scripts/capture/profile-nsight-systems.ps1` / `scripts/cloud/profile.sh`
- Kernel metrics: `scripts/capture/profile-nsight-compute.ps1`
- Mrays/s ≈ (width × height × samples-per-frame) / frame-time. With one primary ray
  per pixel per frame, primary Mrays/s ≈ (width × height) / ms / 1000.
