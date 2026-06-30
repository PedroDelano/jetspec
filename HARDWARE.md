# HARDWARE — detected environment

_Detected 2026-06-30 via `nvidia-smi`, `nvcc`, `lscpu`, `free`, `uname`._

## GPU
- **Model:** NVIDIA RTX PRO 6000 Blackwell **Server Edition**
- **Architecture / compute capability:** Blackwell, **sm_120** (compute_cap 12.0)
- **VRAM:** 97,887 MiB (~96 GB) GDDR7
- **Memory bandwidth:** **1,792 GB/s** (1.8 TB/s; 512-bit bus, GDDR7 @ 28 Gbps) — [NVIDIA spec](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/)
- **Count:** 1 GPU
- **Driver:** 580.126.09 — **runtime CUDA 13.0** (driver-reported max)

## ⚠️ GPU not idle at detection time
- PID 49030 `sglang::scheduler` holding **83,450 MiB** (~83 GB)
- PID 48885 `python` holding 664 MiB
- **Only ~13 GB VRAM free.** GPU-Util 0% (idle but reserving memory).
- This is a **pre-existing, foreign process** (not started by this reproduction). It must be cleared before any non-trivial model will fit. **Pending user decision** — see GOAL.md.

## Toolchain
- **nvcc (CUDA toolkit):** release **12.1** (V12.1.105) — ⚠️ predates Blackwell; sm_120 codegen needs CUDA **≥ 12.8**. Source builds of vLLM/custom kernels for sm_120 will require a newer toolkit or prebuilt wheels.
- **uv:** 0.11.25 (at `/root/.local/bin/uv`)

## CPU / RAM / OS
- **CPU:** 2× AMD EPYC 9555 64-Core (256 logical CPUs, 2 threads/core)
- **RAM:** 1.5 TiB total
- **OS:** Ubuntu 22.04.3 LTS, kernel 6.8.0-106-generic, x86_64
