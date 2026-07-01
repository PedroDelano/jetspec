# STACK LOCK — exact pinned versions (RULE 1 & 2)

Captured 2026-06-30 from the working environment. `requirements.lock.txt` is the full `uv pip freeze`
(189 packages, all `==`-pinned). This file pins the rest of the stack (toolchain, fork, weights) that
a Python lockfile cannot capture.

## Honest note
During the reproduction itself, deps were installed with `uv pip install` **without** pinning at
install time (e.g. `datasets`, `openai`, `transformers` as a range) and **no lockfile was produced** —
a RULE 1/2 violation. This file + `requirements.lock.txt` retroactively pin the exact resolved stack so
the result is reproducible going forward. Re-create the env from the lock, not from the ad-hoc commands
in `cycle/`.

## Python (uv)
- Python: **3.12.13** (uv-managed)
- Full pinned set: **`requirements.lock.txt`** (`uv pip freeze`).
- Install order matters (torch must come from the cu128 index first):
  ```bash
  uv venv --python 3.12 .venv && source .venv/bin/activate
  uv pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.11.0 \
      --index-url https://download.pytorch.org/whl/cu128
  uv pip install "setuptools>=77,<81" setuptools-scm wheel packaging cmake ninja jinja2 regex build
  VLLM_USE_PRECOMPILED=1 uv pip install -e vllm-jetspec --no-build-isolation
  uv pip install --no-deps -r requirements.lock.txt   # exact versions; --no-deps: it's a full freeze
  ```
  `--no-deps` is required: the freeze is a complete list but internally inconsistent for uv's resolver
  (`cuda-python==13.3.1` wants `cuda-bindings>=13.3.1`, env pins `12.9.4`; harmless at runtime).
  The turnkey path is **`bash setup_fresh.sh <dir>`** (handles all of this + verifies `.so` + smoke-tests).
  Build on **local disk** (e.g. `/root`), not the `/workspace` NFS mount, which intermittently throws
  `Stale file handle` during package extraction.
- Headline pins: `torch==2.10.0+cu128`, `torchvision==0.25.0+cu128`, `torchaudio==2.11.0+cu128`,
  `transformers==4.57.6`, `flashinfer-python==0.6.7`, `triton==3.6.0`, `datasets==5.0.0`,
  `openai==2.44.0`, `numpy==2.2.6`, `tokenizers==0.22.2`, `pydantic==2.13.4`,
  `nvidia-cuda-nvcc-cu12==12.9.86`, `nvidia-cuda-cccl-cu12==12.9.27`, `nvidia-cuda-runtime-cu12==12.9.79`.

## Toolchain / system
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, sm_120, 96 GB
- Driver: **580.126.09** (CUDA 13 capable)
- CUDA toolkit for FlashInfer nvcc: **12.9.1** runfile → `/root/cuda-12.9` (nvcc **12.9.86**, supports sm_120).
  `CUDA_HOME=/root/cuda-12.9`; **its `lib64` must NOT be on `LD_LIBRARY_PATH`** (shadows torch's cu128 cublas).
- System nvcc (unused for builds): 12.1
- OS: Ubuntu 22.04.3, kernel 6.8

## vLLM fork
- Repo: `JetSpec-project/vllm-jetspec`
- Fork commit: **`f90d5ca17a2c05f436a80ee2e0984cc7a22e1a16`**
- Precompiled binaries (the `.so`): from upstream vLLM base commit
  **`551b3fb39f3a95ff3dc3feca9528ab4c90649316`** (vLLM `0.18.2rc1.dev57`),
  wheel `wheels.vllm.ai/551b3fb…/cu129/vllm-0.18.2rc1.dev57+g551b3fb39-cp38-abi3-manylinux_2_31_x86_64.whl`.
- Local edits: `patches/vllm-jetspec.local.patch`.

## Model weights / revisions (HF commit hashes)
- Target: **`Qwen/Qwen3-30B-A3B`** @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`
- Draft head: **`JetSpec/jetspec-qwen3-30b-a3b`** @ `67bfdf0a73a34c87efa1a82d4f90023e6bcb819b`
