# RULES — checked at every cycle checkpoint

1. **uv only.** All Python deps via `uv` with exact pinned versions (`uv.lock` / pinned `pyproject.toml`). No bare `pip install`.
2. **Pin the whole stack.** Record and pin CUDA/driver/OS/compiler versions and the exact model weights/revision (target + JetSpec draft head) — not just Python deps.
3. **No silent fallbacks/mocks.** No quiet CPU fallback, stubbed/SM90-only kernel silently skipped, or smaller-model swap to fake success. Any substitution (e.g. forced `--tree-attn-kernel triton`, model size change) is logged and flagged. The decoding must remain the real JetSpec tree path, not vanilla AR.
4. **Stale-process watchdog.** Long/GPU commands run in the background with timeouts. Poll ≥ every 5 min (`nvidia-smi`, process list); kill orphaned processes (incl. our own) still holding VRAM before they wedge the GPU.
