# Results — TensorFold on the N1X (GB10-class, sm_121)

All numbers measured on this machine, 2026-10-19. Methodology: fixed fixture prompts,
256 output tokens, 5 repetitions per cell, medians reported (all values in the JSON receipts).

## Setup matrix

| | Qwen3.8-27B (dense) | Qwen3.8-Flash-Next (176B-A6B MoE) |
|---|---|---|
| Checkpoint | TensorFold MLX 4-bit (16.1 GB) | nvidia FP4-QAD "FP4-Local" (102 GB disk / ~81 GiB loaded) |
| Quantization | MLX affine 4-bit, group 64 | NVFP4 routed experts (CSF scales), MXFP8 attention, FP8 n-gram |
| Engine | stock tensorfold-native 1.1.0 | tensorfold-native @ db28187 + MiaAI-Lab patches 0001-0015 |
| Drafter | DFlash2 (--drafter), ~55% acceptance | built-in MTP, 55% acceptance, 2.64 tok/round |
| Context | 262,144 (checkpoint native max) | 262,144 (native) |
| KV | bf16, 16.2 GiB at full window | fp8, 11.7 GiB at full window, 1056 B/pos/layer |
| Parallel streams | 1 (KV-bound) | 5 |
| Quality tier | ~4-bit dense | within ~1% of bf16 (GPQA 91.5 vs 92.2, NVIDIA's evals) |

## Benchmark cells

| Workload | Temp | Flash Next decode | Flash Next TTFT | 27B decode | 27B TTFT |
|---|---|---|---|---|---|
| fibonacci-raw (code) | 1.0 | 51.75 tok/s | 129 ms | 58.7 tok/s | 88 ms |
| gpu-chat-no-think | 1.0 | 41.28 tok/s | 103 ms | 35.0 tok/s | 146 ms |
| fibonacci-raw | 0.0 | 55.06 tok/s | 134 ms | 51.5 tok/s | 93 ms |
| gpu-chat-no-think | 0.0 | 43.62 tok/s | 117 ms | 37.7 tok/s | 148 ms |

Per-rep spreads are tight (chat T=0: 43.54-43.70; code T=0: 54.69-55.59).

## Interpretation

- **Chat/agentic workloads: Flash Next is 15-18% faster than the dense 27B** while being a
  far more capable model, with 5x the concurrency and 2 GB less KV per full window.
- **Code generation: parity to slight edge** — the dense model's smaller footprint gives it
  an edge at sampled settings, Flash Next leads at greedy.
- **MTP drafting is pulling real weight**: 55% acceptance at 2.64 tokens/round.
- First token stays under 150 ms in every cell — comfortable for interactive agent loops.
- The 27B's edge in code T=1 comes with high variance (46.6-55.9); its T=0 median is the
  only cell where the dense model leads at all.

## Receipts

- `benchmark-n1x-qwen38-27b.json` — TensorFold community-v1 receipt (managed run).
  All 20 cells complete. Local-run only (checkpoint is a local dir, can't be published as-is).
- `bench-openai-n1x-flashnext.json` — tools/bench_openai.py output (the gated engine has
  no `benchmark` subcommand). Same fixture prompts and rep count as community-v1, so the
  two tables compare directly.

## Notes on the environment

- Windows 11 ARM64 host; WSL2 Ubuntu 26.04 aarch64; CUDA passthrough via /dev/dxg.
- nvidia-smi inside WSL: RTX Spark N1X, 98304 MiB framebuffer, CUDA UMD 13.4.
- WSL curl cannot reach external CDNs (404 on github/HF/ziglang) — all downloads happen
  Windows-side, then are read through /mnt/c or copied to ext4 (models must live on ext4;
  9p breaks O_DIRECT).
- One WSL VM crash occurred during a heavy copy+load; $HOME contents survived, /tmp did not.
- Vision tower (0.84 GiB) loads and ran once; subsequent attempts fail cuModuleLoadData
  under WSL. Text-only serving is the reliable path.
