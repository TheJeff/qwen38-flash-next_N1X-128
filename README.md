# TensorFold on the NVIDIA N1X (GB10) — Windows/WSL Port

Portable port of **TensorFold** (ashhart's Zig inference engine for Metal + CUDA) running the
**GB10-class (sm_121)** build on an **NVIDIA RTX Spark N1X** — a Windows 11 ARM64 desktop with a
6144-core Blackwell GPU and 128 GiB unified LPDDR5X — via WSL2 (Ubuntu 26.04, aarch64).

Two models are set up and benchmarked:

| Model | Engine | Context | Concurrency | Result |
|---|---|---|---|---|
| **Qwen3.8-27B** (MLX 4-bit, dense) | stock 1.1.0 release binary | 262,144 | 1 | 35-59 tok/s, TTFT 88-148 ms |
| **Qwen3.8-Flash-Next** (nvidia FP4-QAD, 176B-A6B MoE) | gated engine (commit `db28187` + 15 MiaAI-Lab patches, built from source) | 262,144 | 5 | **41-55 tok/s, TTFT 103-134 ms**, 55% MTP acceptance |

## What's in this repo

| Path | What it is |
|---|---|
| `RUNBOOK-N1X.md` | Complete step-by-step: install, build, kernels, serve, benchmarks, recovery |
| `README.md` | This file |
| `benchmark-n1x-qwen38-27b.json` | Community-v1 benchmark receipt — 27B, all 20 cells complete |
| `bench-openai-n1x-flashnext.json` | Flash Next benchmark (attached-server client, 5 reps/cell) |
| `RESULTS.md` | Side-by-side numbers and interpretation |
| `tf-patches/` | The 15 MiaAI-Lab patches that unlock Flash Next CUDA on commit db28187 |
| `linux-install-script.sh` | Original stock-install script (1.1.0, linux-aarch64) |
| `llama-server-windows` | The original llama.cpp recipe this port replaces |
| `scripts/` | Reusable helpers: download, build, AOT kernels, serve, benchmark |

**Not in git** (too big — see `.gitignore`): the extracted release trees, model weights
(16 GB + 96 GiB), the Zig toolchain tarball, `cuda-keyring.deb`. The runbook has the
exact download URLs and hashes for all of them.

**To publish:**

```
cd git-portfolio
git remote add origin <your-repo-url>
git push -u origin main
```

Current HEAD: `e46b777` (9 files committed + 15 patches, ~8.6 MB).

## Results snapshot

Same fixture prompts, 256-token outputs, 5 reps per cell (medians):

| Workload | Temp | Flash Next 176B-A6B | Qwen3.8-27B dense |
|---|---|---|---|
| code (fibonacci) | 1.0 | 51.8 tok/s | 58.7 tok/s |
| chat (GPU explain) | 1.0 | **41.3 tok/s** | 35.0 tok/s |
| code | 0 | **55.1 tok/s** | 51.5 tok/s |
| chat | 0 | **43.6 tok/s** | 37.7 tok/s |
| TTFT (median) | | 103-134 ms | 88-148 ms |

Flash Next also does MTP speculative decoding at 55% acceptance (2.64 tokens/round)
and serves 5 concurrent streams in 11.7 GiB of fp8 KV. The dense 27B fits only 1
stream at full context.

## Quick start (existing machine)

```
wsl -d Ubuntu -- bash scripts/serve-flashnext.sh
# -> OpenAI-compatible API at http://127.0.0.1:8080/v1 (model id: qwen38-flash-next)
```

## Hardware notes

The N1X is a GB10-class device: one 128 GiB LPDDR5X package partitioned into a
32 GiB CPU-visible region (Windows/WSL) and a 96 GiB GPU framebuffer. Model weights
live GPU-side via `cudaMalloc`; host RAM is staging. TensorFold's engine detects this
("memory shared with the host") and skips device staging for the prompt cache —
the stock binaries already handle it, which is why the 1.1.0 release works unmodified.

## Credits

- [ashhart/TensorFold](https://github.com/ashhart/TensorFold) — the engine (Apache-2.0)
- [MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold) — the DGX Spark recipe and patches adapted here
- [nvidia/Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) — the checkpoint (NVIDIA Open Model Agreement)
- Qwen3.8-27B-MLX-4bit and DFlash2 drafter via the TensorFold HF org
