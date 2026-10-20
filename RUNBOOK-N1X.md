# TensorFold on the N1X — Full Runbook

Machine: NVIDIA RTX Spark N1X (Spark-Jeff) — GB10-class Blackwell (sm_121, 6144 cores),
128 GiB LPDDR5X-9400 partitioned: 32 GiB CPU-side + 96 GiB GPU framebuffer.
Windows 11 ARM64 host; WSL2 Ubuntu 26.04 (aarch64) for the Linux engine.

Portions of this file: exact commands used, in order, verified working 2026-10-19.

## 0. Key gotchas (read first)

- WSL curl 404s on ALL external CDNs (github releases, HF, ziglang.org). apt and git work.
  ALWAYS download via Windows-side curl into /mnt/c, then read from WSL.
- Models must live on ext4 (~/models), NOT /mnt/c — 9p breaks O_DIRECT and is slow.
- The build expects the source tree at the literal path /tensorfold (cubins embed it):
  `ln -sfn $HOME/src/TensorFold /tensorfold`
- /tmp does not survive a WSL VM crash (which can happen under heavy memory pressure).
  $HOME does.
- If WSL netname dies (ERROR_NETNAME_DELETED), just relaunch; everything in $HOME survives.

## 1. Stock engine (1.1.x release) — the 27B test

    # Windows side:
    cd C:/Projects/Tensorfold-n1x
    curl -fSL -o tensorfold-1.1.0-linux-aarch64.tar.gz \
      https://github.com/ashhart/TensorFold/releases/download/v1.1.0/tensorfold-1.1.0-linux-aarch64.tar.gz
    # verify sha256 against .sha256, then:
    tar -xzf tensorfold-1.1.0-linux-aarch64.tar.gz

    # WSL side:
    cd /mnt/c/Projects/Tensorfold-n1x/tensorfold-1.1.0-linux-aarch64
    ./bin/tensorfold-native --version          # 1.1.0
    ./bin/tensorfold-native capabilities --json # chip: nvidia-sm121, backends: [cuda]

Serve the 27B (MLX 4-bit, 16.1 GB, 3 shards from TensorFold/Qwen3.8-27B-MLX-4bit):

    ./bin/tensorfold-native serve <ckpt-dir> --host 127.0.0.1 --port 8080 \
      --name qwen38-27b --context 262144 --parallel 1 --keep-warm 86400

Notes:
- checkpoint max_position_embeddings=262144 — 256K is the native max for THIS model.
- at 256K the KV stream is 16.2 GiB of 20.2 GiB free -> parallel must be 1.
- drafter for 27B: z-lab/Qwen3.8-27B-DFlash2 (3.8 GB), pass --drafter <dir>.
- benchmark receipt from that run: benchmark-n1x-qwen38-27b.json (community-v1,
  all cells complete; code T=1: 58.7 tok/s, chat T=1: 35.0, code T=0: 51.5, chat T=0: 37.7).

## 2. Flash Next engine — gated, build from source

Model: nvidia/Qwen3.8-Flash-Next-FP4-Local @ revision b9ce18a6887e18c089fdea8509a33e63429a3300
(NVIDIA QAD-distilled FP4: NVFP4 experts, MXFP8 attention, MTP head + vision tower baked in.
91.5% GPQA Diamond / 88.7% ToolEvalBench vs 92.2/89.4 for bf16. 176B-A6B qwen4_exp MoE.)
102.2 GB on disk (21 shards under tensors/), ~81 GiB loaded.

Reference recipe: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold

### 2a. Download (Windows side)

    cd C:/Projects/Tensorfold-n1x
    mkdir -p Qwen3.8-Flash-Next-FP4-Local/tensors && cd Qwen3.8-Flash-Next-FP4-Local/tensors
    # 21 shards, 3 parallel, resumable (-C -):
    seq -w 1 21 | xargs -P 3 -I{} curl -fSL --retry 5 --retry-all-errors -C - \
      -o model-000{}-of-00021.safetensors \
      "https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local/resolve/b9ce18a6887e18c089fdea8509a33e63429a3300/tensors/model-000{}-of-00021.safetensors"
    # configs:
    cd .. && for f in config.json generation_config.json chat_template.jinja \
      model.safetensors.index.json tokenizer.json tokenizer_config.json \
      vocab.json merges.txt preprocessor_config.json video_preprocessor_config.json; do
      curl -fsSL -O "https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local/resolve/b9ce18a6887e18c089fdea8509a33e63429a3300/$f"; done

### 2b. Toolchain (WSL, as root: `wsl -d Ubuntu -u root`)

    # Zig 0.17.0 — download WINDOWS-side (ziglang.org 404s from WSL curl):
    #   https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz
    #   sha256 9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8
    cp /mnt/c/Projects/Tensorfold-n1x/zig-aarch64-linux-0.17.0.tar.xz $HOME/ && cd $HOME && tar -xf zig-aarch64-linux-0.17.0.tar.xz
    # verify: $HOME/zig-aarch64-linux-0.17.0/zig version  ->  0.17.0

    # nvcc 13.4 (apt's 12.4 is too old for sm_121):
    dpkg -i /mnt/c/Projects/Tensorfold-n1x/cuda-keyring.deb   # NVIDIA ubuntu2604/sbsa repo
    apt-get update && apt-get install -y cuda-nvcc-13-4 cuda-cudart-dev-13-4
    # /usr/local/cuda-13.4/bin/nvcc

    # python deps (3.14; use --ignore-installed, plain -q fails on uninstall-no-record-file):
    apt-get install -y python3-pip python3-venv
    pip3 install --break-system-packages --ignore-installed \
      torch triton numpy "transformers==5.17.0" "av==19.0.1"

### 2c. Source + patches

    cd $HOME/src && git clone --filter=blob:none https://github.com/ashhart/TensorFold.git
    cd TensorFold && git fetch --depth 1 origin db281878ddb836fd0df510d8771ecb7e0fe47d26
    git checkout -q FETCH_HEAD
    # 15 patches from MiaAI-Lab repo (copies in C:/Projects/Tensorfold-n1x/tf-patches/):
    for p in /mnt/c/Projects/Tensorfold-n1x/tf-patches/*.patch; do git apply "$p"; done
    # (if using a glob directly, cd to the patch dir first — MSYS mangles WSL paths)

### 2d. Build + kernels

    ln -sfn $HOME/src/TensorFold /tensorfold     # literal path the cubins embed
    export PATH=$HOME/zig-aarch64-linux-0.17.0:$PATH
    cd /tensorfold
    zig build -Dnvcc=/usr/local/cuda-13.4/bin/nvcc -Doptimize=fast \
      --prefix $HOME/tf-build -j20 fatbins install native
    # -> $HOME/tf-build/native/bin/tensorfold-native  (reports version 0.6.5, chip nvidia-sm121)

    # Triton AOT kernel set (~5 min, needs torch+triton+numpy importable):
    cd /tensorfold && PYTHONPATH=/tensorfold/src python3 -B tools/zig/flashnext_aot.py build \
      --spec zig/tests/cuda/flashnext/kernels.json --jit zig/tests/cuda/flashnext/jit.json \
      --tp 1 --out $HOME/tf-build/share/tensorfold/cuda/sm121
    # 320 kernels; "hash != captured" PROBLEMs are benign (triton 3.8 vs pinned capture).
    # FATAL are the AttributeErrors (missing module import, usually numpy/torch) — they leave
    # out kernel variants and the engine later refuses to load with MissingTritonVariant.

### 2e. Model onto ext4

    mkdir -p ~/models/Qwen3.8-Flash-Next-FP4-Local/tensors
    cp /mnt/c/Projects/Tensorfold-n1x/Qwen3.8-Flash-Next-FP4-Local/*.json \
       /mnt/c/Projects/Tensorfold-n1x/Qwen3.8-Flash-Next-FP4-Local/*.jinja \
       /mnt/c/Projects/Tensorfold-n1x/Qwen3.8-Flash-Next-FP4-Local/vocab.json \
       /mnt/c/Projects/Tensorfold-n1x/Qwen3.8-Flash-Next-FP4-Local/merges.txt \
       ~/models/Qwen3.8-Flash-Next-FP4-Local/
    cp /mnt/c/Projects/Tensorfold-n1x/Qwen3.8-Flash-Next-FP4-Local/tensors/*.safetensors \
       ~/models/Qwen3.8-Flash-Next-FP4-Local/tensors/

### 2f. Serve (script: tf-serve-fn-text.sh in Hermes scratch)

    export TENSORFOLD_CUDA_KERNELS=$HOME/tf-build/share/tensorfold/cuda/sm121
    export TENSORFOLD_VISION_PYTHONPATH=/tensorfold/src
    cd $HOME/tf-build/native/bin
    ./tensorfold-native serve /home/jeff/models/Qwen3.8-Flash-Next-FP4-Local \
      --host 127.0.0.1 --port 8080 --name qwen38-flash-next \
      --context 262144 --parallel 5 --kv-dtype fp8 [--vision]
    # load ~130-210s. KV: fp8, 1056 B/position/layer, 11.7 GiB full window, 12 GiB kept free.

## 3. Measured results (N1X, with MTP drafts)

### Flash Next FP4 (176B-A6B) vs Qwen3.8-27B dense — same fixtures, 256 tok, 5 reps

| Workload | Temp | Flash Next | 27B dense |
|---|---|---|---|
| code (fibonacci-raw) | 1.0 | 51.8 tok/s | 58.7 tok/s |
| chat (gpu-explain) | 1.0 | **41.3 tok/s** | 35.0 tok/s |
| code | 0 | **55.1 tok/s** | 51.5 tok/s |
| chat | 0 | **43.6 tok/s** | 37.7 tok/s |
| TTFT | | 103-134 ms | 88-148 ms |

Flash Next: 42-44 tok/s general decode, 55% MTP acceptance, 2.64 tokens/round,
5 concurrent streams. Receipts: benchmark-n1x-qwen38-27b.json (community-v1),
bench-openai-n1x-flashnext.json (bench_openai.py, attached-server).

- vision: helper loaded the tower once successfully; other runs fail
  cuModuleLoadData CUDA_ERROR_INVALID_VALUE — flaky under WSL, leave --vision off
  until debugged. Text path is solid.

## 4. Client usage

    http://127.0.0.1:8080/v1  (OpenAI-compatible; model id "qwen38-flash-next")
    /health for status. Thinking on by default; send chat_template_kwargs
    {"enable_thinking": false} to disable per-request.

## 5. Quick recovery (if WSL died)

    wsl -d Ubuntu -- bash /mnt/c/Users/imkoa/AppData/Local/hermes/cache/scratch/tf-serve-fn-text.sh
    # or from a plain shell:
    wsl -d Ubuntu -- bash -c 'ln -sfn $HOME/src/TensorFold /tensorfold; \
      TENSORFOLD_CUDA_KERNELS=$HOME/tf-build/share/tensorfold/cuda/sm121 \
      $HOME/tf-build/native/bin/tensorfold-native serve \
      /home/jeff/models/Qwen3.8-Flash-Next-FP4-Local --host 127.0.0.1 --port 8080 \
      --name qwen38-flash-next --context 262144 --parallel 5 --kv-dtype fp8'
