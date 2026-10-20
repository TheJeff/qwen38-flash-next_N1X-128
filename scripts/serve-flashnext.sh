#!/bin/bash
# Serve Qwen3.8-Flash-Next FP4-Local on the N1X — the verified production config.
# Run INSIDE WSL. Model must be on ext4 (default ~/models/...).
# Text-only: the vision helper is flaky under WSL (see RUNBOOK section 3).
set -euo pipefail
export TENSORFOLD_CUDA_KERNELS="$HOME/tf-build/share/tensorfold/cuda/sm121"
export TENSORFOLD_VISION_PYTHONPATH=/tensorfold/src
MODEL="${1:-$HOME/models/Qwen3.8-Flash-Next-FP4-Local}"
PORT="${PORT:-8080}"

cd "$HOME/tf-build/native/bin"
exec ./tensorfold-native serve "$MODEL" \
  --host 127.0.0.1 --port "$PORT" --name qwen38-flash-next \
  --context 262144 --parallel 5 --kv-dtype fp8 "$@"
# add --vision to attempt image/video input (flaky under WSL — helper may fail to load)
