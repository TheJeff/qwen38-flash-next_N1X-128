#!/bin/bash
# Build the gated Flash-Next engine (commit db28187 + MiaAI-Lab patches) on aarch64 Linux.
# Prereqs: zig 0.17.0 in $HOME/zig-aarch64-linux-0.17.0 (sha 9e8d1166...),
#          nvcc 13.4 (apt cuda-nvcc-13-4 + cuda-keyring), python3 with torch/triton/numpy.
# Source + patches: see RUNBOOK-N1X.md sections 2b-2d. Run INSIDE WSL.
set -euo pipefail
export PATH=$HOME/zig-aarch64-linux-0.17.0:$PATH

# The cubins embed the literal path /tensorfold — the symlink matters.
ln -sfn "$HOME/src/TensorFold" /tensorfold 2>/dev/null || \
  { echo "need: sudo ln -sfn $HOME/src/TensorFold /tensorfold"; exit 1; }

cd /tensorfold
zig build -Dnvcc=/usr/local/cuda-13.4/bin/nvcc -Doptimize=fast \
  --prefix "$HOME/tf-build" -j"$(nproc)" fatbins install native

echo "binary: $HOME/tf-build/native/bin/tensorfold-native"
"$HOME/tf-build/native/bin/tensorfold-native" --version
