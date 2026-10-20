#!/bin/bash
# Generate the sm_121 Triton AOT kernel set for Flash Next.
# Needs: torch, triton, numpy importable; source tree at /tensorfold (see build-engine.sh).
set -euo pipefail
cd /tensorfold
export PYTHONPATH=/tensorfold/src
python3 -B tools/zig/flashnext_aot.py build \
  --spec zig/tests/cuda/flashnext/kernels.json \
  --jit zig/tests/cuda/flashnext/jit.json \
  --tp 1 \
  --out "$HOME/tf-build/share/tensorfold/cuda/sm121"
# Expect ~320 kernels. "Triton hash != captured" PROBLEMs are benign (triton version skew).
# AttributeError PROBLEMs mean a module failed to import (usually missing numpy/torch) —
# fix deps and rerun, or the engine will refuse to load with MissingTritonVariant.
