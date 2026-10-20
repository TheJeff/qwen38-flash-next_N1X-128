#!/bin/bash
# Serve Qwen3.8-27B (MLX 4-bit) with the stock 1.1.x release binary.
# Run inside WSL from the extracted release dir. Needs the DFlash2 drafter dir for drafts.
set -euo pipefail
CKPT="${1:?usage: serve-27b.sh <checkpoint-dir> [drafter-dir]}"
DRAFTER="${2:-}"
EXTRA=()
[ -n "$DRAFTER" ] && EXTRA+=(--drafter "$DRAFTER")
exec ./bin/tensorfold-native serve "$CKPT" \
  --host 127.0.0.1 --port "${PORT:-8080}" --name qwen38-27b \
  --context 262144 --parallel 1 --keep-warm 86400 "${EXTRA[@]}"
