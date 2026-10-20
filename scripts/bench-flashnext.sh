#!/bin/bash
# Benchmark Flash Next via the attached-server client (same fixtures as community-v1).
# Server must already be running (serve-flashnext.sh). Run inside WSL from /tensorfold.
set -euo pipefail
cd /tensorfold
python3 tools/bench_openai.py "${BASE:-http://127.0.0.1:8080}" qwen38-flash-next \
  --tokens 256 --reps 5 --temperatures 1.0,0 \
  --label "${LABEL:-n1x-flashnext-fp4}" \
  --output "${1:-bench-openai-flashnext.json}"
cat "${1:-bench-openai-flashnext.json}"
