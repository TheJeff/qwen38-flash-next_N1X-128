#!/bin/bash
# Download nvidia/Qwen3.8-Flash-Next-FP4-Local @ pinned revision (102 GB).
# Run from Windows (git-bash): WSL curl 404s on the HF CDN. 3 parallel streams, resumable.
set -euo pipefail
REV=b9ce18a6887e18c089fdea8509a33e63429a3300
BASE="https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local/resolve/$REV"
DEST="${1:-Qwen3.8-Flash-Next-FP4-Local}"
mkdir -p "$DEST/tensors"
cd "$DEST/tensors"
seq -w 1 21 | xargs -P 3 -I{} curl -fSL --retry 5 --retry-all-errors -C - \
  -o "model-000{}-of-00021.safetensors" \
  "$BASE/tensors/model-000{}-of-00021.safetensors"
cd ..
for f in config.json generation_config.json chat_template.jinja \
         model.safetensors.index.json tokenizer.json tokenizer_config.json \
         vocab.json merges.txt preprocessor_config.json video_preprocessor_config.json; do
  curl -fsSL --retry 3 --retry-all-errors -o "$f" "$BASE/$f" || echo "WARN: $f failed"
done
echo "done: $(du -sh "$DEST" | cut -f1)"
# Move onto ext4 before serving: cp -r "$DEST" ~/models/
