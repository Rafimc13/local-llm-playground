#!/usr/bin/env bash
set -euo pipefail

if ! command -v ollama >/dev/null 2>&1; then
  echo "ollama was not found. Install it from https://docs.ollama.com/ first." >&2
  exit 1
fi

models=(
  "qwen3:0.6b-q4_K_M"
  "qwen3:1.7b-q4_K_M"
  "qwen3:4b-q4_K_M"
)

for model in "${models[@]}"; do
  ollama pull "${model}"
done
