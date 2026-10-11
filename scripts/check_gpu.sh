#!/usr/bin/env bash
set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found. Install a compatible NVIDIA driver first." >&2
  exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free \
  --format=csv,noheader
