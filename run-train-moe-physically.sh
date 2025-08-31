#!/usr/bin/env bash
set -euo pipefail

# Fixed config per request:
# k=4, num_experts=64, es=[1,2,4,16], gpu=0, heavisides=False

cd "$(dirname "$0")"

echo "[INFO] Starting train-moe_physically.py with k=4, experts=64, es=[1,2,4,16], gpu=0, heavisides=False"
python -u train-moe_physically.py \
  --gpu 0 \
  --ks 4 \
  --num_experts 64 \
  --es 1 2 4 16 \
  --heavisides False
