#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python -u train-moe_physically.py \
  --gpu 0 \
  --ks 128\
  --num_experts 64 \
  --es 1 2 4 8 16 \
  --heavisides False
