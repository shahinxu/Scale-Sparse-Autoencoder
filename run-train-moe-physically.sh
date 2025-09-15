#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Loop through different ks values
for ks in 16 32 64 128; do
    echo "========================================="
    echo "Training with ks=$ks"
    echo "========================================="
    
    python -u train-moe_physically.py \
      --gpu 1 \
      --ks $ks \
      --num_experts 64 \
      --es 16 \
      --heavisides False

    echo "Completed training with ks=$ks"
    echo ""
done

echo "All training completed!"
