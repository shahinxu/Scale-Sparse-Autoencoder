#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Loop through different es values
for es in 2 4 8 16; do
    echo "========================================="
    echo "Training with es=$es"
    echo "========================================="
    
    python -u train-moe_physically_scale.py \
      --gpu 1 \
      --ks 64 \
      --num_experts 64 \
      --es $es \
      --heavisides False
      
    echo "Completed training with es=$es"
    echo ""
done

echo "All training completed!"
