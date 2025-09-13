#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Loop through different ks values
for ks in 2 4 8 16 32 64 128; do
    echo "========================================="
    echo "Training with ks=$ks"
    echo "========================================="
    
    python -u train-gated.py \
      --gpu 4 \
      --l1_penalties 0.01
      # --ks $ks \
      # --num_experts 32 \
      # --es 4 \
      # --heavisides False

    echo "Completed training with ks=$ks"
    echo ""
done

echo "All training completed!"
