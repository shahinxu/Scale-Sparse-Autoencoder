#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Loop through different ks values
# for ks in 64 128 2 4 8 16; do
#     echo "========================================="
#     echo "Training with ks=$ks"
#     echo "========================================="
    
#     # python -u train-moe_physically.py \
#     #   --gpu 1 \
#     #   --ks $ks \
#     #   --num_experts 32 \
#     #   --es 1 \
#     #   --heavisides False
#     # python -u train-topk.py \
#     #   --gpu 2 \
#     #   --ks $ks

#     echo "Completed training with ks=$ks"
#     echo ""
# done

echo "All training completed!"

for l1_penalties in 0.5 1 10 50 100; do
    echo "========================================="
    echo "Training with l1_penalties=$l1_penalties"
    echo "========================================="
    
    python -u train-gated.py \
      --gpu 3 \
      --lr 1e-3 \
      --dict_ratio $((32 / 32)) \
      --l1_penalties $l1_penalties

    echo "Completed training with l1_penalties=$l1_penalties"
    echo ""
done
