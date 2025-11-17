#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

for ks in 32 64 128; do
    echo "========================================="
    echo "Training with ks=$ks"
    echo "========================================="
    
    if ! python -u train-moe_physically_scale.py \
      --gpu 0 \
      --ks $ks \
      --num_experts 64 \
      --es 2 \
      --heavisides False; then
        echo "Error encountered during training with ks=$ks, skipping to next."
        echo ""
        continue
    fi
    # python -u train-topk.py \
    #   --gpu 0 \
    #   --ks $ks

    echo "Completed training with ks=$ks"
    echo ""
done


# for l1_penalties in 3 6 20 30; do
#     echo "========================================="
#     echo "Training with l1_penalties=$l1_penalties"
#     echo "========================================="
    
#     python -u train-gated.py \
#       --gpu 0 \
#       --lr 1e-3 \
#       --dict_ratio $((32 / 32)) \
#       --l1_penalties $l1_penalties

#     echo "Completed training with l1_penalties=$l1_penalties"
#     echo ""
# done
