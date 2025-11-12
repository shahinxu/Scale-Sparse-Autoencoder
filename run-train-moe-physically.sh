#!/usr/bin/env bash
# set -euo pipefail

# cd "$(dirname "$0")"

# for ks in 4; do
#     echo "========================================="
#     echo "Training with ks=$ks"
#     echo "========================================="
    
#     if ! python -u train-moe_physically.py \
#       --gpu 4 \
#       --ks $ks \
#       --num_experts 64 \
#       --es 1 \
#       --heavisides False; then
#         echo "Error encountered during training with ks=$ks, skipping to next."
#         echo ""
#         continue
#     fi

#     echo "Completed training with ks=$ks"
#     echo ""
# done


for l0_penalties in 1.0 0.5 0.1; do
    echo "========================================="
    echo "Training with l0_penalties=$l0_penalties"
    echo "========================================="
    
    python -u train-jump.py \
      --gpu 2 \
      --lr 7e-5 \
      --dict_ratio $((32 / 32)) \
      --l0_penalties $l0_penalties

    echo "Completed training with l0_penalties=$l0_penalties"
    echo ""
done
