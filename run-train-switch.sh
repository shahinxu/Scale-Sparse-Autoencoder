set -euo pipefail

cd "$(dirname "$0")"

for ks in 2 4 8 16 32 64; do
    echo "========================================="
    echo "Training with ks=$ks"
    echo "========================================="
    
    if ! python -u train-moe_physically.py \
    --gpu 0 \
    --ks $ks \
    --num_experts 64 \
    --es 8 \
    --heavisides False; then
        echo "Error encountered during training with ks=$ks, skipping to next."
        echo ""
        continue
    fi

    echo "Completed training with ks=$ks"
    echo ""
done