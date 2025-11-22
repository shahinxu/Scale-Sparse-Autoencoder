set -euo pipefail

cd "$(dirname "$0")"

for ks in 2 32 64; do
    echo "========================================="
    echo "Training with ks=$ks"
    echo "========================================="
    
    if ! python -u train-moe_physically.py \
    --gpu 0 \
    --ks $ks \
    --num_experts 32 \
    --es 1 \
    --heavisides False; then
        echo "Error encountered during training with ks=$ks, skipping to next."
        echo ""
        continue
    fi

    echo "Completed training with ks=$ks"
    echo ""
done