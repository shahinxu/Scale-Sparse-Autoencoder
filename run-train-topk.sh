set -euo pipefail

cd "$(dirname "$0")"

for ks in 4 8 16 32 64 128; do
    echo "========================================="
    echo "Training with ks=$ks"
    echo "========================================="
    if ! python -u train-topk.py \
      --gpu 0 \
      --ks $ks; then
        echo "Error encountered during training with ks=$ks, skipping to next."
        echo ""
        continue
    fi

    echo "Completed training with ks=$ks"
    echo ""
done