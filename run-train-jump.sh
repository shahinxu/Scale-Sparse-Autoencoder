set -euo pipefail

cd "$(dirname "$0")"

for ks in 0.1 0.5 1; do
    echo "========================================="
    echo "Training with ks=$ks"
    echo "========================================="
    if ! python -u train-jump.py \
      --gpu 0 \
      --target_l0s $ks; then
        echo "Error encountered during training with ks=$ks, skipping to next."
        echo ""
        continue
    fi

    echo "Completed training with ks=$ks"
    echo ""
done