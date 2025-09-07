# #!/usr/bin/env bash
# set -euo pipefail
# cd "$(dirname "$0")"

# python -u train-moe_physically.py \
#   --gpu 0 \
#   --ks 128\
#   --num_experts 64 \
#   --es 1 2 4 8 16 \
#   --heavisides False


#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Auto-detect parameters from dictionaries folder
dictionaries_dir="dictionaries"

if [[ ! -d "$dictionaries_dir" ]]; then
    echo "Error: dictionaries folder not found!"
    exit 1
fi

# Find all matching directories and extract parameters
declare -A configs
for dir in "$dictionaries_dir"/MultiExpert*; do
    if [[ -d "$dir" ]]; then
        dirname=$(basename "$dir")
        echo "Found directory: $dirname"
        
        # Parse directory name
        if [[ "$dirname" =~ ^MultiExpert_Scale_([0-9]+)_([0-9]+)_([0-9]+)$ ]]; then
            # Scale variant: MultiExpert_Scale_{ks}_{num_experts}_{es}
            ks="${BASH_REMATCH[1]}"
            num_experts="${BASH_REMATCH[2]}"
            es="${BASH_REMATCH[3]}"
            script="train-moe_physically_scale.py"
            key="scale_${ks}_${num_experts}_${es}"
        elif [[ "$dirname" =~ ^MultiExpert_([0-9]+)_([0-9]+)_([0-9]+)$ ]]; then
            # Regular variant: MultiExpert_{ks}_{num_experts}_{es}
            ks="${BASH_REMATCH[1]}"
            num_experts="${BASH_REMATCH[2]}"
            es="${BASH_REMATCH[3]}"
            script="train-moe_physically.py"
            key="regular_${ks}_${num_experts}_${es}"
        else
            echo "Warning: Directory $dirname doesn't match expected pattern, skipping"
            continue
        fi
        
        # Store config
        configs["$key"]="$script $ks $num_experts $es"
        echo "  -> Detected: script=$script, ks=$ks, num_experts=$num_experts, es=$es"
    fi
done

# Check if any configs were found
if [[ ${#configs[@]} -eq 0 ]]; then
    echo "Error: No valid MultiExpert directories found in $dictionaries_dir"
    exit 1
fi

# Run training for each detected configuration
for key in "${!configs[@]}"; do
    config=(${configs[$key]})
    script="${config[0]}"
    ks="${config[1]}"
    num_experts="${config[2]}"
    es="${config[3]}"
    
    echo ""
    echo "========================================="
    echo "Running training with configuration:"
    echo "  Script: $script"
    echo "  ks: $ks"
    echo "  num_experts: $num_experts" 
    echo "  es: $es"
    echo "========================================="
    
    python -u "$script" \
      --gpu 0 \
      --ks "$ks" \
      --num_experts "$num_experts" \
      --es "$es" \
      --heavisides False
      
    echo "Completed training for $key"
done

echo ""
echo "All training configurations completed!"
