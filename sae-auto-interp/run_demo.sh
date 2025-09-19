#!/bin/bash

BASE_ADVANCE_PATH="MultiExpert_Scale_16_16_2"

# 循环的次数
NUM_RUNS=10

for i in $(seq 0 $((NUM_RUNS-1))); do
    echo "--- Running iteration $i ---"
    
    python my_demo.py \
        --advance_path "${BASE_ADVANCE_PATH}" \
        --iteration "${i}"
    
    echo "--- Iteration $i finished ---"
    echo ""
done

echo "All pipeline runs completed."