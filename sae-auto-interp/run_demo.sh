#!/bin/bash

# --- 配置参数 ---
# 基础 advance_path，例如 "ScaleEncoder"
BASE_ADVANCE_PATH="ScaleAutoEncoder_5k" 

# 循环的次数
NUM_RUNS=1 # 例如，如果你想运行 3 次，从 0 到 2

# (可选) 如果你需要指定 GPU，可以设置 CUDA_VISIBLE_DEVICES
# 例如，跳过 GPU 0，使用 GPU 1,2,3,4
# export CUDA_VISIBLE_DEVICES="1,2,3,4"

# (可选) 如果你希望每次运行使用不同的 GPU，可以在循环内部设置
# 例如，第一次用GPU1，第二次用GPU2...
# GPU_DEVICES=("1" "2" "3") # 对应 $NUM_RUNS 次运行

# --- 循环运行 Python 脚本 ---
for i in $(seq 0 $((NUM_RUNS-1))); do
    echo "--- Running iteration $i ---"

    # (可选) 如果希望每次运行指定不同的 GPU
    # if [ -n "${GPU_DEVICES[$i]}" ]; then
    #     export CUDA_VISIBLE_DEVICES="${GPU_DEVICES[$i]}"
    #     echo "Using CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    # else
    #     echo "Using default CUDA_VISIBLE_DEVICES (or none set for this run)"
    #     unset CUDA_VISIBLE_DEVICES # 取消设置，以便使用所有可见GPU或系统默认
    # fi
    
    python my_demo.py \
        --advance_path "${BASE_ADVANCE_PATH}" \
        --iteration "${i}"
    
    echo "--- Iteration $i finished ---"
    echo ""
done

echo "All pipeline runs completed."