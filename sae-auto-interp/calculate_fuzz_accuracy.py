import os
import json
from collections import defaultdict
import argparse
import glob

def calculate_aggregated_accuracy(base_directory):
    """
    在基础目录中递归搜索所有 gpt2_fuzz 文件夹，
    并计算所有找到的JSON文件的汇总准确率。

    Args:
        base_directory (str): 包含多个实验迭代结果的基础目录路径。
    """
    search_pattern = os.path.join(base_directory, '*', 'gpt2_fuzz')
    print(f"正在使用模式搜索文件夹: {search_pattern}")

    fuzz_directories = glob.glob(search_pattern)

    if not fuzz_directories:
        print("错误: 未找到任何 'gpt2_fuzz' 文件夹。请检查您的目录结构。")
        return

    print(f"找到 {len(fuzz_directories)} 个 'gpt2_fuzz' 文件夹，将进行处理:")
    for d in fuzz_directories:
        print(f"  - {d}")

    # 使用 defaultdict 自动处理新的 distance
    # 结构: {inverted_distance: {'correct': count, 'total': count}}
    results_by_inverted_distance = defaultdict(lambda: {'correct': 0, 'total': 0})
    total_files_processed = 0

    # 遍历所有找到的 gpt2_fuzz 文件夹
    for directory in fuzz_directories:
        json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
        
        # 第一步：收集数据，并在收集中反转 distance 值
        for filename in json_files:
            file_path = os.path.join(directory, filename)
            total_files_processed += 1
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    for record in data:
                        if 'distance' in record and 'correct' in record:
                            original_distance = record['distance']
                            is_correct = record['correct']
                            
                            # 反转 distance: 1->10, 10->1, -1保持不变
                            if original_distance > 0:
                                inverted_distance = 11 - original_distance
                            else:
                                inverted_distance = original_distance
                            
                            # 按反转后的 distance 存储
                            results_by_inverted_distance[inverted_distance]['total'] += 1
                            if is_correct:
                                results_by_inverted_distance[inverted_distance]['correct'] += 1
            except (json.JSONDecodeError, TypeError) as e:
                print(f"警告: 无法处理文件 '{file_path}'. 错误: {e}")
            except Exception as e:
                print(f"处理文件 '{file_path}' 时发生意外错误: {e}")

    if not results_by_inverted_distance:
        print(f"处理了 {total_files_processed} 个文件，但没有找到有效的准确率数据。")
        return

    print(f"\n--- 汇总模糊测试准确率结果 (共处理 {total_files_processed} 个文件) ---")
    
    # 第二步：基于反转后的 distance 计算并打印每个累积 quantile 的准确率
    for quantile in range(1, 11):
        cumulative_correct = 0
        cumulative_total = 0
        
        # 对于每个 quantile，累加所有 inverted_distance >= quantile 的数据
        for inverted_distance, stats in results_by_inverted_distance.items():
            if inverted_distance >= quantile:
                cumulative_correct += stats['correct']
                cumulative_total += stats['total']
        
        if cumulative_total > 0:
            accuracy = (cumulative_correct / cumulative_total) * 100
            print(f"Quantile {quantile} (Inverted Distances >= {quantile}): 准确率 = {accuracy:.2f}% ({cumulative_correct}/{cumulative_total} 正确)")
        else:
            print(f"Quantile {quantile} (Inverted Distances >= {quantile}): 无可用数据。")

    # 单独处理和打印 distance = -1 的情况
    if -1 in results_by_inverted_distance:
        stats = results_by_inverted_distance[-1]
        correct = stats['correct']
        total = stats['total']
        if total > 0:
            accuracy = (correct / total) * 100
            print(f"Distance -1: 准确率 = {accuracy:.2f}% ({correct}/{total} 正确)")
        else:
            print(f"Distance -1: 无可用数据。")
            
    print("-----------------------------------------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从一个基础实验目录中递归计算所有 gpt2_fuzz 文件夹的汇总准确率。"
    )
    parser.add_argument(
        "base_directory",
        type=str,
        help="包含多个实验迭代结果的基础目录路径。"
    )
    
    args = parser.parse_args()
    
    calculate_aggregated_accuracy(args.base_directory)
