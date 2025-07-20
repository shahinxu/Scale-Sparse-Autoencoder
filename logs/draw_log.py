import re
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def parse_training_log(log_file_path):
    steps = []
    l2_losses = []
    
    pattern = r"Step (\d+).*'l2_loss': ([\d.]+)"
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                step = int(match.group(1))
                l2_loss = float(match.group(2))
                steps.append(step)
                l2_losses.append(l2_loss)
    
    return steps, l2_losses

def select_log_files():
    all_log_files = glob.glob("*.log")
    
    if not all_log_files:
        print("未找到任何 .log 文件")
        return []
    
    print(f"\n找到 {len(all_log_files)} 个日志文件:")
    for i, log_file in enumerate(all_log_files, 1):
        print(f"  {i}. {log_file}")
    
    print("\n请选择要处理的文件:")
    print("1. 输入文件编号（用逗号分隔，如：1,3,5）")
    print("2. 输入 'all' 处理所有文件")
    print("3. 输入文件名模式（如：training_*）")
    
    choice = input("\n请输入选择: ").strip()
    
    selected_files = []
    
    if choice.lower() == 'all':
        selected_files = all_log_files
    elif ',' in choice or choice.isdigit():
        try:
            if ',' in choice:
                indices = [int(x.strip()) - 1 for x in choice.split(',')]
            else:
                indices = [int(choice) - 1]
            
            for idx in indices:
                if 0 <= idx < len(all_log_files):
                    selected_files.append(all_log_files[idx])
                else:
                    print(f"警告: 编号 {idx + 1} 超出范围")
        except ValueError:
            print("错误: 请输入有效的编号")
            return []
    else:
        pattern_files = glob.glob(choice)
        if pattern_files:
            selected_files = pattern_files
        else:
            print(f"未找到匹配模式 '{choice}' 的文件")
            return []
    
    return selected_files

def plot_selected_logs_comparison(selected_files, save_path="./l2_loss_comparison.png"):
    if not selected_files:
        print("没有选择任何文件")
        return
    
    print(f"\n将处理以下 {len(selected_files)} 个文件:")
    for log_file in selected_files:
        print(f"  - {log_file}")
    
    plt.rcParams.update({'font.size': 14})
    
    plt.figure(figsize=(16, 12))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']
    
    valid_logs = 0
    
    for i, log_file in enumerate(selected_files):
        try:
            print(f"\n正在处理: {log_file}")
            steps, l2_losses = parse_training_log(log_file)
            
            if not steps:
                print(f"  警告: {log_file} 中未找到有效的训练数据")
                continue
            
            print(f"  成功解析 {len(steps)} 个训练步数的数据")
            print(f"  训练步数范围: {min(steps)} - {max(steps)}")
            print(f"  L2 Loss 范围: {min(l2_losses):.2e} - {max(l2_losses):.2e}")
            
            label = os.path.splitext(os.path.basename(log_file))[0]
            
            color = colors[valid_logs % len(colors)]
            
            plt.plot(steps, l2_losses, linewidth=3, color=color, alpha=0.7, label=label)
            
            valid_logs += 1
            
        except Exception as e:
            print(f"  错误: 处理 {log_file} 时出现错误: {e}")
    
    if valid_logs == 0:
        print("没有找到有效的训练数据")
        return
    
    plt.xlabel('Training Steps', fontsize=18, fontweight='bold')
    plt.ylabel('Reconstruction MSE', fontsize=18, fontweight='bold')
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20, frameon=True, 
               fancybox=True, shadow=True, framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n对比图已保存到: {save_path}")
    
    plt.close()
    
    print(f"\n=== 处理完成 ===")
    print(f"总共处理了 {valid_logs} 个有效的日志文件")

def main():
    """
    主函数：让用户选择文件并绘制对比图
    """
    print("开始绘制选定日志文件的 L2 Loss 对比图...")
    
    # 让用户选择文件
    selected_files = select_log_files()
    
    if selected_files:
        # 绘制选定文件的对比图
        plot_selected_logs_comparison(selected_files)
    else:
        print("未选择任何文件，程序退出")

if __name__ == "__main__":
    main()