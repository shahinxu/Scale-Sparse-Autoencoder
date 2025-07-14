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

def plot_all_logs_comparison(save_path="./l2_loss_comparison.png"):
    """
    在同一张图中绘制所有 log 文件的 L2 loss 对比图
    """
    # 查找当前目录下所有的 .log 文件
    log_files = glob.glob("*.log")
    
    if not log_files:
        print("未找到任何 .log 文件")
        return
    
    print(f"找到 {len(log_files)} 个日志文件:")
    for log_file in log_files:
        print(f"  - {log_file}")
    
    plt.figure(figsize=(15, 10))
    
    # 定义颜色列表
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']
    
    valid_logs = 0
    
    for i, log_file in enumerate(log_files):
        try:
            print(f"\n正在处理: {log_file}")
            steps, l2_losses = parse_training_log(log_file)
            
            if not steps:
                print(f"  警告: {log_file} 中未找到有效的训练数据")
                continue
            
            print(f"  成功解析 {len(steps)} 个训练步数的数据")
            print(f"  训练步数范围: {min(steps)} - {max(steps)}")
            print(f"  L2 Loss 范围: {min(l2_losses):.2e} - {max(l2_losses):.2e}")
            
            # 生成标签（去掉扩展名和路径）
            label = os.path.splitext(os.path.basename(log_file))[0]
            
            # 选择颜色
            color = colors[valid_logs % len(colors)]
            
            # 绘制曲线，设置透明度为0.7
            plt.plot(steps, l2_losses, linewidth=2, color=color, alpha=0.7, label=label)
            
            valid_logs += 1
            
        except Exception as e:
            print(f"  错误: 处理 {log_file} 时出现错误: {e}")
    
    if valid_logs == 0:
        print("没有找到有效的训练数据")
        return
    
    # 设置图表属性
    plt.title('L2 Loss Comparison - All Training Logs', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('L2 Loss', fontsize=14)
    
    # 设置网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 设置科学计数法
    plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 设置图例
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n对比图已保存到: {save_path}")
    
    plt.close()
    
    print(f"\n=== 处理完成 ===")
    print(f"总共处理了 {valid_logs} 个有效的日志文件")

def main():
    """
    主函数：绘制所有日志文件的对比图
    """
    print("开始绘制所有日志文件的 L2 Loss 对比图...")
    plot_all_logs_comparison()

if __name__ == "__main__":
    main()