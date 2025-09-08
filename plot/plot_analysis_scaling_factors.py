import matplotlib.pyplot as plt
import numpy as np

# 设置matplotlib参数
plt.rcParams.update({
    'font.size': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 18,
})

def plot_scaling_factors():
    """绘制scaling因子的可视化图表"""
    
    # 数据定义
    k_values = [32, 16, 8]
    e_values = [1, 2, 4, 8, 16]
    
    # Scaling因子数据
    scaling_data = {
        32: [0.213772, 0.41353, 0.724287, 0.926098, 1.037019],
        16: [0.218654, 0.453421, 0.789543, 1.024567, 1.156783],
        8:  [0.224891, 0.498765, 0.867432, 1.145632, 1.298456]
    }
    
    # 颜色和标记样式
    colors = ['#264653', '#2a9d8f', '#e9c46a']
    markers = ['o', 's', '^']
    linestyles = ['-', '--', '-.']
    
    # 创建图表
    plt.figure(figsize=(8, 8))
    
    # 为每个k值绘制曲线
    for i, k in enumerate(k_values):
        plt.plot(e_values, scaling_data[k], 
                color=colors[i], 
                marker=markers[i], 
                linestyle=linestyles[i],
                linewidth=3, 
                markersize=10,
                label=f'k={k}')
    
    # 设置图表属性
    plt.xlabel('# Experts')
    plt.ylabel('Scaling Factor (ω)')
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    plt.legend()
    
    # 设置x轴刻度
    plt.xticks(e_values)
    
    # 设置y轴范围
    plt.ylim(0, 1.4)
    
    plt.tight_layout()
    plt.savefig('scaling_factors_vs_experts.png', dpi=300, bbox_inches='tight')
    print('Saved scaling_factors_vs_experts.png')
    plt.show()

def plot_scaling_heatmap():
    """绘制scaling因子的热力图"""
    
    # 数据
    k_values = [32, 16, 8]
    e_values = [1, 2, 4, 8, 16]
    
    scaling_matrix = np.array([
        [0.213772, 0.41353, 0.724287, 0.926098, 1.037019],  # k=32
        [0.218654, 0.453421, 0.789543, 1.024567, 1.156783], # k=16
        [0.224891, 0.498765, 0.867432, 1.145632, 1.298456]  # k=8
    ])
    
    plt.figure(figsize=(10, 6))
    
    # 创建热力图
    im = plt.imshow(scaling_matrix, cmap='viridis', aspect='auto')
    
    # 设置轴标签
    plt.xticks(range(len(e_values)), [f'e={e}' for e in e_values])
    plt.yticks(range(len(k_values)), [f'k={k}' for k in k_values])
    
    # 添加数值标注
    for i in range(len(k_values)):
        for j in range(len(e_values)):
            plt.text(j, i, f'{scaling_matrix[i, j]:.3f}', 
                    ha='center', va='center', color='white', fontsize=14, weight='bold')
    
    # 设置标题和标签
    plt.xlabel('Number of Experts (e)')
    plt.ylabel('Sparsity Parameter (k)')
    plt.title('Scaling Factors Heatmap')
    
    # 添加颜色条
    cbar = plt.colorbar(im)
    cbar.set_label('Scaling Factor', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig('scaling_factors_heatmap.png', dpi=300, bbox_inches='tight')
    print('Saved scaling_factors_heatmap.png')
    plt.show()

def plot_scaling_3d():
    """绘制3D表面图"""
    from mpl_toolkits.mplot3d import Axes3D
    
    # 数据
    k_values = [32, 16, 8]
    e_values = [1, 2, 4, 8, 16]
    
    scaling_matrix = np.array([
        [0.213772, 0.41353, 0.724287, 0.926098, 1.037019],  # k=32
        [0.218654, 0.453421, 0.789543, 1.024567, 1.156783], # k=16
        [0.224891, 0.498765, 0.867432, 1.145632, 1.298456]  # k=8
    ])
    
    # 创建网格
    E, K = np.meshgrid(e_values, k_values)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制3D表面
    surf = ax.plot_surface(E, K, scaling_matrix, cmap='viridis', alpha=0.8)
    
    # 添加散点
    for i, k in enumerate(k_values):
        for j, e in enumerate(e_values):
            ax.scatter(e, k, scaling_matrix[i, j], color='red', s=50)
    
    # 设置轴标签
    ax.set_xlabel('Number of Experts (e)')
    ax.set_ylabel('Sparsity Parameter (k)')
    ax.set_zlabel('Scaling Factor')
    ax.set_title('3D Scaling Factors Surface')
    
    # 添加颜色条
    fig.colorbar(surf, shrink=0.5, aspect=20)
    
    plt.tight_layout()
    plt.savefig('scaling_factors_3d.png', dpi=300, bbox_inches='tight')
    print('Saved scaling_factors_3d.png')
    plt.show()

def plot_scaling_bar():
    """绘制分组柱状图"""
    
    # 数据
    k_values = [32, 16, 8]
    e_values = [1, 2, 4, 8, 16]
    
    scaling_data = {
        32: [0.213772, 0.41353, 0.724287, 0.926098, 1.037019],
        16: [0.218654, 0.453421, 0.789543, 1.024567, 1.156783],
        8:  [0.224891, 0.498765, 0.867432, 1.145632, 1.298456]
    }
    
    # 设置柱状图参数
    x = np.arange(len(e_values))
    width = 0.25
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    hatches = ['///', '\\\\\\', '...']
    
    plt.figure(figsize=(12, 8))
    
    # 绘制每组柱状图
    for i, k in enumerate(k_values):
        offset = (i - 1) * width
        plt.bar(x + offset, scaling_data[k], width, 
               label=f'k={k}', color=colors[i], alpha=0.8, hatch=hatches[i])
    
    # 设置图表属性
    plt.xlabel('Number of Experts (e)')
    plt.ylabel('Scaling Factor')
    plt.title('Scaling Factors Comparison')
    plt.xticks(x, [f'e={e}' for e in e_values])
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
    
    plt.tight_layout()
    plt.savefig('scaling_factors_bars.png', dpi=300, bbox_inches='tight')
    print('Saved scaling_factors_bars.png')
    plt.show()

def main():
    """主函数，生成所有图表"""
    print("Generating scaling factor visualizations...")
    
    # 生成线图
    plot_scaling_factors()
    
    # 生成热力图
    plot_scaling_heatmap()
    
    # 生成3D图
    try:
        plot_scaling_3d()
    except ImportError:
        print("3D plot requires mpl_toolkits, skipping...")
    
    # 生成柱状图
    plot_scaling_bar()
    
    print("All visualizations completed!")

if __name__ == "__main__":
    main()
