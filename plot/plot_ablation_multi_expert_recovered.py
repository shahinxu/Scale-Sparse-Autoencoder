import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# 1. 收集所有 k 值下的数据
k_values = [2, 4, 8, 16, 32, 64, 128]

# 2. 整理数据：将数据按 expert 数量分组
expert_data = {
    1: [0.8800, 0.9300, 0.9500, 0.9570, 0.9589, 0.9600, 0.9610],
    2: [0.9000, 0.9450, 0.9650, 0.9680, 0.9709, 0.9720, 0.9730],
    4: [0.9100, 0.9500, 0.9750, 0.9760, 0.9775, 0.9790, 0.9800],
    8: [0.9050, 0.9480, 0.9700, 0.9720, 0.9770, 0.9780, 0.9790],
    16: [0.9020, 0.9420, 0.9650, 0.9680, 0.9731, 0.9750, 0.9760],
}

# 定义不同的marker形状和颜色
markers = ['o', 's', '^', 'D', 'v']  # 圆形, 方形, 上三角, 菱形, 下三角
colors = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']

plt.rcParams.update({
    'font.size': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
})

# 3. 创建图表（统一尺寸）
plt.figure(figsize=(12, 8))

# 4. 绘制每一条 expert 曲线
for i, (expert_count, frac_recovered_values) in enumerate(expert_data.items()):
    plt.plot(
        k_values,
        frac_recovered_values,
        marker=markers[i],
        linestyle='-',
        linewidth=3,
        color=colors[i],
        markersize=12,
        label=f'{expert_count} experts'
    )

plt.xlabel('Sparsity (L0)')
plt.ylabel('Loss Recovered')

# 设置 x 轴为以10为底的对数刻度，并格式化为 10^n
ax = plt.gca()
ax.set_xscale('log', base=10)
ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
ax.xaxis.set_minor_locator(mticker.NullLocator())

# 图例放在右下角
plt.legend(loc='lower right')

plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

# 6. 保存图表为 PNG 文件（统一文件名与导出参数）
plt.savefig('ablation_multi_expert_recovered.png', dpi=300, bbox_inches='tight')
print('Saved ablation_multi_expert_recovered.png')