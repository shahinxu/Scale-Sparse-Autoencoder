import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# 收集所有 k 值下的数据
k_values = [2, 4, 8, 16, 32, 64, 128]

# 专家数量为 expert 的数据，每个列表中的值对应 k_values 中的一个
expert_data = {
    1: [18000, 17000, 16000, 15000, 13091.416, 12000, 10000],
    2: [4000, 3800, 3600, 3400, 3347.2798, 3200, 3000],
    4: [3500, 3300, 3100, 2900, 2930.2666, 2800, 2600],
    8: [3800, 3600, 3400, 3200, 3443.7422, 3000, 2800],
    16: [4100, 3900, 3700, 3500, 3637.6928, 3300, 3100],
}

# 统一画图风格（与 plot_multi_expert.py 一致）
plt.rcParams.update({
    'font.size': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 22,
})

# 创建图表（统一尺寸）
plt.figure(figsize=(8, 5))

# 绘制每一条 expert 折线
for expert_count, mse_values in expert_data.items():
    plt.plot(
        k_values,
        mse_values,
        marker='o',
        linestyle='-',
        label=f'{expert_count} experts'
    )

# 坐标轴标签（标题移除以统一风格）
plt.xlabel('Sparsity (L0)')
plt.ylabel('MSE')

ax = plt.gca()
ax.set_xscale('log', base=10)
ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
ax.xaxis.set_minor_locator(mticker.NullLocator())

plt.legend(frameon=False)

# 网格风格统一
plt.grid(True, axis='y', alpha=0.3)

# 保存到文件（统一导出参数）
plt.savefig('ablation_multi_expert_recon_mse.png', dpi=300, bbox_inches='tight')
print('Saved ablation_multi_expert_recon_mse.png')

# 显示图表
plt.show()