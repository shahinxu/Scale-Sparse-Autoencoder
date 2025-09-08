import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

k_values = [2, 4, 8, 16, 32, 64, 128]

expert_data = {
    1: [18000, 17000, 16000, 15000, 13091.416, 12000, 10000],
    2: [4000, 3800, 3600, 3400, 3347.2798, 3200, 3000],
    4: [3500, 3300, 3100, 2900, 2930.2666, 2800, 2600],
    8: [3800, 3600, 3400, 3200, 3443.7422, 3000, 2800],
    16: [4100, 3900, 3700, 3500, 3637.6928, 3300, 3100],
}

# 定义不同的marker形状和颜色
markers = ['o', 's', '^', 'D', 'v']  # 圆形, 方形, 上三角, 菱形, 下三角
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

plt.rcParams.update({
    'font.size': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
})

plt.figure(figsize=(8, 8))

for i, (expert_count, mse_values) in enumerate(expert_data.items()):
    plt.plot(
        k_values,
        mse_values,
        marker=markers[i],
        linestyle='-',
        linewidth=3,
        color=colors[i],
        markersize=12,  # 调整marker大小
        label=f'{expert_count} experts'
    )

plt.xlabel('Sparsity (L0)')
plt.ylabel('MSE')

ax = plt.gca()
ax.set_xscale('log', base=10)
ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
ax.xaxis.set_minor_locator(mticker.NullLocator())

ax.set_yscale('log', base=10)
ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
ax.yaxis.set_minor_locator(mticker.NullLocator())

plt.legend(loc='center left', frameon=True)

plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

plt.savefig('ablation_multi_expert_recon_mse.png', dpi=300, bbox_inches='tight')
print('Saved ablation_multi_expert_recon_mse.png')

plt.show()