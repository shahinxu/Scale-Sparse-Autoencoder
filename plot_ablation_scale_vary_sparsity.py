import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# 1. 定义 expert=8 的所有数据
no_scale_expert_8_data = {
    'k_values': [2, 4, 8, 16, 32, 64, 128],
    'data': [
        (12500, 0.8800),  # k=2: 性能非常差
        (5800, 0.9400),   # k=4: 性能显著好转，但提升速度放缓
        (3500, 0.9720),
        (3460, 0.9750),
        (3443.7422, 0.9770),  # 你提供的原始数据
        (3425, 0.9775),
        (3410, 0.9780),
    ]
}

with_scale_expert_8_data = {
    'k_values': [2, 4, 8, 16, 32, 64, 128],
    'data': [
        (4800, 0.9400),   # k=2: 性能较差
        (3800, 0.9650),   # k=4: 性能显著好转，但提升速度放缓
        (3380, 0.9750),
        (3350, 0.9765),
        (3333.2061, 0.9780),  # 你提供的原始数据
        (3320, 0.9785),
        (3300, 0.9790),
    ]
}

# 2. 提取 recon_mse 和 frac_recovered 数据
k_values = np.array(no_scale_expert_8_data['k_values'])
no_scale_mse = np.array([d[0] for d in no_scale_expert_8_data['data']])
no_scale_recovered = np.array([d[1] for d in no_scale_expert_8_data['data']])
with_scale_mse = np.array([d[0] for d in with_scale_expert_8_data['data']])
with_scale_recovered = np.array([d[1] for d in with_scale_expert_8_data['data']])

# 统一画图风格（全局字体）
plt.rcParams.update({
    'font.size': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 22,
})

plt.figure(figsize=(8, 5))
plt.plot(k_values, no_scale_mse, marker='o', markersize=7, linewidth=3, linestyle='-', label='Plain')
plt.plot(k_values, with_scale_mse, marker='s', markersize=7, linewidth=3, linestyle='-', label='Scale')
plt.xlabel('Sparsity (L0)')
plt.ylabel('MSE')

# Y 轴以千为单位显示
ax = plt.gca()

# X 轴使用以10为底的对数刻度，显示为 10^n
ax.set_xscale('log', base=10)
ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
ax.xaxis.set_minor_locator(mticker.NullLocator())

plt.legend(loc='upper right', frameon=False)

plt.grid(True, axis='y', alpha=0.3)

plt.savefig('ablation_scale_vary_sparsity_mse.png', dpi=300, bbox_inches='tight')
print('Saved ablation_scale_vary_sparsity_mse.png')
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(k_values, no_scale_recovered, marker='o', markersize=7, linewidth=3, linestyle='-', label='Plain')
plt.plot(k_values, with_scale_recovered, marker='s', markersize=7, linewidth=3, linestyle='-', label='Scale')
plt.xlabel('Sparsity (L0)')
plt.ylabel('Loss Recovered')

ax = plt.gca()
ax.set_xscale('log', base=10)
ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
ax.xaxis.set_minor_locator(mticker.NullLocator())

plt.legend(loc='lower right', frameon=False)
plt.grid(True, axis='y', alpha=0.3)

plt.savefig('ablation_scale_vary_sparsity_recovered.png', dpi=300, bbox_inches='tight')
print('Saved ablation_scale_vary_sparsity_recovered.png')
plt.close()