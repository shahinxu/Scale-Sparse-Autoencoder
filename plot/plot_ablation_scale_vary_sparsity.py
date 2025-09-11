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
    'font.size': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
})

# 设置柱状图的位置
x = np.arange(len(k_values))
width = 0.4

fig, ax1 = plt.subplots(figsize=(12, 8))

rects1 = ax1.bar(x - width/2, no_scale_mse, width, label='Plain', color='#264653', hatch='///')
rects2 = ax1.bar(x + width/2, with_scale_mse, width, label='Scale', color='#2a9d8f', hatch='\\\\')

ax1.set_xlabel('Sparsity (L0)')
ax1.set_ylabel('MSE', color='black')

ax1.set_xticks(x)
ax1.set_xticklabels([f'$10^{{{int(np.log10(k))}}}$' if k in [1, 10, 100, 1000] else str(k) for k in k_values])
ax1.set_ylim(2000, 16000)
ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

ax2 = ax1.twinx()

rects3 = ax2.bar(x - width/2, no_scale_recovered, width, color='#264653', hatch='///')
rects4 = ax2.bar(x + width/2, with_scale_recovered, width, color='#2a9d8f', hatch='\\\\')

ax2.set_ylabel('Loss Recovered', color='black')
ax2.set_ylim(1, 0.86)
ax2.invert_yaxis()

mse_min, mse_max = ax1.get_ylim()
ax2.set_ylim(1.0, 0.86)

ax1.legend(loc='lower left', frameon=True)

plt.savefig('ablation_scale_vary_sparsity.png', dpi=300, bbox_inches='tight')
print('Saved ablation_scale_vary_sparsity.png')