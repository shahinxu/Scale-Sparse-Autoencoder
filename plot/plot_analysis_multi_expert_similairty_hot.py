import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

data_dict = {
    'expert_values': [1, 2, 4, 8, 16],
    'k_values': [2, 4, 8, 16, 32, 64, 128],
    'values': [
        [0.0050, 0.0080, 0.0100, 0.0120, 0.0133, 0.0145, 0.0155],
        [0.0010, 0.0015, 0.0020, 0.0022, 0.0024, 0.0028, 0.0032],
        [0.0008, 0.0012, 0.0018, 0.0022, 0.0026, 0.0030, 0.0035],
        [0.0005, 0.0009, 0.0011, 0.0013, 0.0015, 0.0018, 0.0022],
        [0.0004, 0.0007, 0.0010, 0.0012, 0.0014, 0.0017, 0.0020]
    ]
}

heatmap_data = np.array(data_dict['values'])
experts = data_dict['expert_values']
k_values = data_dict['k_values']

plt.rcParams.update({
    'font.size': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
})

fig, ax = plt.subplots(figsize=(12, 8))

# 创建自定义颜色映射，从264653（深绿）到2a9d8f（浅绿）
colors = ['#264653', '#2a9d8f']
n_bins = 100
custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

im = ax.imshow(heatmap_data, cmap=custom_cmap)

for i in range(len(experts)):
    for j in range(len(k_values)):
        value = heatmap_data[i, j]
        # 在深绿色背景上使用白色文字以确保可读性
        ax.text(j, i, f'{value:.4f}', ha='center', va='center', 
                color='white', fontsize=18, weight='bold')

ax.set_xticks(np.arange(len(k_values)))
ax.set_yticks(np.arange(len(experts)))
ax.set_xticklabels(k_values)
ax.set_yticklabels(experts)

ax.set_xlabel('Sparsity (L0)')
ax.set_ylabel('# Experts')

cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8, aspect=20)

plt.tight_layout()

plt.savefig('analysis_multi_expert_similarity_heatmap.png', dpi=300, bbox_inches='tight')
print('Saved analysis_multi_expert_similarity_heatmap.png')