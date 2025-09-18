import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

data_dict = {
    'expert_values': [1, 2, 4, 8, 16],
    'k_values': [2, 4, 8, 16, 32, 64, 128],
    'values': [
        [0.08521, 0.07892, 0.08115, 0.06943, 0.07159, 0.06813, 0.06822],
        [0.60563, 0.57677, 0.58629, 0.49651, 0.51089, 0.48382, 0.48382],
        [0.69961, 0.54533, 0.5095, 0.50278, 0.49184, 0.52612, 0.56932],
        [0.58239, 0.6763, 0.56059, 0.47736, 0.49431, 0.50805, 0.53598],
        [0.76398, 0.68303, 0.55334, 0.48002, 0.50156, 0.49175, 0.48879]
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

colors = ['#264653', '#2a9d8f']
n_bins = 100
custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

im = ax.imshow(heatmap_data, cmap=custom_cmap)

for i in range(len(experts)):
    for j in range(len(k_values)):
        value = heatmap_data[i, j]
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

plt.savefig('analysis_scale_origin_heatmap.png', dpi=300, bbox_inches='tight')
print('Saved analysis_scale_origin_heatmap.png')