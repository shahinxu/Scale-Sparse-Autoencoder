import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

data_dict = {
    'expert_values': [1, 2, 4, 8, 16],
    'k_values': [2, 4, 8, 16, 32, 64, 128],
    'values': [
        [0.205973307, 0.239624023, 0.106079102, 0.034057617, 0.00394694, 0.00394694, 0],
        [0.002441406, 0.005940755, 0.007161458, 0.029174805, 0.007486979, 0.005411784, 0.003621419],
        [0.001057943, 0.001546224, 0.00398763, 0.006795247, 0.012695313, 0.00797526, 0.009806315],
        [0.000732422, 0.000569661, 0.001383464, 0.002970378, 0.004923503, 0.008422852, 0.010294596],
        [0.001627604, 0.002563477, 0.002156576, 0.001505534, 0.002075195, 0.003580729, 0.006673177]
    ]
}

heatmap_data = np.array(data_dict['values'])
experts = data_dict['expert_values']
k_values = data_dict['k_values']

plt.rcParams.update({
    'font.size': 28,
    'axes.labelsize': 28,
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'legend.fontsize': 24,
})

fig, ax = plt.subplots(figsize=(12, 8))

colors = ['#264653', '#2a9d8f']
n_bins = 100
custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

im = ax.imshow(heatmap_data, cmap=custom_cmap)

for i in range(len(experts)):
    for j in range(len(k_values)):
        value = heatmap_data[i, j]
        ax.text(j, i, f'{value:.3f}', ha='center', va='center', 
                color='white', fontsize=22, weight='bold')

ax.set_xticks(np.arange(len(k_values)))
ax.set_yticks(np.arange(len(experts)))
ax.set_xticklabels(k_values)
ax.set_yticklabels(experts)

ax.set_xlabel('Sparsity (L0)')
ax.set_ylabel('# Activated Experts')

cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8, aspect=20)

plt.tight_layout()

plt.savefig('analysis_multi_expert_similarity_heatmap.png', dpi=300, bbox_inches='tight')
print('Saved analysis_multi_expert_similarity_heatmap.png')