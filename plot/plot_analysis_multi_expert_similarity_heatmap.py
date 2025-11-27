import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

data_dict = {
    'expert_values': [1, 2, 4, 8, 16],
    'k_values': [2, 4, 8, 16, 32, 64, 128],
    'values': [
        [0.205973307, 0.239624023, 0.106079102, 0.034057617, 0.00394694, 0.00394694, 0],
        [0.00114746, 0.00279215, 0.00336588, 0.01371216, 0.00351888, 0.00254354, 0.00170206],
        [0.00049723, 0.00072672, 0.00187418, 0.00319376, 0.00596679, 0.00374837, 0.00460896],
        [0.00034424, 0.00026774, 0.00065023, 0.00139608, 0.00231405, 0.00395874, 0.00483846],
        [0.00076497, 0.00120483, 0.00101359, 0.00070760, 0.00097534, 0.00168294, 0.00313639]
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

fig, ax = plt.subplots(figsize=(12, 6))

colors = ['#264653', '#2a9d8f']
n_bins = 100
custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

im = ax.imshow(heatmap_data, cmap=custom_cmap, aspect='auto')

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
ax.set_ylabel('# Activated Experts (e)')

cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8, aspect=20)

plt.tight_layout()

plt.savefig('analysis_multi_expert_similarity_heatmap.png', dpi=300, bbox_inches='tight')
print('Saved analysis_multi_expert_similarity_heatmap.png')