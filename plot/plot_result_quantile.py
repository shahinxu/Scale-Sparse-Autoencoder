import matplotlib.pyplot as plt
import numpy as np

markers = ['o', 's', '^', 'D', 'v', '*']
colors = ['#264653', '#2a9d8f', '#e9c46a', '#0f4c5c', '#e76f51', '#f4a261']

plt.rcParams.update({
    'font.size': 38,
    'axes.labelsize': 38,
    'xtick.labelsize': 38,
    'ytick.labelsize': 38,
})


quantile_labels = ['not'] + [str(i) for i in np.arange(1, 11)]

data = {
    'Scale SAE, act=2':  [0.7569, 0.5169, 0.5292, 0.5400, 0.5491, 0.5624, 0.5801, 0.6077, 0.6347, 0.7001, 0.7648],
    'Scale SAE, act=4': [0.7597, 0.4699, 0.4797, 0.4909, 0.5037, 0.5183, 0.5346, 0.5606, 0.5916, 0.6648, 0.7468],
    'Scale SAE, act=8': [0.7521, 0.4821, 0.4924, 0.5042, 0.5158, 0.5316, 0.5498, 0.5818, 0.6125, 0.6913, 0.7724],
    'Switch SAE': [0.7494, 0.3231, 0.3466, 0.3563, 0.3802, 0.3998, 0.4201, 0.4567, 0.4949, 0.5900, 0.7027],
    'TopK SAE':   [0.7496, 0.3412, 0.3549, 0.3645, 0.3780, 0.3975, 0.4170, 0.4530, 0.4897, 0.5845, 0.6977],
    'Gated SAE':   [0.7557, 0.3067, 0.3203, 0.3306, 0.3435, 0.3617, 0.3811, 0.4154, 0.4502, 0.5395, 0.6495],
}

x_indices = np.arange(len(quantile_labels))

plt.figure(figsize=(10, 8))

for i, (model, values) in enumerate(data.items()):
    plt.plot(
        x_indices,
        values,
        marker=markers[i],
        linestyle='-',
        linewidth=3,
        color=colors[i],
        markersize=20,
        label=model
    )

plt.xlabel('Quantile')
plt.ylabel('Accuracy')
plt.xticks(x_indices, labels=quantile_labels)
plt.ylim(0.2, 1.0)
# plt.legend(loc='upper center')
plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

plt.savefig('result_quantiles.png', dpi=300, bbox_inches='tight')
print('Saved result_quantiles.png')