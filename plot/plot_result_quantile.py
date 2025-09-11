import matplotlib.pyplot as plt
import numpy as np

markers = ['o', 's', '^', 'D', 'v']
colors = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']

plt.rcParams.update({
    'font.size': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
})

quantile_labels = ['not'] + [str(i) for i in np.arange(1, 11)]

data = {
    'Scale SAE':  [0.8500, 0.4304, 0.4447, 0.4568, 0.4712, 0.4905, 0.5116, 0.5488, 0.5825, 0.6716, 0.7577],
    'Switch SAE': [0.8300, 0.4250, 0.4400, 0.4500, 0.4650, 0.4850, 0.5050, 0.5400, 0.5750, 0.6650, 0.7500],
    'TopK SAE':   [0.8100, 0.4100, 0.4250, 0.4400, 0.4550, 0.4700, 0.4900, 0.5250, 0.5550, 0.6400, 0.7300],
    'ReLU SAE':   [0.8000, 0.3900, 0.4000, 0.4200, 0.4350, 0.4500, 0.4700, 0.5000, 0.5250, 0.6100, 0.7000]
}

x_indices = np.arange(len(quantile_labels))

plt.figure(figsize=(12, 8))

for i, (model, values) in enumerate(data.items()):
    plt.plot(
        x_indices,
        values,
        marker=markers[i],
        linestyle='-',
        linewidth=3,
        color=colors[i],
        markersize=12,
        label=model
    )

plt.xlabel('Quantile')
plt.ylabel('Accuracy')
plt.xticks(x_indices, labels=quantile_labels)
plt.ylim(0.3, 1.0)
plt.legend(loc='upper center')
plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

plt.savefig('result_quantiles.png', dpi=300, bbox_inches='tight')
print('Saved result_quantiles.png')