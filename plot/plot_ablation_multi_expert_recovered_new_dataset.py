import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

k_values = [2, 4, 8, 16, 32, 64, 128]

expert_data = {
    1: [0.75, 0.803, 0.849, 0.877, 0.899, 0.921, 0.949],
    2: [0.774, 0.824, 0.855, 0.909, 0.925, 0.952, 0.968],
    4: [0.782, 0.823, 0.855, 0.902, 0.938, 0.958, 0.974],
    8: [0.77, 0.828, 0.867, 0.903, 0.944, 0.962, 0.976],
    16: [0.773, 0.803, 0.861, 0.902, 0.933, 0.959, 0.973],
}

markers = ['o', 's', '^', 'D', 'v']
colors = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']

plt.rcParams.update({
    'font.size': 38,
    'axes.labelsize': 38,
    'xtick.labelsize': 38,
    'ytick.labelsize': 38,
})

plt.figure(figsize=(10, 8))

for i, (expert_count, frac_recovered_values) in enumerate(expert_data.items()):
    plt.plot(
        k_values,
        frac_recovered_values,
        marker=markers[i],
        linestyle='-',
        linewidth=3,
        color=colors[i],
        markersize=20,
        label=f'act={expert_count}'
    )

plt.xlabel('Sparsity (L0)')
plt.ylabel('Loss Recovered')

ax = plt.gca()
ax.set_xscale('log', base=10)
ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
ax.xaxis.set_minor_locator(mticker.NullLocator())

plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

plt.savefig('ablation_multi_expert_recovered_new_dataset.png', dpi=300, bbox_inches='tight')
print('Saved ablation_multi_expert_recovered_new_dataset.png')