import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

markers = ['o', 's', '^', 'D', 'v']
colors = ['#e9c46a', '#e76f51', '#264653', '#2a9d8f', '#0f4c5c']

plt.rcParams.update({
    'font.size': 42,
    'axes.labelsize': 42,
    'xtick.labelsize': 42,
    'ytick.labelsize': 42,
})


k_values = [2, 4, 8, 16, 32, 64, 128]

data = {
    'Switch SAE': [0.220581055, 0.325236003, 0.240559896, 0.093383789, 0.028564453, 0.00382487, 0],
    'Scale SAE, e=2': [0.003540039, 0.008015951, 0.022135417, 0.011515299, 0.032348633, 0.001180013, 0.003214518],
    'Scale SAE, e=4': [0.001424154, 0.001424154, 0.004516602, 0.011230469, 0.017089844, 0.004191081, 0.008951823],
    'Scale SAE, e=8': [0.000854492, 0.000895182, 0.001627604, 0.003295898, 0.005615234, 0.008260091, 0.005655924],
    'Scale SAE, e=16': [0.004923503, 0.001627604, 0.001302083, 0.001586914, 0.002482096, 0.001871745, 0.000854492]
}

plt.figure(figsize=(10, 8))

for i, (model, values) in enumerate(data.items()):
    clr = colors[i % len(colors)]
    mkr = markers[i % len(markers)]
    plt.plot(
        k_values,
        values,
        marker=mkr,
        linestyle='-',
        linewidth=3,
        color=clr,
        markersize=20,
        label=model
    )

plt.xlabel('Sparsity (L0)')
plt.ylabel('Max Similarity > 0.8')

ax = plt.gca()

ax.set_xscale('log', base=10)
ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
ax.xaxis.set_minor_locator(mticker.NullLocator())

# 不使用 log scale for y-axis，因为相似度值在 0-1 之间
ax.set_ylim(0, 0.4)

# plt.legend(loc='upper right', frameon=True)

plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

plt.savefig('plot_analysis_multi_expert_sim_0.8.png', dpi=300, bbox_inches='tight')
print('Saved plot_analysis_multi_expert_sim_0.8.png')