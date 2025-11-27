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
    'Switch SAE': [0.377352804, 0.54021585, 0.642588019, 0.567230344, 0.439981908, 0.314763, 0.297340043],
    'Scale SAE, e=2': [0.199385494, 0.246667266, 0.332325995, 0.356271982, 0.4364793, 0.264378875, 0.26216045],
    'Scale SAE, e=4': [0.173432082, 0.187408686, 0.217131898, 0.282431692, 0.375346482, 0.306527853, 0.292672813],
    'Scale SAE, e=8': [0.170025498, 0.177756459, 0.188635156, 0.20917654, 0.258278966, 0.313415498, 0.28543359],
    'Scale SAE, e=16': [0.175783157, 0.182457849, 0.185959816, 0.191755056, 0.200798273, 0.219291687, 0.216600418]
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
plt.ylabel('Average Max Similarity')

ax = plt.gca()

ax.set_xscale('log', base=10)
ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
ax.xaxis.set_minor_locator(mticker.NullLocator())

# 不使用 log scale for y-axis，因为相似度值在 0-1 之间
ax.set_ylim(0, 1.0)

# plt.legend(loc='upper right', frameon=True)

plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

plt.savefig('plot_analysis_multi_expert_max_sim.png', dpi=300, bbox_inches='tight')
print('Saved plot_analysis_multi_expert_max_sim.png')