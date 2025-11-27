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
    'Switch SAE': [0.26957194, 0.466389974, 0.48046875, 0.22664388, 0.066162109, 0.009480794, 8.138020833333333e-05],
    'Scale SAE, e=2': [0.01180013, 0.027872721, 0.067789714, 0.038289388, 0.083780924, 0.004394531, 0.009318034],
    'Scale SAE, e=4': [0.00402832, 0.005045573, 0.013549805, 0.031738281, 0.052246094, 0.015299479, 0.021606445],
    'Scale SAE, e=8': [0.002075195, 0.001831055, 0.004109701, 0.010620117, 0.018758138, 0.025309245, 0.019124349],
    'Scale SAE, e=16': [0.009765625, 0.006388346, 0.00374349, 0.004272461, 0.006551107, 0.006876628, 0.002360026]
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
plt.ylabel('Max Similarity > 0.7')

ax = plt.gca()

ax.set_xscale('log', base=10)
ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
ax.xaxis.set_minor_locator(mticker.NullLocator())

# 不使用 log scale for y-axis，因为相似度值在 0-1 之间
ax.set_ylim(0, 0.6)

# plt.legend(loc='upper right', frameon=True)

plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

plt.savefig('plot_analysis_multi_expert_sim_0.7.png', dpi=300, bbox_inches='tight')
print('Saved plot_analysis_multi_expert_sim_0.7.png')