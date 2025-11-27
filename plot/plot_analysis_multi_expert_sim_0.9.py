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
    'Switch SAE': [0.119913737, 0.114054362, 0.049275716, 0.021931966, 0.00769043, 0.000610352, 0.0],
    'Scale SAE, e=2': [0.000325521, 0.001505534, 0.003458659, 0.002075195, 0.005045573, 0.00016276, 0.000447591],
    'Scale SAE, e=4': [0.000528971, 0.00016276, 0.000366211, 0.001383464, 0.001383464, 0.000610352, 0.001180013],
    'Scale SAE, e=8': [0.000244141, 0.000244141, 8.14E-05, 0.000284831, 0.000732422, 0.000569661, 0.000447591],
    'Scale SAE, e=16': [0.001505534, 0.000244141, 0.000406901, 0.000813802, 0.000569661, 0, 0]
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
plt.ylabel('Max Similarity > 0.9')

ax = plt.gca()

ax.set_xscale('log', base=10)
ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
ax.xaxis.set_minor_locator(mticker.NullLocator())

# 不使用 log scale for y-axis，因为相似度值在 0-1 之间
ax.set_ylim(0, 0.2)

# plt.legend(loc='upper right', frameon=True)

plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

plt.savefig('plot_analysis_multi_expert_sim_0.9.png', dpi=300, bbox_inches='tight')
print('Saved plot_analysis_multi_expert_sim_0.9.png')