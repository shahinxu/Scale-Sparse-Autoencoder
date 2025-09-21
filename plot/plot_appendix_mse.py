import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

markers = ['o', 's', '^', 'D', 'v', '*']
colors = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51', '#0f4c5c']

plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})

k_values = [2, 4, 8, 16, 32, 64, 128]

data = {
    'Mean-based (Ours)': [4228.131836, 3385.18334, 2668.222412, 2074.655762, 1446.149658, 1118.567017, 882.0480957],
    'Identity-based': [4610.35, 3622.09, 2881.64, 2245.88, 1655.22, 1252.79, 996.41],
    'Learned-based': [4290.76, 3275.44, 2725.13, 2060.91, 1479.8, 1003.55, 831.23]
}

plt.figure(figsize=(12, 8))

for i, (model, mse_values) in enumerate(data.items()):
    x = k_values
    clr = colors[i % len(colors)]
    mkr = markers[i % len(markers)]
    plt.plot(
        x,
        mse_values,
        marker=mkr,
        linestyle='-',
        linewidth=3,
        color=clr,
        markersize=12,
        label=model
    )

plt.xlabel('Sparsity (L0)')
plt.ylabel('MSE')

ax = plt.gca()

ax.set_xscale('log', base=10)
ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
ax.xaxis.set_minor_locator(mticker.NullLocator())

ax.set_yscale('log', base=10)
ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
ax.yaxis.set_minor_locator(mticker.NullLocator())

plt.legend(loc='upper right', frameon=True)

plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

plt.savefig('appendix_sae.png', dpi=300, bbox_inches='tight')
print('Saved appendix_sae.png')