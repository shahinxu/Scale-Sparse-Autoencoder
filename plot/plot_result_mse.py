import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

markers = ['o', 's', '^', 'D', 'v']  # 圆形, 方形, 上三角, 菱形, 下三角
colors = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']

plt.rcParams.update({
    'font.size': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
})

k_values = [2, 4, 8, 16, 32, 64, 128]

data = {
    'Scale SAE, act=8': [4228.131836, 3385.18334, 2668.222412, 2074.655762, 1446.149658, 1118.567017, 882.0480957],
    'Scale SAE, act=4': [4105.019531, 3318.044678, 2556.581299, 1859.144287, 1344.437134, 1056.427124, 846.8684082],
    'Scale SAE, act=2': [4019.097168, 3142.984863, 2359.544678, 1706.990112, 1303.804199, 1044.636597, 825.8157349],
    'TopK': [4262.05957, 3567.152344, 3083.789795, 2675.5, 2240.794189, 1896.879639, 1437.120483],
    'switch SAE': [4634.476563, 4226.252441, 3455.060059, 2871.674316, 2290.039063, 1800.673828, 1255.720215],
}

plt.figure(figsize=(12, 8))

for i, (model, mse_values) in enumerate(data.items()):
    plt.plot(
        k_values,
        mse_values,
        marker=markers[i],
        linestyle='-',
        linewidth=3,
        color=colors[i],
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

plt.legend(loc='lower left', frameon=True)

plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

plt.savefig('result_sae.png', dpi=300, bbox_inches='tight')
print('Saved result_sae.png')