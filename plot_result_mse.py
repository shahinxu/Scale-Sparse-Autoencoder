import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

markers = ['o', 's', '^', 'D', 'v']  # 圆形, 方形, 上三角, 菱形, 下三角
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

plt.rcParams.update({
    'font.size': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
})

k_values = [2, 4, 8, 16, 32, 64, 128]

data = {
    'Scale SAE': [6800.00, 3900.00, 3500.00, 3380.00, 3333.21, 3300.00, 3280.00],
    'Switch SAE': [12500.00, 5800.00, 4200.00, 3900.00, 3750.00, 3600.00, 3520.00],
    'TopK SAE': [13800.00, 6200.00, 4650.00, 4200.00, 4000.00, 3850.00, 3750.00],
    'ReLU SAE': [15200.00, 7100.00, 5400.00, 4800.00, 4500.00, 4300.00, 4150.00]
}

plt.figure(figsize=(8, 8))

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

plt.legend(loc='upper right', frameon=True)

plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

plt.savefig('result_sae.png', dpi=300, bbox_inches='tight')
print('Saved result_sae.png')