import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

markers = ['o', 's', '^', 'D', 'v', '*']  # Circle, Square, Up Triangle, Diamond, Down Triangle, Star
colors = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51', '#0f4c5c']  # Added a sixth color for the sixth marker

plt.rcParams.update({
    'font.size': 42,
    'axes.labelsize': 42,
    'xtick.labelsize': 42,
    'ytick.labelsize': 42,
})


k_values = [2, 4, 8, 16, 32, 64, 128]

data = {
    'Scale SAE, act=8': [4228.131836, 3385.18334, 2668.222412, 2074.655762, 1446.149658, 1118.567017, 882.0480957],
    'Scale SAE, act=4': [4105.019531, 3318.044678, 2556.581299, 1859.144287, 1344.437134, 1056.427124, 846.8684082],
    'Scale SAE, act=2': [4019.097168, 3142.984863, 2359.544678, 1706.990112, 1303.804199, 1044.636597, 825.8157349],
    'Switch SAE': [4421.600586, 3598.834961, 2908.496582, 2401.701904, 2076.499023, 1800.87085, 1446.733398],
    'TopK SAE': [4262.05957, 3567.152344, 3083.789795, 2675.5, 2240.794189, 1896.879639, 1437.120483],
}

# Gated SAE — uses its own k values (not aligned to the k_values above)
gated_k = [2.046143, 2.858521, 7.129517, 12.41503906, 33.4967041, 82.11975098, 217.4451904]
gated_mse = [7799.085938, 5966.50293, 4370.03418, 3585.277832, 2843.496094, 2202.016846, 1495.226563]

# Add Gated SAE to the plotting sequence (distinct marker/color)
data['Gated SAE'] = gated_mse
data_kmap = { 'Gated SAE': gated_k }

plt.figure(figsize=(10, 8))

for i, (model, mse_values) in enumerate(data.items()):
    # decide x-coordinates: some models use the global k_values, others have explicit k arrays
    x = data_kmap.get(model, k_values)
    clr = colors[i % len(colors)]
    mkr = markers[i % len(markers)]
    plt.plot(
        x,
        mse_values,
        marker=mkr,
        linestyle='-',
        linewidth=3,
        color=clr,
        markersize=20,
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
ax.set_ylim(top=1e4)

# plt.legend(loc='upper right', frameon=True)

plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

plt.savefig('result_mse.png', dpi=300, bbox_inches='tight')
print('Saved result_mse.png')