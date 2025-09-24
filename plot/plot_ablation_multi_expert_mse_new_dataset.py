import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

k_values = [2, 4, 8, 16, 32, 64, 128]

expert_data = {
    1: [6704.312256, 5908.383301, 5239.859619, 4833.949707, 4400.213867, 3978.869629, 3288.005737],
    2: [6164.799316, 5466.811523, 4807.545166, 3968.953247, 3503.194092, 2976.406372, 2277.509766],
    4: [6171.250488, 5448.745361, 4864.013428, 4008.533081, 3208.398804, 2759.023315, 2134.672852],
    8: [7418.414551, 5401.917969, 4667.654785, 3876.642822, 3150.167236, 2536.723755, 2032.005981],
    16: [7450, 6100, 5301, 4022.934937, 3242.151733, 2614.449463, 2070.914673],
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

for i, (expert_count, mse_values) in enumerate(expert_data.items()):
    plt.plot(
        k_values,
        mse_values,
        marker=markers[i],
        linestyle='-',
        linewidth=4,
        color=colors[i],
        markersize=20,
        label=f'act={expert_count}'
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

plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

plt.savefig('ablation_multi_expert_recon_mse_new_dataset.png', dpi=300, bbox_inches='tight')
print('Saved ablation_multi_expert_recon_mse_new_dataset.png')

plt.show()