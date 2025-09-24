import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

k_values = [2, 4, 8, 16, 32, 64, 128]

expert_data = {
    1: [4773.864746, 3942.446289, 3208.585938, 2831.713623, 2582.916504, 2242.923828, 1853.956543],
    2: [4013.604248, 3254.828613, 2640.819092, 1840.053345, 1694.897827, 1348.560669, 938.2376709],
    4: [4182.399414, 3410.270996, 2607.682129, 2058.195801, 1478.701172, 1327.639404, 926.6615601],
    8: [4995.59277, 3489.796875, 2652.856445, 2109.582764, 1549.467529, 1200.161865, 915.9934692],
    16: [5667.7656, 4348.80469, 3326.047852, 2332.625488, 1705.43457, 1320.11853, 1011.082764],
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
        linewidth=3,
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

plt.savefig('ablation_multi_expert_recon_mse.png', dpi=300, bbox_inches='tight')
print('Saved ablation_multi_expert_recon_mse.png')

plt.show()