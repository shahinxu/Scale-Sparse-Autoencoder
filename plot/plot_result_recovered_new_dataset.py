import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

markers = ['o', 's', '^', 'D', 'v', '*']
colors = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51', '#0f4c5c']

plt.rcParams.update({
    'font.size': 38,
    'axes.labelsize': 38,
    'xtick.labelsize': 38,
    'ytick.labelsize': 38,
})

k_values = [2, 4, 8, 16, 32, 64, 128]

data = {
    'Scale SAE, act=8': [0.775369674, 0.818651825, 0.861592829, 0.903992236, 0.942239374, 0.964557976, 0.97557804],
    'Scale SAE, act=4': [0.770707428, 0.825474739, 0.875085533, 0.92136389, 0.954742849, 0.970399886, 0.977922767],
    'Scale SAE, act=2': [0.781481624, 0.835264802, 0.886027902, 0.928794861, 0.953777462, 0.968555063, 0.978331238],
    'Switch SAE': [0.764527589, 0.819699824, 0.860574096, 0.897912025, 0.9229514, 0.944583118, 0.965470165],
    'TopK SAE': [0.794894636, 0.84003368, 0.867943674, 0.898465306, 0.932026476, 0.95340845, 0.97251153],
}

gated_k = [2.046143, 2.858521, 7.129517, 12.41503906, 33.4967041, 82.11975098, 217.4451904]
gated_recovered = [0.646245956, 0.739900529, 0.828819364, 0.882518023, 0.930768758, 0.956792921, 0.972725451]

data['Gated SAE'] = gated_recovered
data_kmap = { 'Gated SAE': gated_k }

plt.figure(figsize=(10, 8))

for i, (model, mse_values) in enumerate(data.items()):
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
plt.ylabel('Loss Recovered')

ax = plt.gca()

ax.set_xscale('log', base=10)
ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
ax.xaxis.set_minor_locator(mticker.NullLocator())

plt.ylim(0.6, 1.0)

# plt.legend(loc='lower right', frameon=True)

plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

plt.savefig('result_recovered_new_dataset.png', dpi=300, bbox_inches='tight')
print('Saved result_recovered_new_dataset.png')