import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

markers = ['o', 's', '^', 'D', 'v', '*']
colors = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51', '#0f4c5c']

plt.rcParams.update({
    'font.size': 42,
    'axes.labelsize': 42,
    'xtick.labelsize': 42,
    'ytick.labelsize': 42,
})

k_values = [2, 4, 8, 16, 32, 64, 128]

# Use the Loss Recovered values provided by the user for Scale and Switch
data = {
    'Scale SAE': [0.800292969, 0.817382813, 0.937988281, 0.964824219, 1.0, 1.0, 1.0],
    'Switch SAE': [0.635253906, 0.767578125, 0.837890625, 0.908203125, 0.924804688, 0.972167969, 0.995],
}

plt.figure(figsize=(10, 8))

for i, (model, recovered_values) in enumerate(data.items()):
    x = k_values
    clr = colors[i % len(colors)]
    mkr = markers[i % len(markers)]
    plt.plot(
        x,
        recovered_values,
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

plt.ylim(0.6, 1.01)

plt.legend(loc='lower right', frameon=True)

plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

plt.savefig('result_gemma_recovered.png', dpi=300, bbox_inches='tight')
print('Saved result_gemma_recovered.png')