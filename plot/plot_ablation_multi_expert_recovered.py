import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

k_values = [2, 4, 8, 16, 32, 64, 128]

expert_data = {
    1: [0.849060953, 0.895408571, 0.929964423, 0.946586311, 0.957092762, 0.968319893, 0.978588164],
    2: [0.880139172, 0.911716878, 0.935240328, 0.968944073, 0.972381473, 0.981268644, 0.991143823],
    4: [0.876437843, 0.911037385, 0.941088855, 0.962358356, 0.9787696, 0.98140949, 0.991612911],
    8: [0.869656086, 0.90976423, 0.940447927, 0.962647676, 0.975598931, 0.984841764, 0.990192473],
    16: [0.858631551, 0.889445186, 0.92530781, 0.956105411, 0.972185969, 0.982759476, 0.989533544],
}

markers = ['o', 's', '^', 'D', 'v']
colors = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']

plt.rcParams.update({
    'font.size': 42,
    'axes.labelsize': 42,
    'xtick.labelsize': 42,
    'ytick.labelsize': 42,
})

plt.figure(figsize=(10, 8))

# 4. 绘制每一条 expert 曲线
for i, (expert_count, frac_recovered_values) in enumerate(expert_data.items()):
    plt.plot(
        k_values,
        frac_recovered_values,
        marker=markers[i],
        linestyle='-',
        linewidth=3,
        color=colors[i],
        markersize=20,
        label=f'act={expert_count}'
    )

plt.xlabel('Sparsity (L0)')
plt.ylabel('Loss Recovered')

ax = plt.gca()
ax.set_xscale('log', base=10)
ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
ax.xaxis.set_minor_locator(mticker.NullLocator())

plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

plt.savefig('ablation_multi_expert_recovered.png', dpi=300, bbox_inches='tight')
print('Saved ablation_multi_expert_recovered.png')