import matplotlib.pyplot as plt
import numpy as np

# Data: (expert, is_scale, value)
rows = [
    (1, False, 0.0133),
    (1, True, 0.0086),
    (2, False, 0.0023),
    (2, True, 0.0050),
    (4, False, 0.0026),
    (4, True, 0.0023),
    (8, False, 0.0015),
    (8, True, 0.0007),
    (16, False, 0.0014),
    (16, True, 0.0006),
]

experts = sorted(list({r[0] for r in rows}))
plain_vals = {e: 0.0 for e in experts}
scale_vals = {e: 0.0 for e in experts}

for e, is_scale, v in rows:
    if is_scale:
        scale_vals[e] = v
    else:
        plain_vals[e] = v

plain_list = [plain_vals[e] for e in experts]
scale_list = [scale_vals[e] for e in experts]

plt.rcParams.update({
    'font.size': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 22,
})

# Plot grouped bar chart
x = np.arange(len(experts))
width = 0.35

plt.figure(figsize=(8,5))
plt.bar(x - width/2, scale_list, width, label='Scale', color='#337AFF', alpha=0.85)
plt.bar(x + width/2, plain_list, width, label='Plain', color='#FF5733', alpha=0.85)
plt.xticks(x, experts)
plt.xlabel('# Experts')
plt.ylabel('Ratio')
# Remove title to match standard style
plt.legend(loc='upper right', frameon=False)
plt.grid(axis='y', alpha=0.3)
outpath = 'analysis_scale_simialrity.png'
plt.savefig(outpath, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved plot to {outpath}')
