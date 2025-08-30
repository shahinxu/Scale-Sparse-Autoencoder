import matplotlib.pyplot as plt
import numpy as np

# Data: (expert, is_scale, value)
rows = [
    (1, False, 0.0133),
    (1, True, 0.0086),
    (2, False, 0.0053),
    (2, True, 0.0050),
    (4, False, 0.0026),
    (4, True, 0.0023),
    (8, False, 0.0015),
    (8, True, 0.0007),
    (16, False, 0.0014),
    (16, True, 0.0006),
]

# Aggregate into dictionaries
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

# Plot grouped bar chart
x = np.arange(len(experts))
width = 0.35

plt.figure(figsize=(8,5))
plt.bar(x - width/2, plain_list, width, label='Plain', color='C1')
plt.bar(x + width/2, scale_list, width, label='Scale', color='C0')
plt.xticks(x, experts)
plt.xlabel('Expert (activation)')
plt.ylabel('Max similarity > 0.9')
plt.title('Max similarity > 0.9 by Expert (Plain vs Scale)')
plt.legend()
plt.grid(axis='y', alpha=0.25)
plt.tight_layout()
outpath = 'scale_vs_plain_max_similarity_bar.png'
plt.savefig(outpath, dpi=300)
plt.close()
print(f'Saved plot to {outpath}')
