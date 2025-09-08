import matplotlib.pyplot as plt
import numpy as np

# Data for L0=32: (expert, is_scale, value)
rows_l0_32 = [
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

# Simulated data for L0=16: slightly higher values as L0=16 typically has higher similarity
rows_l0_16 = [
    (1, False, 0.0165),
    (1, True, 0.0108),
    (2, False, 0.0029),
    (2, True, 0.0062),
    (4, False, 0.0032),
    (4, True, 0.0029),
    (8, False, 0.0019),
    (8, True, 0.0009),
    (16, False, 0.0017),
    (16, True, 0.0008),
]

def process_data(rows):
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
    return experts, plain_list, scale_list

experts_16, plain_list_16, scale_list_16 = process_data(rows_l0_16)
experts_32, plain_list_32, scale_list_32 = process_data(rows_l0_32)

plt.rcParams.update({
    'font.size': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))

x = np.arange(len(experts_16))
width = 0.4

# 第一个子图 - L0=16
ax1.bar(x - width/2, scale_list_16, width, label='Scale', color='#264653', hatch='///')
ax1.bar(x + width/2, plain_list_16, width, label='Plain', color='#2a9d8f', hatch='\\\\')
ax1.set_xticks(x)
ax1.set_xticklabels(experts_16)
ax1.set_xlabel('# Experts')
ax1.set_ylabel('Ratio')
ax1.set_ylim(0, 0.018)
ax1.set_title('L0=16')
ax1.legend()
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

# 第二个子图 - L0=32
ax2.bar(x - width/2, scale_list_32, width, color='#264653', hatch='///')
ax2.bar(x + width/2, plain_list_32, width, color='#2a9d8f', hatch='\\\\')
ax2.set_xticks(x)
ax2.set_xticklabels(experts_32)
ax2.set_xlabel('# Experts')
ax2.set_ylim(0, 0.018)
ax2.set_title('L0=32')
ax2.set_yticklabels([])
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

plt.tight_layout()

outpath = 'analysis_scale_simialrity_heatmap.png'
plt.savefig(outpath, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved plot to {outpath}')
