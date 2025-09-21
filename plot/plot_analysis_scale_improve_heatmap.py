import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import os

data_dict = {
    'expert_values': [1, 2, 4, 8, 16],
    'k_values': [2, 4, 8, 16, 32, 64, 128],
    'values': [
        [-7.30, -3.03, -3.22, -3.21, -5.42, -5.70, -7.05],
        [3.93, 12.63, 13.08, 1.47, 7.47, -1.34, 7.11],
        [18.67, 12.14, 7.46, 8.74, 1.19, 0.85, 4.26],
        [31.25, 10.35, 0.71, 2.24, 1.44, 4.47, 1.40],
        [25.42, 5.67, 2.87, 1.04, 0.52, 3.36, 5.43]
    ]
}

heatmap_data = np.array(data_dict['values'])
experts = data_dict['expert_values']
k_values = data_dict['k_values']

plt.rcParams.update({
    'font.size': 28,
    'axes.labelsize': 28,
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'legend.fontsize': 24,
})

fig, ax = plt.subplots(figsize=(12, 8))

NEG_START = os.environ.get('NEG_START', '#e9c46a')  # 负侧起色（靠近零）
NEG_END = os.environ.get('NEG_END', '#e76f51')      # 负侧终色（更负）
POS_START = os.environ.get('POS_START', '#2a9d8f')  # 正侧起色（较小正值）
POS_END = os.environ.get('POS_END', '#264653')      # 正侧终色（较大正值）
MID_COLOR = '#ffffff'  # 中心白色（用于 0）


half = 128
pos_base = LinearSegmentedColormap.from_list('pos_base', [POS_START, POS_END], N=half)
pos_samples = pos_base(np.linspace(0, 1, half))
neg_base = LinearSegmentedColormap.from_list('neg_base', [NEG_END, NEG_START], N=half)
neg_samples = neg_base(np.linspace(0, 1, half))
listed_colors = np.vstack((neg_samples, pos_samples))
custom_cmap = ListedColormap(listed_colors)

POSITIVE_GAMMA = float(os.environ.get('POSITIVE_GAMMA', '0.6'))
NEGATIVE_GAMMA = float(os.environ.get('NEGATIVE_GAMMA', '0.6'))

vmin = float(np.nanmin(heatmap_data))
vmax = float(np.nanmax(heatmap_data))

mapped = np.zeros_like(heatmap_data, dtype=float)
for ii in range(heatmap_data.shape[0]):
    for jj in range(heatmap_data.shape[1]):
        val = heatmap_data[ii, jj]
        if val == 0 or np.isnan(val):
            mapped[ii, jj] = np.nan
            continue
        if val > 0:
            if vmax == 0:
                t = 1.0
            else:
                t0 = max(0.0, min(1.0, val / vmax))
                t = 0.5 + 0.5 * (t0 ** POSITIVE_GAMMA)
        else:
            if vmin == 0:
                t = 0.0
            else:
                t0 = max(0.0, min(1.0, abs(val) / abs(vmin)))
                t = 0.5 - 0.5 * (t0 ** NEGATIVE_GAMMA)
        mapped[ii, jj] = t

mapped_masked = np.ma.masked_invalid(mapped)
im = ax.imshow(mapped_masked, cmap=custom_cmap, vmin=0.0, vmax=1.0)
im.cmap.set_bad(color='white')

for i in range(len(experts)):
    for j in range(len(k_values)):
        value = heatmap_data[i, j]
        ax.text(j, i, f'{value:.2f}', ha='center', va='center', color='white', fontsize=24, weight='bold')


ax.set_xticks(np.arange(len(k_values)))
ax.set_xticklabels(k_values)
ax.set_yticks(np.arange(len(experts)))
ax.set_yticklabels(experts)

ax.set_xlabel('Sparsity (L0)')
ax.set_ylabel('# Activated Experts')

from matplotlib.ticker import FuncFormatter

def scalar_to_value(scalar: float) -> float:
    """Invert the mapped scalar in [0,1] back to the original data value."""
    if np.isnan(scalar):
        return 0.0
    if scalar == 0.5:
        return 0.0
    if scalar > 0.5:
        t0 = (scalar - 0.5) / 0.5
        val = (t0 ** (1.0 / POSITIVE_GAMMA)) * vmax if vmax != 0 else 0.0
        return float(val)
    else:
        t0 = (0.5 - scalar) / 0.5
        val = - (t0 ** (1.0 / NEGATIVE_GAMMA)) * abs(vmin) if vmin != 0 else 0.0
        return float(val)

desired_values = [-5.0, 0.0, 20.0]

def value_to_scalar(val: float) -> float:
    if val == 0.0:
        return 0.5
    if val > 0:
        if vmax == 0:
            return 1.0
        t0 = max(0.0, min(1.0, val / vmax))
        return 0.5 + 0.5 * (t0 ** POSITIVE_GAMMA)
    else:
        if vmin == 0:
            return 0.0
        t0 = max(0.0, min(1.0, abs(val) / abs(vmin)))
        return 0.5 - 0.5 * (t0 ** NEGATIVE_GAMMA)

tick_scalars = [value_to_scalar(v) for v in desired_values]
cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8, aspect=20, ticks=tick_scalars)

def tick_formatter(x, pos=None):
    v = scalar_to_value(x)
    if abs(v - round(v)) < 1e-6:
        return str(int(round(v)))
    return f"{v:.2f}"

cbar.ax.yaxis.set_major_formatter(FuncFormatter(tick_formatter))

plt.tight_layout()

plt.savefig('analysis_scale_improve_heatmap.png', dpi=300, bbox_inches='tight')
print('Saved analysis_scale_improve_heatmap.png')