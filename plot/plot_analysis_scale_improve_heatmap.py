import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
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
    'font.size': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
})

fig, ax = plt.subplots(figsize=(12, 8))

# 可通过环境变量或直接改下面的默认值来自定义颜色
LEFT_COLOR = os.environ.get('LEFT_COLOR', '#e9c46a')
RIGHT_COLOR = os.environ.get('RIGHT_COLOR', '#2a9d8f')
MID_COLOR = '#ffffff'

custom_cmap = LinearSegmentedColormap.from_list('custom_div',
                                                [LEFT_COLOR, MID_COLOR, RIGHT_COLOR],
                                                N=256)

# Non-linear mapping parameters (control how strongly values are pushed toward ends)
POSITIVE_GAMMA = float(os.environ.get('POSITIVE_GAMMA', '0.6'))  # <1 pushes moderate positives toward the right color
NEGATIVE_GAMMA = float(os.environ.get('NEGATIVE_GAMMA', '1.0'))  # <1 would push moderate negatives toward the left color

vmin = float(np.nanmin(heatmap_data))
vmax = float(np.nanmax(heatmap_data))

# Map raw data into a [0,1] scalar for the colormap, using separate transforms for neg/pos
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

# Mask zeros so they render as white
mapped_masked = np.ma.masked_invalid(mapped)
im = ax.imshow(mapped_masked, cmap=custom_cmap, vmin=0.0, vmax=1.0)
im.cmap.set_bad(color='white')

for i in range(len(experts)):
    for j in range(len(k_values)):
        value = heatmap_data[i, j]
        if value == 0:
            txt_color = 'black'
        else:
            # sample using the mapped scalar we computed
            mapped_scalar = mapped[i, j]
            if np.isnan(mapped_scalar):
                rgba = (1.0, 1.0, 1.0, 1.0)
            else:
                rgba = custom_cmap(mapped_scalar)
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            txt_color = 'black' if luminance > 0.55 else 'white'
        ax.text(j, i, f'{value:.2f}', ha='center', va='center', color=txt_color, fontsize=18, weight='bold')

def _choose_ticks(labels, max_ticks: int = 4):
    n = len(labels)
    num = min(max_ticks, n)
    idxs = np.linspace(0, n - 1, num, dtype=int)
    idxs = np.unique(idxs)
    return idxs, [labels[i] for i in idxs]

xt_idxs, xt_lbls = _choose_ticks(k_values, max_ticks=4)
yt_idxs, yt_lbls = _choose_ticks(experts, max_ticks=3)
ax.set_xticks(xt_idxs)
ax.set_yticks(yt_idxs)
ax.set_xticklabels(xt_lbls)
ax.set_yticklabels(yt_lbls)

ax.set_xlabel('Sparsity (L0)')
ax.set_ylabel('# Experts')

# Create a colorbar that shows original data values by inverting the non-linear mapping
from matplotlib.ticker import FuncFormatter

def scalar_to_value(scalar: float) -> float:
    """Invert the mapped scalar in [0,1] back to the original data value."""
    if np.isnan(scalar):
        return 0.0
    if scalar == 0.5:
        return 0.0
    if scalar > 0.5:
        t0 = (scalar - 0.5) / 0.5
        # invert positive gamma: t0 = (val/vmax)**POSITIVE_GAMMA
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
    # convert scalar back to value and show minimal decimals
    v = scalar_to_value(x)
    if abs(v - round(v)) < 1e-6:
        return str(int(round(v)))
    return f"{v:.2f}"

cbar.ax.yaxis.set_major_formatter(FuncFormatter(tick_formatter))

plt.tight_layout()

plt.savefig('analysis_scale_improve_heatmap.png', dpi=300, bbox_inches='tight')
print('Saved analysis_scale_improve_heatmap.png')