import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

markers = ['o', 's', '^', 'D', 'v', '*']
colors = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51', '#0f4c5c']

plt.rcParams.update({
    'font.size': 28,
    'axes.labelsize': 28,
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'legend.fontsize': 24,
})

k_values = [2, 4, 8, 16, 32, 64, 128]

data = {
    'Scale SAE, act=8': [6936.268555, 5469.526855, 4724.931885, 3875.2854, 3040.996704, 2435.019409, 2010.432129],
    'Scale SAE, act=4': [6171.60791, 5332.038086, 4548.967529, 3610.129028, 2844.769165, 2340.628296, 1934.670837],
    'Scale SAE, act=2': [6196.699707, 5210.7388, 4357.586426, 3456.486938, 2833.997192, 2335.6220703125, 1952.842957],
    'Switch SAE': [6451.283691, 5590.405273, 4853.273926, 4223.412598, 3701.227539, 3270.440918, 2694.414063],
    'TopK SAE': [6101.383789, 5323.825684, 4781.323486, 4214.283936, 3605.59436, 3021.180908, 2400.605347],
}

gated_k = [2.046143, 2.858521, 7.129517, 12.41503906, 33.4967041, 82.11975098, 217.4451904]
gated_mse = [8317.752441, 7208.431885, 5616.853271, 4809.932129, 3869.896484, 3070.947266, 2220.185181]

# Add Gated SAE to the plotting sequence (distinct marker/color)
data['Gated SAE'] = gated_mse
data_kmap = { 'Gated SAE': gated_k }

plt.figure(figsize=(12, 8))

for i, (model, mse_values) in enumerate(data.items()):
    # decide x-coordinates: some models use the global k_values, others have explicit k arrays
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
        markersize=12,
        label=model
    )

plt.xlabel('Sparsity (L0)')
plt.ylabel('MSE')

ax = plt.gca()

ax.set_xscale('log', base=10)
ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
ax.xaxis.set_minor_locator(mticker.NullLocator())

ax.set_yscale('log', base=10)
from math import log10, floor, ceil

# Build fixed major ticks at 2×, 4×, 8× per decade within current y-limits
ymin, ymax = ax.get_ylim()
if ymin <= 0:
    ymin = 1e-6  # safety for log scale
emin = floor(log10(ymin))
emax = ceil(log10(ymax))

tick_vals = []
for e in range(emin - 1, emax + 2):
    for m in (2, 4, 8):
        v = m * (10 ** e)
        if ymin <= v <= ymax:
            tick_vals.append(v)

if tick_vals:
    ax.yaxis.set_major_locator(mticker.FixedLocator(tick_vals))
else:
    # Fallback to a reasonable default in the 10^3 range
    ax.yaxis.set_major_locator(mticker.FixedLocator([2e3, 4e3, 8e3]))

def _fmt_times_pow(y, pos=None):
    if y <= 0:
        return ""
    e = int(floor(log10(y)))
    a = y / (10 ** e)
    # Snap to nearest integer to avoid floating point artifacts (expects 2,4,8)
    a_int = int(round(a))
    return f"${a_int}\\times10^{{{e}}}$"

ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_times_pow))
ax.yaxis.set_minor_locator(mticker.NullLocator())

plt.legend(loc='upper right', frameon=True)

plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

plt.savefig('result_mse_new_dataset.png', dpi=300, bbox_inches='tight')
print('Saved result_mse_new_dataset.png')