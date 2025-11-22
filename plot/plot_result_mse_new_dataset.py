import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

markers = ['o', 's', '^', 'D', 'v', '*', 'P', 'X']
colors = ['#e9c46a', '#f4a261', '#e76f51', '#264653', '#2a9d8f', '#0f4c5c', '#606c38', '#669bbc']

plt.rcParams.update({
    'font.size': 42,
    'axes.labelsize': 42,
    'xtick.labelsize': 42,
    'ytick.labelsize': 42,
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

relu_k = [33.4, 39.7911377, 77.28234863, 359.7]
relu_mse = [9098.353841, 6600.878255, 5146.635091, 2089.457275]

jump_k = [3, 35.65783691, 48.17260742, 50.26379395, 62.65893555, 118.5947266]
jump_mse = [6300, 3784.858643, 3562.219401, 3517.568929, 3302.657796, 2902.764974]

data['Gated SAE'] = gated_mse
data_kmap = { 'Gated SAE': gated_k }
data['ReLU SAE'] = relu_mse
data_kmap['ReLU SAE'] = relu_k
data['Jump SAE'] = jump_mse
data_kmap['Jump SAE'] = jump_k

plt.figure(figsize=(10, 8))

for i, (model, mse_values) in enumerate(data.items()):
    x = data_kmap.get(model, k_values)
    y = mse_values
    clr = colors[i % len(colors)]
    mkr = markers[i % len(markers)]
    plt.plot(    
        x,
        y,
        marker=mkr,
        linestyle='-',
        linewidth=3,
        color=clr,
        markersize=20,
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
ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0,)))
ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0, labelOnlyBase=True))
ax.yaxis.set_minor_locator(mticker.NullLocator())
ax.set_ylim(top=1e4)

# plt.legend(loc='lower left', frameon=True)

plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

plt.savefig('result_mse_new_dataset.png', dpi=300, bbox_inches='tight')
print('Saved result_mse_new_dataset.png')