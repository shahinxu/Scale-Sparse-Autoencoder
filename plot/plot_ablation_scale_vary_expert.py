import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# 数据准备
k_values = np.array([1, 2, 4, 8, 16])

no_scale_data_2 = {
    1: 4773.864746,
    2: 4013.604248,
    4: 4182.399414,
    8: 4995.59277,
    16: 5667.7656,
}
no_scale_data_32 = {
    1: 2582.916504,
    2: 1694.897827,
    4: 1478.701172,
    8: 1549.467529,
    16: 1705.43457,
}
with_scale_data_2 = {
    1: 4712.807129,
    2: 3939.815186,
    4: 4172.773438,
    8: 4228.131836,
    16: 4406.951172,
}
with_scale_data_32 = {
    1: 2505.731445,
    2: 1439.185791,
    4: 1370.657227,
    8: 1446.149658,
    16: 1643.577637,
}
no_scale_loss_recovered_2 = {
    1: 0.849060953,
    2: 0.880139172,
    4: 0.876437843,
    8: 0.869656086,
    16: 0.858631551,
}
no_scale_loss_recovered_32 = {
    1: 0.957092762,
    2: 0.972381473,
    4: 0.9787696,
    8: 0.975598931,
    16: 0.972185969,
}
with_scale_loss_recovered_2 = {
    1: 0.910709739,
    2: 0.917700708,
    4: 0.892298758,
    8: 0.881088138,
    16: 0.870648682,
}
with_scale_loss_recovered_32 = {
    1: 0.960371582,
    2: 0.981371582,
    4: 0.98475194,
    8: 0.982557178,
    16: 0.975340843,
}

plt.rcParams.update({
    'font.size': 28,
    'axes.labelsize': 28,
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'legend.fontsize': 24,
})

x = np.arange(len(k_values))
width = 0.4

# Horizontal layout
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

# L0=2
rects1 = ax1.bar(x - width/2, [no_scale_data_2[k] for k in k_values], width, label='Plain', color='#264653', hatch='///')
rects2 = ax1.bar(x + width/2, [with_scale_data_2[k] for k in k_values], width, label='Scale', color='#2a9d8f', hatch='\\\\')
ax1_twin = ax1.twinx()
# Loss Recovered as line plots (bright colors)
ax1_twin.plot(
    x,
    [no_scale_loss_recovered_2[k] for k in k_values],
    marker='o', linestyle='-', linewidth=2.5, markersize=12,
    color='#e9c46a', label='Plain (Recovered)', zorder=3
)
ax1_twin.plot(
    x,
    [with_scale_loss_recovered_2[k] for k in k_values],
    marker='^', linestyle='-', linewidth=2.5, markersize=12,
    color='#e76f51', label='Scale (Recovered)', zorder=3
)
ax1.set_ylabel('MSE')
ax1.set_title('L0=2', size=28)
ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

MSE_YMIN, MSE_YMAX = 1200, 6000
REC_YMIN, REC_YMAX = 0.84, 1.0
MSE_TICKS = np.linspace(MSE_YMIN, MSE_YMAX, 6)
REC_TICKS = np.linspace(REC_YMIN, REC_YMAX, 4)

ax1.set_ylim(MSE_YMIN, MSE_YMAX)
ax1.set_yticks(MSE_TICKS)
ax1.set_xlabel('Sparsity (L0)')
ax1_twin.set_ylim(REC_YMIN, REC_YMAX)
ax1_twin.set_yticks(REC_TICKS)

rects5 = ax2.bar(x - width/2, [no_scale_data_32[k] for k in k_values], width, label='Plain', color='#264653', hatch='///')
rects6 = ax2.bar(x + width/2, [with_scale_data_32[k] for k in k_values], width, label='Scale', color='#2a9d8f', hatch='\\\\')
ax2_twin = ax2.twinx()
ax2_twin.plot(
    x,
    [no_scale_loss_recovered_32[k] for k in k_values],
    marker='o', linestyle='-', linewidth=2.5, markersize=12,
    color='#e9c46a', label='Plain (Recovered)', zorder=3
)
ax2_twin.plot(
    x,
    [with_scale_loss_recovered_32[k] for k in k_values],
    marker='^', linestyle='-', linewidth=2.5, markersize=12,
    color='#e76f51', label='Scale (Recovered)', zorder=3
)

ax2_twin.set_ylabel('Loss Recovered')
ax2.set_title('L0=32', size=28)
ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax2.set_ylim(MSE_YMIN, MSE_YMAX)
ax2.set_yticks(MSE_TICKS)
ax2_twin.set_ylim(REC_YMIN, REC_YMAX)
ax2_twin.set_yticks(REC_TICKS)
ax2_twin.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

ax2.set_xlabel('Sparsity (L0)')
ax2.set_xticks(x)
ax2.set_xticklabels([str(k) for k in k_values])

ax2.tick_params(axis='y', which='both', left=False, labelleft=False)
ax1_twin.tick_params(axis='y', which='both', right=False, labelright=False)
ax2_twin.tick_params(axis='y', which='both', right=True, labelright=True)

plt.tight_layout()
plt.savefig('ablation_scale_vary_expert.png', dpi=300, bbox_inches='tight')
print('Saved ablation_scale_vary_expert.png')