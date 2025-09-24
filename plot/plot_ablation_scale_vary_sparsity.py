import matplotlib.pyplot as plt
import numpy as np

# 数据准备
k_values = np.array([2, 4, 8, 16, 32, 64, 128])

# expert=8
no_scale_mse_8 = np.array([4995.59277, 3489.796875, 2652.856445, 2109.582764, 1549.467529, 1200.161865, 915.9934692])
no_scale_recovered_8 = np.array([0.869656086, 0.90976423, 0.940447927, 0.962647676, 0.975598931, 0.984841764, 0.990192473])
with_scale_mse_8 = np.array([4228.131836, 3385.18334, 2668.222412, 2074.655762, 1446.149658, 1118.567017, 882.0480957])
with_scale_recovered_8 = np.array([0.892298758, 0.948831856, 0.952491403, 0.96867311, 0.982557178, 0.993911932, 0.996744285])

# expert=16
no_scale_mse_16 = np.array([5667.7656, 4348.80469, 3326.047852, 2332.625488, 1705.43457, 1320.11853, 1011.082764])
no_scale_recovered_16 = np.array([0.858631551, 0.889445186, 0.92530781, 0.956105411, 0.972185969, 0.982759476, 0.989533544])
with_scale_mse_16 = np.array([4406.951172, 3567.803467, 2899.781738, 2230.072266, 1643.577637, 1230.804321, 933.011])
with_scale_recovered_16 = np.array([0.881088138, 0.916454613, 0.967925906, 0.970453382, 0.975340843, 0.982173383, 0.990958631])

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

# expert=8
rects1 = ax1.bar(x - width/2, no_scale_mse_8, width, label='Plain', color='#264653', hatch='///')
rects2 = ax1.bar(x + width/2, with_scale_mse_8, width, label='Scale', color='#2a9d8f', hatch='\\\\')
ax1_twin = ax1.twinx()
# Loss Recovered: line plots on twin axis
ax1_twin.plot(x, no_scale_recovered_8, marker='o', linestyle='-', linewidth=2.5, markersize=12,
              color='#e9c46a', label='Plain (Recovered)', zorder=3)
ax1_twin.plot(x, with_scale_recovered_8, marker='^', linestyle='-', linewidth=2.5, markersize=12,
              color='#e76f51', label='Scale (Recovered)', zorder=3)
ax1.set_ylabel('MSE')
# ax1_twin.set_ylabel('Loss Recovered')
ax1.set_title('e = 8', size=28)
ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax1.set_xlabel('Sparsity (L0)')
# Unify axis limits and ticks across panels
MSE_YMIN, MSE_YMAX = 800, 6000
REC_YMIN, REC_YMAX = 0.85, 1.0
MSE_TICKS = np.linspace(MSE_YMIN, MSE_YMAX, 6)
REC_TICKS = np.linspace(REC_YMIN, REC_YMAX, 4)

ax1.set_ylim(MSE_YMIN, MSE_YMAX)
ax1.set_yticks(MSE_TICKS)
ax1_twin.set_ylim(REC_YMIN, REC_YMAX)
ax1_twin.set_yticks(REC_TICKS)

rects5 = ax2.bar(x - width/2, no_scale_mse_16, width, label='Plain', color='#264653', hatch='///')
rects6 = ax2.bar(x + width/2, with_scale_mse_16, width, label='Scale', color='#2a9d8f', hatch='\\\\')
ax2_twin = ax2.twinx()
ax2_twin.plot(x, no_scale_recovered_16, marker='o', linestyle='-', linewidth=2.5, markersize=12,
              color='#e9c46a', label='Plain (Recovered)', zorder=3)
ax2_twin.plot(x, with_scale_recovered_16, marker='^', linestyle='-', linewidth=2.5, markersize=12,
              color='#e76f51', label='Scale (Recovered)', zorder=3)
# ax2.set_ylabel('MSE')
ax2_twin.set_ylabel('Loss Recovered')
ax2.set_title('e = 16', size=28)
ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax2.set_ylim(MSE_YMIN, MSE_YMAX)
ax2.set_yticks(MSE_TICKS)
ax2_twin.set_ylim(REC_YMIN, REC_YMAX)
ax2_twin.set_yticks(REC_TICKS)

ax2.set_xlabel('Sparsity (L0)')
ax2.set_xticks(x)
ax2.set_xticklabels([str(k) for k in k_values])

# Only show left y-axis (MSE) on first panel, hide on second
ax2.tick_params(axis='y', which='both', left=False, labelleft=False)

# Only show right y-axis (Recovered) on second panel; hide on first
ax1_twin.tick_params(axis='y', which='both', right=False, labelright=False)
ax2_twin.tick_params(axis='y', which='both', right=True, labelright=True)

plt.tight_layout()
plt.savefig('ablation_scale_vary_sparsity.png', dpi=300, bbox_inches='tight')
print('Saved ablation_scale_vary_sparsity.png')