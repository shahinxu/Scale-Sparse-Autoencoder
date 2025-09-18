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
    'font.size': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
})

x = np.arange(len(k_values))
width = 0.4

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

# expert=8
rects1 = ax1.bar(x - width/2, no_scale_mse_8, width, label='Plain', color='#264653', hatch='///')
rects2 = ax1.bar(x + width/2, with_scale_mse_8, width, label='Scale', color='#2a9d8f', hatch='\\\\')
ax1_twin = ax1.twinx()
rects3 = ax1_twin.bar(x - width/2, no_scale_recovered_8, width, color='#264653', hatch='///')
rects4 = ax1_twin.bar(x + width/2, with_scale_recovered_8, width, color='#2a9d8f', hatch='\\\\')
ax1.set_ylabel('MSE')
ax1_twin.set_ylabel('Loss Recovered')
ax1_twin.invert_yaxis()
ax1.set_title('# activated experts = 8')
ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax1.set_ylim(800, 8800)
ax1_twin.set_ylim(1.0, 0.85)
ax1_twin.set_yticks(np.linspace(1, 0.85, 6))
ax1.legend(loc='lower left', frameon=True)
# expert=16
rects5 = ax2.bar(x - width/2, no_scale_mse_16, width, label='Plain', color='#264653', hatch='///')
rects6 = ax2.bar(x + width/2, with_scale_mse_16, width, label='Scale', color='#2a9d8f', hatch='\\\\')
ax2_twin = ax2.twinx()
rects7 = ax2_twin.bar(x - width/2, no_scale_recovered_16, width, color='#264653', hatch='///')
rects8 = ax2_twin.bar(x + width/2, with_scale_recovered_16, width, color='#2a9d8f', hatch='\\\\')
ax2.set_ylabel('MSE')
ax2_twin.set_ylabel('Loss Recovered')
ax2_twin.invert_yaxis()
ax2.set_title('# activated experts = 16')
ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax2.set_ylim(800, 10800)
ax2_twin.set_ylim(1.0, 0.85)
ax2_twin.set_yticks(np.linspace(1, 0.85, 6))

# 只在最下面的图显示横坐标和标签
ax2.set_xlabel('Sparsity (L0)')
ax2.set_xticks(x)
ax2.set_xticklabels([str(k) for k in k_values])

plt.tight_layout()
plt.savefig('ablation_scale_vary_sparsity.png', dpi=300, bbox_inches='tight')
print('Saved ablation_scale_vary_sparsity.png')