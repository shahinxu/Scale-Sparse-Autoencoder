import matplotlib.pyplot as plt
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
    'font.size': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
})

x = np.arange(len(k_values))
width = 0.4

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

# L0=2
rects1 = ax1.bar(x - width/2, [no_scale_data_2[k] for k in k_values], width, label='Plain', color='#264653', hatch='///')
rects2 = ax1.bar(x + width/2, [with_scale_data_2[k] for k in k_values], width, label='Scale', color='#2a9d8f', hatch='\\\\')
ax1_twin = ax1.twinx()
rects3 = ax1_twin.bar(x - width/2, [no_scale_loss_recovered_2[k] for k in k_values], width, color='#264653', hatch='///')
rects4 = ax1_twin.bar(x + width/2, [with_scale_loss_recovered_2[k] for k in k_values], width, color='#2a9d8f', hatch='\\\\')
ax1.set_ylabel('MSE')
ax1_twin.set_ylabel('Loss Recovered')
ax1_twin.invert_yaxis()
ax1.set_title('L0=2')
ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax1.set_ylim(3800, 6000)
ax1_twin.set_ylim(1.0, 0.84)
ax1_twin.set_yticks(np.linspace(1, 0.84, 3))
ax1.legend(loc='lower left', frameon=True)

# L0=32
rects5 = ax2.bar(x - width/2, [no_scale_data_32[k] for k in k_values], width, label='Plain', color='#264653', hatch='///')
rects6 = ax2.bar(x + width/2, [with_scale_data_32[k] for k in k_values], width, label='Scale', color='#2a9d8f', hatch='\\\\')
ax2_twin = ax2.twinx()
rects7 = ax2_twin.bar(x - width/2, [no_scale_loss_recovered_32[k] for k in k_values], width, color='#264653', hatch='///')
rects8 = ax2_twin.bar(x + width/2, [with_scale_loss_recovered_32[k] for k in k_values], width, color='#2a9d8f', hatch='\\\\')
ax2.set_ylabel('MSE')
ax2_twin.set_ylabel('Loss Recovered')
ax2_twin.invert_yaxis()
ax2.set_title('L0=32')
ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax2.set_ylim(1200, 3000)
ax2_twin.set_ylim(1.0, 0.95)
ax2_twin.set_yticks(np.linspace(1, 0.95, 6))

# 只在最下面的图显示横坐标和标签
ax2.set_xlabel('Sparsity (L0)')
ax2.set_xticks(x)
ax2.set_xticklabels([str(k) for k in k_values])

plt.tight_layout()
plt.savefig('ablation_scale_vary_expert.png', dpi=300, bbox_inches='tight')
print('Saved ablation_scale_vary_expert.png')