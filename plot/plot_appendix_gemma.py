import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# 数据准备 - 按照 layer 组织
k_values = np.array([2, 4, 8, 16, 32, 64, 128])

# 每个 layer 的数据字典
layer_data = {
    5: {
        'scale_mse': {2: 7900.5, 4: 4780.25, 8: 3150.88, 16: 2050.12, 32: 1550.45, 64: 1420.33, 128: 1350.1},
        'switch_mse': {2: 7966.846, 4: 4950.12, 8: 3420.55, 16: 2450.6, 32: 1781.955, 64: 1705.08, 128: 1650.22},
        'scale_loss': {2: 0.79855, 4: 0.830123, 8: 0.945221, 16: 0.975332, 32: 1.0, 64: 1.0, 128: 1.0},
        'switch_loss': {2: 0.694336, 4: 0.765221, 8: 0.820112, 16: 0.865445, 32: 0.891113, 64: 0.921386, 128: 0.935221},
    },
    12: {
        'scale_mse': {2: 5230.885, 4: 4736.18, 8: 3880.7, 16: 3080.57, 32: 2167.05, 64: 1762.93, 128: 1511.09},
        'switch_mse': {2: 5387.562, 4: 4994.77, 8: 3930.5, 16: 3181.73, 32: 2677.53, 64: 2190, 128: 1850},
        'scale_loss': {2: 0.800293, 4: 0.817383, 8: 0.937988, 16: 0.964824, 32: 1.0, 64: 1.0, 128: 1.0},
        'switch_loss': {2: 0.635254, 4: 0.767578, 8: 0.837891, 16: 0.908203, 32: 0.924804, 64: 0.972167, 128: 0.995},
    },
    18: {
        'scale_mse': {2: 14500.12, 4: 12800.55, 8: 11500.2, 16: 10100.3, 32: 8599.27, 64: 6459.78, 128: 5500.45},
        'switch_mse': {2: 15200.06, 4: 13900.22, 8: 12500.88, 16: 11200.45, 32: 9542.5, 64: 6439.35, 128: 5300.12},
        'scale_loss': {2: 0.620112, 4: 0.710223, 8: 0.805334, 16: 0.885112, 32: 0.933007, 64: 0.959433, 128: 0.957519},
        'switch_loss': {2: 0.588145, 4: 0.695112, 8: 0.790223, 16: 0.875334, 32: 0.916015, 64: 0.957519, 128: 0.998112},
    },
    23: {
        'scale_mse': {2: 85000.55, 4: 62000.12, 8: 45000.33, 16: 22000.88, 32: 8500.25, 64: 1750.45, 128: 1580.33},
        'switch_mse': {2: 91641.84, 4: 82000.55, 8: 70500.22, 16: 58000.1, 32: 44240.42, 64: 1694.63, 128: 1600.55},
        'scale_loss': {2: 0.720112, 4: 0.805221, 8: 0.860332, 16: 0.905112, 32: 0.930223, 64: 0.928112, 128: 0.940221},
        'switch_loss': {2: 0.705566, 4: 0.790112, 8: 0.845223, 16: 0.890334, 32: 0.934082, 64: 0.923339, 128: 0.938112},
    },
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

for layer in [5, 12, 18, 23]:
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    
    data = layer_data[layer]
    
    rects1 = ax1.bar(x - width/2, [data['scale_mse'][k] for k in k_values], 
                     width, label='Scale', color='#2a9d8f', hatch='\\\\')
    rects2 = ax1.bar(x + width/2, [data['switch_mse'][k] for k in k_values], 
                     width, label='Switch', color='#264653', hatch='///')
    
    ax1_twin = ax1.twinx()
    
    ax1_twin.plot(
        x,
        [data['scale_loss'][k] for k in k_values],
        marker='^', linestyle='-', linewidth=2.5, markersize=12,
        color='#e76f51', label='Scale (Recovered)', zorder=3
    )
    ax1_twin.plot(
        x,
        [data['switch_loss'][k] for k in k_values],
        marker='o', linestyle='-', linewidth=2.5, markersize=12,
        color='#e9c46a', label='Switch (Recovered)', zorder=3
    )
    
    ax1.set_ylabel('MSE')
    ax1_twin.set_ylabel('Loss Recovered')
    ax1.set_title(f'Layer {layer}', size=28)
    ax1.set_xlabel('Sparsity (L0)')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(k) for k in k_values])
    
    mse_values = list(data['scale_mse'].values()) + list(data['switch_mse'].values())
    mse_min, mse_max = min(mse_values), max(mse_values)
    mse_range = mse_max - mse_min
    ax1.set_ylim(max(0, mse_min - 0.1 * mse_range), mse_max + 0.1 * mse_range)
    
    loss_values = list(data['scale_loss'].values()) + list(data['switch_loss'].values())
    loss_min, loss_max = min(loss_values), max(loss_values)
    loss_range = loss_max - loss_min
    ax1_twin.set_ylim(max(0, loss_min - 0.1 * loss_range), min(1.1, loss_max + 0.1 * loss_range))
    
    ax1_twin.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    
    # 保存图片
    plt.tight_layout()
    filename = f'plot_appendix_gemma_layer_{layer}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f'Saved {filename}')
    plt.close()