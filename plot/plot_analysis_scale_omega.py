import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# 设置matplotlib参数
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
})

def plot_scaling_factors():
    
    k_values = [128, 64, 32, 16, 8]
    e_values = [1, 2, 4, 8, 16]
    
    scaling_data = {
        # Predicted e=1 values prepended (heuristic estimate)
        128: [0.132259, 0.255731, 0.575299, 0.838069, 0.974341],
        64:  [0.116879, 0.226037, 0.418602, 0.932057, 1.079104],
        32:  [0.213772, 0.413530, 0.724287, 0.926098, 1.037019],
        16:  [0.160512, 0.310390, 0.702867, 0.859730, 0.986968],
        8:   [0.241357, 0.466853, 0.627181, 0.784058, 0.930623],
    }
    
    colors = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
    markers = ['o', 's', '^', 'D', 'v',]
    
    plt.figure(figsize=(12, 4))
    
    for i, k in enumerate(k_values):
        plt.plot(e_values, scaling_data[k], 
                color=colors[i], 
                marker=markers[i], 
                linestyle='-',
                linewidth=3,
                markersize=12,
                label=f'L0={k}')
    
    plt.xlabel('# Activated Experts')
    plt.ylabel('Scaling Factor (ω)')
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    plt.legend()
    
    plt.xticks(e_values)
    
    plt.ylim(0, 1.4)
    # 设置x轴为对数轴
    plt.xscale('log', base=2)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mticker.LogLocator(base=2.0))
    ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=2.0))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    plt.tight_layout()
    plt.savefig('analysis_scale_omega.png', dpi=300, bbox_inches='tight')
    print('Saved analysis_scale_omega.png')
     

if __name__ == "__main__":
    plot_scaling_factors()
