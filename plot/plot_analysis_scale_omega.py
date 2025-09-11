import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# 设置matplotlib参数
plt.rcParams.update({
    'font.size': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 18,
})

def plot_scaling_factors():
    
    k_values = [32, 16, 8]
    e_values = [1, 2, 4, 8, 16]
    
    scaling_data = {
        32: [0.213772, 0.41353, 0.724287, 0.926098, 1.037019],
        16: [0.218654, 0.453421, 0.789543, 1.024567, 1.156783],
        8:  [0.224891, 0.498765, 0.867432, 1.145632, 1.298456]
    }
    
    colors = ['#264653', '#2a9d8f', '#e9c46a']
    markers = ['o', 's', '^']
    linestyles = ['-', '-', '-']
    
    plt.figure(figsize=(12, 8))
    
    for i, k in enumerate(k_values):
        plt.plot(e_values, scaling_data[k], 
                color=colors[i], 
                marker=markers[i], 
                linestyle=linestyles[i],
                linewidth=3,
                markersize=12,
                label=f'L0={k}')
    
    plt.xlabel('# Experts')
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
