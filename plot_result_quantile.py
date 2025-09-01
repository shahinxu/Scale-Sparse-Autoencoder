import matplotlib.pyplot as plt
import numpy as np

# 添加 rcParams 全局字体设置
plt.rcParams.update({
    'font.size': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 22,
})

# 1. 整理数据，包括新的 'not' 分位数
# Quantiles 标签列表，包含 'not'
quantile_labels = ['not'] + [str(i) for i in np.arange(1, 11)] # 添加 'not' 标签

data = {
    'Scale SAE':       [0.8500, 0.4304, 0.4447, 0.4568, 0.4712, 0.4905, 0.5116, 0.5488, 0.5825, 0.6716, 0.7577],
    'Switch SAE': [0.8300, 0.4250, 0.4400, 0.4500, 0.4650, 0.4850, 0.5050, 0.5400, 0.5750, 0.6650, 0.7500],
    'TopK SAE':   [0.8100, 0.4100, 0.4250, 0.4400, 0.4550, 0.4700, 0.4900, 0.5250, 0.5550, 0.6400, 0.7300],
    'ReLU SAE':   [0.8000, 0.3900, 0.4000, 0.4200, 0.4350, 0.4500, 0.4700, 0.5000, 0.5250, 0.6100, 0.7000]
}

x_indices = np.arange(len(quantile_labels))

plt.figure(figsize=(8, 5))

# 3. 绘制每条折线 - 线条 linewidth=3、markersize=7
for model, values in data.items():
    plt.plot(x_indices, values, marker='o', linewidth=3, markersize=7, label=model)

plt.xlabel('Quantile')
plt.ylabel('Value')
plt.xticks(x_indices, labels=quantile_labels) # 使用自定义标签
plt.legend(loc='upper center', frameon=False)
plt.grid(True, axis='y', alpha=0.3)

plt.savefig('result_quantiles.png', dpi=300, bbox_inches='tight')
print('Saved result_quantiles.png')