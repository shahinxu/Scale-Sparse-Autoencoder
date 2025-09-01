import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# 统一画图风格
plt.rcParams.update({
    'font.size': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 22,
})

# 1. 整理数据
k_values = [2, 4, 8, 16, 32, 64, 128]

data = {
    'Scale SAE': [6800.00, 3900.00, 3500.00, 3380.00, 3333.21, 3300.00, 3280.00],
    'Switch SAE': [7200.00, 4200.00, 3700.00, 3550.00, 3450.00, 3400.00, 3380.00],
    'TopK SAE': [7800.00, 4500.00, 3950.00, 3800.00, 3650.00, 3580.00, 3550.00],
    'ReLU SAE': [8500.00, 5100.00, 4400.00, 4250.00, 4100.00, 4000.00, 3950.00]
}

# 2. 创建图表（标准尺寸）
plt.figure(figsize=(8, 5))

# 3. 绘制每条折线（加粗线条与标记）
for model, mse_values in data.items():
    plt.plot(k_values, mse_values, marker='o', linewidth=3, markersize=7, label=model)

# 4. 设置图表属性（去掉标题，英文标签）
plt.xlabel('Sparsity (L0)')
plt.ylabel('Reconstruction MSE (×10^3)')

# Y 轴以千为单位显示
ax = plt.gca()
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, pos: f"{y/1000:g}"))

# X 轴使用以10为底的对数刻度，显示为 10^n
ax.set_xscale('log', base=10)
ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
ax.xaxis.set_minor_locator(mticker.NullLocator())

# 图例放在右上角，无边框
plt.legend(loc='upper right', frameon=False)

# 网格（仅 y 轴）
plt.grid(True, axis='y', alpha=0.3)

# 5. 保存图表为 PNG 文件（标准导出参数）
plt.savefig('sae_mse_comparison.png', dpi=300, bbox_inches='tight')
print('Saved sae_mse_comparison.png')