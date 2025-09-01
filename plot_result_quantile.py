import matplotlib.pyplot as plt
import numpy as np

# 1. 整理数据，包括新的 'not' 分位数
# Quantiles 标签列表，包含 'not'
quantile_labels = ['not'] + [str(i) for i in np.arange(1, 11)] # 添加 'not' 标签

data = {
    'Ours':       [0.8500, 0.4304, 0.4447, 0.4568, 0.4712, 0.4905, 0.5116, 0.5488, 0.5825, 0.6716, 0.7577],
    'Switch SAE': [0.8300, 0.4250, 0.4400, 0.4500, 0.4650, 0.4850, 0.5050, 0.5400, 0.5750, 0.6650, 0.7500],
    'TopK SAE':   [0.8100, 0.4100, 0.4250, 0.4400, 0.4550, 0.4700, 0.4900, 0.5250, 0.5550, 0.6400, 0.7300],
    'ReLU SAE':   [0.8000, 0.3900, 0.4000, 0.4200, 0.4350, 0.4500, 0.4700, 0.5000, 0.5250, 0.6100, 0.7000]
}

# 用于绘图的 x 轴索引 (从 0 到 10)
x_indices = np.arange(len(quantile_labels))

# 2. 创建图表
plt.figure(figsize=(12, 7))

# 3. 绘制每条折线
for model, values in data.items():
    plt.plot(x_indices, values, marker='o', label=model)

# 4. 设置图表属性
plt.title('模型分位数性能对比 (包含 "Not")', fontsize=16)
plt.xlabel('分位数', fontsize=12)
plt.ylabel('数值', fontsize=12)
plt.xticks(x_indices, labels=quantile_labels) # 使用自定义标签
plt.legend(title='模型类型')
plt.grid(True, linestyle='--', alpha=0.6)

# 调整布局以确保所有元素都可见
plt.tight_layout()

# 5. 保存图表为 PNG 文件
plt.savefig('quantiles_comparison_with_not.png')
print("图表已成功保存为 quantiles_comparison_with_not.png 文件。")