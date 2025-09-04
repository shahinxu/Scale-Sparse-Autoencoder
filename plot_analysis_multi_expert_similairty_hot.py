import matplotlib.pyplot as plt
import numpy as np

# 不使用 Scale 的虚构数据 (max similarity > 0.9)
# 行: experts (1, 2, 4, 8, 16)
# 列: k (2, 4, 8, 16, 32, 64, 128)
data_dict = {
    'expert_values': [1, 2, 4, 8, 16],
    'k_values': [2, 4, 8, 16, 32, 64, 128],
    'values': [
        [0.0050, 0.0080, 0.0100, 0.0120, 0.0133, 0.0145, 0.0155],
        [0.0010, 0.0015, 0.0020, 0.0022, 0.0024, 0.0028, 0.0032],
        [0.0008, 0.0012, 0.0018, 0.0022, 0.0026, 0.0030, 0.0035],
        [0.0005, 0.0009, 0.0011, 0.0013, 0.0015, 0.0018, 0.0022],
        [0.0004, 0.0007, 0.0010, 0.0012, 0.0014, 0.0017, 0.0020]
    ]
}

# 转换为 NumPy 数组
heatmap_data = np.array(data_dict['values'])
experts = data_dict['expert_values']
k_values = data_dict['k_values']

# 统一画图风格
plt.rcParams.update({
    'font.size': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 22,
})

# 创建图表和子图（统一尺寸）
fig, ax = plt.subplots(figsize=(8, 5))

# 使用反转的 colormap，使值越高颜色越深
im = ax.imshow(heatmap_data, cmap='viridis_r')

# 设置刻度和标签
ax.set_xticks(np.arange(len(k_values)))
ax.set_yticks(np.arange(len(experts)))
ax.set_xticklabels(k_values)
ax.set_yticklabels(experts)

ax.set_xlabel('Sparsity (L0)')
ax.set_ylabel('# Experts')

# 添加颜色条
cbar = ax.figure.colorbar(im, ax=ax)

# 调整布局以确保一切都可见
plt.tight_layout()

# 保存图表（统一导出参数）
plt.savefig('analysis_multi_expert_similarity_heatmap.png', dpi=300, bbox_inches='tight')
print('Saved analysis_multi_expert_similarity_heatmap.png')