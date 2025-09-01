import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Global plotting style to match plot_multi_expert.py
plt.rcParams.update({
    'font.size': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 22,
})


def main():
    activations = np.array([2, 4, 8, 16])
    scale = np.array([7.809635, 7.171092, 7.794615, 8.114607]) / 16.0
    plain = np.array([8.259099, 7.591157, 8.601194, 8.260692]) / 16.0

    # 简单误差（示例）：按数值的 5% 作为误差
    scale_err = 0.05 * scale
    plain_err = 0.05 * plain

    out = "analysis_feature_overlap.png"
    plt.figure(figsize=(8, 5))

    # 转为分组柱状图
    x = np.arange(len(activations))
    width = 0.35
    plt.bar(
        x - width/2,
        scale,
        width,
        yerr=scale_err,
        capsize=4,
        label='Scale',
        color='#337AFF',
        alpha=0.85,
    )
    plt.bar(
        x + width/2,
        plain,
        width,
        yerr=plain_err,
        capsize=4,
        label='Plain',
        color='#FF5733',
        alpha=0.85,
    )

    # 类别型 x 轴（显示为 2/4/8/16）
    plt.xticks(x, [str(int(a)) for a in activations])

    plt.xlabel('# Experts')
    plt.ylabel('Similarity')
    # Y 轴从 0.3 开始（如需 0.2，可将 bottom 改为 0.2）
    plt.ylim(bottom=0.4)
    plt.grid(True, axis='y', alpha=0.4)
    plt.legend(loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved {os.path.abspath(out)}")


if __name__ == '__main__':
    main()
