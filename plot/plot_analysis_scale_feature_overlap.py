import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.size': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
})


def main():
    activations = np.array([1, 2, 4, 8, 16])
    scale = np.array([1.569235, 7.809635, 7.171092, 7.794615, 8.114607]) / 16.0
    plain = np.array([1.485429, 8.259099, 7.591157, 8.601194, 8.260692]) / 16.0

    out = "analysis_scale_feature_overlap.png"
    plt.figure(figsize=(8, 8))

    x = np.arange(len(activations))
    width = 0.4
    plt.bar(
        x - width/2,
        scale,
        width,
        label='Scale',
        color='#264653',
        hatch='///'
    )
    plt.bar(
        x + width/2,
        plain,
        width,
        label='Plain',
        color='#2a9d8f',
        hatch='\\\\'
    )

    plt.xticks(x, [str(int(a)) for a in activations])

    plt.xlabel('# Experts')
    plt.ylabel('Similarity')
    plt.ylim(0, 0.6)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved {os.path.abspath(out)}")


if __name__ == '__main__':
    main()
