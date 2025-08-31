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

    out = "feature_overlap_vs_activation.png"
    plt.figure(figsize=(8, 5))
    plt.plot(activations, scale, marker='o', markersize=7, linewidth=3, label='Scale', color='#1f77b4')
    plt.plot(activations, plain, marker='s', markersize=7, linewidth=3, label='Plain', color='#ff7f0e')

    plt.xscale('log', base=2)
    plt.xticks(activations, [str(int(a)) for a in activations])
    plt.gca().get_xaxis().set_minor_locator(plt.FixedLocator([]))

    plt.xlabel('Activation (E)')
    plt.ylabel('Similarity')
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved {os.path.abspath(out)}")


if __name__ == '__main__':
    main()
