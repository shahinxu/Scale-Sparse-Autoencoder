#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os


def main():
    # Activations and average overlap counts (user-provided)
    activations = np.array([2, 4, 8, 16])
    scale = np.array([7.809635, 7.171092, 7.794615, 8.114607])
    plain = np.array([8.259099, 7.591157, 8.601194, 8.260692])

    out = "feature_overlap_vs_activation.png"
    plt.figure(figsize=(7, 4.5))
    plt.plot(activations, scale, marker='o', markersize=7, linewidth=2, label='Scale', color='#1f77b4')
    plt.plot(activations, plain, marker='s', markersize=7, linewidth=2, label='Plain', color='#ff7f0e')

    # Use log2 scale so 2,4,8,16 are evenly spaced
    plt.xscale('log', base=2)
    plt.xticks(activations, [str(int(a)) for a in activations])
    # set minor ticks at powers of two if desired
    plt.gca().get_xaxis().set_minor_locator(plt.FixedLocator([]))

    plt.xlabel('Activation (E)')
    plt.ylabel('Average overlap count')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out, dpi=300)
    print(f"Saved {os.path.abspath(out)}")


if __name__ == '__main__':
    main()
