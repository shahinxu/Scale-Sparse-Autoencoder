import matplotlib.pyplot as plt
import numpy as np

def draw_symmetric_final_analogy():
    fig = plt.figure(figsize=(12, 7), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # === 辅助函数：画波形 ===
    def draw_wave(x_start, y_start, width, height, type='input'):
        t = np.linspace(0, 4*np.pi, 100)
        trend = np.sin(t) * 0.8
        noise = np.random.normal(0, 0.2, 100) * np.sin(t*3)
        
        if type == 'input': y, color, lw = trend + noise, 'black', 2
        elif type == 'low': y, color, lw = trend, 'blue', 2
        elif type == 'high': y, color, lw = noise, 'red', 1.5
            
        x = np.linspace(x_start, x_start + width, 100)
        y = y_start + (y * 0.5) + height/2
        ax.plot(x, y, color=color, lw=lw)
        # 边框
        rect = plt.Rectangle((x_start, y_start), width, height, fill=False, edgecolor='#888', lw=1, linestyle=':')
        ax.add_patch(rect)

    # ==========================
    # 左侧：Signal Processing
    # ==========================
    ax.text(3, 7.5, "Signal Processing Domain", ha='center', fontsize=16, fontweight='bold')

    # 1. Input
    draw_wave(1.5, 5.8, 3, 1.0, type='input')
    ax.text(3, 6.9, "Raw Signal", ha='center', fontsize=11)

    # Arrows Down
    ax.annotate("", xy=(1.8, 4.8), xytext=(2.5, 5.7), arrowprops=dict(arrowstyle="->", lw=2, color='#555'))
    ax.annotate("", xy=(4.2, 4.8), xytext=(3.5, 5.7), arrowprops=dict(arrowstyle="->", lw=2, color='#555'))

    # 2. Components
    draw_wave(0.5, 3.5, 2.5, 1.0, type='low')
    ax.text(1.75, 3.2, "Low Freq (Trend)", ha='center', fontsize=11, color='blue', fontweight='bold')

    draw_wave(3.5, 3.5, 2.5, 1.0, type='high')
    ax.text(4.75, 3.2, "High Freq (Details)", ha='center', fontsize=11, color='red', fontweight='bold')
    
    # 3. Filter Operation (The Missing Part!)
    # Circle
    circle_left = plt.Circle((3, 2.0), 0.6, color='white', ec='black', lw=1.5, zorder=10)
    ax.add_patch(circle_left)
    ax.text(3, 2.0, "Filter", ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows to Filter
    ax.annotate("", xy=(2.6, 2.4), xytext=(1.8, 3.1), arrowprops=dict(arrowstyle="->", lw=1.5)) # Low in
    ax.text(2.3, 2.5, "Block/Remove", ha='right', fontsize=8, color='blue', style='italic') # Label

    ax.annotate("", xy=(3.4, 2.4), xytext=(4.2, 3.1), arrowprops=dict(arrowstyle="->", lw=1.5)) # High in
    ax.text(3.7, 2.5, "Pass", ha='left', fontsize=8, color='red', style='italic') # Label

    # 4. Final Output
    ax.annotate("", xy=(3, 1.1), xytext=(3, 1.4), arrowprops=dict(arrowstyle="->", lw=1.5))
    draw_wave(1.5, 0.0, 3, 1.0, type='high') # Output looks like High Freq
    ax.text(3, -0.3, "Filtered Signal", ha='center', fontsize=12, fontweight='bold', bbox=dict(fc='#f0f0f0', ec='black', boxstyle='round'))


    # ==========================
    # 分隔线
    # ==========================
    ax.plot([6, 6], [0, 8], 'k--', lw=1, alpha=0.2)


    # ==========================
    # 右侧：Feature Scaling
    # ==========================
    ax.text(9, 7.5, "Feature Scaling Domain (Ours)", ha='center', fontsize=16, fontweight='bold')

    # 1. Input
    ax.text(9, 6.3, "Encoder Weight\n$W$", ha="center", va="center", 
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black"), fontsize=11)

    # Arrows Down
    ax.annotate("", xy=(7.8, 4.8), xytext=(8.5, 5.7), arrowprops=dict(arrowstyle="->", lw=2, color='black'))
    ax.annotate("", xy=(10.2, 4.8), xytext=(9.5, 5.7), arrowprops=dict(arrowstyle="->", lw=2, color='black'))

    # 2. Components
    ax.text(7.5, 4.0, "Mean Feature $\\bar{W}$", ha="center", va="center", 
            bbox=dict(boxstyle="round,pad=0.5", fc="#E3F2FD", ec="blue"), fontsize=11, color='blue')

    ax.text(10.5, 4.0, "Residual $\\Delta W$", ha="center", va="center", 
            bbox=dict(boxstyle="round,pad=0.5", fc="#FFEBEE", ec="red"), fontsize=11, color='red')

    # 3. Scaling Operation
    # Circle
    circle_right = plt.Circle((9, 2.0), 0.6, color='white', ec='black', lw=1.5, zorder=10)
    ax.add_patch(circle_right)
    ax.text(9, 2.0, "Subtract\n& Scale", ha='center', va='center', fontsize=9)

    # Arrows to Operation
    ax.annotate("-", xy=(8.6, 2.4), xytext=(7.8, 3.5), arrowprops=dict(arrowstyle="->", lw=1.5)) # Mean in
    ax.annotate("+", xy=(9.4, 2.4), xytext=(10.2, 3.5), arrowprops=dict(arrowstyle="->", lw=1.5)) # Residual in

    # 4. Final Output
    ax.annotate("", xy=(9, 1.1), xytext=(9, 1.4), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(9, 0.5, "Scaled Feature\n$\\bar{W}+(1+\\omega) \\cdot \\Delta W$", ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.5", fc="#FFF3E0", ec="orange"), fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('plot_high_pass.png', dpi=300, bbox_inches='tight')

draw_symmetric_final_analogy()