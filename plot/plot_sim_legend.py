import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Styling aligned with other plotting scripts
markers = ['o', 's', '^', 'D', 'v']
colors = ['#e9c46a', '#e76f51', '#264653', '#2a9d8f', '#0f4c5c']

plt.rcParams.update({
    'font.size': 42,
    'axes.labelsize': 42,
    'xtick.labelsize': 42,
    'ytick.labelsize': 42,
})

def draw_legend_with_frame():
    # 1. 定义精确的颜色和样式
    # 颜色代码根据原图进行了微调，使其更接近
    legend_config = [
        {'label': 'Switch SAE', 'color': colors[0], 'marker': markers[0], 'size': 26},
        {'label': 'Scale SAE, e=2', 'color': colors[1], 'marker': markers[1], 'size': 26},
        {'label': 'Scale SAE, e=4', 'color': colors[2], 'marker': markers[2], 'size': 26},
        {'label': 'Scale SAE, e=8', 'color': colors[3], 'marker': markers[3], 'size': 26},
        {'label': 'Scale SAE, e=16', 'color': colors[4], 'marker': markers[4], 'size': 26},
    ]

    # 2. 创建画布
    # 增加画布宽度以容纳大字体和间距
    fig, ax = plt.subplots(figsize=(12, 1.2))

    # 3. 创建句柄
    handles = []
    for item in legend_config:
        h = mlines.Line2D(
            [], [], 
            color=item['color'], 
            marker=item['marker'],
            linestyle='-', 
            linewidth=5,
            markersize=item['size'], 
            label=item['label']
        )
        handles.append(h)

    legend = ax.legend(
        handles=handles,
        loc='center',
        ncol=len(legend_config),
        
        frameon=True,           # 开启边框
        fancybox=True,          # 开启圆角效果
        framealpha=0.5,           # 边框不透明度
        
        fontsize=28,
    )

    frame = legend.get_frame()

    ax.axis('off')

    # 7. 保存及显示
    plt.tight_layout()
    plt.savefig('legend_with_frame.png', bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == '__main__':
    draw_legend_with_frame()