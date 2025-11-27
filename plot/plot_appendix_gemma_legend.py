import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

plt.rcParams.update({
    'font.size': 42,
    'axes.labelsize': 42,
    'xtick.labelsize': 42,
    'ytick.labelsize': 42,
})

def draw_legend_with_frame():
    # 定义图例配置，与 plot_appendix_gemma.py 中的颜色和样式保持一致
    legend_config = [
        # 柱状图图例（使用 Patch）
        {'label': 'MSE, Scale', 'type': 'bar', 'color': '#2a9d8f', 'hatch': '\\\\'},
        {'label': 'MSE, Switch', 'type': 'bar', 'color': '#264653', 'hatch': '///'},
        # 折线图图例（使用 Line）
        {'label': 'Loss Recovered, Scale', 'type': 'line', 'color': '#e76f51', 'marker': '^'},
        {'label': 'Loss Recovered, Switch', 'type': 'line', 'color': '#e9c46a', 'marker': 'o'},
    ]

    # 创建画布
    fig, ax = plt.subplots(figsize=(20, 1.2))

    # 创建句柄
    handles = []
    for item in legend_config:
        if item['type'] == 'bar':
            # 柱状图使用 Patch
            h = mpatches.Patch(
                facecolor=item['color'],
                edgecolor='black',
                hatch=item['hatch'],
                label=item['label']
            )
        else:  # line
            # 折线图使用 Line2D
            h = mlines.Line2D(
                [], [], 
                color=item['color'], 
                marker=item['marker'],
                linestyle='-', 
                linewidth=5,
                markersize=26, 
                label=item['label']
            )
        handles.append(h)

    legend = ax.legend(
        handles=handles,
        loc='center',
        ncol=len(legend_config),
        frameon=True,           # 开启边框
        fancybox=True,          # 开启圆角效果
        framealpha=0.5,         # 边框不透明度
        fontsize=28,
    )

    frame = legend.get_frame()

    ax.axis('off')

    # 保存
    plt.tight_layout()
    plt.savefig('plot_appendix_gemma_legend.png', bbox_inches='tight', dpi=300)
    print('Saved plot_appendix_gemma_legend.png')
    plt.close()

if __name__ == '__main__':
    draw_legend_with_frame()
