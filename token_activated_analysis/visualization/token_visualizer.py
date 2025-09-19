import matplotlib.pyplot as plt
import numpy as np
import os
from ..utils.file_utils import sanitize_filename

def create_token_visualization(token_text, batch_idx, token_pos, 
                             top_k_indices, top_k_values,
                             expert_activations, expert_activation_counts, expert_total_activation,
                             recon_error, original_activation, reconstructed_activation, 
                             text_folder, original_text, k, model_info):
    """创建token可视化 - 支持单专家和多专家模型"""
    
    try:
        top_k_indices = top_k_indices.cpu()
        top_k_values = top_k_values.cpu()
        original_activation = original_activation.cpu()
        reconstructed_activation = reconstructed_activation.cpu()
        
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1.2, 1], width_ratios=[1, 1, 1, 1])
        
        # 1. Expert概览 - 根据模型类型调整
        ax1 = fig.add_subplot(gs[0, :2])
        if expert_activation_counts:
            expert_ids = sorted(expert_activation_counts.keys())
            counts = [expert_activation_counts[eid] for eid in expert_ids]
            total_activations = [expert_total_activation[eid] for eid in expert_ids]
            
            # 为单专家模型调整标题和标签
            if model_info['is_multi_expert']:
                title_suffix = f"Token: \"{token_text}\" (Pos: {token_pos})"
                xlabel = 'Expert ID'
            else:
                title_suffix = f"Token: \"{token_text}\" (Pos: {token_pos}) - Single Expert Model"
                xlabel = 'Expert (Single Model)'
            
            # 创建双轴图表
            bars1 = ax1.bar([x - 0.2 for x in expert_ids], counts, width=0.4, 
                           color='steelblue', label='Feature Count')
            ax1_twin = ax1.twinx()
            bars2 = ax1_twin.bar([x + 0.2 for x in expert_ids], total_activations, width=0.4,
                                color='orange', label='Total Strength')
            
            ax1.set_title(f'Expert Activations Overview\n{title_suffix}', 
                          fontsize=14, fontweight='bold')
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel('Feature Count', color='steelblue')
            ax1_twin.set_ylabel('Total Activation', color='orange')
            ax1.grid(True, alpha=0.3)
            
            # 添加数值标注
            for bar, count in zip(bars1, counts):
                height = bar.get_height()
                ax1.annotate(f'{count}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, color='steelblue')
            
            for bar, val in zip(bars2, total_activations):
                height = bar.get_height()
                ax1_twin.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                                 xytext=(0, 3), textcoords="offset points",
                                 ha='center', va='bottom', fontsize=9, color='orange')
            
            # 添加图例
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # 2. 被激活的Feature详细列表 - 根据模型类型调整格式
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.axis('off')
        
        if expert_activations:
            # 收集所有激活的feature
            all_features = []
            for expert_id, features in expert_activations.items():
                for feature_id, local_id, value in features:
                    all_features.append((feature_id, expert_id, local_id, value))
            
            # 按激活值排序
            all_features.sort(key=lambda x: x[3], reverse=True)
            
            # 创建表格文本 - 根据模型类型调整格式
            feature_text = f"Top-{min(15, len(all_features))} Activated Features:\n"
            if model_info['is_multi_expert']:
                feature_text += "Rank | Feature ID | Expert | Local ID | Value\n"
                feature_text += "-" * 45 + "\n"
                
                for i, (fid, eid, lid, val) in enumerate(all_features[:15]):
                    feature_text += f"{i+1:3d}  | {fid:8d} | {eid:5d} | {lid:7d} | {val:6.3f}\n"
            else:
                feature_text += "Rank | Feature ID | Value\n"
                feature_text += "-" * 25 + "\n"
                
                for i, (fid, eid, lid, val) in enumerate(all_features[:15]):
                    feature_text += f"{i+1:3d}  | {fid:8d} | {val:6.3f}\n"
            
            if len(all_features) > 15:
                feature_text += f"... and {len(all_features) - 15} more features"
            
            ax2.text(0.05, 0.95, feature_text, transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        # 3. Feature activation values - 根据模型类型调整
        ax3 = fig.add_subplot(gs[1, :])
        if expert_activations:
            expert_ids_with_features = sorted(expert_activations.keys())
            x_offset = 0
            x_ticks = []
            x_labels = []
            
            for expert_id in expert_ids_with_features:
                features = expert_activations[expert_id]
                features.sort(key=lambda x: x[2], reverse=True)
                
                feature_values = [f[2] for f in features]
                feature_ids = [f[0] for f in features]
                x_positions = np.arange(len(feature_values)) + x_offset
                
                if model_info['is_multi_expert']:
                    color = plt.cm.Set3(expert_id / max(64, model_info['num_experts']))
                    label = f'Expert {expert_id} ({len(features)} features)'
                    center_label = f'E{expert_id}'
                else:
                    color = plt.cm.Set3(0.5)  # 固定颜色
                    label = f'Single Expert ({len(features)} features)'
                    center_label = 'Expert'
                
                bars = ax3.bar(x_positions, feature_values, color=color, label=label)
                
                # 在每个bar上标注feature ID
                for bar, fid, val in zip(bars, feature_ids, feature_values):
                    if val > 0.1:  # 只标注较大的激活值
                        ax3.annotate(f'{fid}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                    xytext=(0, 3), textcoords="offset points",
                                    ha='center', va='bottom', fontsize=8, rotation=45)
                
                if len(feature_values) > 0:
                    center_pos = x_offset + len(feature_values) / 2 - 0.5
                    x_ticks.append(center_pos)
                    x_labels.append(center_label)
                
                x_offset += len(feature_values) + 1
            
            if model_info['is_multi_expert']:
                title = 'Feature Activation Values by Expert (with Feature IDs)'
                xlabel = 'Features (grouped by Expert)'
            else:
                title = 'Feature Activation Values (Single Expert Model)'
                xlabel = 'Features'
            
            ax3.set_title(title, fontsize=14, fontweight='bold')
            ax3.set_xlabel(xlabel)
            ax3.set_ylabel('Activation Value')
            if x_ticks:
                ax3.set_xticks(x_ticks)
                ax3.set_xticklabels(x_labels)
            ax3.grid(True, alpha=0.3)
            
            if len(expert_ids_with_features) <= 10:
                ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # 4. 统计信息 - 添加模型类型信息
        ax4 = fig.add_subplot(gs[2, :2])
        ax4.axis('off')
        
        total_active_features = sum(len(features) for features in expert_activations.values())
        
        # 计算更多统计信息
        if top_k_values.numel() > 0:
            positive_values = top_k_values[top_k_values > 0]
            if len(positive_values) > 0:
                mean_activation = positive_values.mean().item()
                max_activation = positive_values.max().item()
                min_activation = positive_values.min().item()
                std_activation = positive_values.std().item()
            else:
                mean_activation = max_activation = min_activation = std_activation = 0.0
        else:
            mean_activation = max_activation = min_activation = std_activation = 0.0
        
        info_text = f"""Token: "{token_text}" (Position: {token_pos})
Original Text: {original_text[:50]}{'...' if len(original_text) > 50 else ''}
Text Index: {batch_idx}

=== Model Info ===
Type: {model_info['model_type']}
Multi-Expert: {model_info['is_multi_expert']}
Experts: {model_info['num_experts']}

=== Reconstruction ===
Error: {recon_error:.6f}

=== Feature Activation ===
Total Active: {total_active_features}/{k}
Activated Experts: {len(expert_activations)}
Mean Activation: {mean_activation:.4f}
Max Activation: {max_activation:.4f}
Std Activation: {std_activation:.4f}

=== Original Activation ===
Mean: {original_activation.mean().item():.4f}
Std:  {original_activation.std().item():.4f}
Range: [{original_activation.min().item():.4f}, {original_activation.max().item():.4f}]"""
        
        ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        # 5. Feature值分布直方图
        ax5 = fig.add_subplot(gs[2, 2:])
        if top_k_values.numel() > 0:
            positive_values = top_k_values[top_k_values > 0].numpy()
            if len(positive_values) > 0:
                n_bins = min(15, len(positive_values))
                n, bins, patches = ax5.hist(positive_values, bins=n_bins, 
                                          alpha=0.7, color='green', edgecolor='black')
                
                # 为不同范围的bin使用不同颜色
                cm = plt.cm.YlOrRd
                for i, (patch, value) in enumerate(zip(patches, n)):
                    patch.set_facecolor(cm(i / len(patches)))
                
                ax5.set_title(f'Feature Activation Distribution\n({len(positive_values)} active features)', 
                             fontsize=12, fontweight='bold')
                ax5.set_xlabel('Activation Value')
                ax5.set_ylabel('Count')
                ax5.grid(True, alpha=0.3)
                
                mean_val = np.mean(positive_values)
                median_val = np.median(positive_values)
                ax5.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
                ax5.axvline(median_val, color='blue', linestyle='--', label=f'Median: {median_val:.3f}')
                ax5.legend(fontsize=9)
        
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, 
                           wspace=0.3, hspace=0.4)
        
        safe_token_name = sanitize_filename(token_text)
        if not safe_token_name:
            safe_token_name = f"pos_{token_pos}"
        
        visualization_file = os.path.join(text_folder, f"token_{token_pos:02d}_{safe_token_name}_analysis.png")
        plt.savefig(visualization_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      Visualization saved to: {visualization_file}")
        
    except Exception as e:
        print(f"      Error creating visualization for token {token_pos}: {e}")
