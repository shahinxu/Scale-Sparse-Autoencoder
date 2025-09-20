import matplotlib.pyplot as plt
import numpy as np
import os
from ..utils.file_utils import sanitize_filename

def create_token_visualization(token_text, batch_idx, token_pos, 
                             top_k_indices, top_k_values,
                             expert_activations, expert_activation_counts, expert_total_activation,
                             recon_error, original_activation, reconstructed_activation, 
                             text_folder, original_text, k, model_info):
    """创建token可视化 - 将每个子图保存为独立的图像文件。"""
    
    plt.rcParams.update({
        'font.size': 22,
        'axes.labelsize': 22,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
    })
    
    try:
        top_k_indices = top_k_indices.cpu()
        top_k_values = top_k_values.cpu()
        original_activation = original_activation.cpu()
        reconstructed_activation = reconstructed_activation.cpu()

        safe_token_name = sanitize_filename(token_text)
        if not safe_token_name:
            safe_token_name = f"pos_{token_pos}"
        
        base_filename = os.path.join(text_folder, f"token_{token_pos:02d}_{safe_token_name}")

        if expert_activation_counts:
            fig1, ax1 = plt.subplots(figsize=(12, 7))
            expert_ids = sorted(expert_activation_counts.keys())
            counts = [expert_activation_counts[eid] for eid in expert_ids]
            total_activations = [expert_total_activation[eid] for eid in expert_ids]
            
            expert_ids = [int(eid) for eid in expert_ids]
            xlabel = 'Expert ID' if model_info['is_multi_expert'] else 'Expert'

            bars1 = ax1.bar(
                [x - 0.2 for x in expert_ids], 
                counts, 
                width=0.4, 
                color='#264653', 
                label='Feature Count'
            )
            ax1_twin = ax1.twinx()
            bars2 = ax1_twin.bar(
                [x + 0.2 for x in expert_ids], 
                total_activations, 
                width=0.4, 
                color='#2a9d8f', 
                label='Activation Strength'
            )
            
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel('Feature Count')
            ax1_twin.set_ylabel('Activation Strength')
            ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

            ax1.set_xlim(0.5, 64.5)
            ax1.set_xticks([1, 16, 32, 48, 64])
            
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            fig1.savefig(f"{base_filename}_1_expert_overview.png", dpi=300, bbox_inches='tight')
            plt.close(fig1)

        if expert_activations:
            fig3, ax3 = plt.subplots(figsize=(max(15, len(top_k_indices) * 0.2), 7))
            expert_ids_with_features = sorted(expert_activations.keys())
            x_offset = 0
            x_ticks, x_labels = [], []
            
            # 使用您指定的颜色列表
            color_palette = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51', '#0f4c5c']

            for i, expert_id in enumerate(expert_ids_with_features):
                features = sorted(expert_activations[expert_id], key=lambda x: x[2], reverse=True)
                feature_values = [f[2] for f in features]
                x_positions = np.arange(len(feature_values)) + x_offset
                
                color = color_palette[i % len(color_palette)] if model_info['is_multi_expert'] else '#2a9d8f'
                label = f'Expert {expert_id} ({len(features)} features)' if model_info['is_multi_expert'] else f'Single Expert ({len(features)} features)'
                center_label = f'E{expert_id}' if model_info['is_multi_expert'] else 'Expert'
                
                ax3.bar(x_positions, feature_values, color=color, label=label)
                
                if len(feature_values) > 0:
                    x_ticks.append(x_offset + len(feature_values) / 2 - 0.5)
                    x_labels.append(center_label)
                x_offset += len(feature_values) + 2 # 增加组间距
            
            xlabel = 'Features (grouped by Expert)' if model_info['is_multi_expert'] else 'Features'
            
            ax3.set_xlabel(xlabel)
            ax3.set_ylabel('Activation Value')
            if x_ticks:
                ax3.set_xticks(x_ticks)
                ax3.set_xticklabels(x_labels)
            ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
            
            # 将图例放在右上角
            if len(expert_ids_with_features) <= 10:
                ax3.legend(loc='upper right')
            
            fig3.savefig(f"{base_filename}_3_activation_values.png", dpi=300, bbox_inches='tight')
            plt.close(fig3)

        positive_values = top_k_values[top_k_values > 0]
        if top_k_values.numel() > 0:
            positive_values_np = positive_values.numpy()
            if len(positive_values_np) > 0:
                fig5, ax5 = plt.subplots(figsize=(8, 6))
                n_bins = min(15, len(positive_values_np))
                n, bins, patches = ax5.hist(positive_values_np, bins=n_bins, alpha=0.7, color='green', edgecolor='black')
                
                cm = plt.cm.YlOrRd
                for i, patch in enumerate(patches):
                    patch.set_facecolor(cm(i / len(patches)))
                
                ax5.set_xlabel('Activation Value')
                ax5.set_ylabel('Count')
                ax5.grid(True, alpha=0.3)
                
                mean_val, median_val = np.mean(positive_values_np), np.median(positive_values_np)
                ax5.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
                ax5.axvline(median_val, color='blue', linestyle='--', label=f'Median: {median_val:.3f}')
                ax5.legend()
                
                fig5.savefig(f"{base_filename}_5_activation_distribution.png", dpi=300, bbox_inches='tight')
                plt.close(fig5)

        print(f"      Visualizations saved for token '{token_text}' in folder: {text_folder}")
        
    except Exception as e:
        print(f"      Error creating visualization for token {token_pos}: {e}")
