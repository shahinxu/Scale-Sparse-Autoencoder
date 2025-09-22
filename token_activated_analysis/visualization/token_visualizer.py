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
        'font.size': 28,
        'axes.labelsize': 28,
        'xtick.labelsize': 26,
        'ytick.labelsize': 28,
        'legend.fontsize': 26,
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

        # --- 新增：生成Top 8特征文件 ---
        if expert_activations:
            all_features = []
            for expert_id, features in expert_activations.items():
                for feature_data in features:
                    # feature_data is a tuple, e.g., (feature_idx_in_expert, global_idx, activation_value)
                    activation_value = feature_data[2]
                    feature_idx = feature_data[0]
                    all_features.append({
                        'activation': activation_value,
                        'expert_id': expert_id,
                        'feature_id': feature_idx
                    })
            
            # 按激活值从高到低排序
            sorted_features = sorted(all_features, key=lambda x: x['activation'], reverse=True)
            
            # 获取前8个特征
            top_8_features = sorted_features[:8]
            
            # 准备写入文件的内容
            output_lines = [f"Top 8 features for token: '{token_text}' (Position: {token_pos})\n\n"]
            for i, feature in enumerate(top_8_features):
                line = (f"{i+1}. Activation: {feature['activation']:.4f}\n"
                        f"   - Expert ID: {feature['expert_id']}\n"
                        f"   - Feature Index: {feature['feature_id']}\n")
                output_lines.append(line)
            
            # 写入文件
            txt_filename = f"{base_filename}_top_8_features.txt"
            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write("\n".join(output_lines))
            print(f"      Top 8 features info saved to: {txt_filename}")

    except Exception as e:
        print(f"      Error during visualization or file creation for token {token_pos}: {e}")
        # Fallback to original exception handling if needed
        print(f"      Error creating visualization for token {token_pos}: {e}")
    
    try:
        if expert_activation_counts:
            fig1, ax1 = plt.subplots(figsize=(12, 8))
            expert_ids = sorted(expert_activation_counts.keys())
            counts = [expert_activation_counts[eid] for eid in expert_ids]
            total_activations = [expert_total_activation[eid] for eid in expert_ids]
            
            expert_ids = [int(eid) for eid in expert_ids]
            xlabel = 'Expert ID' if model_info['is_multi_expert'] else 'Expert'

            ax1.bar(
                [x + 1 - 0.2 for x in expert_ids], 
                counts, 
                width=0.4, 
                color='#264653', 
                label='Feature Count'
            )
            ax1_twin = ax1.twinx()
            ax1_twin.bar(
                [x + 1 + 0.2 for x in expert_ids], 
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

            fig1.savefig(f"{base_filename}_1_expert_overview.png", dpi=300)
            plt.close(fig1)

        if expert_activations:
            fig3, ax3 = plt.subplots(figsize=(12, 8))
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
                center_label = f'E{expert_id}' if model_info['is_multi_expert'] else 'Expert'

                ax3.bar(x_positions, feature_values, color=color)

                if len(feature_values) > 0:
                    x_ticks.append(x_offset + len(feature_values) / 2 - 0.5)
                    x_labels.append(center_label)
                x_offset += len(feature_values) + 2
            
            xlabel = 'Features (grouped by Expert)' if model_info['is_multi_expert'] else 'Features'
            
            ax3.set_xlabel(xlabel)
            ax3.set_ylabel('Activation Value')
            if x_ticks:
                ax3.set_xticks(x_ticks)
                ax3.set_xticklabels(x_labels)
            ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
            
            fig3.savefig(f"{base_filename}_3_activation_values.png", dpi=300)
            plt.close(fig3)

        positive_values = top_k_values[top_k_values > 0]
        if top_k_values.numel() > 0:
            positive_values_np = positive_values.numpy()
            if len(positive_values_np) > 0:
                fig5, ax5 = plt.subplots(figsize=(12, 8))
                n_bins = min(15, len(positive_values_np))
                n, bins, patches = ax5.hist(positive_values_np, bins=n_bins, alpha=0.7, color='green', edgecolor='black', density=True)
                
                cm = plt.cm.YlOrRd
                for i, patch in enumerate(patches):
                    patch.set_facecolor(cm(i / len(patches)))
                
                ax5.set_xlabel('Activation Value')
                ax5.set_ylabel('Proportion')
                ax5.grid(True, alpha=0.3)
                
                mean_val, median_val = np.mean(positive_values_np), np.median(positive_values_np)
                ax5.axvline(mean_val, color='#264653', linestyle='--')
                ax5.axvline(median_val, color='#2a9d8f', linestyle='--')
                ax5.legend([f'Mean: {mean_val:.4f}', f'Median: {median_val:.4f}'])
                
                fig5.savefig(f"{base_filename}_5_activation_distribution.png", dpi=300, bbox_inches='tight')
                plt.close(fig5)

        print(f"Visualizations saved for token '{token_text}' in folder: {text_folder}")
    except Exception as e:
        print(f"      Error creating visualization for token {token_pos}: {e}")

