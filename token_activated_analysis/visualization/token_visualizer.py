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
    
    try:
        # --- 初始设置 ---
        top_k_indices = top_k_indices.cpu()
        top_k_values = top_k_values.cpu()
        original_activation = original_activation.cpu()
        reconstructed_activation = reconstructed_activation.cpu()

        safe_token_name = sanitize_filename(token_text)
        if not safe_token_name:
            safe_token_name = f"pos_{token_pos}"
        
        base_filename = os.path.join(text_folder, f"token_{token_pos:02d}_{safe_token_name}")

        # --- 1. Expert概览 ---
        if expert_activation_counts:
            fig1, ax1 = plt.subplots(figsize=(12, 7))
            expert_ids = sorted(expert_activation_counts.keys())
            counts = [expert_activation_counts[eid] for eid in expert_ids]
            total_activations = [expert_total_activation[eid] for eid in expert_ids]
            
            title_suffix = f"Token: \"{token_text}\" (Pos: {token_pos})"
            xlabel = 'Expert ID' if model_info['is_multi_expert'] else 'Expert (Single Model)'
            
            bars1 = ax1.bar([x - 0.2 for x in expert_ids], counts, width=0.4, color='steelblue', label='Feature Count')
            ax1_twin = ax1.twinx()
            bars2 = ax1_twin.bar([x + 0.2 for x in expert_ids], total_activations, width=0.4, color='orange', label='Total Strength')
            
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel('Feature Count', color='steelblue')
            ax1_twin.set_ylabel('Total Activation', color='orange')
            ax1.grid(True, alpha=0.3)
            
            for bar, count in zip(bars1, counts):
                height = bar.get_height()
                ax1.annotate(f'{count}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, color='steelblue')
            
            for bar, val in zip(bars2, total_activations):
                height = bar.get_height()
                ax1_twin.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, color='orange')
            
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            fig1.savefig(f"{base_filename}_1_expert_overview.png", dpi=300, bbox_inches='tight')
            plt.close(fig1)

        # --- 2. 被激活的Feature详细列表 ---
        if expert_activations:
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            ax2.axis('off')
            all_features = []
            for expert_id, features in expert_activations.items():
                for feature_id, local_id, value in features:
                    all_features.append((feature_id, expert_id, local_id, value))
            all_features.sort(key=lambda x: x[3], reverse=True)
            
            feature_text = f"Top-{min(15, len(all_features))} Activated Features:\n"
            if model_info['is_multi_expert']:
                feature_text += "Rank | Feature ID | Expert | Local ID | Value\n" + "-" * 45 + "\n"
                for i, (fid, eid, lid, val) in enumerate(all_features[:15]):
                    feature_text += f"{i+1:3d}  | {fid:8d} | {eid:5d} | {lid:7d} | {val:6.3f}\n"
            else:
                feature_text += "Rank | Feature ID | Value\n" + "-" * 25 + "\n"
                for i, (fid, eid, lid, val) in enumerate(all_features[:15]):
                    feature_text += f"{i+1:3d}  | {fid:8d} | {val:6.3f}\n"
            
            if len(all_features) > 15:
                feature_text += f"... and {len(all_features) - 15} more features"
            
            ax2.text(0.05, 0.95, feature_text, transform=ax2.transAxes, fontsize=10, verticalalignment='top', fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            fig2.savefig(f"{base_filename}_2_feature_list.png", dpi=300, bbox_inches='tight')
            plt.close(fig2)

        # --- 3. Feature activation values ---
        if expert_activations:
            fig3, ax3 = plt.subplots(figsize=(max(15, len(top_k_indices) * 0.2), 7))
            expert_ids_with_features = sorted(expert_activations.keys())
            x_offset = 0
            x_ticks, x_labels = [], []
            
            for expert_id in expert_ids_with_features:
                features = sorted(expert_activations[expert_id], key=lambda x: x[2], reverse=True)
                feature_values = [f[2] for f in features]
                feature_ids = [f[0] for f in features]
                x_positions = np.arange(len(feature_values)) + x_offset
                
                color = plt.cm.Set3(expert_id / max(64, model_info['num_experts'])) if model_info['is_multi_expert'] else plt.cm.Set3(0.5)
                label = f'Expert {expert_id} ({len(features)} features)' if model_info['is_multi_expert'] else f'Single Expert ({len(features)} features)'
                center_label = f'E{expert_id}' if model_info['is_multi_expert'] else 'Expert'
                
                bars = ax3.bar(x_positions, feature_values, color=color, label=label)
                
                for bar, fid, val in zip(bars, feature_ids, feature_values):
                    if val > 0.1:
                        ax3.annotate(f'{fid}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, rotation=45)
                
                if len(feature_values) > 0:
                    x_ticks.append(x_offset + len(feature_values) / 2 - 0.5)
                    x_labels.append(center_label)
                x_offset += len(feature_values) + 1
            
            xlabel = 'Features (grouped by Expert)' if model_info['is_multi_expert'] else 'Features'
            
            ax3.set_xlabel(xlabel)
            ax3.set_ylabel('Activation Value')
            if x_ticks:
                ax3.set_xticks(x_ticks)
                ax3.set_xticklabels(x_labels)
            ax3.grid(True, alpha=0.3)
            if len(expert_ids_with_features) <= 10:
                ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            fig3.savefig(f"{base_filename}_3_activation_values.png", dpi=300, bbox_inches='tight')
            plt.close(fig3)

        # --- 4. 统计信息 ---
        fig4, ax4 = plt.subplots(figsize=(8, 10))
        ax4.axis('off')
        total_active_features = sum(len(features) for features in expert_activations.values())
        positive_values = top_k_values[top_k_values > 0]
        mean_activation, max_activation, std_activation = (positive_values.mean().item(), positive_values.max().item(), positive_values.std().item()) if len(positive_values) > 0 else (0.0, 0.0, 0.0)
        
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
        
        ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=10, verticalalignment='top', fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        fig4.savefig(f"{base_filename}_4_statistics.png", dpi=300, bbox_inches='tight')
        plt.close(fig4)

        # --- 5. Feature值分布直方图 ---
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
                ax5.legend(fontsize=9)
                
                fig5.savefig(f"{base_filename}_5_activation_distribution.png", dpi=300, bbox_inches='tight')
                plt.close(fig5)

        print(f"      Visualizations saved for token '{token_text}' in folder: {text_folder}")
        
    except Exception as e:
        print(f"      Error creating visualization for token {token_pos}: {e}")
