import torch as t
from nnsight import LanguageModel
from dictionary_learning.trainers.moe_physically import MultiExpertAutoEncoder
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer
from config import lm, layer
from collections import defaultdict
import os
import re

GPU = "3"
MODEL = "MultiExpert_32_8_1"
LAYER = 8
MODEL_PATH = f"/home/xuzhen/switch_sae/dictionaries/{MODEL}/{LAYER}.pt"
OUTPUT_ROOT = f"sae_analysis_results_{MODEL}_{LAYER}"
k = 32

def sanitize_filename(text):
    """Clean filename by removing unsafe characters"""
    safe_text = re.sub(r'[^\w\s-]', '', text)
    safe_text = re.sub(r'[-\s]+', '_', safe_text)
    return safe_text.strip('_')

class FixedOrderBuffer:
    """
    确保token顺序正确的简单buffer
    一次处理一个文本，确保激活和token的完美对应
    """
    
    def __init__(self, model, layer_name, device="cpu", max_length=20):
        self.model = model
        self.layer_name = layer_name
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(lm)
        
        # 获取对应的layer module
        layer_parts = layer_name.split('.')
        self.layer_module = model
        for part in layer_parts:
            if part.isdigit():
                self.layer_module = self.layer_module[int(part)]
            else:
                self.layer_module = getattr(self.layer_module, part)
        
        print(f"FixedOrderBuffer initialized for layer: {layer_name}")
    
    def process_text(self, text):
        """
        处理单个文本，返回所有token的激活
        
        Returns:
            activations: tensor [seq_len, hidden_dim]
            tokens: list of token strings
            token_ids: list of token ids
        """
        print(f"\nProcessing text: '{text[:50]}...'")
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=False
        )
        
        token_ids = inputs['input_ids'][0].tolist()
        tokens = [self.tokenizer.decode([tid]) for tid in token_ids]
        
        print(f"  Tokens ({len(tokens)}): {tokens}")
        
        # 通过模型获取激活
        with t.no_grad():
            with self.model.trace(text, scan=False, validate=False) as tracer:
                hidden_states = self.layer_module.output.save()
            
            # 修复：直接获取激活，不使用.value
            activations = hidden_states
            if isinstance(activations, tuple):
                activations = activations[0]
            
            print(f"  Raw activations shape: {activations.shape}")
            
            # 形状: [1, seq_len, hidden_dim] -> [seq_len, hidden_dim]
            if len(activations.shape) == 3:
                activations = activations[0]  # 去掉batch维度
            
            # 确保长度匹配
            min_len = min(len(tokens), activations.shape[0])
            activations = activations[:min_len].to(self.device)
            tokens = tokens[:min_len]
            token_ids = token_ids[:min_len]
            
            print(f"  Final activations shape: {activations.shape}")
            print(f"  Matched {min_len} tokens with activations")
        
        return activations, tokens, token_ids

@t.no_grad()
def analyze_with_fixed_buffer(dictionary, buffer, texts, device="cpu"):
    """使用固定顺序的buffer分析文本"""
    
    print(f"\n{'='*80}")
    print(f"Analyzing {len(texts)} texts with Fixed Order Buffer")
    print(f"{'='*80}")
    
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    for text_idx, text in enumerate(texts):
        # 处理单个文本
        try:
            activations, tokens, token_ids = buffer.process_text(text)
        except Exception as e:
            print(f"Error processing text {text_idx}: {e}")
            continue
        
        # 创建文本文件夹
        safe_text_name = sanitize_filename(text[:50])
        if not safe_text_name:
            safe_text_name = f"text_{text_idx}"
        
        text_folder = os.path.join(OUTPUT_ROOT, f"text_{text_idx:02d}_{safe_text_name}")
        os.makedirs(text_folder, exist_ok=True)
        
        print(f"\nAnalyzing Text {text_idx}: '{text}'")
        print(f"Output folder: {text_folder}")
        print(f"Total tokens: {len(tokens)}")
        
        # 保存token信息到文件
        token_info_file = os.path.join(text_folder, "token_info.txt")
        with open(token_info_file, 'w', encoding='utf-8') as f:
            f.write(f"Text: {text}\n")
            f.write(f"Total tokens: {len(tokens)}\n\n")
            f.write("Token mapping:\n")
            for i, (token, tid) in enumerate(zip(tokens, token_ids)):
                f.write(f"  {i:2d}: '{token}' (id: {tid})\n")
        
        # 分析每个token
        for token_pos, (activation, token_text) in enumerate(zip(activations, tokens)):
            print(f"  Processing token {token_pos}: '{token_text}'")
            
            try:
                # SAE分析
                x_hat, f = dictionary(activation.unsqueeze(0), output_features=True)
                token_reconstructed = x_hat[0]
                token_features = f[0]
                
                # 获取top-k激活
                top_k_values, top_k_indices = token_features.topk(k, sorted=True)
                
                # 计算重构误差
                recon_error = t.linalg.norm(activation - token_reconstructed).item()
                
                # 分析expert激活
                expert_dict_size = dictionary.expert_dict_size
                expert_activations = defaultdict(list)
                expert_activation_counts = defaultdict(int)
                expert_total_activation = defaultdict(float)
                
                for fid, fval in zip(top_k_indices, top_k_values):
                    if fval.item() > 0:
                        expert_id = fid.item() // expert_dict_size
                        local_feature_id = fid.item() % expert_dict_size
                        expert_activations[expert_id].append((fid.item(), local_feature_id, fval.item()))
                        expert_activation_counts[expert_id] += 1
                        expert_total_activation[expert_id] += fval.item()
                
                print(f"    Reconstruction Error: {recon_error:.6f}")
                print(f"    Activated Experts: {len(expert_activations)}")
                print(f"    Expert Distribution: {dict(expert_activation_counts)}")
                
                # 保存详细的token分析
                token_analysis_file = os.path.join(text_folder, f"token_{token_pos:02d}_{sanitize_filename(token_text)}_analysis.txt")
                with open(token_analysis_file, 'w', encoding='utf-8') as f:
                    f.write(f"Token Analysis\n")
                    f.write(f"="*50 + "\n")
                    f.write(f"Position: {token_pos}\n")
                    f.write(f"Token: '{token_text}'\n")
                    f.write(f"Token ID: {token_ids[token_pos]}\n")
                    f.write(f"Text: {text}\n")
                    f.write(f"Reconstruction Error: {recon_error:.6f}\n\n")
                    
                    f.write(f"Expert Activations:\n")
                    for expert_id in sorted(expert_activations.keys()):
                        features = expert_activations[expert_id]
                        f.write(f"  Expert {expert_id}: {len(features)} features, total={expert_total_activation[expert_id]:.4f}\n")
                        for feature_id, local_id, value in sorted(features, key=lambda x: x[2], reverse=True):
                            f.write(f"    Feature {feature_id} (local {local_id}): {value:.6f}\n")
                    f.write(f"\n")
                    
                    f.write(f"Top-{k} Features:\n")
                    for i, (fid, fval) in enumerate(zip(top_k_indices, top_k_values)):
                        if fval.item() > 0:
                            expert_id = fid.item() // expert_dict_size
                            local_id = fid.item() % expert_dict_size
                            f.write(f"  {i+1:2d}. Feature {fid.item():5d} (Expert {expert_id}, Local {local_id:4d}): {fval.item():.6f}\n")
                
                # 创建可视化
                create_token_visualization(
                    token_text, text_idx, token_pos,
                    top_k_indices, top_k_values,
                    expert_activations, expert_activation_counts, expert_total_activation,
                    recon_error, activation, token_reconstructed,
                    text_folder, text, k
                )
                
            except Exception as e:
                print(f"    Error analyzing token {token_pos}: {e}")
                continue

def create_token_visualization(token_text, batch_idx, token_pos, 
                             top_k_indices, top_k_values,
                             expert_activations, expert_activation_counts, expert_total_activation,
                             recon_error, original_activation, reconstructed_activation, 
                             text_folder, original_text, k):
    """创建token可视化 - 优化版本"""
    
    try:
        top_k_indices = top_k_indices.cpu()
        top_k_values = top_k_values.cpu()
        original_activation = original_activation.cpu()
        reconstructed_activation = reconstructed_activation.cpu()
        
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1.2, 1], width_ratios=[1, 1, 1, 1])
        
        # 1. 合并的Expert概览 (将原来的两个expert图合并为一个)
        ax1 = fig.add_subplot(gs[0, :2])
        if expert_activation_counts:
            expert_ids = sorted(expert_activation_counts.keys())
            counts = [expert_activation_counts[eid] for eid in expert_ids]
            total_activations = [expert_total_activation[eid] for eid in expert_ids]
            
            # 创建双轴图表
            bars1 = ax1.bar([x - 0.2 for x in expert_ids], counts, width=0.4, 
                           color='steelblue', alpha=0.8, label='Feature Count')
            ax1_twin = ax1.twinx()
            bars2 = ax1_twin.bar([x + 0.2 for x in expert_ids], total_activations, width=0.4,
                                color='orange', alpha=0.8, label='Total Strength')
            
            ax1.set_title(f'Expert Activations Overview\nToken: "{token_text}" (Pos: {token_pos})', 
                          fontsize=14, fontweight='bold')
            ax1.set_xlabel('Expert ID')
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
        
        # 2. 被激活的Feature详细列表
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
            
            # 创建表格文本
            feature_text = f"Top-{min(15, len(all_features))} Activated Features:\n"
            feature_text += "Rank | Feature ID | Expert | Local ID | Value\n"
            feature_text += "-" * 45 + "\n"
            
            for i, (fid, eid, lid, val) in enumerate(all_features[:15]):
                feature_text += f"{i+1:3d}  | {fid:8d} | {eid:5d} | {lid:7d} | {val:6.3f}\n"
            
            if len(all_features) > 15:
                feature_text += f"... and {len(all_features) - 15} more features"
            
            ax2.text(0.05, 0.95, feature_text, transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        # 3. Feature activation values by expert (保留这个主要的可视化)
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
                feature_ids = [f[0] for f in features]  # 获取完整的feature ID
                x_positions = np.arange(len(feature_values)) + x_offset
                
                color = plt.cm.Set3(expert_id / 64)
                bars = ax3.bar(x_positions, feature_values, color=color, alpha=0.8, 
                              label=f'Expert {expert_id} ({len(features)} features)')
                
                # 在每个bar上标注feature ID
                for bar, fid, val in zip(bars, feature_ids, feature_values):
                    if val > 0.1:  # 只标注较大的激活值
                        ax3.annotate(f'{fid}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                    xytext=(0, 3), textcoords="offset points",
                                    ha='center', va='bottom', fontsize=8, rotation=45)
                
                if len(feature_values) > 0:
                    center_pos = x_offset + len(feature_values) / 2 - 0.5
                    x_ticks.append(center_pos)
                    x_labels.append(f'E{expert_id}')
                
                x_offset += len(feature_values) + 1
            
            ax3.set_title(f'Feature Activation Values by Expert (with Feature IDs)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Features (grouped by Expert)')
            ax3.set_ylabel('Activation Value')
            if x_ticks:
                ax3.set_xticks(x_ticks)
                ax3.set_xticklabels(x_labels)
            ax3.grid(True, alpha=0.3)
            
            if len(expert_ids_with_features) <= 10:
                ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # 4. 统计信息 (增强版)
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

def main():
    device = f'cuda:{GPU}'
    
    print("Loading language model...")
    model = LanguageModel(lm, dispatch=True, device_map=device)
    
    custom_texts = [
        "The cat sat on the mat.",
        "She walked to the store yesterday.",
        
        "The answer is 42 plus 17 equals 59.",
        "In 2023, the temperature was -15.7 degrees.",
        
        "Barack Obama was born in Hawaii.",
        "Microsoft Corporation develops Windows.",
        
        "Twenty-first-century nano-technology is amazing.",
        "Self-driving cars use state-of-the-art AI.",
        
        "Wait... what? Really?! That's @incredible#hashtag!",
        "Visit https://www.example.com for more info.",
        
        "DNA contains adenine, thymine, guanine, and cytosine.",
        "The algorithm uses gradient descent optimization.",
        
        "I feel extremely happy and grateful today.",
        "Love, hope, and freedom are universal values.",
        
        "I love apples but hate oranges completely.",
        "This is not impossible, it's just difficult.",
        
        # 9. 时间和地点
        "On Monday morning at 9:30 AM in New York.",
        "During the Renaissance period in Florence, Italy.",
        
        # 10. 重复词汇测试 (测试context效果)
        "Apple pie, apple juice, apple tree, green apple.",
        "The king's crown, the king's castle, the king's sword.",
        
        # 11. 语言混合和外来词
        "The restaurant serves delicious sushi and pizza.",
        "Café, naïve, résumé, and other français words.",
        
        # 12. 长句子和复杂结构
        "Although the weather was terrible, we decided to go hiking because the mountain views would be spectacular.",
        
        # 13. 短句和单词
        "Yes.",
        "No way!",
        "Absolutely.",
        
        # 14. 引用和对话
        "He said, 'Hello there!' and she replied, 'Hi!'",
        "According to Einstein, 'Imagination is more important than knowledge.'",
        
        # 15. 技术代码风格
        "function calculateSum(a, b) { return a + b; }",
        "import torch.nn.functional as F",
    ]
    
    print("Creating Fixed Order Buffer...")
    buffer = FixedOrderBuffer(
        model=model,
        layer_name=f"transformer.h.{layer}",
        device=device,
        max_length=20
    )
    
    print(f"Loading SAE from {MODEL_PATH}...")
    ae = MultiExpertAutoEncoder(
        activation_dim=768,
        dict_size=32*768,
        k=32,
        experts=8,
        e=1,
        heaviside=False
    )
    ae.load_state_dict(t.load(MODEL_PATH))
    ae.to(device)
    ae.eval()
    
    print("Analyzing with Fixed Order Buffer...")
    analyze_with_fixed_buffer(ae, buffer, custom_texts, device=device)
    
    print(f"\nAnalysis complete!")
    print(f"Results saved to: {OUTPUT_ROOT}/")
    print("Check token_info.txt in each folder to verify token order is correct!")

if __name__ == "__main__":
    main()