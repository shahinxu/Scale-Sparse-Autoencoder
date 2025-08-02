import torch as t
from nnsight import LanguageModel
from dictionary_learning.test_buffer import ActivationBuffer
from dictionary_learning.trainers.moe_physically import MultiExpertAutoEncoder
import json
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer
from config import lm, activation_dim, layer, n_ctxs
from collections import defaultdict
import os
import re

GPU = "0"
MODEL = "MultiExpert_64_1"
MODEL_PATH = f"/home/xuzhen/switch_sae/dictionaries/{MODEL}/8.pt"
OUTPUT_ROOT = f"sae_analysis_results_{MODEL}"


def sanitize_filename(text):
    """Clean filename by removing unsafe characters"""
    safe_text = re.sub(r'[^\w\s-]', '', text)
    safe_text = re.sub(r'[-\s]+', '_', safe_text)
    return safe_text.strip('_')


def parse_custom_texts(custom_texts):
    """Parse custom_texts, supporting both string and tuple formats"""
    parsed_data = []
    
    for item in custom_texts:
        if isinstance(item, tuple) and len(item) == 2:
            text, token_indices = item
            parsed_data.append({
                'text': text,
                'token_indices': token_indices,
                'analyze_all': False
            })
        elif isinstance(item, str):
            parsed_data.append({
                'text': item,
                'token_indices': None,
                'analyze_all': True
            })
        else:
            raise ValueError(f"Unsupported format: {item}")
    
    return parsed_data


@t.no_grad()
def analyze_specified_tokens(
    dictionary,  
    activations, 
    device="cpu",
    n_batches: int = 1,
    custom_texts=None
):
    """Analyze specified token positions in each batch"""
    assert n_batches > 0
    
    tokenizer = AutoTokenizer.from_pretrained(lm)
    parsed_texts = parse_custom_texts(custom_texts) if custom_texts else []
    
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    for batch_idx in range(n_batches):
        try:
            x = next(activations).to(device)
        except StopIteration:
            raise StopIteration("Not enough activations in buffer.")
        
        if batch_idx < len(parsed_texts):
            current_config = parsed_texts[batch_idx]
            text = current_config['text']
            analyze_all = current_config['analyze_all']
            specified_indices = current_config['token_indices']
        else:
            text = f"Default_text_{batch_idx}"
            analyze_all = True
            specified_indices = None
        
        safe_text_name = sanitize_filename(text[:50])
        if not safe_text_name:
            safe_text_name = f"text_{batch_idx}"
        
        text_folder = os.path.join(OUTPUT_ROOT, f"text_{batch_idx:02d}_{safe_text_name}")
        os.makedirs(text_folder, exist_ok=True)
        
        input_ids = tokenizer.encode(text, truncation=True, max_length=20)
        tokens = [tokenizer.decode([token_id]) for token_id in input_ids]
        
        if analyze_all:
            token_indices = list(range(min(len(x), len(tokens))))
        else:
            token_indices = specified_indices if specified_indices else [0]
        
        print(f"\n{'='*80}")
        print(f"BATCH {batch_idx} - TEXT: '{text}'")
        print(f"{'='*80}")
        print(f"Output folder: {text_folder}")
        print(f"Tokens: {tokens}")
        if analyze_all:
            print(f"Analyzing ALL tokens (positions 0-{len(token_indices)-1})")
        else:
            print(f"Analyzing specified token positions: {token_indices}")
        
        for token_pos in token_indices:
            if token_pos >= len(x) or token_pos >= len(tokens):
                print(f"Warning: Token position {token_pos} out of range (max: {min(len(x)-1, len(tokens)-1)})")
                continue
            
            token_activation = x[token_pos]
            token_text = tokens[token_pos] if token_pos < len(tokens) else f"pos_{token_pos}"
            
            x_hat, f = dictionary(token_activation.unsqueeze(0), output_features=True)
            token_reconstructed = x_hat[0]
            token_features = f[0]
            
            # Get top-k (32) activations from f
            top_k_values, top_k_indices = token_features.topk(dictionary.k, sorted=True)
            
            # Calculate reconstruction error
            recon_error = t.linalg.norm(token_activation - token_reconstructed).item()
            
            # Analyze experts and features from top-k activations
            expert_dict_size = dictionary.expert_dict_size
            expert_activations = defaultdict(list)  # expert_id -> [(feature_id, local_id, value), ...]
            expert_activation_counts = defaultdict(int)
            expert_total_activation = defaultdict(float)
            
            for fid, fval in zip(top_k_indices, top_k_values):
                if fval.item() > 0:  # Only consider positive activations
                    expert_id = fid.item() // expert_dict_size
                    local_feature_id = fid.item() % expert_dict_size
                    expert_activations[expert_id].append((fid.item(), local_feature_id, fval.item()))
                    expert_activation_counts[expert_id] += 1
                    expert_total_activation[expert_id] += fval.item()
            
            print(f"\n  Token Position {token_pos}: '{token_text}'")
            print(f"    Reconstruction Error: {recon_error:.6f}")
            print(f"    Top-{dictionary.k} Features Analyzed")
            print(f"    Activated Experts: {len(expert_activations)}")
            print(f"    Expert Distribution: {dict(expert_activation_counts)}")
            
            create_token_visualization(
                token_text, batch_idx, token_pos,
                top_k_indices, top_k_values, 
                expert_activations, expert_activation_counts, expert_total_activation,
                recon_error, token_activation, token_reconstructed,
                text_folder, text, dictionary.k
            )


def create_token_visualization(token_text, batch_idx, token_pos, 
                             top_k_indices, top_k_values,
                             expert_activations, expert_activation_counts, expert_total_activation,
                             recon_error, original_activation, reconstructed_activation, 
                             text_folder, original_text, k):
    """Create detailed visualization for specified token and save to text folder"""
    
    top_k_indices = top_k_indices.cpu()
    top_k_values = top_k_values.cpu()
    original_activation = original_activation.cpu()
    reconstructed_activation = reconstructed_activation.cpu()
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1])
    
    # 1. Expert activation counts (how many features from each expert)
    ax1 = fig.add_subplot(gs[0, :2])
    if expert_activation_counts:
        expert_ids = sorted(expert_activation_counts.keys())
        counts = [expert_activation_counts[eid] for eid in expert_ids]
        
        bars = ax1.bar(expert_ids, counts, color='steelblue', alpha=0.8)
        ax1.set_title(f'Expert Activation Counts (Top-{k} Features)\nToken: "{token_text}" (Pos: {token_pos})', 
                      fontsize=14, fontweight='bold')
        ax1.set_xlabel('Expert ID')
        ax1.set_ylabel('Number of Activated Features')
        ax1.grid(True, alpha=0.3)
        
        # Annotate bars with counts
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.annotate(f'{count}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Expert total activation strength
    ax2 = fig.add_subplot(gs[0, 2:])
    if expert_total_activation:
        expert_ids = sorted(expert_total_activation.keys())
        total_activations = [expert_total_activation[eid] for eid in expert_ids]
        
        bars = ax2.barh(range(len(expert_ids)), total_activations, color='orange', alpha=0.8)
        ax2.set_yticks(range(len(expert_ids)))
        ax2.set_yticklabels([f'Expert {eid}' for eid in expert_ids])
        ax2.set_title(f'Expert Total Activation Strength', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Total Activation Value')
        ax2.grid(True, alpha=0.3)
        
        # Annotate bars with values
        for i, (bar, val) in enumerate(zip(bars, total_activations)):
            width = bar.get_width()
            ax2.annotate(f'{val:.2f}', xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0), textcoords="offset points",
                        ha='left', va='center', fontsize=9, fontweight='bold')
    
    # 3. Feature activation values by expert
    ax3 = fig.add_subplot(gs[1, :])
    if expert_activations:
        expert_ids_with_features = sorted(expert_activations.keys())
        x_offset = 0
        x_ticks = []
        x_labels = []
        
        for expert_id in expert_ids_with_features:
            features = expert_activations[expert_id]
            features.sort(key=lambda x: x[2], reverse=True)  # Sort by activation value
            
            feature_values = [f[2] for f in features]
            x_positions = np.arange(len(feature_values)) + x_offset
            
            color = plt.cm.Set3(expert_id / 64)
            bars = ax3.bar(x_positions, feature_values, color=color, alpha=0.8, 
                          label=f'Expert {expert_id} ({len(features)} features)')
            
            if len(feature_values) > 0:
                center_pos = x_offset + len(feature_values) / 2 - 0.5
                x_ticks.append(center_pos)
                x_labels.append(f'E{expert_id}')
            
            x_offset += len(feature_values) + 1
        
        ax3.set_title(f'Top-{k} Feature Activation Values by Expert', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Features (grouped by Expert)')
        ax3.set_ylabel('Activation Value')
        if x_ticks:
            ax3.set_xticks(x_ticks)
            ax3.set_xticklabels(x_labels)
        ax3.grid(True, alpha=0.3)
        
        if len(expert_ids_with_features) <= 10:
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 4. Token and reconstruction statistics
    ax4 = fig.add_subplot(gs[2, :2])
    ax4.axis('off')
    
    total_active_features = sum(len(features) for features in expert_activations.values())
    
    info_text = f"""Token: "{token_text}" (Position: {token_pos})
Original Text: {original_text[:50]}{'...' if len(original_text) > 50 else ''}
Batch: {batch_idx}

Reconstruction Error: {recon_error:.6f}
Top-{k} Active Features: {total_active_features}
Activated Experts: {len(expert_activations)}

Original Activation Stats:
  Mean: {original_activation.mean().item():.4f}
  Std:  {original_activation.std().item():.4f}
  Min:  {original_activation.min().item():.4f}
  Max:  {original_activation.max().item():.4f}"""
    
    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    # 5. Feature activation value distribution
    ax5 = fig.add_subplot(gs[2, 2:])
    if top_k_values.numel() > 0:
        positive_values = top_k_values[top_k_values > 0].numpy()
        if len(positive_values) > 0:
            ax5.hist(positive_values, bins=min(20, len(positive_values)), 
                    alpha=0.7, color='green', edgecolor='black')
            ax5.set_title(f'Top-{k} Feature Value Distribution', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Activation Value')
            ax5.set_ylabel('Count')
            ax5.grid(True, alpha=0.3)
            
            mean_val = np.mean(positive_values)
            max_val = np.max(positive_values)
            ax5.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
            ax5.axvline(max_val, color='orange', linestyle='--', label=f'Max: {max_val:.3f}')
            ax5.legend()
        else:
            ax5.text(0.5, 0.5, 'No positive activations', transform=ax5.transAxes, 
                    ha='center', va='center', fontsize=12)
    else:
        ax5.text(0.5, 0.5, 'No activations', transform=ax5.transAxes, 
                ha='center', va='center', fontsize=12)
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, 
                       wspace=0.3, hspace=0.3)
    
    safe_token_name = sanitize_filename(token_text)
    if not safe_token_name:
        safe_token_name = f"pos_{token_pos}"
    
    visualization_file = os.path.join(text_folder, f"token_{token_pos:02d}_{safe_token_name}_analysis.png")
    plt.savefig(visualization_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"      Visualization saved to: {visualization_file}")


def create_custom_data_generator(custom_texts, max_length=20):
    tokenizer = AutoTokenizer.from_pretrained(lm)
    parsed_texts = parse_custom_texts(custom_texts)
    
    def gen():
        while True:
            for config in parsed_texts:
                text = config['text']
                input_ids = tokenizer.encode(text, truncation=True, max_length=max_length)
                processed_text = tokenizer.decode(input_ids, skip_special_tokens=True)
                yield processed_text
    return gen()

def main():
    device = f'cuda:{GPU}'
    
    print("Loading language model...")
    model = LanguageModel(lm, dispatch=True, device_map=device)
    submodule = model.transformer.h[layer]
    
    custom_texts = [
        # 1. 语义渐变测试 - 从具体到抽象
        "The red apple tastes sweet and delicious for breakfast today.",
        
        # 2. 重复模式测试 - 测试专家专门化
        "Python Python Python Python Python programming language Python",
        
        # 3. 语法结构测试 - 测试句法专家
        "Although the weather was terrible, we decided to go hiking anyway.",
        
        # 4. 数字和逻辑测试 - 测试数值专家
        "The equation 2 + 3 = 5 is a simple mathematical fact.",
        
        # 5. 情感极性测试 - 测试情感专家激活
        "I absolutely love this wonderful beautiful amazing fantastic day!",
        
        # 6. 技术词汇测试 - 测试专业领域专家
        "Neural networks use backpropagation algorithms for gradient descent optimization.",
        
        # 7. 时间和因果测试 - 测试时序专家
        "First we prepare ingredients, then cook the meal, finally we eat.",
        
        # 8. 否定和对比测试 - 测试对比逻辑专家
        "This is not wrong but rather completely right and absolutely correct."
    ]
    
    data = create_custom_data_generator(custom_texts)
    
    print("Creating activation buffer with custom data...")
    buffer = ActivationBuffer(
        data, 
        model, 
        submodule, 
        d_submodule=activation_dim, 
        n_ctxs=n_ctxs, 
        device=device,
        sequential=True
    )
    
    print(f"Loading SAE from {MODEL_PATH}...")
    ae = MultiExpertAutoEncoder(
        activation_dim=768, 
        dict_size=32*768, 
        k=32, 
        experts=64, 
        e=1, 
        heaviside=False
    )
    ae.load_state_dict(t.load(MODEL_PATH))
    ae.to(device)
    ae.eval()
    
    print("Analyzing specified tokens with SAE...")
    analyze_specified_tokens(ae, buffer, device=device, n_batches=len(custom_texts), custom_texts=custom_texts)

    print(f"\nToken analysis complete!")
    print(f"All results saved to: {OUTPUT_ROOT}/")
    print("Each token generates one PNG file with comprehensive analysis.")

if __name__ == "__main__":
    main()