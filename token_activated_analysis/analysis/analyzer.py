import torch as t
import os
from collections import defaultdict
from ..visualization.token_visualizer import create_token_visualization
from ..utils.file_utils import sanitize_filename

@t.no_grad()
def analyze_with_fixed_buffer(dictionary, buffer, texts, model_info, output_root, k, device="cpu"):
    """使用固定顺序的buffer分析文本"""
    
    print(f"\n{'='*80}")
    print(f"Analyzing {len(texts)} texts with Fixed Order Buffer")
    print(f"Model Type: {model_info['model_type']}")
    print(f"Multi-Expert: {model_info['is_multi_expert']}")
    print(f"Number of Experts: {model_info['num_experts']}")
    print(f"{'='*80}")
    
    os.makedirs(output_root, exist_ok=True)
    
    # 保存模型信息
    model_info_file = os.path.join(output_root, "model_info.txt")
    with open(model_info_file, 'w', encoding='utf-8') as f:
        f.write("Model Analysis Information\n")
        f.write("="*50 + "\n")
        f.write(f"Model Type: {model_info['model_type']}\n")
        f.write(f"Multi-Expert: {model_info['is_multi_expert']}\n")
        f.write(f"Number of Experts: {model_info['num_experts']}\n")
        f.write(f"Expert Dict Size: {model_info['expert_dict_size']}\n")
        f.write(f"Total Dict Size: {model_info['total_dict_size']}\n")
        f.write(f"Analysis Date: August 10, 2025\n")
    
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
        
        text_folder = os.path.join(output_root, f"text_{text_idx:02d}_{safe_text_name}")
        os.makedirs(text_folder, exist_ok=True)
        
        print(f"\nAnalyzing Text {text_idx}: '{text}'")
        print(f"Output folder: {text_folder}")
        print(f"Total tokens: {len(tokens)}")
        
        # 保存token信息到文件
        token_info_file = os.path.join(text_folder, "token_info.txt")
        with open(token_info_file, 'w', encoding='utf-8') as f:
            f.write(f"Text: {text}\n")
            f.write(f"Total tokens: {len(tokens)}\n")
            f.write(f"Model Type: {model_info['model_type']}\n")
            f.write(f"Multi-Expert: {model_info['is_multi_expert']}\n\n")
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
                
                # 分析expert激活 - 根据模型类型调整
                expert_activations = defaultdict(list)
                expert_activation_counts = defaultdict(int)
                expert_total_activation = defaultdict(float)
                
                for fid, fval in zip(top_k_indices, top_k_values):
                    if fval.item() > 0:
                        if model_info['is_multi_expert']:
                            # 多专家模型：计算expert_id和local_feature_id
                            expert_id = fid.item() // model_info['expert_dict_size']
                            local_feature_id = fid.item() % model_info['expert_dict_size']
                        else:
                            # 单专家模型：所有特征都属于专家0
                            expert_id = 0
                            local_feature_id = fid.item()
                        
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
                    f.write(f"Model Type: {model_info['model_type']}\n")
                    f.write(f"Reconstruction Error: {recon_error:.6f}\n\n")
                    
                    f.write(f"Expert Activations:\n")
                    if model_info['is_multi_expert']:
                        for expert_id in sorted(expert_activations.keys()):
                            features = expert_activations[expert_id]
                            f.write(f"  Expert {expert_id}: {len(features)} features, total={expert_total_activation[expert_id]:.4f}\n")
                            for feature_id, local_id, value in sorted(features, key=lambda x: x[2], reverse=True):
                                f.write(f"    Feature {feature_id} (local {local_id}): {value:.6f}\n")
                    else:
                        # 单专家模型的特殊显示
                        if 0 in expert_activations:
                            features = expert_activations[0]
                            f.write(f"  Single Expert (ID 0): {len(features)} features, total={expert_total_activation[0]:.4f}\n")
                            for feature_id, local_id, value in sorted(features, key=lambda x: x[2], reverse=True):
                                f.write(f"    Feature {feature_id}: {value:.6f}\n")
                    f.write(f"\n")
                    
                    f.write(f"Top-{k} Features:\n")
                    for i, (fid, fval) in enumerate(zip(top_k_indices, top_k_values)):
                        if fval.item() > 0:
                            if model_info['is_multi_expert']:
                                expert_id = fid.item() // model_info['expert_dict_size']
                                local_id = fid.item() % model_info['expert_dict_size']
                                f.write(f"  {i+1:2d}. Feature {fid.item():5d} (Expert {expert_id}, Local {local_id:4d}): {fval.item():.6f}\n")
                            else:
                                f.write(f"  {i+1:2d}. Feature {fid.item():5d}: {fval.item():.6f}\n")
                
                # 创建可视化
                create_token_visualization(
                    token_text, text_idx, token_pos,
                    top_k_indices, top_k_values,
                    expert_activations, expert_activation_counts, expert_total_activation,
                    recon_error, activation, token_reconstructed,
                    text_folder, text, k, model_info
                )
                
            except Exception as e:
                print(f"    Error analyzing token {token_pos}: {e}")
                continue
