import torch as t
from dictionary_learning.trainers.moe_physically import MultiExpertAutoEncoder
import os
import random
import numpy as np

def save_detailed_report(results, report_path):
    with open(report_path, 'w', encoding='utf-8') as f:
        config = results['config']
        
        f.write("Feature Max Similarity Analysis Report\n")
        f.write("="*50 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Feature type: {config['feature_type']}\n")
        f.write(f"  Number of experts: {config['n_experts']}\n") 
        f.write(f"  Features per expert: {config['features_per_expert']}\n")
        f.write(f"  Random seed: {config['seed']}\n\n")
        
        comparison = results['comparison']
        f.write("Summary Statistics:\n")
        f.write("-"*20 + "\n")
        f.write(f"  Avg expert mean max similarity: {comparison['avg_expert_mean_max_sim']:.6f}\n")
        f.write(f"  Global mean max similarity: {comparison['global_mean_max_sim']:.6f}\n")
        f.write(f"  Expert/Global ratio: {comparison['expert_vs_global_ratio']:.6f}\n\n")
        
        f.write("Feature Clustering Analysis:\n")
        f.write("-"*30 + "\n")
        f.write(f"  Expert avg features >0.8: {comparison['avg_expert_features_above_0.8_ratio']:.1%}\n")
        f.write(f"  Global features >0.8: {comparison['global_features_above_0.8_ratio']:.1%}\n")
        f.write(f"  Clustering score (>0.8): {comparison['clustering_score_0.8']:.3f}\n")
        f.write(f"  Expert avg features >0.7: {comparison['avg_expert_features_above_0.7_ratio']:.1%}\n")
        f.write(f"  Global features >0.7: {comparison['global_features_above_0.7_ratio']:.1%}\n")
        f.write(f"  Clustering score (>0.7): {comparison['clustering_score_0.7']:.3f}\n\n")
        
        f.write("Individual Expert Statistics:\n")
        f.write("-"*35 + "\n")
        f.write("Expert | Features | Mean | >0.9 | >0.8  | >0.7  | Global Max\n")
        f.write("-------|----------|------|------|-------|-------|----------\n")
        
        for expert_id in range(config['n_experts']):
            stats = results['expert_max_similarities'][expert_id]
            f.write(f"  {expert_id:2d}   | {stats['n_features']:8d} | {stats['mean_max_similarity']:.3f} | "
                   f"{stats['features_above_0.9']['ratio']:4.1%} | {stats['features_above_0.8']['ratio']:5.1%} | "
                   f"{stats['features_above_0.7']['ratio']:5.1%} | {stats['global_max_similarity']:.3f}\n")
        
        global_stats = results['global_max_similarities']
        f.write(f"\nGlobal Sample Statistics:\n")
        f.write("-"*25 + "\n")
        f.write(f"  Sampled features: {global_stats['n_sampled_features']}\n")
        f.write(f"  Mean max similarity: {global_stats['mean_max_similarity']:.6f}\n")
        f.write(f"  Std max similarity: {global_stats['std_max_similarity']:.6f}\n")
        f.write(f"  Median max similarity: {global_stats['median_max_similarity']:.6f}\n")
        f.write(f"  Global max: {global_stats['global_max_similarity']:.6f}\n")
        f.write(f"  Features >0.9: {global_stats['features_above_0.9']['count']} ({global_stats['features_above_0.9']['ratio']:.1%})\n")
        f.write(f"  Features >0.8: {global_stats['features_above_0.8']['count']} ({global_stats['features_above_0.8']['ratio']:.1%})\n")
        f.write(f"  Features >0.7: {global_stats['features_above_0.7']['count']} ({global_stats['features_above_0.7']['ratio']:.1%})\n")
        
        f.write(f"\nInterpretation:\n")
        f.write("-"*15 + "\n")
        clustering_score = comparison['clustering_score_0.8']
        if clustering_score > 0.1:
            f.write("  🎯 Strong evidence of feature clustering within experts\n")
        elif clustering_score > 0.05:
            f.write("  📊 Moderate evidence of feature clustering within experts\n")
        elif clustering_score > 0:
            f.write("  📈 Weak evidence of feature clustering within experts\n")
        else:
            f.write("  ❌ No clear evidence of expert-specific feature clustering\n")


def compute_feature_max_similarity(
    dictionary,
    feature_type="decoder",
    seed=42,
    device="cpu"
):
    if not hasattr(dictionary, "expert_modules"):
        raise AttributeError("Dictionary must have 'expert_modules' attribute.")
    
    random.seed(seed)
    t.manual_seed(seed)
    np.random.seed(seed)
    
    expert_modules = dictionary.expert_modules
    n_experts = len(expert_modules)
    
    if feature_type == "decoder":
        feature_matrices = []
        for expert in expert_modules:
            if hasattr(expert, 'decoder'):
                decoder_weight = expert.decoder.weight if hasattr(expert.decoder, 'weight') else expert.decoder
                feature_matrices.append(decoder_weight.to(device))
            else:
                raise AttributeError(f"Expert does not have decoder attribute")
        feature_name = "decoder"
    elif feature_type == "encoder":
        feature_matrices = []
        for expert in expert_modules:
            if hasattr(expert, 'encoder'):
                encoder_weight = expert.encoder.weight if hasattr(expert.encoder, 'weight') else expert.encoder
                feature_matrices.append(encoder_weight.T.to(device))  # 转置encoder权重
            else:
                raise AttributeError(f"Expert does not have encoder attribute")
        feature_name = "encoder"
    else:
        raise ValueError("feature_type must be 'decoder' or 'encoder'")
    
    # 检查每个expert的feature数量
    features_per_expert = [mat.shape[0] for mat in feature_matrices]
    print(f"Computing {feature_name} max similarity analysis:")
    print(f"  - Experts: {n_experts}")
    print(f"  - Features per expert: {features_per_expert}")
    
    # 存储结果
    results = {
        'expert_max_similarities': {},   # expert_id -> max similarity stats
        'global_max_similarities': {},   # 全局最大相似度统计
        'comparison': {},                # 比较统计
        'config': {
            'n_experts': n_experts,
            'features_per_expert': features_per_expert,
            'feature_type': feature_type,
            'seed': seed
        }
    }
    
    # 1. 计算每个expert内部feature的最大相似度
    print("\n1. Computing intra-expert max similarities...")
    
    all_expert_max_sims = []  # 存储所有expert的最大相似度
    
    for expert_id, feature_matrix in enumerate(feature_matrices):
        n_features = feature_matrix.shape[0]
        print(f"  Processing Expert {expert_id} ({n_features} features)...")
        
        # 标准化features
        features_normed = feature_matrix / feature_matrix.norm(dim=1, keepdim=True)
        
        # 计算相似性矩阵
        sim_matrix = features_normed @ features_normed.T
        
        # 去除对角线元素（自己与自己的相似度）
        sim_matrix.fill_diagonal_(-float('inf'))
        
        # 每个feature的最大相似度
        max_similarities, _ = sim_matrix.max(dim=1)
        max_similarities = max_similarities.detach().cpu().numpy()  # 修复：添加detach()
        
        # 统计超过阈值的feature数量
        n_features_total = len(max_similarities)
        above_09 = np.sum(max_similarities > 0.9)
        above_08 = np.sum(max_similarities > 0.8)
        above_07 = np.sum(max_similarities > 0.7)
        above_06 = np.sum(max_similarities > 0.6)
        above_05 = np.sum(max_similarities > 0.5)
        
        # 计算占比
        ratio_09 = above_09 / n_features_total
        ratio_08 = above_08 / n_features_total
        ratio_07 = above_07 / n_features_total
        ratio_06 = above_06 / n_features_total
        ratio_05 = above_05 / n_features_total
        
        expert_stats = {
            'n_features': n_features_total,
            'mean_max_similarity': float(np.mean(max_similarities)),
            'std_max_similarity': float(np.std(max_similarities)),
            'median_max_similarity': float(np.median(max_similarities)),
            'global_max_similarity': float(np.max(max_similarities)),
            'min_max_similarity': float(np.min(max_similarities)),
            'features_above_0.9': {'count': int(above_09), 'ratio': float(ratio_09)},
            'features_above_0.8': {'count': int(above_08), 'ratio': float(ratio_08)},
            'features_above_0.7': {'count': int(above_07), 'ratio': float(ratio_07)},
            'features_above_0.6': {'count': int(above_06), 'ratio': float(ratio_06)},
            'features_above_0.5': {'count': int(above_05), 'ratio': float(ratio_05)},
            'max_similarities': max_similarities.tolist()  # 保存所有最大相似度
        }
        
        results['expert_max_similarities'][expert_id] = expert_stats
        all_expert_max_sims.extend(max_similarities.tolist())
        
        print(f"    Mean max sim: {expert_stats['mean_max_similarity']:.4f}")
        print(f"    Features >0.9: {above_09}/{n_features_total} ({ratio_09:.1%})")
        print(f"    Features >0.8: {above_08}/{n_features_total} ({ratio_08:.1%})")
        print(f"    Features >0.7: {above_07}/{n_features_total} ({ratio_07:.1%})")
        
        # 清理GPU内存
        del features_normed, sim_matrix, max_similarities
        t.cuda.empty_cache()
    
    # 2. 计算全局feature的最大相似度（采样）
    print("\n2. Computing global max similarities...")
    
    # 合并所有expert的features
    all_features = t.cat(feature_matrices, dim=0)  # [total_features, feature_dim]
    total_features = all_features.shape[0]
    
    # 采样数量等于单个expert的feature数量
    sample_size = features_per_expert[0]  # 使用第一个expert的feature数量作为采样数量
    print(f"  Total features: {total_features}")
    print(f"  Sample size: {sample_size}")
    
    # 随机采样
    sampled_indices = random.sample(range(total_features), min(sample_size, total_features))
    sampled_features = all_features[sampled_indices]
    
    # 标准化
    sampled_features_normed = sampled_features / sampled_features.norm(dim=1, keepdim=True)
    
    # 计算相似性矩阵
    global_sim_matrix = sampled_features_normed @ sampled_features_normed.T
    global_sim_matrix.fill_diagonal_(-float('inf'))
    
    # 每个feature的最大相似度
    global_max_similarities, _ = global_sim_matrix.max(dim=1)
    global_max_similarities = global_max_similarities.detach().cpu().numpy()  # 修复：添加detach()
    
    # 统计全局feature的最大相似度分布
    n_global_features = len(global_max_similarities)
    global_above_09 = np.sum(global_max_similarities > 0.9)
    global_above_08 = np.sum(global_max_similarities > 0.8)
    global_above_07 = np.sum(global_max_similarities > 0.7)
    global_above_06 = np.sum(global_max_similarities > 0.6)
    global_above_05 = np.sum(global_max_similarities > 0.5)
    
    global_ratio_09 = global_above_09 / n_global_features
    global_ratio_08 = global_above_08 / n_global_features
    global_ratio_07 = global_above_07 / n_global_features
    global_ratio_06 = global_above_06 / n_global_features
    global_ratio_05 = global_above_05 / n_global_features
    
    global_stats = {
        'n_sampled_features': int(n_global_features),
        'sample_size': int(sample_size),
        'total_available_features': int(total_features),
        'mean_max_similarity': float(np.mean(global_max_similarities)),
        'std_max_similarity': float(np.std(global_max_similarities)),
        'median_max_similarity': float(np.median(global_max_similarities)),
        'global_max_similarity': float(np.max(global_max_similarities)),
        'min_max_similarity': float(np.min(global_max_similarities)),
        'features_above_0.9': {'count': int(global_above_09), 'ratio': float(global_ratio_09)},
        'features_above_0.8': {'count': int(global_above_08), 'ratio': float(global_ratio_08)},
        'features_above_0.7': {'count': int(global_above_07), 'ratio': float(global_ratio_07)},
        'features_above_0.6': {'count': int(global_above_06), 'ratio': float(global_ratio_06)},
        'features_above_0.5': {'count': int(global_above_05), 'ratio': float(global_ratio_05)},
        'max_similarities': global_max_similarities.tolist()
    }
    
    results['global_max_similarities'] = global_stats
    
    print(f"  Global mean max sim: {global_stats['mean_max_similarity']:.4f}")
    print(f"  Global features >0.9: {global_above_09}/{n_global_features} ({global_ratio_09:.1%})")
    print(f"  Global features >0.8: {global_above_08}/{n_global_features} ({global_ratio_08:.1%})")
    print(f"  Global features >0.7: {global_above_07}/{n_global_features} ({global_ratio_07:.1%})")
    
    # 3. 计算比较统计
    print("\n3. Computing comparison statistics...")
    
    # Expert内部平均统计
    expert_mean_max_sims = [results['expert_max_similarities'][i]['mean_max_similarity'] 
                           for i in range(n_experts)]
    avg_expert_mean_max_sim = np.mean(expert_mean_max_sims)
    
    # 计算平均占比
    avg_expert_ratio_08 = np.mean([results['expert_max_similarities'][i]['features_above_0.8']['ratio'] 
                                  for i in range(n_experts)])
    avg_expert_ratio_07 = np.mean([results['expert_max_similarities'][i]['features_above_0.7']['ratio'] 
                                  for i in range(n_experts)])
    
    comparison_stats = {
        'avg_expert_mean_max_sim': float(avg_expert_mean_max_sim),
        'global_mean_max_sim': global_stats['mean_max_similarity'],
        'expert_vs_global_ratio': float(avg_expert_mean_max_sim / max(global_stats['mean_max_similarity'], 1e-8)),
        'avg_expert_features_above_0.8_ratio': float(avg_expert_ratio_08),
        'global_features_above_0.8_ratio': float(global_ratio_08),
        'avg_expert_features_above_0.7_ratio': float(avg_expert_ratio_07),
        'global_features_above_0.7_ratio': float(global_ratio_07),
        'clustering_score_0.8': float(avg_expert_ratio_08 - global_ratio_08),
        'clustering_score_0.7': float(avg_expert_ratio_07 - global_ratio_07)
    }
    
    results['comparison'] = comparison_stats
    
    print(f"  Avg expert mean max sim: {avg_expert_mean_max_sim:.4f}")
    print(f"  Global mean max sim: {global_stats['mean_max_similarity']:.4f}")
    print(f"  Expert/Global ratio: {comparison_stats['expert_vs_global_ratio']:.4f}")
    print(f"  Expert clustering score (>0.8): {comparison_stats['clustering_score_0.8']:.3f}")
    print(f"  Expert clustering score (>0.7): {comparison_stats['clustering_score_0.7']:.3f}")
    
    # 清理内存
    del all_features, sampled_features, sampled_features_normed, global_sim_matrix, feature_matrices
    t.cuda.empty_cache()
    
    return results


def save_similarity_analysis(results, save_path):
    """保存相似性分析结果"""
    import json
    import os
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Max similarity analysis saved to: {save_path}")


def print_similarity_summary(results):
    """打印相似性分析摘要"""
    print("\n" + "="*70)
    print("FEATURE MAX SIMILARITY ANALYSIS SUMMARY")
    print("="*70)
    
    config = results['config']
    print(f"Configuration:")
    print(f"  - Feature type: {config['feature_type']}")
    print(f"  - Number of experts: {config['n_experts']}")
    print(f"  - Features per expert: {config['features_per_expert']}")
    print(f"  - Random seed: {config['seed']}")
    
    print(f"\n📊 Key Results:")
    comparison = results['comparison']
    print(f"  - Avg expert mean max similarity: {comparison['avg_expert_mean_max_sim']:.4f}")
    print(f"  - Global mean max similarity: {comparison['global_mean_max_sim']:.4f}")
    print(f"  - Expert/Global ratio: {comparison['expert_vs_global_ratio']:.4f}")
    
    print(f"\n📈 Feature Clustering Statistics:")
    print(f"  Expert avg >0.8 ratio: {comparison['avg_expert_features_above_0.8_ratio']:.1%}")
    print(f"  Global >0.8 ratio: {comparison['global_features_above_0.8_ratio']:.1%}")
    print(f"  Clustering score (>0.8): {comparison['clustering_score_0.8']:.3f}")
    print(f"  Expert avg >0.7 ratio: {comparison['avg_expert_features_above_0.7_ratio']:.1%}")
    print(f"  Global >0.7 ratio: {comparison['global_features_above_0.7_ratio']:.1%}")
    print(f"  Clustering score (>0.7): {comparison['clustering_score_0.7']:.3f}")
    
    print(f"\n🔍 Expert Ranking by Feature Clustering (>0.8):")
    expert_clustering = [(i, results['expert_max_similarities'][i]['features_above_0.8']['ratio']) 
                        for i in range(config['n_experts'])]
    expert_clustering.sort(key=lambda x: x[1], reverse=True)
    
    for i, (expert_id, ratio) in enumerate(expert_clustering[:10]):
        stats = results['expert_max_similarities'][expert_id]
        count = stats['features_above_0.8']['count']
        total = stats['n_features']
        mean_sim = stats['mean_max_similarity']
        print(f"  {i+1:2d}. Expert {expert_id}: {count}/{total} ({ratio:.1%}), mean={mean_sim:.3f}")
    
    print(f"\n💡 Interpretation:")
    clustering_score = comparison['clustering_score_0.8']
    if clustering_score > 0.1:
        print(f"  🎯 Strong feature clustering within experts")
    elif clustering_score > 0.05:
        print(f"  📊 Moderate feature clustering within experts")
    elif clustering_score > 0:
        print(f"  📈 Weak feature clustering within experts")
    else:
        print(f"  ❌ No evidence of expert-specific feature clustering")
    
    expert_global_ratio = comparison['expert_vs_global_ratio']
    if expert_global_ratio > 1.2:
        print(f"  ✅ Expert features are more similar to each other than random")
    elif expert_global_ratio > 1.0:
        print(f"  ⚡ Slight tendency for expert feature similarity")
    else:
        print(f"  ⚠️ Expert features not more similar than random")
    
    print("="*70)


GPU = "5"
MODEL = "MultiExpert_64_8"  
MODEL_PATH = f"/home/xuzhen/switch_sae/dictionaries/{MODEL}/8.pt"
OUTPUT_DIR = f"feature_max_similarity_analysis_{MODEL}"

def main():
    device = f'cuda:{GPU}'
    
    print("🔍 Feature Max Similarity Analysis")
    print("="*50)
    print(f"Model: {MODEL}")
    print(f"Device: {device}")
    
    print("\n📥 Loading SAE model...")
    ae = MultiExpertAutoEncoder(
        activation_dim=768,
        dict_size=32*768, 
        k=32,
        experts=64,
        e=8,
        heaviside=False
    )
    ae.load_state_dict(t.load(MODEL_PATH))
    ae.to(device)
    ae.eval()
    
    print(f"✅ Model loaded: {len(ae.expert_modules)} experts")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for feature_type in ["decoder"]:
        print(f"\n🔬 Analyzing {feature_type} features...")
        
        try:
            results = compute_feature_max_similarity(
                dictionary=ae,
                feature_type=feature_type,
                seed=42,
                device=device
            )
            
            save_path = os.path.join(OUTPUT_DIR, f"{feature_type}_max_similarity.json")
            save_similarity_analysis(results, save_path)
            
            print_similarity_summary(results)
            
            report_path = os.path.join(OUTPUT_DIR, f"{feature_type}_max_similarity_report.txt")
            save_detailed_report(results, report_path)
            
        except Exception as e:
            print(f"❌ Error analyzing {feature_type} features: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✅ Analysis complete! Results saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()