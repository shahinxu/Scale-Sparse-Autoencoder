import torch as t
from dictionary_learning.trainers.moe_physically import MultiExpertAutoEncoder
from dictionary_learning.trainers.moe_physically_scale import MultiExpertScaleAutoEncoder
import os
import random
import numpy as np
import json

def compute_feature_max_similarity(
    dictionary,
    feature_type="encoder",
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
    elif feature_type == "encoder":
        feature_matrices = []
        for expert in expert_modules:
            if hasattr(expert, 'encoder'):
                encoder_weight = expert.encoder.weight if hasattr(expert.encoder, 'weight') else expert.encoder
                feature_matrices.append(encoder_weight.T.to(device))
            else:
                raise AttributeError(f"Expert does not have encoder attribute")
    else:
        raise ValueError("feature_type must be 'decoder' or 'encoder'")
    
    features_per_expert = [mat.shape[0] for mat in feature_matrices]
    
    results = {
        'config': {
            'n_experts': n_experts,
            'features_per_expert': features_per_expert,
            'feature_type': feature_type,
            'seed': seed
        },
        'expert_stats': {},
        'global_stats': {},
        'summary': {}
    }
    
    # 1. Expert内部相似度
    all_expert_max_sims = []
    
    for expert_id, feature_matrix in enumerate(feature_matrices):
        n_features = feature_matrix.shape[0]
        
        features_normed = feature_matrix / feature_matrix.norm(dim=1, keepdim=True)
        sim_matrix = features_normed @ features_normed.T
        sim_matrix.fill_diagonal_(-float('inf'))
        
        max_similarities, _ = sim_matrix.max(dim=1)
        max_similarities = max_similarities.detach().cpu().numpy()
        
        above_08 = np.sum(max_similarities > 0.8)
        above_07 = np.sum(max_similarities > 0.7)
        
        results['expert_stats'][expert_id] = {
            'n_features': n_features,
            'mean_max_similarity': float(np.mean(max_similarities)),
            'features_above_0.8': {'count': int(above_08), 'ratio': float(above_08 / n_features)},
            'features_above_0.7': {'count': int(above_07), 'ratio': float(above_07 / n_features)}
        }
        
        all_expert_max_sims.extend(max_similarities.tolist())
        
        del features_normed, sim_matrix, max_similarities
        t.cuda.empty_cache()
    
    # 2. 全局相似度（采样）
    all_features = t.cat(feature_matrices, dim=0)
    total_features = all_features.shape[0]
    sample_size = features_per_expert[0]
    
    sampled_indices = random.sample(range(total_features), min(sample_size, total_features))
    sampled_features = all_features[sampled_indices]
    
    sampled_features_normed = sampled_features / sampled_features.norm(dim=1, keepdim=True)
    global_sim_matrix = sampled_features_normed @ sampled_features_normed.T
    global_sim_matrix.fill_diagonal_(-float('inf'))
    
    global_max_similarities, _ = global_sim_matrix.max(dim=1)
    global_max_similarities = global_max_similarities.detach().cpu().numpy()
    
    n_global_features = len(global_max_similarities)
    global_above_08 = np.sum(global_max_similarities > 0.8)
    global_above_07 = np.sum(global_max_similarities > 0.7)
    
    results['global_stats'] = {
        'n_sampled_features': int(n_global_features),
        'mean_max_similarity': float(np.mean(global_max_similarities)),
        'features_above_0.8': {'count': int(global_above_08), 'ratio': float(global_above_08 / n_global_features)},
        'features_above_0.7': {'count': int(global_above_07), 'ratio': float(global_above_07 / n_global_features)}
    }
    
    # 3. 比较统计
    expert_mean_max_sims = [results['expert_stats'][i]['mean_max_similarity'] for i in range(n_experts)]
    avg_expert_mean_max_sim = np.mean(expert_mean_max_sims)
    
    avg_expert_ratio_08 = np.mean([results['expert_stats'][i]['features_above_0.8']['ratio'] for i in range(n_experts)])
    avg_expert_ratio_07 = np.mean([results['expert_stats'][i]['features_above_0.7']['ratio'] for i in range(n_experts)])
    
    global_ratio_08 = results['global_stats']['features_above_0.8']['ratio']
    global_ratio_07 = results['global_stats']['features_above_0.7']['ratio']
    
    results['summary'] = {
        'avg_expert_mean_max_sim': float(avg_expert_mean_max_sim),
        'global_mean_max_sim': results['global_stats']['mean_max_similarity'],
        'expert_vs_global_ratio': float(avg_expert_mean_max_sim / max(results['global_stats']['mean_max_similarity'], 1e-8)),
        'avg_expert_features_above_0.8_ratio': float(avg_expert_ratio_08),
        'global_features_above_0.8_ratio': float(global_ratio_08),
        'avg_expert_features_above_0.7_ratio': float(avg_expert_ratio_07),
        'global_features_above_0.7_ratio': float(global_ratio_07),
        'clustering_score_0.8': float(avg_expert_ratio_08 - global_ratio_08),
        'clustering_score_0.7': float(avg_expert_ratio_07 - global_ratio_07)
    }
    
    del all_features, sampled_features, sampled_features_normed, global_sim_matrix, feature_matrices
    t.cuda.empty_cache()
    
    return results

def print_results(results):
    """简化的结果打印"""
    config = results['config']
    summary = results['summary']
    
    print(f"\nResults for {config['feature_type']} features ({config['n_experts']} experts):")
    print(f"Expert avg similarity: {summary['avg_expert_mean_max_sim']:.4f}")
    print(f"Global avg similarity: {summary['global_mean_max_sim']:.4f}")
    print(f"Expert/Global ratio: {summary['expert_vs_global_ratio']:.4f}")
    print(f"Clustering score (>0.8): {summary['clustering_score_0.8']:.3f}")
    print(f"Clustering score (>0.7): {summary['clustering_score_0.7']:.3f}")
    print(f"Interpretation: {results['interpretation']['clustering_interpretation']}")

GPU = "5"
MODEL = "MultiExpert_Scale_64_4"  
MODEL_PATH = f"/home/xuzhen/switch_sae/dictionaries/{MODEL}/8.pt"

def main():
    device = f'cuda:{GPU}'
    
    ae = MultiExpertScaleAutoEncoder(
        activation_dim=768,
        dict_size=32*768, 
        k=32,
        experts=64,
        e=4,
        heaviside=False
    )
    ae.load_state_dict(t.load(MODEL_PATH))
    ae.to(device)
    ae.eval()
    
    for feature_type in ["decoder"]:
        results = compute_feature_max_similarity(
            dictionary=ae,
            feature_type=feature_type,
            seed=42,
            device=device
        )
        
        print_results(results)
    
if __name__ == "__main__":
    main()