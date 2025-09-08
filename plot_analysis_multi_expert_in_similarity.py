
import torch as t
import numpy as np
from dictionary_learning.trainers.moe_physically import MultiExpertAutoEncoder
import matplotlib.pyplot as plt
import csv

GPU = "5"

plt.rcParams.update({
    'font.size': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
})


def compute_in_expert_metrics(dictionary, device="cpu"):
    expert_modules = dictionary.expert_modules
    feature_matrices = [expert.decoder.weight if hasattr(expert.decoder, 'weight') else expert.decoder for expert in expert_modules]
    feature_matrices = [mat.to(device) for mat in feature_matrices]
    all_mean = []
    all_max_mean = []
    all_max_ratio = []
    for mat in feature_matrices:
        F = mat / mat.norm(dim=1, keepdim=True)
        S = F @ F.T
        S.fill_diagonal_(-float('inf'))
        mean_sim = S[S != -float('inf')].mean().item()
        max_sim = S.max(dim=1)[0].detach().cpu().numpy()
        all_mean.append(mean_sim)
        all_max_mean.append(np.mean(max_sim))
        all_max_ratio.append(np.mean(max_sim > 0.9))
    in_expert_mean = np.mean(all_mean)
    in_expert_max_mean = np.mean(all_max_mean)
    in_expert_max_mean_ratio = np.mean(all_max_ratio)
    return in_expert_mean, in_expert_max_mean, in_expert_max_mean_ratio


def compute_inter_expert_metrics(dictionary, device="cpu"):
    import random
    expert_modules = dictionary.expert_modules
    feature_matrices = [expert.decoder.weight if hasattr(expert.decoder, 'weight') else expert.decoder for expert in expert_modules]
    feature_matrices = [mat.to(device) for mat in feature_matrices]
    feature_counts = [mat.shape[0] for mat in feature_matrices]
    all_features = t.cat(feature_matrices, dim=0)
    total_features = all_features.shape[0]
    all_mean = []
    all_max_mean = []
    all_max_ratio = []
    for count in feature_counts:
        idx = random.sample(range(total_features), count)
        F = all_features[idx]
        F = F / F.norm(dim=1, keepdim=True)
        S = F @ F.T
        S.fill_diagonal_(-float('inf'))
        mean_sim = S[S != -float('inf')].mean().item()
        max_sim = S.max(dim=1)[0].detach().cpu().numpy()
        all_mean.append(mean_sim)
        all_max_mean.append(np.mean(max_sim))
        all_max_ratio.append(np.mean(max_sim > 0.9))
    inter_expert_mean = np.mean(all_mean)
    inter_expert_max_mean = np.mean(all_max_mean)
    inter_expert_max_mean_ratio = np.mean(all_max_ratio)
    return inter_expert_mean, inter_expert_max_mean, inter_expert_max_mean_ratio


def main():
    device = f'cuda:{GPU}'
    activations = [1, 2, 4, 8, 16]
    in_mean_list = []
    in_max_mean_list = []
    in_max_ratio_list = []
    inter_mean_list = []
    inter_max_mean_list = []
    inter_max_ratio_list = []

    for activation in activations:
        MODEL = f"MultiExpert_32_64_{activation}"
        MODEL_PATH = f"/home/xuzhen/switch_sae/dictionaries/{MODEL}/8.pt"
        ae = MultiExpertAutoEncoder(
            activation_dim=768,
            dict_size=32*768,
            k=32,
            experts=64,
            e=activation,
            heaviside=False
        )
        ae.load_state_dict(t.load(MODEL_PATH))
        ae.to(device)
        ae.eval()
        in_expert_mean, in_expert_max_mean, in_expert_max_mean_ratio = compute_in_expert_metrics(ae, device=device)
        inter_expert_mean, inter_expert_max_mean, inter_expert_max_mean_ratio = compute_inter_expert_metrics(ae, device=device)
        in_mean_list.append(in_expert_mean)
        in_max_mean_list.append(in_expert_max_mean)
        in_max_ratio_list.append(in_expert_max_mean_ratio)
        inter_mean_list.append(inter_expert_mean)
        inter_max_mean_list.append(inter_expert_max_mean)
        inter_max_ratio_list.append(inter_expert_max_mean_ratio)

    x = np.arange(len(activations))

    def draw_lollipop_single(x, inter_vals, in_vals, activations, title, ylabel, outpath):
        plt.figure(figsize=(8,5))
        for xi, inter_v, in_v in zip(x, inter_vals, in_vals):
            plt.plot([xi, xi], [inter_v, in_v], color='gray', linewidth=3, zorder=1)
            plt.scatter(xi, inter_v, color='C1', s=100, zorder=2, label='inter-expert' if xi==0 else '')
            plt.scatter(xi, in_v, color='C0', s=100, zorder=3, label='in-expert' if xi==0 else '')
            delta = in_v - inter_v
            y_annot = max(inter_v, in_v) + 0.01 * (abs(max(inter_vals.max(), in_vals.max())) + 1e-8)
            plt.text(xi, y_annot, f'{delta:.3f}', ha='center', va='bottom', fontsize=12, color='black')
        plt.xticks(x, activations)
        plt.xlabel('activation')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        plt.close()

    in_mean_arr = np.array(in_mean_list)
    inter_mean_arr = np.array(inter_mean_list)
    in_max_mean_arr = np.array(in_max_mean_list)
    inter_max_mean_arr = np.array(inter_max_mean_list)
    in_max_ratio_arr = np.array(in_max_ratio_list)
    inter_max_ratio_arr = np.array(inter_max_ratio_list)

    draw_lollipop_single(x, inter_mean_arr, in_mean_arr, activations, 'Mean Similarity', 'Mean Similarity', 'expert_feature_similarity_mean_lollipop.png')
    draw_lollipop_single(x, inter_max_mean_arr, in_max_mean_arr, activations, 'Max Similarity', 'Mean Similarity', 'expert_feature_similarity_max_mean_lollipop.png')
    draw_lollipop_single(x, inter_max_ratio_arr, in_max_ratio_arr, activations, 'Ratio', 'Ratio', 'expert_feature_similarity_max_mean_ratio_lollipop.png')

    width = 0.4
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))
    
    ax1.bar(x - width/2, in_max_ratio_list, width, label='in-expert', color='#337AFF', hatch='///')
    ax1.bar(x + width/2, inter_max_ratio_list, width, label='inter-expert', color='#FF5733', hatch='\\\\\\')
    ax1.set_xticks(x)
    ax1.set_xticklabels(activations)
    ax1.set_xlabel('# Experts')
    ax1.set_ylabel('Ratio')
    ax1.set_ylim(0, 0.006)
    ax1.set_title('L0=16')
    ax1.legend()
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    ax2.bar(x - width/2, in_max_ratio_list, width, color='#337AFF', hatch='///')
    ax2.bar(x + width/2, inter_max_ratio_list, width, color='#FF5733', hatch='\\\\\\')
    ax2.set_xticks(x)
    ax2.set_xticklabels(activations)
    ax2.set_xlabel('# Experts')
    ax2.set_ylim(0, 0.006)
    ax2.set_title('L0=32')
    ax2.set_yticklabels([])
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig('analysis_multi_expert_ratio_bar.png', dpi=300, bbox_inches='tight')
    print("Saved analysis_multi_expert_ratio_bar.png")
    plt.close()

if __name__ == "__main__":
    main()