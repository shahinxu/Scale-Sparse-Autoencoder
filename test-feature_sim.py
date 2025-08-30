
import torch as t
import numpy as np
from dictionary_learning.trainers.moe_physically import MultiExpertAutoEncoder
import matplotlib.pyplot as plt
import csv

GPU = "5"


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
        MODEL = f"MultiExpert_64_{activation}"
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
            plt.plot([xi, xi], [inter_v, in_v], color='gray', linewidth=2, zorder=1)
            plt.scatter(xi, inter_v, color='C1', s=80, zorder=2, label='inter-expert' if xi==0 else '')
            plt.scatter(xi, in_v, color='C0', s=80, zorder=3, label='in-expert' if xi==0 else '')
            delta = in_v - inter_v
            y_annot = max(inter_v, in_v) + 0.01 * (abs(max(inter_vals.max(), in_vals.max())) + 1e-8)
            plt.text(xi, y_annot, f'{delta:.3f}', ha='center', va='bottom', fontsize=8, color='black')
        plt.xticks(x, activations)
        plt.xlabel('activation')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(axis='y', alpha=0.25)
        plt.tight_layout()
        plt.savefig(outpath, dpi=300)
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

    width = 0.35
    # mean bar
    plt.figure(figsize=(8,5))
    plt.bar(x - width/2, in_mean_list, width, label='in-expert', color='C0')
    plt.bar(x + width/2, inter_mean_list, width, label='inter-expert', color='C1')
    plt.xticks(x, activations)
    plt.xlabel('activation')
    plt.ylabel('Mean Similairty')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('expert_feature_similarity_mean_bar.png', dpi=300)
    plt.close()
    # max mean bar
    plt.figure(figsize=(8,5))
    plt.bar(x - width/2, in_max_mean_list, width, label='in-expert', color='C0')
    plt.bar(x + width/2, inter_max_mean_list, width, label='inter-expert', color='C1')
    plt.xticks(x, activations)
    plt.xlabel('activation')
    plt.ylabel('Max Similarity')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('expert_feature_similarity_max_mean_bar.png', dpi=300)
    plt.close()
    # max mean>0.9 ratio bar
    plt.figure(figsize=(8,5))
    plt.bar(x - width/2, in_max_ratio_list, width, label='in-expert', color='C0')
    plt.bar(x + width/2, inter_max_ratio_list, width, label='inter-expert', color='C1')
    plt.xticks(x, activations)
    plt.xlabel('activation')
    plt.ylabel('Ratio')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('expert_feature_similarity_max_mean_ratio_bar.png', dpi=300)
    plt.close()

    # write results to CSV
    csv_path = 'expert_feature_similarity_results.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['activation', 'metric', 'in_expert', 'inter_expert', 'delta'])
        for i, act in enumerate(activations):
            writer.writerow([act, 'mean', float(in_mean_list[i]), float(inter_mean_list[i]), float(in_mean_list[i] - inter_mean_list[i])])
            writer.writerow([act, 'max_mean', float(in_max_mean_list[i]), float(inter_max_mean_list[i]), float(in_max_mean_list[i] - inter_max_mean_list[i])])
            writer.writerow([act, 'max_mean_ratio', float(in_max_ratio_list[i]), float(inter_max_ratio_list[i]), float(in_max_ratio_list[i] - inter_max_ratio_list[i])])

    print(f'Lollipop plots saved and CSV written to {csv_path}')

if __name__ == "__main__":
    main()