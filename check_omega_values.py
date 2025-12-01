import torch as t
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from dictionary_learning.trainers.scale import MultiExpertScaleAutoEncoder

def check_omega_values(model_path, activation_dim=768, dict_size=32*768, k=32, experts=64, e=1, heaviside=False, save_fig=False, output_dir=None, map_location='cpu'):
    """
    检查已训练模型的omega参数值
    
    Args:
        model_path: 模型文件路径
        activation_dim: 激活维度
        dict_size: 字典大小
        k: top-k参数
        experts: 专家数量
        e: 选择的专家数量
        heaviside: 是否使用heaviside函数
    """
    
    # 加载模型
    print(f"Loading model from: {model_path}")
    # For batch runs we default to loading on CPU to avoid GPU memory pressure.
    device = 'cpu' if map_location == 'cpu' else ('cuda' if t.cuda.is_available() else 'cpu')
    
    try:
        # 创建模型实例
        model = MultiExpertScaleAutoEncoder(
            activation_dim=activation_dim,
            dict_size=dict_size,
            k=k,
            experts=experts,
            e=e,
            heaviside=heaviside
        )
        
        # 加载权重
        # load to CPU (map_location) by default, then move model to requested device
        state_dict = t.load(model_path, map_location=map_location)
        # allow wrappers where checkpoint contains {'state_dict': ...}
        if isinstance(state_dict, dict) and 'state_dict' in state_dict and isinstance(state_dict['state_dict'], dict):
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
        model.to(device)
        
        print(f"Model loaded successfully!")
        print(f"Device: {device}")
        print(f"Number of experts: {experts}")
        
        # 获取omega值
        omega_values = model.omega.data.cpu().numpy()
        beta_values = model.beta.data.cpu().numpy()
        
        print(f"\nOmega values shape: {omega_values.shape}")
        print(f"Beta values shape: {beta_values.shape}")
        
        # 打印omega值的统计信息
        print(f"\nOmega Statistics:")
        print(f"Mean: {np.mean(omega_values):.6f}")
        print(f"Std: {np.std(omega_values):.6f}")
        print(f"Min: {np.min(omega_values):.6f}")
        print(f"Max: {np.max(omega_values):.6f}")
        
        print(f"\nBeta Statistics:")
        print(f"Mean: {np.mean(beta_values):.6f}")
        print(f"Std: {np.std(beta_values):.6f}")
        print(f"Min: {np.min(beta_values):.6f}")
        print(f"Max: {np.max(beta_values):.6f}")
        
        # 打印每个专家的omega值
        print(f"\nOmega values for each expert:")
        for i, omega in enumerate(omega_values):
            print(f"Expert {i:2d}: {omega:8.6f}")
        
        print(f"\nBeta values for each expert:")
        for i, beta in enumerate(beta_values):
            print(f"Expert {i:2d}: {beta:8.6f}")
        
        # 可视化omega和beta值（仅在save_fig=True时生成）
        if save_fig:
            plt.figure(figsize=(15, 5))
            # Omega值分布
            plt.subplot(1, 3, 1)
            expert_ids = np.arange(len(omega_values))
            plt.bar(expert_ids, omega_values, alpha=0.7, color='blue')
            plt.xlabel('Expert ID')
            plt.ylabel('Omega Value')
            plt.title('Omega Values per Expert')
            plt.grid(True, alpha=0.3)
            
            # Beta值分布
            plt.subplot(1, 3, 2)
            plt.bar(expert_ids, beta_values, alpha=0.7, color='red')
            plt.xlabel('Expert ID')
            plt.ylabel('Beta Value')
            plt.title('Beta Values per Expert')
            plt.grid(True, alpha=0.3)
            
            # Omega和Beta的散点图
            plt.subplot(1, 3, 3)
            plt.scatter(omega_values, beta_values, alpha=0.7)
            plt.xlabel('Omega Value')
            plt.ylabel('Beta Value')
            plt.title('Omega vs Beta Values')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            # 保存图像
            if output_dir is None:
                output_dir = '.'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"omega_beta_k{k}_e{e}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\nVisualization saved to: {os.path.abspath(output_path)}")
        
        return omega_values, beta_values
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def main():
    # Batch configuration: grid of k and e to test
    LAYER = 8
    experts = 64
    # user-requested grid (you can adjust these lists)
    k_list = [128, 64, 32, 16, 8, 4]
    e_list = [1, 2, 4, 8, 16]

    results = {}
    import csv

    for k in k_list:
        results[k] = {}
        for e in e_list:
            # build candidate paths: prefer Scale variant, then plain
            path_scale = f"dictionaries/MultiExpert_Scale_{k}_{experts}_{e}/{LAYER}.pt"
            path_plain = f"dictionaries/MultiExpert_{k}_{experts}_{e}/{LAYER}.pt"
            model_path = None
            if os.path.exists(path_scale):
                model_path = path_scale
            elif os.path.exists(path_plain):
                model_path = path_plain

            if model_path is None:
                print(f"Skipping k={k}, e={e}: no checkpoint found (tried scale/plain)")
                results[k][e] = None
                continue

            print(f"Processing k={k}, e={e} -> {model_path}")
            omega_vals, beta_vals = check_omega_values(
                model_path=model_path,
                activation_dim=768,
                dict_size=32 * 768,
                k=k,
                experts=experts,
                e=e,
                heaviside=False,
                save_fig=False,
                output_dir='omega_outputs',
                map_location='cpu'
            )

            if omega_vals is None:
                results[k][e] = None
                continue

            # summarize
            res = {
                'omega_mean': float(np.mean(omega_vals)),
                'omega_std': float(np.std(omega_vals)),
                'omega_min': float(np.min(omega_vals)),
                'omega_max': float(np.max(omega_vals)),
                'beta_mean': float(np.mean(beta_vals)),
                'beta_std': float(np.std(beta_vals)),
                'beta_min': float(np.min(beta_vals)),
                'beta_max': float(np.max(beta_vals)),
                'n_experts': int(len(omega_vals))
            }
            results[k][e] = res

    # save JSON and CSV
    out_json = 'omega_grid_results.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wrote JSON results to {out_json}")

    # CSV: rows k, cols e, fill with omega_mean (or empty)
    out_csv = 'omega_grid_results.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['k\\e'] + [str(e) for e in e_list]
        writer.writerow(header)
        for k in k_list:
            row = [str(k)]
            for e in e_list:
                cell = ''
                if results.get(k) and results[k].get(e):
                    cell = f"{results[k][e]['omega_mean']:.6f}"
                row.append(cell)
            writer.writerow(row)
    print(f"Wrote CSV results to {out_csv}")

if __name__ == "__main__":
    main()
