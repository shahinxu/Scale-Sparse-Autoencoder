import torch as t
import numpy as np
import matplotlib.pyplot as plt
import os
from dictionary_learning.trainers.moe_physically_scale import MultiExpertScaleAutoEncoder

def check_omega_values(model_path, activation_dim=768, dict_size=32*768, k=32, experts=64, e=1, heaviside=False):
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
    device = 'cuda' if t.cuda.is_available() else 'cpu'
    
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
        state_dict = t.load(model_path, map_location=device)
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
        
        # 可视化omega和beta值
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
        output_path = "omega_beta_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {os.path.abspath(output_path)}")
        
        return omega_values, beta_values
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def main():
    # 配置参数
    LAYER = 8
    E = 16
    k = 32
    experts = 64
    
    # 模型路径
    model_path = f"dictionaries/MultiExpert_Scale_{k}_{experts}_{E}/{LAYER}.pt"
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please check the path and make sure the model exists.")
        
        print("\nLooking for alternative model files...")
        dictionaries_dir = "dictionaries"
        if os.path.exists(dictionaries_dir):
            for item in os.listdir(dictionaries_dir):
                if "MultiExpert_Scale" in item:
                    full_path = os.path.join(dictionaries_dir, item)
                    if os.path.isdir(full_path):
                        model_file = os.path.join(full_path, f"{LAYER}.pt")
                        if os.path.exists(model_file):
                            print(f"Found: {model_file}")
        return
    
    print(f"Checking omega values for model: {model_path}")
    omega_values, beta_values = check_omega_values(
        model_path=model_path,
        activation_dim=768,
        dict_size=32 * 768,  # 根据实际情况调整
        k=k,
        experts=experts,
        e=E,
        heaviside=False
    )
    
    if omega_values is not None:
        print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()
