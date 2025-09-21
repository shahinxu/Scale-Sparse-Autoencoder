import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch as t
from nnsight import LanguageModel
from dictionary_learning.trainers.moe_physically_scale import MultiExpertScaleAutoEncoder
from dictionary_learning.trainers.top_k import AutoEncoderTopK
from config import lm, layer

from .utils.model_utils import detect_model_type
from .analysis.buffer import FixedOrderBuffer
from .analysis.analyzer import analyze_with_fixed_buffer
from .visualization.token_visualizer import create_token_visualization

GPU = "0"
K = 32
EXPERT = 64
E = 8
MODEL = f"MultiExpert_Scale_{K}_{EXPERT}_{E}"
LAYER = 8
MODEL_PATH = f"/home/xuzhen/switch_sae/dictionaries/{MODEL}/{LAYER}.pt"
OUTPUT_ROOT = f"sae_analysis_results_{MODEL}_{LAYER}"


def main():
    device = f'cuda:{GPU}'
    
    print("Loading language model...")
    model = LanguageModel(lm, dispatch=True, device_map=device)
    
    custom_texts = [
        # "The cat sat on the mat.",
        # "She walked to the store yesterday.",
        
        # "The answer is 42 plus 17 equals 59.",
        # "In 2023, the temperature was -15.7 degrees.",
        
        # "Barack Obama was born in Hawaii.",
        # "Microsoft Corporation develops Windows.",
        
        # "Twenty-first-century nano-technology is amazing.",
        # "Self-driving cars use state-of-the-art AI.",
        
        # "Wait... what? Really?! That's @incredible#hashtag!",
        # "Visit https://www.example.com for more info.",
        
        # "DNA contains adenine, thymine, guanine, and cytosine.",
        # "The algorithm uses gradient descent optimization.",
        
        # "I feel extremely happy and grateful today.",
        # "Love, hope, and freedom are universal values.",
        
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
    
    ae = MultiExpertScaleAutoEncoder(
        activation_dim=768,
        dict_size=32*768,
        k=K,
        experts=EXPERT,
        e=E,
        heaviside=False
    )
    # ae = AutoEncoderTopK(
    #     activation_dim=768,
    #     dict_size=1*768,
    #     k=32
    # )
    
    ae.load_state_dict(t.load(MODEL_PATH))
    ae.to(device)
    ae.eval()
    
    model_info = detect_model_type(ae)
    
    print("Analyzing with Fixed Order Buffer...")
    analyze_with_fixed_buffer(ae, buffer, custom_texts, model_info, OUTPUT_ROOT, K, device=device)
    
    print(f"\nAnalysis complete!")
    print(f"Results saved to: {OUTPUT_ROOT}/")
    print(f"Model information saved to: {OUTPUT_ROOT}/model_info.txt")

if __name__ == "__main__":
    main()