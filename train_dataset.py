import torch as t
from nnsight import LanguageModel
from dictionary_learning.test_buffer import ActivationBuffer
from dictionary_learning.trainers.moe_physically import MultiExpertAutoEncoder
import json
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from config import lm, activation_dim, layer, n_ctxs
from collections import defaultdict
import os
import re

GPU = "0"
MODEL = "MultiExpert_64_8"
MODEL_PATH = f"/home/xuzhen/switch_sae/dictionaries/{MODEL}/8.pt"
OUTPUT_ROOT = f"expert_feature_analysis_{MODEL}_synthetic"

TARGET_EXPERTS = [0, 1]  # 只分析前两个expert


def generate_synthetic_sentences():
    """生成100个测试句子"""
    sentences = [
        # 基础语法结构
        "The cat sat on the mat.",
        "She walked to the store yesterday.",
        "They are playing football in the park.",
        "I will go to school tomorrow.",
        "He has been working here for five years.",
        
        # 复合词和专有名词
        "The Ardhanarishvara sculpture is beautiful.",
        "McDonald's restaurant serves hamburgers.",
        "Spider-Man is a popular superhero.",
        "New York City has many skyscrapers.",
        "Twenty-first century technology is advancing rapidly.",
        
        # 数字和日期
        "The meeting is scheduled for 3:30 PM.",
        "She was born on January 15, 1990.",
        "The temperature reached 25 degrees Celsius.",
        "Chapter 11 discusses important concepts.",
        "The company earned $2.5 million last year.",
        
        # 标点符号和特殊字符
        "Wait... what did you say?",
        "The email address is user@example.com.",
        "Use Ctrl+C to copy the text.",
        "The ratio is 3:1 in favor of the team.",
        "She said, 'Hello, how are you?'",
        
        # 科学和技术术语
        "DNA contains genetic information.",
        "The algorithm processes data efficiently.",
        "Photosynthesis occurs in plant leaves.",
        "Machine learning requires large datasets.",
        "The HTTP protocol enables web communication.",
        
        # 文学和艺术
        "Shakespeare wrote many famous plays.",
        "The Mona Lisa is displayed in the Louvre.",
        "Jazz music originated in New Orleans.",
        "Impressionist painters used vibrant colors.",
        "The novel explores themes of love and loss.",
        
        # 历史和地理
        "World War II ended in 1945.",
        "The Amazon rainforest spans multiple countries.",
        "Ancient Rome had a powerful military.",
        "Mount Everest is the highest peak.",
        "The Silk Road connected East and West.",
        
        # 动物和自然
        "Lions live in the African savanna.",
        "Dolphins are intelligent marine mammals.",
        "Earthquakes occur along tectonic plates.",
        "Rainbows appear after storms.",
        "Butterflies undergo metamorphosis.",
        
        # 食物和文化
        "Italian cuisine features pasta and pizza.",
        "Sushi is a traditional Japanese dish.",
        "Thanksgiving is celebrated in November.",
        "Coffee beans are grown in tropical regions.",
        "Chocolate comes from cacao trees.",
        
        # 运动和娱乐
        "The Olympics occur every four years.",
        "Basketball players need good coordination.",
        "Movies are shown in theaters worldwide.",
        "Video games have become increasingly popular.",
        "Musicians perform concerts for audiences.",
        
        # 职业和工作
        "Doctors diagnose and treat patients.",
        "Engineers design bridges and buildings.",
        "Teachers educate students in schools.",
        "Farmers grow crops to feed people.",
        "Artists create beautiful works of art.",
        
        # 复杂句子结构
        "Although it was raining, they decided to go hiking.",
        "The book, which was published last year, became a bestseller.",
        "Neither the students nor the teacher understood the problem.",
        "If you study hard, you will pass the exam.",
        "The more you practice, the better you become.",
        
        # 抽象概念
        "Freedom is a fundamental human right.",
        "Love conquers all obstacles in life.",
        "Knowledge is power in modern society.",
        "Time heals all wounds eventually.",
        "Beauty lies in the eye of the beholder.",
        
        # 技术和创新
        "Artificial intelligence is transforming industries.",
        "Smartphones connect people around the world.",
        "Solar panels convert sunlight into electricity.",
        "3D printing creates objects from digital designs.",
        "Virtual reality provides immersive experiences.",
        
        # 教育和学习
        "Students learn through reading and practice.",
        "Universities offer degrees in various fields.",
        "Online courses provide flexible learning options.",
        "Research contributes to scientific knowledge.",
        "Libraries preserve books and information.",
        
        # 健康和医学
        "Exercise improves physical and mental health.",
        "Vaccines prevent dangerous diseases.",
        "Healthy eating includes fruits and vegetables.",
        "Sleep is essential for brain function.",
        "Medical research saves countless lives.",
        
        # 环境和生态
        "Climate change affects global weather patterns.",
        "Recycling reduces waste and pollution.",
        "Forests provide oxygen and habitats.",
        "Renewable energy sources are sustainable.",
        "Conservation protects endangered species.",
        
        # 商业和经济
        "Companies compete for market share.",
        "Investments can generate returns over time.",
        "Supply and demand determine prices.",
        "Innovation drives economic growth.",
        "Trade connects countries and cultures.",
        
        # 社会和政治
        "Democracy allows citizens to vote.",
        "Laws maintain order in society.",
        "Human rights protect individual freedoms.",
        "Communities work together for common goals.",
        "Leadership requires vision and integrity.",
        
        # 混合复杂句子
        "The quick brown fox jumps over the lazy dog.",
        "Anti-establishment protesters gathered in downtown Manhattan.",
        "Twenty-first-century nano-technology revolutionizes medicine.",
        "Self-driving cars use AI-powered computer vision systems.",
        "Multi-billion-dollar corporations influence global markets.",
    ]
    
    return sentences[:100]  # 确保只返回100个句子


class ExpertFeatureCollector:
    """收集指定expert的每个feature的激活token信息"""
    
    def __init__(self, target_experts, expert_dict_size=768*32):
        self.target_experts = set(target_experts)
        self.expert_dict_size = expert_dict_size
        
        # expert_id -> feature_id -> list of token activations
        self.expert_feature_tokens = defaultdict(lambda: defaultdict(list))
        
        # expert_id -> feature_id -> max activation strength
        self.expert_feature_max_strength = defaultdict(lambda: defaultdict(float))
        
        # expert_id -> feature_id -> best example
        self.expert_feature_best_example = defaultdict(dict)
        
        # 统计信息
        self.expert_stats = defaultdict(lambda: {
            'total_features_activated': 0,
            'total_token_activations': 0,
            'texts_processed': 0
        })
        
        self.total_texts_processed = 0
    
    def add_feature_activation(self, expert_id, feature_id, token_text, activation_strength, 
                             text_id, token_pos, original_text):
        """添加一个feature的token激活记录"""
        
        if expert_id not in self.target_experts:
            return
        
        # 计算相对feature ID (在该expert内的ID)
        relative_feature_id = feature_id % self.expert_dict_size
        
        # 添加token激活记录
        token_record = {
            'token': token_text,
            'strength': activation_strength,
            'text_id': text_id,
            'token_pos': token_pos,
            'original_text': original_text
        }
        
        self.expert_feature_tokens[expert_id][relative_feature_id].append(token_record)
        
        # 更新最大激活强度和最佳示例
        if activation_strength > self.expert_feature_max_strength[expert_id][relative_feature_id]:
            self.expert_feature_max_strength[expert_id][relative_feature_id] = activation_strength
            self.expert_feature_best_example[expert_id][relative_feature_id] = token_record
        
        # 更新统计信息
        self.expert_stats[expert_id]['total_token_activations'] += 1
        
        # 更新feature计数（只计算已激活的feature）
        self.expert_stats[expert_id]['total_features_activated'] = len(self.expert_feature_tokens[expert_id])
    
    def update_stats(self, total_texts):
        """更新统计信息"""
        self.total_texts_processed = total_texts
        
        for expert_id in self.target_experts:
            self.expert_stats[expert_id]['texts_processed'] = total_texts
    
    def get_expert_feature_summary(self, expert_id, top_n=20):
        """获取指定expert的feature摘要"""
        if expert_id not in self.expert_feature_tokens:
            return None
        
        features_data = []
        
        for feature_id, token_records in self.expert_feature_tokens[expert_id].items():
            max_strength = self.expert_feature_max_strength[expert_id][feature_id]
            best_example = self.expert_feature_best_example[expert_id][feature_id]
            
            # 统计unique tokens
            unique_tokens = set(record['token'] for record in token_records)
            
            # 获取最强激活的前几个token
            sorted_records = sorted(token_records, key=lambda x: x['strength'], reverse=True)
            top_tokens = sorted_records[:top_n]
            
            features_data.append({
                'feature_id': feature_id,
                'max_activation': max_strength,
                'total_activations': len(token_records),
                'unique_tokens_count': len(unique_tokens),
                'unique_tokens': list(unique_tokens),
                'top_activations': top_tokens,
                'best_example': best_example
            })
        
        # 按最大激活强度排序
        features_data.sort(key=lambda x: x['max_activation'], reverse=True)
        
        return {
            'expert_id': expert_id,
            'statistics': self.expert_stats[expert_id],
            'total_features': len(features_data),
            'features': features_data
        }
    
    def save_expert_feature_analysis(self, output_dir):
        """保存expert feature分析结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 全局统计
        global_stats = {
            'target_experts': list(self.target_experts),
            'total_texts_processed': self.total_texts_processed,
            'expert_feature_counts': {
                expert_id: len(self.expert_feature_tokens[expert_id]) 
                for expert_id in self.target_experts
            },
            'dataset_info': {
                'source': 'Synthetic Sentences',
                'total_sentences': 100,
                'description': 'Hand-crafted sentences for SAE analysis'
            }
        }
        
        with open(os.path.join(output_dir, 'global_statistics.json'), 'w', encoding='utf-8') as f:
            json.dump(global_stats, f, indent=2, ensure_ascii=False)
        
        # 为每个target expert生成详细分析
        for expert_id in self.target_experts:
            if expert_id not in self.expert_feature_tokens:
                continue
                
            expert_dir = os.path.join(output_dir, f'expert_{expert_id:02d}')
            os.makedirs(expert_dir, exist_ok=True)
            
            # 获取完整的feature摘要
            summary = self.get_expert_feature_summary(expert_id, top_n=50)
            
            # 保存完整的JSON分析
            with open(os.path.join(expert_dir, 'feature_analysis.json'), 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            # 保存可读的文本报告
            self._save_readable_feature_report(expert_dir, summary)
            
            # 保存每个feature的详细token列表
            self._save_feature_token_details(expert_dir, expert_id)
    
    def _save_readable_feature_report(self, expert_dir, summary):
        """保存可读的feature报告"""
        with open(os.path.join(expert_dir, 'feature_report.txt'), 'w', encoding='utf-8') as f:
            expert_id = summary['expert_id']
            stats = summary['statistics']
            
            f.write(f"Expert {expert_id} - Feature Analysis Report (Synthetic Data)\n")
            f.write("="*70 + "\n\n")
            
            f.write("📊 Statistics:\n")
            f.write(f"  Total Features Activated: {stats['total_features_activated']}\n")
            f.write(f"  Total Token Activations: {stats['total_token_activations']}\n")
            f.write(f"  Texts Processed: {stats['texts_processed']}\n")
            if stats['total_features_activated'] > 0:
                f.write(f"  Avg Activations per Feature: {stats['total_token_activations']/stats['total_features_activated']:.2f}\n\n")
            
            f.write("🔥 Top Features by Maximum Activation:\n")
            f.write("-"*70 + "\n")
            
            for i, feature_data in enumerate(summary['features'][:20], 1):
                feature_id = feature_data['feature_id']
                max_activation = feature_data['max_activation']
                total_activations = feature_data['total_activations']
                unique_tokens = feature_data['unique_tokens_count']
                best_example = feature_data['best_example']
                
                f.write(f"\n{i:2d}. Feature {feature_id:4d}:\n")
                f.write(f"    Max Activation: {max_activation:.4f}\n")
                f.write(f"    Total Activations: {total_activations}\n")
                f.write(f"    Unique Tokens: {unique_tokens}\n")
                f.write(f"    Best Token: '{best_example['token']}' (strength: {best_example['strength']:.4f})\n")
                f.write(f"    Context: {best_example['original_text']}\n")
                
                # 显示该feature的前10个最强token
                top_tokens = feature_data['top_activations'][:10]
                token_list = [f"'{record['token']}'({record['strength']:.3f})" for record in top_tokens]
                f.write(f"    Top Tokens: {', '.join(token_list)}\n")
    
    def _save_feature_token_details(self, expert_dir, expert_id):
        """保存每个feature的详细token信息"""
        features_dir = os.path.join(expert_dir, 'features')
        os.makedirs(features_dir, exist_ok=True)
        
        for feature_id, token_records in self.expert_feature_tokens[expert_id].items():
            feature_file = os.path.join(features_dir, f'feature_{feature_id:04d}.txt')
            
            with open(feature_file, 'w', encoding='utf-8') as f:
                f.write(f"Expert {expert_id} - Feature {feature_id} - Token Activations\n")
                f.write("="*60 + "\n\n")
                
                # 按激活强度排序
                sorted_records = sorted(token_records, key=lambda x: x['strength'], reverse=True)
                
                f.write(f"Total Activations: {len(token_records)}\n")
                f.write(f"Unique Tokens: {len(set(record['token'] for record in token_records))}\n")
                f.write(f"Max Activation: {max(record['strength'] for record in token_records):.4f}\n\n")
                
                f.write("All Token Activations (sorted by strength):\n")
                f.write("-"*60 + "\n")
                
                for i, record in enumerate(sorted_records, 1):
                    f.write(f"{i:3d}. '{record['token']}' ")
                    f.write(f"(strength: {record['strength']:.4f}, ")
                    f.write(f"text_{record['text_id']}, pos_{record['token_pos']})\n")
                    f.write(f"     Context: {record['original_text']}\n\n")


@t.no_grad()
def analyze_synthetic_sentences(dictionary, model, submodule, device, sentences, collector):
    """分析合成句子，收集target experts的feature激活信息"""
    
    print(f"\n{'='*60}")
    print(f"Processing {len(sentences)} synthetic sentences")
    print(f"Target experts: {TARGET_EXPERTS}")
    print(f"{'='*60}")
    
    tokenizer = AutoTokenizer.from_pretrained(lm)
    
    def gen():
        while True:
            for sentence in sentences:
                yield sentence
    
    buffer = ActivationBuffer(
        gen(), 
        model, 
        submodule, 
        d_submodule=activation_dim, 
        n_ctxs=min(n_ctxs, len(sentences) * 50),  # 每个句子平均50个token
        device=device,
        sequential=True
    )
    
    sentence_feature_activations = defaultdict(int)
    
    for text_id, sentence in enumerate(sentences):
        try:
            x = next(buffer).to(device)
        except StopIteration:
            print(f"Warning: Not enough activations for sentence {text_id}")
            break
        
        input_ids = tokenizer.encode(sentence, truncation=True, max_length=128)
        tokens = [tokenizer.decode([token_id]) for token_id in input_ids]
        
        if text_id % 10 == 0:
            print(f"  Processing sentence {text_id}/{len(sentences)}: '{sentence[:60]}...'")
        
        for token_pos in range(min(len(x), len(tokens))):
            token_activation = x[token_pos]
            token_text = tokens[token_pos]
            
            # 获取SAE的feature激活
            _, f = dictionary(token_activation.unsqueeze(0), output_features=True)
            token_features = f[0]
            
            # 获取top-k激活的features
            top_k_values, top_k_indices = token_features.topk(dictionary.k, sorted=True)
            expert_dict_size = dictionary.expert_dict_size
            
            for fid, fval in zip(top_k_indices, top_k_values):
                if fval.item() > 0:
                    expert_id = fid.item() // expert_dict_size
                    
                    # 只处理target experts
                    if expert_id in TARGET_EXPERTS:
                        collector.add_feature_activation(
                            expert_id=expert_id,
                            feature_id=fid.item(),
                            token_text=token_text,
                            activation_strength=fval.item(),
                            text_id=text_id,
                            token_pos=token_pos,
                            original_text=sentence
                        )
                        sentence_feature_activations[expert_id] += 1
    
    collector.update_stats(len(sentences))
    
    print(f"  Analysis completed:")
    for expert_id in TARGET_EXPERTS:
        activations = sentence_feature_activations.get(expert_id, 0)
        total_features = len(collector.expert_feature_tokens[expert_id])
        print(f"    Expert {expert_id}: {activations} activations, {total_features} features")
    
    del buffer
    t.cuda.empty_cache()


def main():
    device = f'cuda:{GPU}'
    
    print(f"Expert Feature Analysis Configuration (Synthetic Data):")
    print(f"  Model: {MODEL}")
    print(f"  Device: {device}")
    print(f"  Target Experts: {TARGET_EXPERTS}")
    print(f"  Synthetic sentences: 100")
    
    # 生成测试句子
    sentences = generate_synthetic_sentences()
    print(f"\n📝 Generated {len(sentences)} synthetic sentences")
    print("Sample sentences:")
    for i, sentence in enumerate(sentences[:5], 1):
        print(f"  {i}. {sentence}")
    print("  ...")
    
    print("\nLoading language model...")
    model = LanguageModel(lm, dispatch=True, device_map=device)
    submodule = model.transformer.h[layer]
    
    print(f"Loading SAE from {MODEL_PATH}...")
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
    
    # 初始化feature收集器
    collector = ExpertFeatureCollector(
        target_experts=TARGET_EXPERTS,
        expert_dict_size=ae.expert_dict_size
    )
    
    # 分析合成句子
    try:
        analyze_synthetic_sentences(ae, model, submodule, device, sentences, collector)
    except Exception as e:
        print(f"Error during analysis: {e}")
        return
    
    # 保存结果
    print(f"\nSaving expert feature analysis results...")
    collector.save_expert_feature_analysis(OUTPUT_ROOT)
    
    # 最终统计
    print(f"\n✅ Expert Feature Analysis Complete!")
    print(f"📊 Final Statistics:")
    print(f"  Total sentences processed: {collector.total_texts_processed}")
    
    for expert_id in TARGET_EXPERTS:
        if expert_id in collector.expert_feature_tokens:
            stats = collector.expert_stats[expert_id]
            feature_count = len(collector.expert_feature_tokens[expert_id])
            print(f"\n  Expert {expert_id}:")
            print(f"    Features activated: {feature_count}")
            print(f"    Total activations: {stats['total_token_activations']}")
            if feature_count > 0:
                print(f"    Avg activations per feature: {stats['total_token_activations']/feature_count:.2f}")
    
    print(f"\n📁 Results saved to: {OUTPUT_ROOT}/")
    print(f"  - global_statistics.json: 全局统计信息")
    print(f"  - expert_XX/feature_analysis.json: 每个expert的完整分析")
    print(f"  - expert_XX/feature_report.txt: 可读的分析报告")
    print(f"  - expert_XX/features/feature_XXXX.txt: 每个feature的详细token列表")


if __name__ == "__main__":
    main()