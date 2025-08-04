import torch as t
from nnsight import LanguageModel
from dictionary_learning.test_buffer import ActivationBuffer
from dictionary_learning.trainers.moe_physically import MultiExpertAutoEncoder
import json
from transformers import AutoTokenizer
from config import lm, activation_dim, layer, n_ctxs
from collections import defaultdict
import os
import pandas as pd

GPU = "5"
MODEL = "MultiExpert_64_8"
MODEL_PATH = f"/home/xuzhen/switch_sae/dictionaries/{MODEL}/2.pt"
OUTPUT_ROOT = f"expert_feature_analysis_{MODEL}_wikitext"

WIKITEXT_PATH = "/home/xuzhen/switch_sae/wikitext"
WIKITEXT_VERSION = "wikitext-2-raw-v1"
SPLIT = "train"

BATCH_SIZE = 200
TOTAL_BATCHES = 10
TARGET_EXPERTS = [0, 1]


def load_wikitext_batch(wikitext_path, version="wikitext-2-raw-v1", split="train", 
                       batch_size=200, batch_idx=0, min_length=20, max_length=200):
    """批次加载WikiText数据集"""
    
    dataset_path = os.path.join(wikitext_path, version)
    parquet_files = []
    
    if os.path.exists(dataset_path):
        all_files = os.listdir(dataset_path)
        parquet_files = [f for f in all_files if f.startswith(f"{split}-") and f.endswith(".parquet")]
        parquet_files.sort()
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found for {split} split in {dataset_path}")
    
    print(f"Loading batch {batch_idx} (size: {batch_size})")
    
    all_texts = []
    texts_read = 0
    start_idx = batch_idx * batch_size
    end_idx = start_idx + batch_size
    
    for parquet_file in parquet_files:
        full_path = os.path.join(dataset_path, parquet_file)
        df = pd.read_parquet(full_path)
        
        for _, row in df.iterrows():
            text = row['text'].strip()
            
            if len(text) < min_length:
                continue
            if len(text) > max_length:
                text = text[:max_length]
            if text.startswith('=') and text.endswith('='):
                continue
            if not text or text.isspace():
                continue
            if len(text.split()) < 3:
                continue
            
            if texts_read < start_idx:
                texts_read += 1
                continue
                
            if texts_read < end_idx:
                all_texts.append(text)
                texts_read += 1
            else:
                break
        
        if len(all_texts) >= batch_size:
            break
    
    print(f"  Loaded {len(all_texts)} texts for batch {batch_idx}")
    return all_texts


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
        self.total_batches_processed = 0
    
    def add_feature_activation(self, expert_id, feature_id, token_text, activation_strength, 
                             text_id, token_pos, original_text):
        """添加一个feature的token激活记录"""
        
        if expert_id not in self.target_experts:
            return
        
        global_text_id = self.total_texts_processed + text_id
        
        # 计算相对feature ID (在该expert内的ID)
        relative_feature_id = feature_id % self.expert_dict_size
        
        # 添加token激活记录
        token_record = {
            'token': token_text,
            'strength': activation_strength,
            'text_id': global_text_id,
            'token_pos': token_pos,
            'original_text': original_text[:100] + '...' if len(original_text) > 100 else original_text
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
    
    def update_batch_stats(self, batch_size):
        """更新批次统计信息"""
        self.total_texts_processed += batch_size
        self.total_batches_processed += 1
        
        for expert_id in self.target_experts:
            self.expert_stats[expert_id]['texts_processed'] = self.total_texts_processed
    
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
            'total_batches_processed': self.total_batches_processed,
            'expert_feature_counts': {
                expert_id: len(self.expert_feature_tokens[expert_id]) 
                for expert_id in self.target_experts
            },
            'dataset_info': {
                'source': 'WikiText',
                'version': WIKITEXT_VERSION,
                'split': SPLIT,
                'batch_size': BATCH_SIZE,
                'total_batches': TOTAL_BATCHES
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
            
            f.write(f"Expert {expert_id} - Feature Analysis Report (WikiText)\n")
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
def analyze_batch(dictionary, model, submodule, device, texts, batch_idx, collector):
    """分析一个批次的文本，收集target experts的feature激活信息"""
    
    print(f"\n{'='*60}")
    print(f"Processing Batch {batch_idx + 1}/{TOTAL_BATCHES}")
    print(f"Batch size: {len(texts)} texts")
    print(f"Target experts: {TARGET_EXPERTS}")
    print(f"{'='*60}")
    
    tokenizer = AutoTokenizer.from_pretrained(lm)
    
    def gen():
        while True:
            for text in texts:
                input_ids = tokenizer.encode(text, truncation=True, max_length=128)
                processed_text = tokenizer.decode(input_ids, skip_special_tokens=True)
                yield processed_text
    
    buffer = ActivationBuffer(
        gen(), 
        model, 
        submodule, 
        d_submodule=activation_dim, 
        n_ctxs=min(n_ctxs, len(texts) * 150),
        device=device,
        sequential=True
    )
    
    batch_feature_activations = defaultdict(int)
    
    for text_id, text in enumerate(texts):
        try:
            x = next(buffer).to(device)
        except StopIteration:
            print(f"Warning: Not enough activations for text {text_id}")
            break
        
        input_ids = tokenizer.encode(text, truncation=True, max_length=128)
        tokens = [tokenizer.decode([token_id]) for token_id in input_ids]
        
        if text_id % 50 == 0:
            print(f"  Processing text {text_id}/{len(texts)}: '{text[:50]}...'")
        
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
                            original_text=text
                        )
                        batch_feature_activations[expert_id] += 1
    
    collector.update_batch_stats(len(texts))
    
    print(f"  Batch {batch_idx + 1} completed:")
    for expert_id in TARGET_EXPERTS:
        activations = batch_feature_activations.get(expert_id, 0)
        total_features = len(collector.expert_feature_tokens[expert_id])
        print(f"    Expert {expert_id}: {activations} activations, {total_features} features")
    
    del buffer
    t.cuda.empty_cache()


def main():
    device = f'cuda:{GPU}'
    
    print(f"Expert Feature Analysis Configuration:")
    print(f"  Dataset: {WIKITEXT_VERSION}")
    print(f"  Split: {SPLIT}")
    print(f"  Model: {MODEL}")
    print(f"  Device: {device}")
    print(f"  Target Experts: {TARGET_EXPERTS}")
    print(f"  Batch processing: {BATCH_SIZE} texts per batch, {TOTAL_BATCHES} batches")
    print(f"  Total texts to process: {BATCH_SIZE * TOTAL_BATCHES}")
    
    print("\nLoading language model...")
    model = LanguageModel(lm, dispatch=True, device_map=device)
    submodule = model.transformer.h[layer]
    
    print(f"Loading SAE from {MODEL_PATH}...")
    ae = MultiExpertAutoEncoder(
        activation_dim=768, 
        dict_size=32*768, 
        k=4, 
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
    
    # 批次处理
    for batch_idx in range(TOTAL_BATCHES):
        try:
            # 加载当前批次的数据
            batch_texts = load_wikitext_batch(
                wikitext_path=WIKITEXT_PATH,
                version=WIKITEXT_VERSION,
                split=SPLIT,
                batch_size=BATCH_SIZE,
                batch_idx=batch_idx
            )
            
            if not batch_texts:
                print(f"No more texts available at batch {batch_idx}")
                break
            
            # 分析当前批次
            analyze_batch(ae, model, submodule, device, batch_texts, batch_idx, collector)
            
            # 每处理几个批次保存一次中间结果
            if (batch_idx + 1) % 2 == 0:
                print(f"\nSaving intermediate results after batch {batch_idx + 1}...")
                collector.save_expert_feature_analysis(OUTPUT_ROOT)
            
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            continue
    
    # 最终保存
    print(f"\nSaving final expert feature analysis results...")
    collector.save_expert_feature_analysis(OUTPUT_ROOT)
    
    # 最终统计
    print(f"\n✅ Expert Feature Analysis Complete!")
    print(f"📊 Final Statistics:")
    print(f"  Total texts processed: {collector.total_texts_processed}")
    print(f"  Total batches processed: {collector.total_batches_processed}")
    
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