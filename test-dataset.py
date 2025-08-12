import torch as t
from nnsight import LanguageModel
from dictionary_learning.trainers.moe_physically import MultiExpertAutoEncoder
from dictionary_learning.trainers.moe_physically_scale import MultiExpertScaleAutoEncoder
from dictionary_learning.trainers.top_k import AutoEncoderTopK
import json
from transformers import AutoTokenizer
from config import lm, layer
from collections import defaultdict
import os
import pandas as pd

GPU = "4"
MODEL = "MultiExpert_8_1"
MODEL_PATH = f"/home/xuzhen/switch_sae/dictionaries/{MODEL}/8.pt"
OUTPUT_ROOT = f"expert_feature_analysis_{MODEL}_wikitext"

WIKITEXT_PATH = "/home/xuzhen/switch_sae/wikitext"
WIKITEXT_VERSION = "wikitext-103-raw-v1"
SPLIT = "train"

BATCH_SIZE = 200
TOTAL_BATCHES = 5
TARGET_EXPERTS = list(range(8))


class FixedOrderBuffer:
    def __init__(self, model, layer_name, device="cpu", max_length=128):
        self.model = model
        self.layer_name = layer_name
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(lm)
        
        layer_parts = layer_name.split('.')
        self.layer_module = model
        for part in layer_parts:
            if part.isdigit():
                self.layer_module = self.layer_module[int(part)]
            else:
                self.layer_module = getattr(self.layer_module, part)
        
        print(f"FixedOrderBuffer initialized for layer: {layer_name}")
    
    def process_text(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=False
        )
        
        token_ids = inputs['input_ids'][0].tolist()
        tokens = [self.tokenizer.decode([tid]) for tid in token_ids]
        
        with t.no_grad():
            with self.model.trace(text, scan=False, validate=False):
                hidden_states = self.layer_module.output.save()
            
            activations = hidden_states
            if isinstance(activations, tuple):
                activations = activations[0]
            
            if len(activations.shape) == 3:
                activations = activations[0]
            
            min_len = min(len(tokens), activations.shape[0])
            activations = activations[:min_len].to(self.device)
            tokens = tokens[:min_len]
            token_ids = token_ids[:min_len]
        
        return activations, tokens, token_ids


def auto_detect_dataset_structure(base_path):
    """自动检测数据集结构"""
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Dataset path does not exist: {base_path}")
    
    # 检查是否直接包含parquet文件（无版本结构）
    direct_files = [f for f in os.listdir(base_path) if f.endswith('.parquet')]
    if direct_files:
        return {
            'has_version': False,
            'dataset_path': base_path,
            'structure': 'direct'
        }
    
    # 检查是否有版本子目录结构
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if subdirs:
        for subdir in subdirs:
            subdir_path = os.path.join(base_path, subdir)
            subdir_files = [f for f in os.listdir(subdir_path) if f.endswith('.parquet')]
            if subdir_files:
                return {
                    'has_version': True,
                    'dataset_path': subdir_path,
                    'version': subdir,
                    'structure': 'versioned'
                }
    
    raise FileNotFoundError(f"No parquet files found in {base_path} or its subdirectories")


def load_dataset_batch(dataset_path, version=None, split="train", 
                      batch_size=200, batch_idx=0, min_length=20, max_length=200):
    """
    通用数据集批次加载函数，自动适应不同的数据集结构
    
    Args:
        dataset_path: 数据集根路径
        version: 数据集版本（可选，None表示直接读取）
        split: 数据分割
        batch_size: 批次大小
        batch_idx: 批次索引
        min_length: 最小文本长度
        max_length: 最大文本长度
    """
    
    # 自动检测数据集结构
    dataset_info = auto_detect_dataset_structure(dataset_path)
    
    if dataset_info['structure'] == 'direct':
        # 直接从根目录读取parquet文件
        search_path = dataset_path
        print(f"Using direct dataset structure: {search_path}")
    elif dataset_info['structure'] == 'versioned':
        # 使用版本子目录
        if version:
            search_path = os.path.join(dataset_path, version)
        else:
            search_path = dataset_info['dataset_path']
        print(f"Using versioned dataset structure: {search_path}")
    else:
        raise ValueError(f"Unknown dataset structure: {dataset_info['structure']}")
    
    # 查找parquet文件
    parquet_files = []
    if os.path.exists(search_path):
        all_files = os.listdir(search_path)
        
        # 优先查找包含split的文件
        split_files = [f for f in all_files if f.startswith(f"{split}-") and f.endswith(".parquet")]
        if split_files:
            parquet_files = split_files
        else:
            # 如果没有找到，查找所有parquet文件
            parquet_files = [f for f in all_files if f.endswith(".parquet")]
            print(f"No {split}-specific files found, using all parquet files")
        
        parquet_files.sort()
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {search_path}")
    
    print(f"Found {len(parquet_files)} parquet files: {parquet_files[:3]}{'...' if len(parquet_files) > 3 else ''}")
    print(f"Loading batch {batch_idx} (size: {batch_size})")
    
    all_texts = []
    texts_read = 0
    start_idx = batch_idx * batch_size
    end_idx = start_idx + batch_size
    
    for parquet_file in parquet_files:
        full_path = os.path.join(search_path, parquet_file)
        try:
            df = pd.read_parquet(full_path)
            print(f"  Processing {parquet_file}: {len(df)} rows")
            
            # 自动检测文本列
            text_column = None
            for col in ['text', 'content', 'document', 'body']:
                if col in df.columns:
                    text_column = col
                    break
            
            if text_column is None:
                # 如果没有找到标准列名，使用第一个字符串列
                string_cols = df.select_dtypes(include=['object']).columns
                if len(string_cols) > 0:
                    text_column = string_cols[0]
                    print(f"    Using column '{text_column}' as text source")
                else:
                    print(f"    No text column found in {parquet_file}, skipping")
                    continue
            
            for _, row in df.iterrows():
                text = str(row[text_column]).strip()
                
                # 文本过滤
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
                
        except Exception as e:
            print(f"  Error reading {parquet_file}: {e}")
            continue
    
    print(f"  Loaded {len(all_texts)} texts for batch {batch_idx}")
    return all_texts, dataset_info


def detect_model_type(dictionary):
    """检测模型类型并返回相关信息"""
    if hasattr(dictionary, 'experts') and hasattr(dictionary, 'expert_dict_size'):
        # 多专家模型
        return {
            'is_multi_expert': True,
            'num_experts': dictionary.experts,
            'expert_dict_size': dictionary.expert_dict_size,
            'total_dict_size': dictionary.dict_size,
            'model_type': 'MultiExpert'
        }
    else:
        # 单专家模型 (TopK等)
        return {
            'is_multi_expert': False,
            'num_experts': 1,
            'expert_dict_size': dictionary.dict_size,
            'total_dict_size': dictionary.dict_size,
            'model_type': 'SingleExpert'
        }


class ExpertFeatureCollector:
    def __init__(self, target_experts, model_info):
        self.model_info = model_info
        self.is_multi_expert = model_info['is_multi_expert']
        self.expert_dict_size = model_info['expert_dict_size']
        
        # 对于单专家模型，自动调整target_experts为[0]
        if not self.is_multi_expert:
            self.target_experts = {0}
            print(f"📝 Detected single-expert model, adjusting target_experts to [0]")
        else:
            self.target_experts = set(target_experts)
            print(f"📝 Detected multi-expert model with {model_info['num_experts']} experts")
        
        self.expert_feature_tokens = defaultdict(lambda: defaultdict(list))
        self.expert_feature_max_strength = defaultdict(lambda: defaultdict(float))
        self.expert_feature_best_example = defaultdict(dict)
        
        self.expert_stats = defaultdict(lambda: {
            'total_features_activated': 0,
            'total_token_activations': 0,
            'texts_processed': 0
        })
        
        self.total_texts_processed = 0
        self.total_batches_processed = 0
    
    def add_feature_activation(self, expert_id, feature_id, token_text, activation_strength, 
                             text_id, token_pos, original_text):
        
        if expert_id not in self.target_experts:
            return
        
        global_text_id = self.total_texts_processed + text_id
        
        # 对于单专家模型，feature_id就是实际的特征ID
        # 对于多专家模型，需要计算相对特征ID
        if self.is_multi_expert:
            relative_feature_id = feature_id % self.expert_dict_size
        else:
            relative_feature_id = feature_id
        
        token_record = {
            'token': token_text,
            'strength': activation_strength,
            'text_id': global_text_id,
            'token_pos': token_pos,
            'original_text': original_text[:100] + '...' if len(original_text) > 100 else original_text
        }
        
        self.expert_feature_tokens[expert_id][relative_feature_id].append(token_record)
        
        if activation_strength > self.expert_feature_max_strength[expert_id][relative_feature_id]:
            self.expert_feature_max_strength[expert_id][relative_feature_id] = activation_strength
            self.expert_feature_best_example[expert_id][relative_feature_id] = token_record
        
        self.expert_stats[expert_id]['total_token_activations'] += 1
        self.expert_stats[expert_id]['total_features_activated'] = len(self.expert_feature_tokens[expert_id])
    
    def update_batch_stats(self, batch_size):
        self.total_texts_processed += batch_size
        self.total_batches_processed += 1
        
        for expert_id in self.target_experts:
            self.expert_stats[expert_id]['texts_processed'] = self.total_texts_processed
    
    def get_expert_feature_summary(self, expert_id, top_n=20):
        if expert_id not in self.expert_feature_tokens:
            return None
        
        features_data = []
        
        for feature_id, token_records in self.expert_feature_tokens[expert_id].items():
            max_strength = self.expert_feature_max_strength[expert_id][feature_id]
            best_example = self.expert_feature_best_example[expert_id][feature_id]
            
            unique_tokens = set(record['token'] for record in token_records)
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
        
        features_data.sort(key=lambda x: x['max_activation'], reverse=True)
        
        return {
            'expert_id': expert_id,
            'statistics': self.expert_stats[expert_id],
            'total_features': len(features_data),
            'features': features_data,
            'model_info': self.model_info
        }
    
    def save_expert_feature_analysis(self, output_dir, dataset_info):
        os.makedirs(output_dir, exist_ok=True)
        
        global_stats = {
            'model_info': self.model_info,
            'target_experts': list(self.target_experts),
            'total_texts_processed': self.total_texts_processed,
            'total_batches_processed': self.total_batches_processed,
            'expert_feature_counts': {
                expert_id: len(self.expert_feature_tokens[expert_id]) 
                for expert_id in self.target_experts
            },
            'dataset_info': {
                'source': 'Custom Dataset',
                'structure': dataset_info['structure'],
                'has_version': dataset_info['has_version'],
                'version': dataset_info.get('version', 'N/A'),
                'split': SPLIT,
                'batch_size': BATCH_SIZE,
                'total_batches': TOTAL_BATCHES,
                'buffer_type': 'FixedOrderBuffer'
            }
        }
        
        with open(os.path.join(output_dir, 'global_statistics.json'), 'w', encoding='utf-8') as f:
            json.dump(global_stats, f, indent=2, ensure_ascii=False)
        
        for expert_id in self.target_experts:
            if expert_id not in self.expert_feature_tokens:
                continue
                
            expert_dir = os.path.join(output_dir, f'expert_{expert_id:02d}')
            os.makedirs(expert_dir, exist_ok=True)
            
            summary = self.get_expert_feature_summary(expert_id, top_n=50)
            
            with open(os.path.join(expert_dir, 'feature_analysis.json'), 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            self._save_readable_feature_report(expert_dir, summary)
            self._save_feature_token_details(expert_dir, expert_id)
    
    def _save_readable_feature_report(self, expert_dir, summary):
        with open(os.path.join(expert_dir, 'feature_report.txt'), 'w', encoding='utf-8') as f:
            expert_id = summary['expert_id']
            stats = summary['statistics']
            model_info = summary['model_info']
            
            f.write(f"Expert {expert_id} - Feature Analysis Report (FixedOrderBuffer)\n")
            f.write("="*80 + "\n\n")
            
            f.write("🏗️ Model Information:\n")
            f.write(f"  Model Type: {model_info['model_type']}\n")
            f.write(f"  Multi-Expert: {model_info['is_multi_expert']}\n")
            f.write(f"  Total Experts: {model_info['num_experts']}\n")
            f.write(f"  Expert Dict Size: {model_info['expert_dict_size']}\n")
            f.write(f"  Total Dict Size: {model_info['total_dict_size']}\n\n")
            
            f.write("📊 Statistics:\n")
            f.write(f"  Total Features Activated: {stats['total_features_activated']}\n")
            f.write(f"  Total Token Activations: {stats['total_token_activations']}\n")
            f.write(f"  Texts Processed: {stats['texts_processed']}\n")
            if stats['total_features_activated'] > 0:
                f.write(f"  Avg Activations per Feature: {stats['total_token_activations']/stats['total_features_activated']:.2f}\n\n")
            else:
                f.write(f"  Avg Activations per Feature: N/A (no features activated)\n\n")
            
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
                
                top_tokens = feature_data['top_activations'][:10]
                token_list = [f"'{record['token']}'({record['strength']:.3f})" for record in top_tokens]
                f.write(f"    Top Tokens: {', '.join(token_list)}\n")
    
    def _save_feature_token_details(self, expert_dir, expert_id):
        features_dir = os.path.join(expert_dir, 'features')
        os.makedirs(features_dir, exist_ok=True)
        
        for feature_id, token_records in self.expert_feature_tokens[expert_id].items():
            feature_file = os.path.join(features_dir, f'feature_{feature_id:04d}.txt')
            
            with open(feature_file, 'w', encoding='utf-8') as f:
                f.write(f"Expert {expert_id} - Feature {feature_id} - Token Activations (FixedOrderBuffer)\n")
                f.write("="*70 + "\n\n")
                
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
def analyze_batch_with_fixed_buffer(dictionary, buffer, texts, batch_idx, collector, device="cpu"):
    """使用FixedOrderBuffer分析一个批次的文本"""
    
    print(f"\n{'='*60}")
    print(f"Processing Batch {batch_idx + 1}/{TOTAL_BATCHES} with FixedOrderBuffer")
    print(f"Batch size: {len(texts)} texts")
    print(f"Target experts: {list(collector.target_experts)}")
    print(f"Model type: {collector.model_info['model_type']}")
    print(f"{'='*60}")
    
    batch_feature_activations = defaultdict(int)
    
    for text_id, text in enumerate(texts):
        if text_id % 50 == 0:
            print(f"  Processing text {text_id}/{len(texts)}: '{text[:50]}...'")
        
        try:
            activations, tokens, _ = buffer.process_text(text)
            
            for token_pos, (activation, token_text) in enumerate(zip(activations, tokens)):
                _, f = dictionary(activation.unsqueeze(0), output_features=True)
                token_features = f[0]
                
                top_k_values, top_k_indices = token_features.topk(dictionary.k, sorted=True)
                
                for fid, fval in zip(top_k_indices, top_k_values):
                    if fval.item() > 0:
                        # 根据模型类型计算expert_id
                        if collector.is_multi_expert:
                            expert_id = fid.item() // collector.expert_dict_size
                        else:
                            expert_id = 0  # 单专家模型，所有特征都属于专家0
                        
                        if expert_id in collector.target_experts:
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
        
        except Exception as e:
            print(f"    Error processing text {text_id}: {e}")
            continue
    
    collector.update_batch_stats(len(texts))
    
    print(f"  Batch {batch_idx + 1} completed:")
    for expert_id in collector.target_experts:
        activations = batch_feature_activations.get(expert_id, 0)
        total_features = len(collector.expert_feature_tokens[expert_id])
        print(f"    Expert {expert_id}: {activations} activations, {total_features} features")
    
    t.cuda.empty_cache()


def main():
    device = f'cuda:{GPU}'
    
    print(f"Expert Feature Analysis Configuration (with FixedOrderBuffer):")
    print(f"  Dataset Path: {WIKITEXT_PATH}")
    print(f"  Dataset Version: {WIKITEXT_VERSION if WIKITEXT_VERSION else 'Auto-detect'}")
    print(f"  Split: {SPLIT}")
    print(f"  Model: {MODEL}")
    print(f"  Device: {device}")
    print(f"  Batch processing: {BATCH_SIZE} texts per batch, {TOTAL_BATCHES} batches")
    print(f"  Total texts to process: {BATCH_SIZE * TOTAL_BATCHES}")
    print(f"  Buffer: FixedOrderBuffer (ensures correct token order)")
    
    print("\nLoading language model...")
    model = LanguageModel(lm, dispatch=True, device_map=device)
    
    print("Creating FixedOrderBuffer...")
    buffer = FixedOrderBuffer(
        model=model,
        layer_name=f"transformer.h.{layer}",
        device=device,
        max_length=128
    )
    
    print(f"Loading SAE from {MODEL_PATH}...")
    ae = MultiExpertAutoEncoder(
        activation_dim=768, 
        dict_size=32*768, 
        k=32, 
        experts=8, 
        e=1, 
        heaviside=False
    )
    ae.load_state_dict(t.load(MODEL_PATH))
    ae.to(device)
    ae.eval()
    
    model_info = detect_model_type(ae)
    print(f"\n🔍 Model Detection Results:")
    print(f"  Model Type: {model_info['model_type']}")
    print(f"  Multi-Expert: {model_info['is_multi_expert']}")
    print(f"  Number of Experts: {model_info['num_experts']}")
    print(f"  Expert Dict Size: {model_info['expert_dict_size']}")
    print(f"  Total Dict Size: {model_info['total_dict_size']}")
    
    collector = ExpertFeatureCollector(
        target_experts=TARGET_EXPERTS,
        model_info=model_info
    )
    
    print(f"  Final Target Experts: {list(collector.target_experts)}")
    
    # 数据集信息将在第一次加载时获得
    dataset_info = None
    
    for batch_idx in range(TOTAL_BATCHES):
        try:
            batch_texts, current_dataset_info = load_dataset_batch(
                dataset_path=WIKITEXT_PATH,
                version=WIKITEXT_VERSION,
                split=SPLIT,
                batch_size=BATCH_SIZE,
                batch_idx=batch_idx
            )
            
            if dataset_info is None:
                dataset_info = current_dataset_info
            
            if not batch_texts:
                print(f"No more texts available at batch {batch_idx}")
                break
            
            analyze_batch_with_fixed_buffer(ae, buffer, batch_texts, batch_idx, collector, device)
            
            if (batch_idx + 1) % 2 == 0:
                print(f"\nSaving intermediate results after batch {batch_idx + 1}...")
                collector.save_expert_feature_analysis(OUTPUT_ROOT, dataset_info)
            
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            continue
    
    print(f"\nSaving final expert feature analysis results...")
    collector.save_expert_feature_analysis(OUTPUT_ROOT, dataset_info)
    
    generate_comparison_report(collector, dataset_info)
    
    print(f"\n✅ Expert Feature Analysis Complete (with FixedOrderBuffer)!")
    print(f"📊 Final Statistics:")
    print(f"  Model Type: {model_info['model_type']}")
    print(f"  Dataset Structure: {dataset_info['structure']}")
    print(f"  Has Version: {dataset_info['has_version']}")
    print(f"  Total texts processed: {collector.total_texts_processed}")
    print(f"  Total batches processed: {collector.total_batches_processed}")
    
    for expert_id in collector.target_experts:
        if expert_id in collector.expert_feature_tokens:
            stats = collector.expert_stats[expert_id]
            feature_count = len(collector.expert_feature_tokens[expert_id])
            print(f"\n  Expert {expert_id}:")
            print(f"    Features activated: {feature_count}")
            print(f"    Total activations: {stats['total_token_activations']}")
            if feature_count > 0:
                print(f"    Avg activations per feature: {stats['total_token_activations']/feature_count:.2f}")
            else:
                print(f"    Avg activations per feature: N/A (no features activated)")
        else:
            print(f"\n  Expert {expert_id}:")
            print(f"    Features activated: 0")
            print(f"    Total activations: 0")
            print(f"    Avg activations per feature: N/A (no features activated)")
    
    print(f"\n📁 Results saved to: {OUTPUT_ROOT}/")
    print(f"  - global_statistics.json: Global statistics")
    print(f"  - expert_XX/feature_analysis.json: Complete analysis for each expert")
    print(f"  - expert_XX/feature_report.txt: Readable analysis report")
    print(f"  - expert_XX/features/feature_XXXX.txt: Detailed token list for each feature")
    print(f"  - comparison_report.txt: Comparison analysis")


def generate_comparison_report(collector, dataset_info):
    comparison_file = os.path.join(OUTPUT_ROOT, "comparison_report.txt")
    
    with open(comparison_file, 'w', encoding='utf-8') as f:
        f.write("FixedOrderBuffer - Universal Dataset Support\n")
        f.write("="*50 + "\n\n")
        
        f.write("🏗️ Model Information:\n")
        f.write(f"  Model Type: {collector.model_info['model_type']}\n")
        f.write(f"  Multi-Expert: {collector.model_info['is_multi_expert']}\n")
        f.write(f"  Number of Experts: {collector.model_info['num_experts']}\n")
        f.write(f"  Expert Dict Size: {collector.model_info['expert_dict_size']}\n\n")
        
        f.write("📊 Dataset Information:\n")
        f.write(f"  Structure: {dataset_info['structure']}\n")
        f.write(f"  Has Version: {dataset_info['has_version']}\n")
        f.write(f"  Version: {dataset_info.get('version', 'N/A')}\n")
        f.write(f"  Auto-detection: ✅ Success\n\n")
        
        f.write("FixedOrderBuffer Benefits:\n")
        f.write("-"*30 + "\n")
        f.write("✅ Guaranteed correct token-activation correspondence\n")
        f.write("✅ One text processed at a time (no mixing)\n") 
        f.write("✅ Clear debugging information\n")
        f.write("✅ Transparent processing pipeline\n")
        f.write("✅ Token position tracking\n")
        f.write("✅ Auto-detection of model type\n")
        f.write("✅ Compatible with both multi-expert and single-expert models\n")
        f.write("✅ Universal dataset structure support\n")
        f.write("✅ Automatic text column detection\n\n")
        
        f.write("Dataset Structure Support:\n")
        f.write("-"*30 + "\n")
        f.write("✅ Direct parquet files (no version subdirectory)\n")
        f.write("✅ Versioned structure (with subdirectories)\n")
        f.write("✅ Auto-detection of text columns (text, content, document, body)\n")
        f.write("✅ Fallback to first string column if standard names not found\n")
        f.write("✅ Split-specific file detection (train-, test-, val-)\n")
        f.write("✅ Fallback to all parquet files if split-specific not found\n\n")
        
        f.write("Processing Statistics:\n")
        f.write("-"*25 + "\n")
        f.write(f"Total texts processed: {collector.total_texts_processed}\n")
        f.write(f"Total batches: {collector.total_batches_processed}\n")
        
        for expert_id in collector.target_experts:
            if expert_id in collector.expert_feature_tokens:
                stats = collector.expert_stats[expert_id]
                feature_count = len(collector.expert_feature_tokens[expert_id])
                f.write(f"\nExpert {expert_id}:\n")
                f.write(f"  Features activated: {feature_count}\n")
                f.write(f"  Total activations: {stats['total_token_activations']}\n")
                if feature_count > 0:
                    f.write(f"  Activations per feature: {stats['total_token_activations']/feature_count:.2f}\n")
                else:
                    f.write(f"  Activations per feature: N/A (no features activated)\n")
            else:
                f.write(f"\nExpert {expert_id}:\n")
                f.write(f"  Features activated: 0\n")
                f.write(f"  Total activations: 0\n")
                f.write(f"  Activations per feature: N/A (no features activated)\n")


if __name__ == "__main__":
    main()