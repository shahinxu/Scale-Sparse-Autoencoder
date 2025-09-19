import torch as t
from transformers import AutoTokenizer
from config import lm

class FixedOrderBuffer:
    """
    确保token顺序正确的简单buffer
    一次处理一个文本，确保激活和token的完美对应
    """
    
    def __init__(self, model, layer_name, device="cpu", max_length=20):
        self.model = model
        self.layer_name = layer_name
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(lm)
        
        # 获取对应的layer module
        layer_parts = layer_name.split('.')
        self.layer_module = model
        for part in layer_parts:
            if part.isdigit():
                self.layer_module = self.layer_module[int(part)]
            else:
                self.layer_module = getattr(self.layer_module, part)
        
        print(f"FixedOrderBuffer initialized for layer: {layer_name}")
    
    def process_text(self, text):
        """
        处理单个文本，返回所有token的激活
        
        Returns:
            activations: tensor [seq_len, hidden_dim]
            tokens: list of token strings
            token_ids: list of token ids
        """
        print(f"\nProcessing text: '{text[:50]}...'")
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=False
        )
        
        token_ids = inputs['input_ids'][0].tolist()
        tokens = [self.tokenizer.decode([tid]) for tid in token_ids]
        
        print(f"  Tokens ({len(tokens)}): {tokens}")
        
        # 通过模型获取激活
        with t.no_grad():
            with self.model.trace(text, scan=False, validate=False) as tracer:
                hidden_states = self.layer_module.output.save()
            
            # 修复：直接获取激活，不使用.value
            activations = hidden_states
            if isinstance(activations, tuple):
                activations = activations[0]
            
            print(f"  Raw activations shape: {activations.shape}")
            
            # 形状: [1, seq_len, hidden_dim] -> [seq_len, hidden_dim]
            if len(activations.shape) == 3:
                activations = activations[0]  # 去掉batch维度
            
            # 确保长度匹配
            min_len = min(len(tokens), activations.shape[0])
            activations = activations[:min_len].to(self.device)
            tokens = tokens[:min_len]
            token_ids = token_ids[:min_len]
            
            print(f"  Final activations shape: {activations.shape}")
            print(f"  Matched {min_len} tokens with activations")
        
        return activations, tokens, token_ids
