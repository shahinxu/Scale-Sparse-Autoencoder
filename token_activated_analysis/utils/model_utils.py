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
