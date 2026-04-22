"""
超参搜索空间 — Agent可探索的优化维度

比赛要求Agent自动探索：
- 训练/微调策略（冻结层、学习率、batch、负采样等）
- 数据处理/增强（构象采样、去冗余、采样比例等）
- 推理/排序策略（多模型融合、重排序、top-K选择策略）
"""

from typing import Dict, List, Any
import random


# ===== 搜索空间定义 =====

SEARCH_SPACE = {
    # 训练策略
    'training': {
        'learning_rate': [1e-5, 3e-5, 5e-5, 1e-4, 3e-4, 5e-4],
        'weight_decay': [1e-6, 1e-5, 1e-4, 1e-3],
        'batch_size': [32, 64, 128, 256],
        'epochs': [5, 10, 15, 20, 30],
        'warmup_ratio': [0.0, 0.05, 0.1, 0.15],
        'gradient_clip': [0.5, 1.0, 2.0, 5.0],
    },
    
    # 模型结构
    'architecture': {
        'mol_hidden_dim': [128, 256, 384, 512],
        'mol_num_layers': [4, 6, 8, 10],
        'mol_num_heads': [4, 8, 12],
        'pocket_hidden_dim': [128, 256, 384, 512],
        'pocket_num_layers': [2, 4, 6, 8],
        'embed_dim': [64, 128, 256, 512],
        'temperature': [0.03, 0.05, 0.07, 0.1, 0.15],
        'dropout': [0.0, 0.05, 0.1, 0.15, 0.2],
    },
    
    # 冻结策略
    'freezing': {
        'freeze_mol_layers': [0, 1, 2, 3, 4],
        'freeze_pocket_layers': [0, 1, 2, 3],
        'unfreeze_after_epochs': [0, 5, 10],
    },
    
    # 数据增强
    'data_augmentation': {
        'smiles_augmentation': [True, False],  # SMILES随机化
        'pocket_noise_std': [0.0, 0.1, 0.5, 1.0],  # 口袋坐标噪声
        'negative_sampling': ['random', 'hard', 'mixed'],  # 负采样策略
    },
    
    # 推理策略
    'inference': {
        'ensemble_models': [1, 2, 3, 5],  # 模型融合数
        'rerank_top_k': [0, 100, 500, 1000],  # 重排序top-k
        'multi_pocket_agg': ['mean', 'max', 'attention'],  # 多口袋聚合
    },
}


class SearchSpace:
    """搜索空间管理器"""
    
    def __init__(self, space: Dict = None):
        self.space = space or SEARCH_SPACE
    
    def sample(self, category: str = None) -> Dict[str, Any]:
        """
        随机采样一组超参
        
        Args:
            category: 指定类别（training/architecture/freezing等），None=全部
        Returns:
            采样的超参字典
        """
        if category:
            categories = {category: self.space[category]}
        else:
            categories = self.space
        
        result = {}
        for cat_name, cat_space in categories.items():
            result[cat_name] = {}
            for param, values in cat_space.items():
                result[cat_name][param] = random.choice(values)
        
        return result
    
    def mutate(self, config: Dict, category: str = None, 
               num_mutations: int = 1) -> Dict[str, Any]:
        """
        对现有配置进行变异
        
        Args:
            config: 当前配置
            num_mutations: 变异数量
        Returns:
            变异后的配置
        """
        import copy
        new_config = copy.deepcopy(config)
        
        # 选择要变异的类别
        if category:
            cats = [category]
        else:
            cats = list(self.space.keys())
        
        for _ in range(num_mutations):
            cat = random.choice(cats)
            if cat not in new_config:
                new_config[cat] = {}
            
            param = random.choice(list(self.space[cat].keys()))
            new_config[cat][param] = random.choice(self.space[cat][param])
        
        return new_config
    
    def get_dimensions(self) -> int:
        """获取搜索空间总维度"""
        return sum(len(v) for v in self.space.values())
    
    def describe(self) -> str:
        """描述搜索空间"""
        lines = ["搜索空间:"]
        for cat, params in self.space.items():
            lines.append(f"  {cat}:")
            for param, values in params.items():
                lines.append(f"    {param}: {len(values)} options")
        lines.append(f"  总维度: {self.get_dimensions()}")
        return "\n".join(lines)
