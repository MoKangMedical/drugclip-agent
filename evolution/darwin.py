"""
Darwin进化引擎 — 让Agent自主进化

核心机制：变异→选择→遗传→棘轮（只保留改进）
基于 MoKangMedical/darwin-framework
"""

import json
import random
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvolutionResult:
    """进化结果"""
    generation: int
    fitness: float
    mutations: List[str]
    kept: bool
    reason: str
    details: Dict = field(default_factory=dict)


class DarwinEngine:
    """
    达尔文进化引擎
    
    核心原则：
    1. 变异（Mutation）：对当前最优配置进行随机修改
    2. 选择（Selection）：只保留有改进的变异
    3. 遗传（Inheritance）：新一代继承上一代的最优基因
    4. 棘轮（Ratchet）：fitness只升不降，改进被永久锁定
    """
    
    def __init__(self, min_improvement: float = 0.005):
        """
        Args:
            min_improvement: 最小改进阈值，低于此值视为无效改进
        """
        self.min_improvement = min_improvement
        self.best_fitness = 0.0
        self.generation = 0
        self.history: List[EvolutionResult] = []
        self.mutation_pool: List[Dict] = []
    
    def evolve(self, fitness: float, mutations: List[str], 
               details: Dict = None) -> EvolutionResult:
        """
        执行一轮进化选择
        
        Args:
            fitness: 当前fitness（EF1%）
            mutations: 本轮应用的变异描述
            details: 额外细节
        Returns:
            EvolutionResult
        """
        self.generation += 1
        improvement = fitness - self.best_fitness
        kept = improvement >= self.min_improvement
        
        if kept:
            self.best_fitness = fitness
            reason = f"改进 {improvement*100:.2f}%，保留"
        else:
            reason = f"改进 {improvement*100:.2f}% 未达阈值 {self.min_improvement*100:.1f}%，丢弃"
        
        result = EvolutionResult(
            generation=self.generation,
            fitness=fitness,
            mutations=mutations,
            kept=kept,
            reason=reason,
            details=details or {},
        )
        
        self.history.append(result)
        return result
    
    def get_stats(self) -> Dict:
        """获取进化统计"""
        kept_count = sum(1 for r in self.history if r.kept)
        return {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'total_mutations': sum(len(r.mutations) for r in self.history),
            'kept_mutations': kept_count,
            'success_rate': kept_count / max(self.generation, 1),
        }
    
    def export_history(self, path: str):
        """导出进化历史"""
        data = {
            'stats': self.get_stats(),
            'history': [
                {
                    'generation': r.generation,
                    'fitness': r.fitness,
                    'mutations': r.mutations,
                    'kept': r.kept,
                    'reason': r.reason,
                }
                for r in self.history
            ]
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
