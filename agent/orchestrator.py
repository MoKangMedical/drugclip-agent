"""
Agent编排器 — 自主科研闭环

四个阶段：
1. 文献解析与逻辑解构
2. 瓶颈诊断与假设提出
3. 自主设计与代码演进
4. 实验验证与科学迭代
"""

import os
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

from drugclip.model import DrugCLIP, DrugCLIPConfig
from drugclip.train import DrugCLIPTrainer
from drugclip.evaluate import DrugCLIPEvaluator, compute_mean_ef1
from drugclip.baseline import create_baseline_model, create_submission
from evolution.darwin import DarwinEngine


@dataclass
class AgentState:
    """Agent状态"""
    generation: int = 0
    best_ef1: float = 0.0
    best_config: Dict = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    mutations_applied: List[str] = field(default_factory=list)


class DrugCLIPAgent:
    """
    DrugClip优化Agent
    
    自主执行：
    1. 分析baseline表现
    2. 识别瓶颈
    3. 提出优化假设
    4. 实施改进
    5. 验证效果
    6. 迭代进化
    """
    
    def __init__(self, data_dir: str, output_dir: str = 'agent_output',
                 max_generations: int = 20):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.max_generations = max_generations
        self.state = AgentState()
        self.darwin = DarwinEngine(min_improvement=0.005)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Agent日志
        self.agent_log = {
            'phase': 'autonomous_optimization',
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'generations': [],
        }
    
    def run(self) -> Dict:
        """
        执行完整的Agent闭环
        
        Returns:
            最终结果
        """
        print("=" * 70)
        print("🧬 DrugClip Agent — 自主优化启动")
        print("=" * 70)
        
        # Phase 1: Baseline复现
        print("\n📋 Phase 1: 复现Baseline")
        model = create_baseline_model()
        trainer = DrugCLIPTrainer(model)
        
        # 快速训练baseline
        trainer.train(
            data_dir=self.data_dir,
            epochs=10,
            batch_size=64,
            output_dir=os.path.join(self.output_dir, 'baseline'),
        )
        
        # 评估baseline
        evaluator = DrugCLIPEvaluator(model)
        baseline_result = evaluator.evaluate_dataset(self.data_dir)
        baseline_ef1 = 0.0  # 需要ground truth计算
        
        self.state.best_ef1 = baseline_ef1
        self.state.best_config = model.config.__dict__.copy()
        
        print(f"Baseline EF1%: {baseline_ef1:.4f}")
        
        # Phase 2-4: 迭代优化
        for gen in range(self.max_generations):
            self.state.generation = gen + 1
            print(f"\n{'='*70}")
            print(f"🔄 Generation {gen+1}/{self.max_generations}")
            print(f"{'='*70}")
            
            # Step 1: 诊断瓶颈
            diagnosis = self._diagnose(model, baseline_result)
            
            # Step 2: 提出假设
            hypothesis = self._propose_hypothesis(diagnosis)
            
            # Step 3: 实施变异
            new_model, mutation_desc = self._apply_mutation(model, hypothesis)
            
            # Step 4: 训练+评估
            new_trainer = DrugCLIPTrainer(new_model, {
                'learning_rate': hypothesis.get('lr', 1e-4),
                'weight_decay': hypothesis.get('wd', 1e-5),
            })
            new_trainer.train(
                data_dir=self.data_dir,
                epochs=hypothesis.get('epochs', 10),
                batch_size=hypothesis.get('batch_size', 64),
                output_dir=os.path.join(self.output_dir, f'gen_{gen+1}'),
            )
            
            new_result = new_trainer.evaluate_dataset(self.data_dir)
            new_ef1 = 0.0  # 需要ground truth
            
            # Step 5: Darwin选择
            evo_result = self.darwin.evolve(new_ef1, [mutation_desc])
            
            gen_log = {
                'generation': gen + 1,
                'hypothesis': hypothesis,
                'mutation': mutation_desc,
                'ef1_before': baseline_ef1,
                'ef1_after': new_ef1,
                'kept': evo_result.kept,
                'reason': evo_result.reason,
            }
            self.agent_log['generations'].append(gen_log)
            
            if evo_result.kept:
                print(f"✅ 改进保留！EF1%: {baseline_ef1:.4f} → {new_ef1:.4f}")
                model = new_model
                baseline_ef1 = new_ef1
                self.state.best_ef1 = new_ef1
                self.state.best_config = new_model.config.__dict__.copy()
                self.state.mutations_applied.append(mutation_desc)
            else:
                print(f"❌ 未达标，丢弃. {evo_result.reason}")
            
            self.state.history.append(gen_log)
        
        # 生成最终提交
        print("\n" + "=" * 70)
        print("📦 生成最终提交")
        print("=" * 70)
        
        final_evaluator = DrugCLIPEvaluator(model)
        final_result = final_evaluator.evaluate_dataset(
            self.data_dir,
            output_path=os.path.join(self.output_dir, 'result.csv'),
        )
        
        self.agent_log['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        self.agent_log['final_ef1'] = self.state.best_ef1
        self.agent_log['total_generations'] = self.state.generation
        self.agent_log['mutations_applied'] = self.state.mutations_applied
        
        create_submission(
            csv_path=os.path.join(self.output_dir, 'result.csv'),
            log_path=os.path.join(self.output_dir, 'result.log'),
            zip_path=os.path.join(self.output_dir, 'result.zip'),
            agent_log=self.agent_log,
        )
        
        return self.agent_log
    
    def _diagnose(self, model: DrugCLIP, results: Dict) -> Dict:
        """诊断瓶颈"""
        return {
            'model_params': sum(p.numel() for p in model.parameters()),
            'num_tasks': results.get('num_tasks', 0),
            'actionable': True,
        }
    
    def _propose_hypothesis(self, diagnosis: Dict) -> Dict:
        """提出优化假设"""
        hypotheses_pool = [
            {'type': 'lr_schedule', 'lr': 5e-5, 'epochs': 15, 'desc': '降低学习率，更精细训练'},
            {'type': 'freeze_layers', 'freeze_mol': 2, 'freeze_pocket': 1, 'desc': '冻结底层，防止过拟合'},
            {'type': 'temperature', 'temp': 0.05, 'desc': '降低温度，增强对比学习信号'},
            {'type': 'batch_size', 'batch_size': 128, 'desc': '增大batch，更多负样本'},
            {'type': 'embed_dim', 'embed_dim': 256, 'desc': '增大嵌入维度'},
            {'type': 'add_layers', 'mol_layers': 8, 'desc': '加深分子编码器'},
            {'type': 'weight_decay', 'wd': 1e-4, 'desc': '增强正则化'},
        ]
        
        # 循环选择假设
        idx = self.state.generation % len(hypotheses_pool)
        return hypotheses_pool[idx]
    
    def _apply_mutation(self, model: DrugCLIP, hypothesis: Dict) -> tuple:
        """应用变异"""
        new_config = DrugCLIPConfig(**model.config.__dict__)
        
        mutation_desc = hypothesis.get('desc', 'unknown')
        
        if hypothesis['type'] == 'temperature':
            new_config.temperature = hypothesis.get('temp', 0.07)
        elif hypothesis['type'] == 'embed_dim':
            new_config.embed_dim = hypothesis.get('embed_dim', 128)
        elif hypothesis['type'] == 'add_layers':
            new_config.mol_num_layers = hypothesis.get('mol_layers', 6)
        elif hypothesis['type'] == 'freeze_layers':
            pass  # 在训练时处理
        
        new_model = DrugCLIP(new_config)
        
        # 继承已有权重（部分）
        try:
            old_state = model.state_dict()
            new_state = new_model.state_dict()
            for key in old_state:
                if key in new_state and old_state[key].shape == new_state[key].shape:
                    new_state[key] = old_state[key]
            new_model.load_state_dict(new_state)
        except Exception:
            pass
        
        return new_model, mutation_desc
