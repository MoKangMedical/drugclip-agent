"""
EF1% 评估 — Enrichment Factor @ top 1%

衡量模型在排名前1%的候选里，相比随机抽样富集到活性分子的倍数
"""

import csv
import json
import numpy as np
import torch
from typing import Dict, List, Tuple
from pathlib import Path
from collections import defaultdict

from drugclip.model import DrugCLIP
from drugclip.data import encode_smiles, parse_pdb_residues


def compute_ef1(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    计算单个靶点的EF1%
    
    EF1% = (hits_top1% / N_top1%) / (hits_total / N_total)
    
    Args:
        y_true: 0/1 标签 (1=active, 0=decoy)
        y_scores: 模型打分 (越高越可能是active)
    Returns:
        ef1: Enrichment Factor @ 1%
    """
    n_total = len(y_true)
    n_hits_total = y_true.sum()
    
    if n_hits_total == 0 or n_total == 0:
        return 0.0
    
    # 按分数降序排列
    sorted_idx = np.argsort(-y_scores)
    n_top1 = max(1, int(n_total * 0.01))
    
    # top 1%中的活性分子数
    hits_top1 = y_true[sorted_idx[:n_top1]].sum()
    
    # EF1%
    ef1 = (hits_top1 / n_top1) / (n_hits_total / n_total)
    return float(ef1)


def compute_mean_ef1(results: Dict[str, List[Tuple[str, float, int]]]) -> Dict:
    """
    计算所有靶点的Mean EF1%
    
    Args:
        results: {task_id: [(ligand_id, score, is_active), ...]}
    Returns:
        {
            'mean_ef1': float,
            'per_task': {task_id: ef1},
            'num_tasks': int,
        }
    """
    per_task = {}
    for task_id, ligands in results.items():
        y_true = np.array([l[2] for l in ligands], dtype=np.float32)
        y_scores = np.array([l[1] for l in ligands], dtype=np.float32)
        per_task[task_id] = compute_ef1(y_true, y_scores)
    
    mean_ef1 = np.mean(list(per_task.values())) if per_task else 0.0
    
    return {
        'mean_ef1': float(mean_ef1),
        'per_task': per_task,
        'num_tasks': len(per_task),
    }


class DrugCLIPEvaluator:
    """DrugCLIP评估器"""
    
    def __init__(self, model: DrugCLIP, device: str = None):
        self.model = model
        self.device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 
                       'mps' if torch.backends.mps.is_available() else 'cpu')
        )
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def score_task(self, task_dir: str) -> List[Tuple[str, float]]:
        """
        对单个靶点的所有配体打分
        
        Args:
            task_dir: 任务目录路径
        Returns:
            [(ligand_id, score), ...]
        """
        task_dir = Path(task_dir)
        
        # 读取配体
        ligands = []
        with open(task_dir / 'ligands.csv') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ligands.append((row['ligand_id'], row['smiles']))
        
        # 获取口袋特征
        receptors_dir = task_dir / 'receptors'
        pdb_files = list(receptors_dir.glob('*.pdb')) if receptors_dir.exists() else []
        
        if not pdb_files:
            # 没有receptor，用随机分数
            return [(lid, float(np.random.randn())) for lid, _ in ligands]
        
        # 编码口袋（使用所有receptor取平均）
        pocket_embeds = []
        for pdb_file in pdb_files:
            features, mask = parse_pdb_residues(str(pdb_file))
            features_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            mask_t = torch.tensor([mask], dtype=torch.bool).to(self.device)
            embed = self.model.encode_pocket(features_t, mask_t)
            pocket_embeds.append(embed)
        
        pocket_embed = torch.stack(pocket_embeds).mean(dim=0)  # [1, embed_dim]
        
        # 逐批编码分子并打分
        batch_size = 256
        scores = []
        
        for i in range(0, len(ligands), batch_size):
            batch_ligands = ligands[i:i+batch_size]
            
            mol_tokens_list = []
            mol_mask_list = []
            for _, smiles in batch_ligands:
                tokens, mask = encode_smiles(smiles)
                mol_tokens_list.append(tokens)
                mol_mask_list.append(mask)
            
            mol_tokens = torch.tensor(mol_tokens_list, dtype=torch.long).to(self.device)
            mol_mask = torch.tensor(mol_mask_list, dtype=torch.bool).to(self.device)
            
            mol_embeds = self.model.encode_molecule(mol_tokens, mol_mask)
            
            # 余弦相似度
            sim = self.model.score(pocket_embed, mol_embeds)  # [1, batch]
            scores.extend(sim.squeeze(0).cpu().tolist())
        
        return [(ligands[i][0], scores[i]) for i in range(len(ligands))]
    
    def evaluate_dataset(self, data_dir: str, output_path: str = None) -> Dict:
        """
        评估整个数据集
        
        Args:
            data_dir: 数据根目录
            output_path: 结果CSV输出路径
        Returns:
            评估结果字典
        """
        data_dir = Path(data_dir)
        
        # 读取manifest
        manifest_path = data_dir / 'manifest.jsonl'
        tasks = []
        if manifest_path.exists():
            with open(manifest_path) as f:
                tasks = [json.loads(line) for line in f]
        else:
            tasks_dir = data_dir / 'tasks'
            if tasks_dir.exists():
                for d in sorted(tasks_dir.iterdir()):
                    if d.is_dir() and (d / 'task.json').exists():
                        with open(d / 'task.json') as f:
                            tasks.append(json.loads(f.read()))
        
        # 逐靶点打分
        all_results = []
        results_by_task = defaultdict(list)
        
        for task in tasks:
            task_id = task['task_id']
            task_dir = data_dir / 'tasks' / task_id
            
            print(f"Scoring {task_id}...")
            scored = self.score_task(str(task_dir))
            
            for ligand_id, score in scored:
                all_results.append({
                    'task_id': task_id,
                    'ligand_id': ligand_id,
                    'score': score,
                })
                results_by_task[task_id].append((ligand_id, score, 0))  # is_active未知
        
        # 输出CSV
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['task_id', 'ligand_id', 'score'])
                writer.writeheader()
                writer.writerows(all_results)
            print(f"Results saved to {output_path}")
        
        return {
            'total_scores': len(all_results),
            'num_tasks': len(tasks),
            'results': dict(results_by_task),
        }


import os
