"""
数据加载器 — DUD-E + LIT-PCBA

处理比赛提供的：
- manifest.jsonl (任务索引)
- tasks/<task_id>/task.json (任务元信息)
- tasks/<task_id>/ligands.csv (配体列表)
- tasks/<task_id>/receptors/ (受体结构)
- tasks/<task_id>/refs/ (参考共晶配体)
"""

import os
import csv
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


# ===== SMILES 编码 =====

SMILES_CHARSET = list("CNOSFClBrIcnos()[]=#-+\\/@.:0123456789")
CHAR2IDX = {c: i+1 for i, c in enumerate(SMILES_CHARSET)}  # 0=PAD
IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}
VOCAB_SIZE = len(SMILES_CHARSET) + 1  # +1 for PAD

def encode_smiles(smiles: str, max_len: int = 256) -> Tuple[List[int], List[bool]]:
    """SMILES → token indices + mask"""
    tokens = [CHAR2IDX.get(c, 0) for c in smiles[:max_len]]
    mask = [False] * len(tokens)
    # Padding
    while len(tokens) < max_len:
        tokens.append(0)
        mask.append(True)
    return tokens, mask


# ===== PDB 处理 =====

AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}
AA_LIST = list("ARNDCEQGHILKMFPSTWYV")
AA2IDX = {aa: i for i, aa in enumerate(AA_LIST)}

def parse_pdb_residues(pdb_path: str, max_residues: int = 256) -> Tuple[np.ndarray, List[bool]]:
    """
    解析PDB文件，提取残基特征（氨基酸one-hot + CA坐标）
    Returns:
        features: [max_residues, 23]  (20 AA one-hot + 3 xyz)
        mask: [max_residues] padding mask (True=valid, False=pad)
    """
    residues = []
    current_residue = None
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain_id = line[21]
            res_seq = int(line[22:26].strip())
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            
            res_key = (chain_id, res_seq, res_name)
            if res_key != current_residue:
                if res_name in AA_3TO1:
                    aa = AA_3TO1[res_name]
                    residues.append({'aa': aa, 'coords': [x, y, z]})
                current_residue = res_key
            
            # 更新CA坐标
            if atom_name == 'CA' and residues:
                residues[-1]['coords'] = [x, y, z]
    
    # 截断到max_residues
    residues = residues[:max_residues]
    
    # 构建特征矩阵
    features = np.zeros((max_residues, 23), dtype=np.float32)
    mask = [False] * max_residues
    
    for i, res in enumerate(residues):
        aa_idx = AA2IDX.get(res['aa'], 0)
        features[i, aa_idx] = 1.0  # one-hot
        features[i, 20:23] = res['coords']  # xyz
        mask[i] = True
    
    return features, mask


# ===== Dataset =====

class DrugCLIPDataset(Dataset):
    """比赛数据集"""
    
    def __init__(self, data_dir: str, task_ids: List[str] = None, 
                 max_mol_len: int = 256, max_residues: int = 256,
                 mode: str = 'train'):
        self.data_dir = Path(data_dir)
        self.max_mol_len = max_mol_len
        self.max_residues = max_residues
        self.mode = mode
        
        # 读取manifest
        manifest_path = self.data_dir / 'manifest.jsonl'
        if manifest_path.exists():
            with open(manifest_path) as f:
                self.manifest = [json.loads(line) for line in f]
        else:
            # 扫描tasks目录
            tasks_dir = self.data_dir / 'tasks'
            self.manifest = []
            if tasks_dir.exists():
                for d in sorted(tasks_dir.iterdir()):
                    if d.is_dir() and (d / 'task.json').exists():
                        with open(d / 'task.json') as f:
                            self.manifest.append(json.loads(f.read()))
        
        if task_ids:
            self.manifest = [t for t in self.manifest if t.get('task_id') in task_ids]
        
        # 构建样本列表
        self.samples = []
        for task in self.manifest:
            task_id = task['task_id']
            task_dir = self.data_dir / 'tasks' / task_id
            ligands_path = task_dir / 'ligands.csv'
            
            if not ligands_path.exists():
                continue
            
            # 读取配体
            with open(ligands_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.samples.append({
                        'task_id': task_id,
                        'ligand_id': row['ligand_id'],
                        'smiles': row['smiles'],
                        'task_dir': str(task_dir),
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 编码SMILES
        mol_tokens, mol_mask = encode_smiles(sample['smiles'], self.max_mol_len)
        
        # 获取口袋特征（使用第一个receptor）
        task_dir = Path(sample['task_dir'])
        receptors_dir = task_dir / 'receptors'
        pocket_features = np.zeros((self.max_residues, 23), dtype=np.float32)
        pocket_mask = [False] * self.max_residues
        
        if receptors_dir.exists():
            pdb_files = list(receptors_dir.glob('*.pdb'))
            if pdb_files:
                pocket_features, pocket_mask = parse_pdb_residues(
                    str(pdb_files[0]), self.max_residues
                )
        
        return {
            'task_id': sample['task_id'],
            'ligand_id': sample['ligand_id'],
            'mol_tokens': torch.tensor(mol_tokens, dtype=torch.long),
            'mol_mask': torch.tensor(mol_mask, dtype=torch.bool),
            'pocket_features': torch.tensor(pocket_features, dtype=torch.float32),
            'pocket_mask': torch.tensor(pocket_mask, dtype=torch.bool),
        }


def create_dataloader(data_dir: str, task_ids: List[str] = None,
                      batch_size: int = 64, num_workers: int = 4,
                      mode: str = 'train') -> DataLoader:
    """创建DataLoader"""
    dataset = DrugCLIPDataset(data_dir, task_ids, mode=mode)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(mode == 'train'),
    )
