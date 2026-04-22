"""
DrugCLIP Model — 对比学习口袋-分子编码器

基于 Science 论文 "Deep contrastive learning enables genome-wide virtual screening"
将蛋白口袋和小分子分别编码为向量，通过余弦相似度实现超快筛选
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DrugCLIPConfig:
    """模型配置"""
    # 分子编码器
    mol_hidden_dim: int = 256
    mol_num_layers: int = 6
    mol_num_heads: int = 8
    mol_vocab_size: int = 100  # SMILES字符表
    
    # 口袋编码器
    pocket_hidden_dim: int = 256
    pocket_num_layers: int = 4
    pocket_num_heads: int = 8
    pocket_max_residues: int = 256
    
    # 对比学习
    embed_dim: int = 128       # 投影维度
    temperature: float = 0.07  # InfoNCE温度
    dropout: float = 0.1
    
    # 训练
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    max_epochs: int = 50


class MoleculeEncoder(nn.Module):
    """分子SMILES编码器 — Transformer-based"""
    
    def __init__(self, config: DrugCLIPConfig):
        super().__init__()
        self.config = config
        
        # SMILES字符嵌入
        self.token_embed = nn.Embedding(config.mol_vocab_size, config.mol_hidden_dim)
        self.pos_embed = nn.Embedding(512, config.mol_hidden_dim)  # 最大长度512
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.mol_hidden_dim,
            nhead=config.mol_num_heads,
            dim_feedforward=config.mol_hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.mol_num_layers)
        
        # 投影头
        self.projector = nn.Sequential(
            nn.Linear(config.mol_hidden_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.embed_dim)
        )
    
    def forward(self, smiles_tokens: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            smiles_tokens: [batch, seq_len] SMILES token indices
            mask: [batch, seq_len] padding mask
        Returns:
            embeddings: [batch, embed_dim] 分子向量
        """
        batch_size, seq_len = smiles_tokens.shape
        
        # 嵌入
        x = self.token_embed(smiles_tokens)
        pos = self.pos_embed(torch.arange(seq_len, device=x.device))
        x = x + pos.unsqueeze(0)
        
        # Transformer编码
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # 池化（mask-aware mean pooling）
        if mask is not None:
            mask_expanded = (~mask).unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        
        # 投影并归一化
        x = self.projector(x)
        x = F.normalize(x, dim=-1)
        return x


class PocketEncoder(nn.Module):
    """蛋白口袋编码器 — 残基级Transformer"""
    
    def __init__(self, config: DrugCLIPConfig):
        super().__init__()
        self.config = config
        
        # 残基特征（氨基酸类型 + 3D坐标）
        self.residue_embed = nn.Linear(20 + 3, config.pocket_hidden_dim)  # 20种氨基酸 + xyz
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.pocket_hidden_dim,
            nhead=config.pocket_num_heads,
            dim_feedforward=config.pocket_hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.pocket_num_layers)
        
        # 投影头
        self.projector = nn.Sequential(
            nn.Linear(config.pocket_hidden_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.embed_dim)
        )
    
    def forward(self, residue_features: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            residue_features: [batch, num_residues, 23] 氨基酸one-hot + xyz坐标
            mask: [batch, num_residues] padding mask
        Returns:
            embeddings: [batch, embed_dim] 口袋向量
        """
        x = self.residue_embed(residue_features)
        x = self.transformer(x, src_key_padding_mask=mask)
        
        if mask is not None:
            mask_expanded = (~mask).unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        
        x = self.projector(x)
        x = F.normalize(x, dim=-1)
        return x


class DrugCLIP(nn.Module):
    """
    DrugCLIP — 对比学习虚拟筛选模型
    
    将口袋和分子映射到同一向量空间，
    通过余弦相似度实现超快虚拟筛选
    """
    
    def __init__(self, config: DrugCLIPConfig = None):
        super().__init__()
        self.config = config or DrugCLIPConfig()
        self.mol_encoder = MoleculeEncoder(self.config)
        self.pocket_encoder = PocketEncoder(self.config)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1.0 / self.config.temperature)))
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """训练前向"""
        mol_embeds = self.mol_encoder(batch['mol_tokens'], batch.get('mol_mask'))
        pocket_embeds = self.pocket_encoder(batch['pocket_features'], batch.get('pocket_mask'))
        
        # 对比损失（InfoNCE）
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * pocket_embeds @ mol_embeds.T
        labels = torch.arange(logits.shape[0], device=logits.device)
        
        loss_p2m = F.cross_entropy(logits, labels)
        loss_m2p = F.cross_entropy(logits.T, labels)
        loss = (loss_p2m + loss_m2p) / 2
        
        return {
            'loss': loss,
            'logit_scale': logit_scale,
            'mol_embeds': mol_embeds,
            'pocket_embeds': pocket_embeds,
        }
    
    @torch.no_grad()
    def encode_molecule(self, smiles_tokens: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """推理：编码分子"""
        self.eval()
        return self.mol_encoder(smiles_tokens, mask)
    
    @torch.no_grad()
    def encode_pocket(self, residue_features: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """推理：编码口袋"""
        self.eval()
        return self.pocket_encoder(residue_features, mask)
    
    @torch.no_grad()
    def score(self, pocket_embeds: torch.Tensor, mol_embeds: torch.Tensor) -> torch.Tensor:
        """计算口袋-分子相似度分数"""
        return F.cosine_similarity(
            pocket_embeds.unsqueeze(1),  # [N, 1, dim]
            mol_embeds.unsqueeze(0),     # [1, M, dim]
            dim=-1
        )  # [N, M]
