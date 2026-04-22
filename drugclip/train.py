"""
训练/微调 DrugCLIP

支持：
- 从checkpoint恢复
- 冻结层控制
- 学习率调度
- 混合精度训练
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import Dict, Optional
from pathlib import Path

from drugclip.model import DrugCLIP, DrugCLIPConfig
from drugclip.data import create_dataloader


class DrugCLIPTrainer:
    """DrugCLIP训练器"""
    
    def __init__(self, model: DrugCLIP, config: Dict = None):
        self.model = model
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                   'mps' if torch.backends.mps.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        # 学习率调度
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # 混合精度
        self.scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else None
        
        # 训练状态
        self.global_step = 0
        self.best_loss = float('inf')
        self.history = []
    
    def freeze_layers(self, encoder: str = 'both', num_layers: int = 0):
        """冻结编码器层"""
        if encoder in ('mol', 'both'):
            for i, layer in enumerate(self.model.mol_encoder.transformer.layers):
                if i < num_layers:
                    for p in layer.parameters():
                        p.requires_grad = False
        
        if encoder in ('pocket', 'both'):
            for i, layer in enumerate(self.model.pocket_encoder.transformer.layers):
                if i < num_layers:
                    for p in layer.parameters():
                        p.requires_grad = False
    
    def train_epoch(self, dataloader, epoch: int) -> Dict:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            # 移到device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(batch)
                    loss = outputs['loss']
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch)
                loss = outputs['loss']
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            self.scheduler.step(epoch + num_batches / len(dataloader))
            self.global_step += 1
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        return {'loss': avg_loss, 'steps': num_batches}
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict = None):
        """保存checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'config': self.config,
            'history': self.history,
            'metrics': metrics or {},
        }, path)
    
    def load_checkpoint(self, path: str):
        """加载checkpoint"""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.global_step = ckpt['global_step']
        self.best_loss = ckpt['best_loss']
        self.history = ckpt.get('history', [])
        return ckpt.get('epoch', 0)
    
    def train(self, data_dir: str, epochs: int = 50, batch_size: int = 64,
              output_dir: str = 'checkpoints', task_ids: list = None):
        """完整训练流程"""
        os.makedirs(output_dir, exist_ok=True)
        
        dataloader = create_dataloader(data_dir, task_ids, batch_size, mode='train')
        
        log = {
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': self.config,
            'epochs': [],
        }
        
        for epoch in range(epochs):
            metrics = self.train_epoch(dataloader, epoch)
            self.history.append(metrics)
            
            log['epochs'].append({
                'epoch': epoch + 1,
                'loss': metrics['loss'],
                'lr': self.optimizer.param_groups[0]['lr'],
            })
            
            print(f"Epoch {epoch+1}/{epochs} | Loss: {metrics['loss']:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # 保存best
            if metrics['loss'] < self.best_loss:
                self.best_loss = metrics['loss']
                self.save_checkpoint(os.path.join(output_dir, 'best.pt'), epoch, metrics)
            
            # 定期保存
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(os.path.join(output_dir, f'epoch_{epoch+1}.pt'), epoch, metrics)
        
        # 保存训练日志
        log['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(os.path.join(output_dir, 'train_log.json'), 'w') as f:
            json.dump(log, f, indent=2)
        
        return log
