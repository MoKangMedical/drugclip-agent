"""
Baseline复现 — DrugCLIP原始模型

比赛策略：先复现baseline，建立完整实验闭环
"""

import os
import json
import time
import torch
from pathlib import Path
from typing import Dict

from drugclip.model import DrugCLIP, DrugCLIPConfig
from drugclip.train import DrugCLIPTrainer
from drugclip.evaluate import DrugCLIPEvaluator


def create_baseline_model() -> DrugCLIP:
    """创建baseline模型（原始DrugCLIP配置）"""
    config = DrugCLIPConfig(
        mol_hidden_dim=256,
        mol_num_layers=6,
        mol_num_heads=8,
        pocket_hidden_dim=256,
        pocket_num_layers=4,
        pocket_num_heads=8,
        embed_dim=128,
        temperature=0.07,
        learning_rate=1e-4,
        weight_decay=1e-5,
    )
    return DrugCLIP(config)


def run_baseline(data_dir: str, output_dir: str = 'baseline_output',
                 epochs: int = 30, batch_size: int = 64) -> Dict:
    """
    执行baseline复现
    
    1. 创建模型
    2. 训练
    3. 评估
    4. 生成提交文件
    
    Returns:
        baseline结果字典
    """
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    
    log = {
        'phase': 'baseline_reproduction',
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'steps': [],
    }
    
    # Step 1: 创建模型
    print("=" * 60)
    print("Step 1: 创建DrugCLIP Baseline模型")
    print("=" * 60)
    model = create_baseline_model()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {param_count:,}")
    log['steps'].append({'step': 'create_model', 'params': param_count})
    
    # Step 2: 训练
    print("\n" + "=" * 60)
    print("Step 2: 训练Baseline")
    print("=" * 60)
    trainer = DrugCLIPTrainer(model, {
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
    })
    
    train_log = trainer.train(
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=os.path.join(output_dir, 'checkpoints'),
    )
    log['steps'].append({'step': 'train', 'epochs': epochs, 'final_loss': train_log['epochs'][-1]['loss'] if train_log['epochs'] else None})
    
    # Step 3: 评估
    print("\n" + "=" * 60)
    print("Step 3: 评估Baseline")
    print("=" * 60)
    evaluator = DrugCLIPEvaluator(model)
    eval_result = evaluator.evaluate_dataset(
        data_dir=data_dir,
        output_path=os.path.join(output_dir, 'result.csv'),
    )
    log['steps'].append({'step': 'evaluate', 'num_tasks': eval_result['num_tasks']})
    
    # Step 4: 生成提交文件
    print("\n" + "=" * 60)
    print("Step 4: 生成提交文件")
    print("=" * 60)
    submission_path = os.path.join(output_dir, 'result.zip')
    create_submission(
        csv_path=os.path.join(output_dir, 'result.csv'),
        log_path=os.path.join(output_dir, 'result.log'),
        zip_path=submission_path,
        agent_log=log,
    )
    
    elapsed = time.time() - start_time
    log['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    log['elapsed_seconds'] = elapsed
    
    print(f"\n✅ Baseline复现完成！耗时: {elapsed:.0f}s")
    print(f"📦 提交文件: {submission_path}")
    
    return log


def create_submission(csv_path: str, log_path: str, zip_path: str, agent_log: Dict = None):
    """创建比赛提交的result.zip"""
    import zipfile
    
    # 写入log
    with open(log_path, 'w') as f:
        if agent_log:
            json.dump(agent_log, f, indent=2, ensure_ascii=False)
        else:
            f.write("Baseline reproduction completed.\n")
    
    # 打包zip
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, 'result.csv')
        zf.write(log_path, 'result.log')
    
    print(f"📦 提交文件已创建: {zip_path}")
    print(f"   - result.csv")
    print(f"   - result.log")
