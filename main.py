"""
DrugClip-Agent 入口

AI4S智能体CNS挑战赛 —— 任务1：DrugClip高通量虚拟筛选优化智能体

用法：
    # 1. 先复现baseline
    python main.py --mode baseline --data_dir /path/to/data
    
    # 2. 启动Agent自主优化
    python main.py --mode agent --data_dir /path/to/data --max_generations 20
    
    # 3. 生成提交文件
    python main.py --mode submit --data_dir /path/to/data --checkpoint best.pt
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

# 添加项目根目录到path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drugclip.model import DrugCLIP, DrugCLIPConfig
from drugclip.baseline import run_baseline, create_submission
from drugclip.train import DrugCLIPTrainer
from drugclip.evaluate import DrugCLIPEvaluator
from agent.orchestrator import DrugCLIPAgent
from evolution.search_space import SearchSpace


def main():
    parser = argparse.ArgumentParser(description='DrugClip-Agent — AI4S CNS挑战赛')
    parser.add_argument('--mode', choices=['baseline', 'agent', 'submit', 'eval'],
                        default='agent', help='运行模式')
    parser.add_argument('--data_dir', required=True, help='数据目录')
    parser.add_argument('--output_dir', default='output', help='输出目录')
    parser.add_argument('--checkpoint', default=None, help='模型checkpoint路径')
    parser.add_argument('--max_generations', type=int, default=20, help='最大进化代数')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批大小')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🧬 DrugClip-Agent — AI4S智能体CNS挑战赛")
    print("   任务1：DrugClip高通量虚拟筛选优化智能体")
    print("=" * 70)
    print(f"模式: {args.mode}")
    print(f"数据: {args.data_dir}")
    print(f"输出: {args.output_dir}")
    print()
    
    if args.mode == 'baseline':
        # 模式1: Baseline复现
        print("📋 模式: Baseline复现")
        run_baseline(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    
    elif args.mode == 'agent':
        # 模式2: Agent自主优化
        print("🤖 模式: Agent自主优化")
        agent = DrugCLIPAgent(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            max_generations=args.max_generations,
        )
        result = agent.run()
        print(f"\n✅ Agent优化完成！")
        print(f"   最终EF1%: {result.get('final_ef1', 'N/A')}")
        print(f"   总代数: {result.get('total_generations', 0)}")
        print(f"   提交文件: {args.output_dir}/result.zip")
    
    elif args.mode == 'submit':
        # 模式3: 生成提交
        print("📦 模式: 生成提交文件")
        if not args.checkpoint:
            print("错误: --checkpoint 必须指定")
            return
        
        model = DrugCLIP()
        ckpt = torch.load(args.checkpoint, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        
        evaluator = DrugCLIPEvaluator(model)
        evaluator.evaluate_dataset(
            args.data_dir,
            output_path=os.path.join(args.output_dir, 'result.csv'),
        )
        
        create_submission(
            csv_path=os.path.join(args.output_dir, 'result.csv'),
            log_path=os.path.join(args.output_dir, 'result.log'),
            zip_path=os.path.join(args.output_dir, 'result.zip'),
        )
    
    elif args.mode == 'eval':
        # 模式4: 评估
        print("📊 模式: 评估")
        if args.checkpoint:
            model = DrugCLIP()
            ckpt = torch.load(args.checkpoint, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model = DrugCLIP()
        
        evaluator = DrugCLIPEvaluator(model)
        result = evaluator.evaluate_dataset(args.data_dir)
        print(f"评估完成: {result['num_tasks']} tasks, {result['total_scores']} scores")


if __name__ == '__main__':
    main()
