#!/bin/bash
# DrugClip-Agent — Agent自主优化

# 设置环境
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

# 数据目录
DATA_DIR="${1:-./data}"

# Agent自主优化
echo "=========================================="
echo "🤖 Agent自主优化"
echo "=========================================="
python main.py --mode agent \
    --data_dir "$DATA_DIR" \
    --output_dir output/agent \
    --max_generations 20 \
    --epochs 15 \
    --batch_size 64

echo ""
echo "✅ Agent优化完成！"
echo "📦 提交文件: output/agent/result.zip"
