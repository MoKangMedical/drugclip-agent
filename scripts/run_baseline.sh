#!/bin/bash
# DrugClip-Agent — 运行脚本

# 设置环境
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

# 数据目录（修改为实际路径）
DATA_DIR="${1:-./data}"

# Step 1: 复现Baseline
echo "=========================================="
echo "📋 Step 1: 复现Baseline"
echo "=========================================="
python main.py --mode baseline \
    --data_dir "$DATA_DIR" \
    --output_dir output/baseline \
    --epochs 30 \
    --batch_size 64

echo ""
echo "✅ Baseline复现完成！"
echo "📦 提交文件: output/baseline/result.zip"
