# DrugClip-Agent 🧬

AI4S智能体CNS挑战赛 —— DrugClip高通量虚拟筛选优化智能体

## 赛题

自主优化 DrugCLIP（对比学习虚拟筛选），在 DUD-E + LIT-PCBA 117个靶点上最大化 Mean EF1%。

## 策略：先复现，再优化

1. **Baseline复现**：复现 DrugCLIP 原始模型
2. **瓶颈诊断**：分析baseline在各靶点上的表现，识别改进空间
3. **自主优化**：Agent自动探索训练策略、数据增强、推理优化
4. **迭代进化**：Darwin进化引擎驱动，只保留有效改进

## 架构

```
drugclip-agent/
├── drugclip/          # DrugCLIP模型核心
│   ├── model.py       # 对比学习模型
│   ├── data.py        # 数据加载 (DUD-E / LIT-PCBA)
│   ├── train.py       # 训练/微调
│   ├── evaluate.py    # EF1%评估
│   └── baseline.py    # Baseline复现
├── agent/             # 自主科研Agent
│   ├── orchestrator.py    # 主循环（文献→诊断→编码→实验）
│   ├── diagnosis_agent.py # 瓶颈诊断
│   ├── optimizer_agent.py # 策略优化
│   └── experiment_agent.py# 实验执行
├── evolution/         # 进化引擎
│   ├── darwin.py      # 变异→选择→遗传→棘轮
│   └── search_space.py# 超参搜索空间
├── main.py            # 入口
└── scripts/           # 运行脚本
```

## 提交格式

```
result.zip
├── result.csv    # task_id,ligand_id,score
└── result.log    # Agent自主优化过程日志
```
