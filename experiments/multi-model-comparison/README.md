# 多模型对比实验计划

## 🎯 实验目标

基于之前的 `mh-sharegpt` 数据集，对比多个开源大语言模型在心理学对话任务上的表现，包括：
- 训练效率对比
- 推理性能对比  
- 心理学专业能力评估
- Human Judge 主观评价

## 🏗️ 实验架构

```
experiments/multi-model-comparison/
├── README.md                    # 本文件
├── models/                      # 模型配置
│   ├── qwen2.5-7b.md          # Qwen2.5-7B 配置
│   ├── llama3.1-8b.md         # Llama3.1-8B 配置
│   ├── chatglm3-6b.md         # ChatGLM3-6B 配置
│   └── internlm2-7b.md        # InternLM2-7B 配置
├── scripts/                     # 实验脚本
│   ├── train_all_models.sh    # 批量训练脚本
│   ├── evaluate_all_models.sh # 批量评估脚本
│   └── human_judge_test.py    # Human Judge 测试脚本
├── configs/                     # 配置文件
│   ├── train_configs/          # 训练配置
│   ├── eval_configs/           # 评估配置
│   └── human_judge_configs/   # Human Judge 配置
├── results/                     # 实验结果
│   ├── training_logs/          # 训练日志
│   ├── evaluation_results/     # 评估结果
│   └── human_judge_results/   # Human Judge 结果
└── analysis/                    # 结果分析
    ├── performance_comparison.md
    └── human_judge_analysis.md
```

## 🚀 模型选择策略

### 选择标准
1. **模型大小**：7B-13B 参数，适合单 H100 训练
2. **开源许可**：Apache 2.0 或 MIT 等宽松许可
3. **中文能力**：原生支持中文或经过中文优化
4. **社区活跃度**：有良好的维护和更新
5. **推理效率**：适合生产环境部署

### 候选模型列表

| 模型 | 参数量 | 许可证 | 中文支持 | 特点 |
|------|--------|--------|----------|------|
| **Qwen2.5-7B-Instruct** | 7B | Apache 2.0 | ✅ 原生 | 阿里云开源，中文优秀 |
| **Llama3.1-8B-Instruct** | 8B | Meta License | ⚠️ 需微调 | Meta 最新，性能优秀 |
| **ChatGLM3-6B** | 6B | Apache 2.0 | ✅ 原生 | 清华开源，中文对话强 |
| **InternLM2-7B-Chat** | 7B | Apache 2.0 | ✅ 原生 | 上海AI Lab，中文优秀 |
| **Baichuan2-7B-Chat** | 7B | Apache 2.0 | ✅ 原生 | 百川智能，中文对话强 |

## 📊 实验设计

### 阶段 1：模型训练对比
- **数据集**：CodyWhy/mh-sharegpt-20250820
- **训练方式**：LoRA 微调（参数高效）
- **训练参数**：统一配置，确保公平对比
- **监控指标**：Loss 曲线、训练速度、显存使用

### 阶段 2：模型评估对比
- **评估工具**：ModelScope EvalScope
- **评估基准**：心理学相关数据集
- **评估维度**：知识准确性、对话质量、安全性

### 阶段 3：Human Judge 测试
- **测试方式**：AI 与模型对话，人工评分
- **评分维度**：专业性、同理心、实用性、安全性
- **测试场景**：心理咨询、心理教育、危机干预

## 🔧 技术实现

### 训练配置
- **硬件**：NVIDIA H100 80GB
- **框架**：ms-swift + Flash Attention
- **优化**：LoRA + Gradient Checkpointing
- **监控**：SwanLab 实验跟踪

### 评估配置
- **服务部署**：LMDeploy 后端
- **批量评估**：EvalScope + 自定义脚本
- **结果存储**：结构化 JSON + 可视化报告

### Human Judge 系统
- **对话界面**：Web 应用 + API 接口
- **评分系统**：多维度 Likert 量表
- **数据收集**：匿名化 + 统计分析

## 📅 实验时间线

| 阶段 | 预计时间 | 主要任务 |
|------|----------|----------|
| **准备阶段** | 1-2 天 | 环境配置、模型下载、脚本准备 |
| **训练阶段** | 3-5 天 | 批量训练所有模型 |
| **评估阶段** | 2-3 天 | 自动化评估、结果收集 |
| **Human Judge** | 3-5 天 | 测试设计、数据收集、分析 |
| **总结阶段** | 1-2 天 | 结果分析、报告撰写 |

## 🎯 预期成果

1. **多模型性能对比报告**
2. **训练效率分析**
3. **心理学任务能力排名**
4. **Human Judge 主观评价结果**
5. **最佳实践建议**

## 📝 下一步行动

1. 创建详细的模型配置文件
2. 编写自动化训练和评估脚本
3. 设计 Human Judge 测试界面
4. 准备实验环境和依赖
5. 开始第一轮实验

---

*最后更新：2025-08-20*
*实验负责人：AI Assistant*
