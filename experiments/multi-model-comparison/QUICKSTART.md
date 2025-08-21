# 🚀 多模型对比实验 - 快速开始指南

## 📋 实验概述

本实验旨在对比多个开源大语言模型在心理学对话任务上的表现，包括：
- **5个候选模型**：Qwen2.5-7B、Llama3.1-8B、ChatGLM3-6B、InternLM2-7B、Baichuan2-7B
- **统一训练配置**：基于 `mh-sharegpt` 数据集的 LoRA 微调
- **自动化评估**：使用 EvalScope 进行客观评估
- **人工评判**：Web 界面的多维度主观评分

## ⚡ 快速开始

### 1. 环境准备

```bash
# 安装依赖包
pip install 'ms-swift[all]' evalscope 'evalscope[app]' gradio pandas requests

# 检查 GPU 环境
nvidia-smi

# 确保显存充足 (建议 80GB+)
```

### 2. 一键启动完整实验

```bash
# 进入实验目录
cd experiments/multi-model-comparison

# 运行完整实验流程
./run_experiment.sh
```

### 3. 分阶段运行

```bash
# 仅运行训练阶段
./run_experiment.sh --train

# 仅运行评估阶段  
./run_experiment.sh --eval

# 仅运行 Human Judge 测试
./run_experiment.sh --human
```

## 🔧 详细配置

### 模型配置

每个模型的详细配置在 `models/` 目录下：
- `qwen2.5-7b.md` - Qwen2.5-7B 配置
- `llama3.1-8b.md` - Llama3.1-8B 配置
- `chatglm3-6b.md` - ChatGLM3-6B 配置
- `internlm2-7b.md` - InternLM2-7B 配置
- `baichuan2-7b.md` - Baichuan2-7B 配置

### 训练配置

基础训练配置在 `configs/train_configs/base_config.yaml`：
- 数据集：CodyWhy/mh-sharegpt-20250820
- 训练方式：LoRA 微调
- 优化策略：Flash Attention + 混合精度
- 监控工具：SwanLab

### 评估配置

评估使用 ModelScope 的 EvalScope：
- 评估基准：ceval、cmmlu、pceb、psyqa
- 评估方式：服务模式评估
- 结果格式：JSON + 可视化报告

## 📊 实验流程

```
1. 环境检查 → 2. 模型训练 → 3. 模型评估 → 4. Human Judge → 5. 结果分析
```

### 阶段 1：模型训练 (3-5 小时)
- 自动下载基础模型
- LoRA 微调训练
- SwanLab 监控
- 检查点保存

### 阶段 2：模型评估 (2-3 小时)
- 模型导出和部署
- EvalScope 批量评估
- 结果收集和存储

### 阶段 3：Human Judge (3-5 小时)
- Web 界面启动
- 人工评分收集
- 多维度评价
- 结果统计分析

## 🎯 预期结果

### 客观指标
- **训练效率**：Loss 曲线、训练速度、显存使用
- **推理性能**：响应时间、吞吐量
- **评估分数**：各基准测试的准确率

### 主观指标
- **专业性**：心理学知识准确性
- **同理心**：情感理解和支持能力
- **实用性**：实际咨询场景适用性
- **安全性**：避免有害建议

## 🔍 监控和调试

### SwanLab 监控
```bash
# 查看训练进度
# 访问 https://swanlab.ai
# 项目：multi-model-psychology
```

### 日志文件
```bash
# 训练日志
logs/*_training.log

# 评估日志  
logs/*_eval.log

# 部署日志
logs/*_deploy.log
```

### 常见问题

#### 1. 显存不足
```bash
# 调整批次大小
--per_device_train_batch_size 4
--gradient_accumulation_steps 8
```

#### 2. 网络问题
```bash
# 设置代理
export NO_PROXY=127.0.0.1,localhost,::1
export no_proxy=$NO_PROXY
```

#### 3. 依赖缺失
```bash
# 安装缺失的包
pip install -r requirements.txt
```

## 📁 目录结构

```
experiments/multi-model-comparison/
├── README.md                    # 实验计划文档
├── QUICKSTART.md               # 本文件
├── models/                      # 模型配置
├── scripts/                     # 实验脚本
├── configs/                     # 配置文件
├── run_experiment.sh           # 主启动脚本
└── results/                     # 实验结果 (运行时创建)
```

## 🚨 注意事项

1. **硬件要求**：建议使用 H100 或 A100 GPU (80GB+ 显存)
2. **时间预算**：完整实验需要 8-13 小时
3. **网络要求**：需要稳定的网络连接访问 ModelScope
4. **存储空间**：预留至少 100GB 存储空间
5. **许可证**：Llama3.1 需要申请 Meta 许可证

## 📞 技术支持

如遇到问题，请检查：
1. 日志文件中的错误信息
2. 环境依赖是否完整
3. GPU 显存是否充足
4. 网络连接是否正常

## 🎉 开始实验

准备好环境后，运行：

```bash
cd experiments/multi-model-comparison
./run_experiment.sh
```

祝实验顺利！🚀

---

*最后更新：2025-08-20*
*实验版本：v1.0*
