# GLM-4-9B SFT 实验报告

## 1. 实验概述

**实验目标**: 使用 ShareGPT 格式数据对 GLM-4-9B 模型进行监督微调（SFT），提升模型在中文对话和心理学任务上的表现。

**实验时间**: 2025-08-24
**实验环境**: H100 GPU (80GB 显存)
**模型规模**: 9B 参数
**训练方式**: LoRA 微调

## 2. 环境准备

### 2.1 基础环境
```bash
# 激活 conda 环境
conda activate swift_llm

# 检查 GPU
nvidia-smi

# 检查 swift 版本
swift --version
```

### 2.2 环境变量设置
```bash
# SwanLab 配置
export SWANLAB_TOKEN="P2PI8lAMWL1fF90kZAoXj"
export SWANLAB_PROJECT="multi-model-psychology"

# 网络代理配置（避免本地服务被拦截）
export NO_PROXY=127.0.0.1,localhost,::1
export no_proxy=$NO_PROXY

# 验证设置
echo "SWANLAB_TOKEN: $SWANLAB_TOKEN"
echo "SWANLAB_PROJECT: $SWANLAB_PROJECT"
echo "NO_PROXY: $NO_PROXY"
```

## 3. 数据准备

### 3.1 数据集信息
- **数据集**: `CodyWhy/mh-sharegpt-20250820`
- **格式**: ShareGPT JSONL 格式
- **内容**: 中文心理学对话数据
- **规模**: 约 5000 条对话

### 3.2 数据格式示例
```json
{
  "conversations": [
    {
      "from": "human",
      "value": "我最近总是感到焦虑，该怎么办？"
    },
    {
      "from": "gpt",
      "value": "焦虑是一种常见的情绪反应，我理解你的感受。建议你可以尝试深呼吸、运动或者寻求专业心理咨询师的帮助。"
    }
  ]
}
```

## 4. 训练配置

### 4.1 训练命令（优化版）
```bash
CUDA_VISIBLE_DEVICES=6 \
swift sft \
  --model ZhipuAI/glm-4-9b-chat-hf \
  --train_type lora \
  --dataset CodyWhy/mh-sharegpt-20250820 \
  --torch_dtype bfloat16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 6 \
  --per_device_eval_batch_size 3 \
  --router_aux_loss_coef 1e-3 \
  --learning_rate 1e-4 \
  --lora_rank 12 \
  --lora_alpha 48 \
  --target_modules all-linear \
  --gradient_accumulation_steps 6 \
  --eval_steps 100 \
  --save_steps 100 \
  --save_total_limit 5 \
  --logging_steps 20 \
  --max_length 4096 \
  --output_dir ./output/glm4-9b-sft/v1-$(date +%Y%m%d-%H%M%S) \
  --warmup_ratio 0.1 \
  --dataloader_num_workers 8 \
  --report_to swanlab \
  --swanlab_token $SWANLAB_TOKEN \
  --swanlab_project $SWANLAB_PROJECT \
  --swanlab_mode cloud \
  --swanlab_exp_name glm4-9b-sft-v1
```

### 4.2 关键参数说明

#### **模型配置**
- `--model ZhipuAI/glm-4-9b-chat-hf`: 9B 参数的 GLM-4 模型
- `--train_type lora`: 使用 LoRA 进行参数高效微调
- `--torch_dtype bfloat16`: 使用 bfloat16 精度，节省显存

#### **LoRA 配置**
- `--lora_rank 12`: LoRA 秩，比 7B 模型稍大，平衡效果和效率
- `--lora_alpha 48`: LoRA 缩放因子，通常为 rank 的 4 倍
- `--target_modules all-linear`: 目标所有线性层，更全面的微调

#### **训练优化**
- `--num_train_epochs 3`: 增加训练轮数，提升模型效果
- `--per_device_train_batch_size 6`: 单批次大小（H100 显存充足）
- `--gradient_accumulation_steps 6`: 梯度累积步数，总批次大小 = 6 × 6 = 36
- `--max_length 4096`: 增大序列长度，提升长对话处理能力
- `--dataloader_num_workers 8`: 数据加载器工作进程数

#### **监控与保存**
- `--eval_steps 100`: 每 100 步进行一次评估
- `--save_steps 100`: 每 100 步保存一次检查点
- `--save_total_limit 5`: 保存更多检查点，便于选择最佳模型
- `--logging_steps 20`: 每 20 步记录一次日志

## 5. 训练过程

### 5.1 训练监控
```bash
# 查看 SwanLab 实验
# 项目地址: https://swanlab.ai/multi-model-psychology
# 实验名称: glm4-9b-sft-v1

# 查看训练日志
tail -f ./output/glm4-9b-sft/v1-*/trainer_state.json
```

### 5.2 关键指标
- **Loss 趋势**: 训练损失下降情况
- **Learning Rate**: 学习率变化（预热和衰减）
- **GPU 利用率**: 显存使用和计算效率
- **训练速度**: 每秒处理的样本数

### 5.3 预期训练时间
- **总步数**: 约 1500-3000 步（3 epochs）
- **训练时间**: 约 4-6 小时（H100 GPU）
- **显存占用**: 约 60-70GB

## 6. 模型推理测试

### 6.1 使用基座模型 + 适配器进行推理（推荐用于快速测试）

```bash
# 推理命令（不永久合并权重）
CUDA_VISIBLE_DEVICES=2 \
swift infer \
  --model ZhipuAI/glm-4-9b-chat-hf \
  --adapters ./output/glm4-9b-sft/v1-*/checkpoint-* \
  --merge_lora true \
  --infer_backend pt
```

**参数说明**:
- `--model ZhipuAI/glm-4-9b-chat-hf`: 加载原始 9B 基座权重
- `--adapters`: 指定 LoRA 适配器路径
- `--merge_lora true`: 在运行时合并适配器权重到基座
- `--infer_backend pt`: 使用 PyTorch 后端

**优势**:
- ✅ 快速测试，无需等待导出
- ✅ 不永久写入合并权重
- ✅ 适合开发和调试阶段

### 6.2 推理测试示例
```bash
# 启动推理服务
CUDA_VISIBLE_DEVICES=2 \
swift infer \
  --model ZhipuAI/glm-4-9b-chat-hf \
  --adapters ./output/glm4-9b-sft/v1-*/checkpoint-* \
  --merge_lora true \
  --infer_backend pt \
  --max_new_tokens 4096 \
  --served_model_name glm4-9b-sft
```

## 7. 模型部署

### 7.1 使用 Swift Deploy 部署

#### **方式 A: 部署已导出的合并模型**
```bash
swift deploy \
  --model output/glm4-9b-sft/v1-*/checkpoint-*-merged \
  --infer_backend pt \
  --max_new_tokens 4096 \
  --served_model_name glm4-9b-sft
```

```bash
swift deploy \
  --model output/glm4-9b-sft/v1-*/checkpoint-*-merged \
  --infer_backend lmdeploy \
  --max_new_tokens 4096 \
  --served_model_name glm4-9b-sft
```

#### **方式 B: 部署基座 + 适配器**
```bash
swift deploy \
  --model ZhipuAI/glm-4-9b-chat-hf \
  --adapters ./output/glm4-9b-sft/v1-*/checkpoint-* \
  --infer_backend pt \
  --max_new_tokens 4096 \
  --served_model_name glm4-9b-sft
```

### 7.2 服务健康检查
```bash
# 测试模型列表接口
curl --noproxy '*' http://127.0.0.1:8000/v1/models

# 测试推理接口
curl --noproxy '*' http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm4-9b-sft","messages":[{"role":"user","content":"你好"}]}'
```

## 8. 模型评测

### 8.1 使用 EvalScope 评测

#### **安装 EvalScope**
```bash
pip install 'evalscope[app]'
```

#### **心理学专项评测（简化版）**

创建专门的心理学评测脚本 `eval_glm4_psy.sh`：

```bash
#!/bin/bash
# GLM-4-9B 心理学专项评测脚本
# 评测多个心理学相关的测试集

evalscope eval \
  --model glm4-9b-sft \
  --api-url http://127.0.0.1:8000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets mmlu mmlu_pro mmlu_redux super_gpqa \
  --dataset-args '{
    "mmlu": {
      "subset_list": ["professional_psychology", "high_school_psychology"],
      "few_shot_num": 0,
      "few_shot_random": false,
      "metric_list": ["AverageAccuracy"]
    },
    "mmlu_pro": {
      "subset_list": ["psychology"],
      "few_shot_num": 5,
      "few_shot_random": false,
      "metric_list": ["AverageAccuracy"]
    },
    "mmlu_redux": {
      "subset_list": ["professional_psychology", "high_school_psychology"],
      "few_shot_num": 0,
      "few_shot_random": false,
      "metric_list": ["AverageAccuracy"]
    },
    "super_gpqa": {
      "subset_list": ["Psychology"],
      "few_shot_num": 0,
      "few_shot_random": false,
      "metric_list": ["AverageAccuracy"]
    }
  }'
```

#### **数据集说明**

| 数据集 | 子集 | 评测内容 | Few-shot | 备注 |
|--------|------|----------|----------|------|
| **MMLU** | professional_psychology | 专业心理学知识 | 0 | 标准心理学基准 |
| **MMLU** | high_school_psychology | 高中心理学知识 | 0 | 基础心理学概念 |
| **MMLU-Pro** | psychology | 心理学专业测试 | 5 | 高级心理学评估 |
| **MMLU-Redux** | professional_psychology | 专业心理学（精简版） | 0 | 优化后的专业测试 |
| **MMLU-Redux** | high_school_psychology | 高中心理学（精简版） | 0 | 优化后的基础测试 |
| **Super-GPQA** | Psychology | 心理学综合测试 | 0 | 跨领域心理学评估 |

#### **使用方法**

1. **保存脚本**：
```bash
# 将脚本保存到 temp 目录
cp temp/eval_glm4_psy.sh ./eval_glm4_psy.sh
chmod +x ./eval_glm4_psy.sh
```

2. **执行评测**：
```bash
# 确保模型服务已启动
./eval_glm4_psy.sh
```

3. **结果分析**：
```bash
# 查看评测结果
evalscope app --lang zh

# 分析心理学专项表现
# 重点关注专业心理学和高中心理学的准确率差异
```

### 8.2 结果可视化
```bash
# 启动 EvalScope 可视化界面
evalscope app --lang zh

# 默认地址: http://127.0.0.1:7860
# 上传评测结果文件进行可视化分析
```

## 9. 实验结果

### 9.1 训练结果
- **训练轮数**: 3 epochs
- **总步数**: [待补充]
- **最终 Loss**: [待补充]
- **训练时间**: [待补充]

### 9.2 推理测试结果
- **基座模型加载**: [待测试]
- **LoRA 适配器加载**: [待测试]
- **推理速度**: [待测试]
- **显存占用**: [待测试]

### 9.3 心理学专项评测结果
| 数据集 | 子集 | 样本数 | 准确率 | Few-shot | 备注 |
|--------|------|--------|--------|----------|------|
| **MMLU** | professional_psychology | [待评测] | [待评测] | 0 | 专业心理学知识 |
| **MMLU** | high_school_psychology | [待评测] | [待评测] | 0 | 高中心理学概念 |
| **MMLU-Pro** | psychology | [待评测] | [待评测] | 5 | 高级心理学评估 |
| **MMLU-Redux** | professional_psychology | [待评测] | [待评测] | 0 | 专业心理学（精简版） |
| **MMLU-Redux** | high_school_psychology | [待评测] | [待评测] | 0 | 高中心理学（精简版） |
| **Super-GPQA** | Psychology | [待评测] | [待评测] | 0 | 跨领域心理学评估 |

### 9.4 性能对比
- **基座模型 vs 微调后模型**: [待对比]
- **与其他 9B 模型对比**: [待对比]
- **显存效率**: [待分析]

## 10. 问题与解决

### 10.1 训练阶段
- **显存不足**: 已通过调整批次大小和梯度累积解决
- **LoRA 配置**: 使用 `rank=12, alpha=48` 平衡效果和效率
- **序列长度**: 设置为 4096，提升长对话处理能力
- **训练轮数**: 增加到 3 epochs，提升模型效果

### 10.2 推理阶段
- **模型加载**: 使用 `--merge_lora true` 在运行时合并权重
- **显存管理**: 推理时需要足够的显存（建议 60GB+）
- **后端选择**: PyTorch 后端提供最佳兼容性

### 10.3 部署阶段
- **端口配置**: 确保端口 8000 可用
- **服务名称**: 使用描述性的模型名称便于识别
- **健康检查**: 定期测试服务可用性

## 11. 经验总结

### 11.1 9B 模型训练要点
1. **显存管理**: 合理设置批次大小和梯度累积
2. **LoRA 配置**: 使用适中的 rank 值平衡效果和效率
3. **序列长度**: 可以设置较大的最大长度（4096）
4. **训练轮数**: 适当增加训练轮数提升效果
5. **监控指标**: 密切关注显存使用和训练速度

### 11.2 推理优化建议
- **快速测试**: 使用 `--merge_lora true` 进行临时推理
- **生产部署**: 导出合并后的权重进行稳定服务
- **性能监控**: 跟踪推理延迟和显存使用

### 11.3 评测策略
- **服务评测**: 使用 `--eval-type service` 评测已部署模型
- **心理学专项**: 专注于心理学相关数据集的评测
- **结果分析**: 重点关注专业心理学和基础心理学的表现差异

## 12. 后续计划

### 12.1 短期目标
- [ ] 完成模型推理测试，验证 LoRA 加载
- [ ] 部署推理服务，进行在线评测
- [ ] 获得具体性能数据，分析模型效果
- [ ] 执行心理学专项评测，验证模型在心理学任务上的表现

### 12.2 中期目标
- [ ] 尝试不同的 LoRA 配置（rank, alpha）
- [ ] 测试不同的序列长度设置
- [ ] 对比不同训练轮数的效果

### 12.3 长期目标
- [ ] 扩展到其他 9B+ 模型
- [ ] 研究更高效的微调方法
- [ ] 建立完整的模型评测体系

## 13. 变更日志

- **2025-08-24**: 创建实验报告，记录 GLM-4-9B 微调实验
- **待更新**: 训练结果、推理测试、评测数据、性能分析等

---

**注意**: 本实验报告记录了 GLM-4-9B 模型的完整微调流程，包括训练、推理测试、部署和评测。特别关注了增加训练轮数、优化序列长度和简化评测流程，适合快速验证和开发阶段。
