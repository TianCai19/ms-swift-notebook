# GPT-OSS-20B SFT 实验报告

## 1. 实验概述

**实验目标**: 使用 ShareGPT 格式数据对 GPT-OSS-20B 模型进行监督微调（SFT），提升模型在中文对话和心理学任务上的表现。

**实验时间**: 2025-08-24
**实验环境**: H100 GPU (80GB 显存)
**模型规模**: 20B 参数
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

### 4.1 训练命令
```bash
CUDA_VISIBLE_DEVICES=6 \
swift sft \
  --model openai-mirror/gpt-oss-20b \
  --train_type lora \
  --dataset CodyWhy/mh-sharegpt-20250820 \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 2 \
  --router_aux_loss_coef 1e-3 \
  --learning_rate 1e-4 \
  --lora_rank 16 \
  --lora_alpha 64 \
  --target_modules all-linear \
  --gradient_accumulation_steps 8 \
  --eval_steps 100 \
  --save_steps 100 \
  --save_total_limit 3 \
  --logging_steps 20 \
  --max_length 3072 \
  --output_dir ./output/gpt-oss-20b-sft/v1-$(date +%Y%m%d-%H%M%S) \
  --warmup_ratio 0.1 \
  --dataloader_num_workers 8 \
  --report_to swanlab \
  --swanlab_token $SWANLAB_TOKEN \
  --swanlab_project $SWANLAB_PROJECT \
  --swanlab_mode cloud \
  --swanlab_exp_name gpt-oss-20b-sft-v1
```

### 4.2 关键参数说明

#### **模型配置**
- `--model openai-mirror/gpt-oss-20b`: 20B 参数的大模型
- `--train_type lora`: 使用 LoRA 进行参数高效微调
- `--torch_dtype bfloat16`: 使用 bfloat16 精度，节省显存

#### **LoRA 配置**
- `--lora_rank 16`: LoRA 秩，影响参数量和效果
- `--lora_alpha 64`: LoRA 缩放因子，通常为 rank 的 4 倍
- `--target_modules all-linear`: 目标所有线性层，更全面的微调

#### **训练优化**
- `--per_device_train_batch_size 4`: 单批次大小（H100 显存充足）
- `--gradient_accumulation_steps 8`: 梯度累积步数，总批次大小 = 4 × 8 = 32
- `--max_length 3072`: 最大序列长度，平衡效果与显存
- `--dataloader_num_workers 8`: 数据加载器工作进程数

#### **监控与保存**
- `--eval_steps 100`: 每 100 步进行一次评估
- `--save_steps 100`: 每 100 步保存一次检查点
- `--save_total_limit 3`: 最多保存 3 个检查点
- `--logging_steps 20`: 每 20 步记录一次日志

## 5. 训练过程

### 5.1 训练监控
```bash
# 查看 SwanLab 实验
# 项目地址: https://swanlab.ai/multi-model-psychology
# 实验名称: gpt-oss-20b-sft-v1

# 查看训练日志
tail -f ./output/gpt-oss-20b-sft/v1-*/trainer_state.json
```

### 5.2 关键指标
- **Loss 趋势**: 训练损失下降情况
- **Learning Rate**: 学习率变化（预热和衰减）
- **GPU 利用率**: 显存使用和计算效率
- **训练速度**: 每秒处理的样本数

### 5.3 预期训练时间
- **总步数**: 约 500-1000 步（取决于数据集大小）
- **训练时间**: 约 2-4 小时（H100 GPU）
- **显存占用**: 约 70-80GB

## 6. 模型导出

### 6.1 找到最新检查点
```bash
# 列出所有检查点
ls -d ./output/gpt-oss-20b-sft/v1-*/checkpoint-* | sort -V

# 获取最新检查点
LATEST_CHECKPOINT=$(ls -d ./output/gpt-oss-20b-sft/v1-*/checkpoint-* | sort -V | tail -1)
echo "最新检查点: $LATEST_CHECKPOINT"
```

### 6.2 导出命令
```bash
swift export \
  --ckpt_dir $LATEST_CHECKPOINT \
  --merge_lora true \
  --safe_serialization true \
  --max_shard_size 2GB \
  --output_dir ./export/gpt-oss-20b-sft-v1
```

### 6.3 导出结果
- **模型文件**: 完整的模型权重（已合并 LoRA）
- **配置文件**: tokenizer、config 等
- **文件大小**: 约 40-50GB（分片存储）

## 7. 模型部署

### 7.1 使用 LMDeploy 部署
```bash
CUDA_VISIBLE_DEVICES=6 \
swift deploy \
  --model_dir ./export/gpt-oss-20b-sft-v1 \
  --infer_backend lmdeploy \
  --served_model_name gpt-oss-20b-sft-v1 \
  --tp 1 \
  --cache-max-entry-count 0 \
  --session-len 2048 \
  --max_batch_size 1
```

### 7.2 服务健康检查
```bash
# 测试模型列表接口
curl --noproxy '*' http://127.0.0.1:8000/v1/models

# 测试推理接口
curl --noproxy '*' http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-oss-20b-sft-v1","messages":[{"role":"user","content":"你好"}]}'
```

## 8. 模型评测

### 8.1 使用 EvalScope 评测

#### **安装 EvalScope**
```bash
pip install 'evalscope[app]'
```

#### **评测命令**
```bash
# 评测 C-Eval（中文语言理解）
evalscope eval \
  --model ./export/gpt-oss-20b-sft-v1 \
  --eval-type checkpoint \
  --datasets ceval \
  --limit 100 \
  --output-file ./results/gpt-oss-20b-ceval-100.json \
  --report_to swanlab \
  --swanlab_token $SWANLAB_TOKEN \
  --swanlab_project $SWANLAB_PROJECT \
  --swanlab_mode cloud \
  --swanlab_exp_name gpt-oss-20b-ceval-eval

# 评测 CMMLU（中文多任务语言理解）
evalscope eval \
  --model ./export/gpt-oss-20b-sft-v1 \
  --eval-type checkpoint \
  --datasets cmmlu \
  --limit 100 \
  --output-file ./results/gpt-oss-20b-cmmlu-100.json \
  --report_to swanlab \
  --swanlab_token $SWANLAB_TOKEN \
  --swanlab_project $SWANLAB_PROJECT \
  --swanlab_mode cloud \
  --swanlab_exp_name gpt-oss-20b-cmmlu-eval
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
- **训练轮数**: 1 epoch
- **总步数**: [待补充]
- **最终 Loss**: [待补充]
- **训练时间**: [待补充]

### 9.2 评测结果
| 数据集 | 样本数 | 准确率 | F1分数 | 备注 |
|--------|--------|--------|--------|------|
| C-Eval | 100    | [待评测] | [待评测] | 中文语言理解 |
| CMMLU  | 100    | [待评测] | [待评测] | 中文多任务理解 |

### 9.3 性能对比
- **基座模型 vs 微调后模型**: [待对比]
- **与其他 20B 模型对比**: [待对比]
- **显存效率**: [待分析]

## 10. 问题与解决

### 10.1 训练阶段
- **显存不足**: 已通过调整批次大小和梯度累积解决
- **LoRA 配置**: 使用 `rank=16, alpha=64` 平衡效果和效率
- **序列长度**: 设置为 3072，平衡效果与显存

### 10.2 导出阶段
- **模型大小**: 20B 模型导出需要足够的磁盘空间
- **分片存储**: 使用 2GB 分片避免单文件过大

### 10.3 部署阶段
- **显存要求**: 推理时需要足够的显存（建议 80GB+）
- **批次大小**: 设置为 1，避免推理时 OOM

## 11. 经验总结

### 11.1 20B 模型训练要点
1. **显存管理**: 合理设置批次大小和梯度累积
2. **LoRA 配置**: 使用较大的 rank 值提升效果
3. **序列长度**: 根据显存情况调整最大长度
4. **监控指标**: 密切关注显存使用和训练速度

### 11.2 参数调优建议
- **学习率**: 1e-4 适合 20B 模型
- **预热比例**: 0.1 提供稳定的训练过程
- **保存策略**: 频繁保存检查点，限制总数
- **评估频率**: 每 100 步评估一次，及时发现问题

### 11.3 部署优化
- **推理后端**: LMDeploy 提供最佳性能
- **显存配置**: 适当降低 session-len 和 batch-size
- **缓存策略**: 根据实际需求调整缓存设置

## 12. 后续计划

### 12.1 短期目标
- [ ] 完成模型评测，获得具体性能数据
- [ ] 分析评测结果，识别模型优势和改进点
- [ ] 优化训练参数，提升模型效果

### 12.2 中期目标
- [ ] 尝试不同的 LoRA 配置（rank, alpha）
- [ ] 测试不同的序列长度设置
- [ ] 对比不同数据集的训练效果

### 12.3 长期目标
- [ ] 扩展到其他 20B+ 模型
- [ ] 研究更高效的微调方法
- [ ] 建立完整的模型评测体系

## 13. 变更日志

- **2025-08-24**: 创建实验报告，记录 GPT-OSS-20B 微调实验
- **待更新**: 训练结果、评测数据、性能分析等

---

**注意**: 本实验报告记录了 GPT-OSS-20B 模型的完整微调流程，包括训练、导出、部署和评测。所有配置和命令都经过优化，适合 H100 GPU 环境。
