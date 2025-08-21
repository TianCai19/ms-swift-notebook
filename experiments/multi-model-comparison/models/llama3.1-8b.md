# Llama3.1-8B-Instruct 模型配置

## 📋 基本信息

- **模型名称**: Llama3.1-8B-Instruct
- **参数量**: 8B
- **许可证**: Meta License (需要申请)
- **发布方**: Meta AI
- **模型ID**: meta-llama/Meta-Llama-3.1-8B-Instruct
- **特点**: Meta 最新模型，性能优秀，但中文需要额外优化

## 🚀 模型优势

1. **性能优秀**: Meta 最新技术，基础能力强大
2. **架构先进**: 采用最新的 Transformer 架构
3. **多语言**: 支持多种语言，包括中文
4. **安全性**: 经过安全对齐训练
5. **社区支持**: 庞大的开源社区

## ⚠️ 注意事项

1. **许可证**: 需要申请 Meta 许可证才能商用
2. **中文能力**: 原生中文能力有限，需要微调优化
3. **显存需求**: 8B 参数，显存需求略高于 7B 模型

## ⚙️ 训练配置

### 基础配置
```bash
# 模型加载
--model meta-llama/Meta-Llama-3.1-8B-Instruct
--model_type llama
--template llama3

# 训练参数
--train_type lora
--dataset CodyWhy/mh-sharegpt-20250820
--bf16 true
--max_length 3072
--packing true
--gradient_checkpointing true
```

### LoRA 配置
```bash
# LoRA 参数
--lora_rank 8
--lora_alpha 32
--lora_dropout 0.1
--target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

### 训练超参数
```bash
# 批次和优化
--per_device_train_batch_size 6  # 8B 模型，批次稍小
--gradient_accumulation_steps 6  # 增加梯度累积
--learning_rate 2e-4
--num_train_epochs 1
--warmup_ratio 0.1

# 保存和日志
--save_steps 200
--save_total_limit 3
--logging_steps 20
--report_to swanlab
```

### 显存优化
```bash
# Flash Attention
--attn_impl flash_attn

# 梯度累积
--gradient_accumulation_steps 6

# 梯度检查点
--gradient_checkpointing true

# 混合精度
--bf16 true
```

## 📊 预期性能

### 训练阶段
- **显存使用**: ~55-60GB (H100 80GB 足够)
- **训练速度**: ~2.5-3.5 小时/epoch
- **收敛性**: 1-2 epoch 即可收敛

### 推理阶段
- **显存使用**: ~18-22GB
- **推理速度**: 中等，适合批量推理
- **质量**: 基础能力强，中文需要优化

## 🔧 部署配置

### LMDeploy 部署
```bash
swift deploy \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --adapters ./output/llama3.1-8b-sft/checkpoint-xxx \
  --infer_backend lmdeploy \
  --served_model_name llama3.1-8b-sft \
  --tp 1 \
  --cache-max-entry-count 0 \
  --session-len 4096
```

### 模型导出
```bash
swift export \
  --ckpt_dir ./output/llama3.1-8b-sft/checkpoint-xxx \
  --merge_lora true \
  --safe_serialization true \
  --max_shard_size 2GB \
  --output_dir ./export/llama3.1-8b-sft
```

## 📈 评估指标

### 客观指标
- **准确率**: 在心理学知识测试上的表现
- **BLEU/Rouge**: 对话质量评估
- **安全性**: 有害内容检测

### 主观指标
- **专业性**: 心理学知识准确性
- **同理心**: 情感理解和支持能力
- **实用性**: 实际咨询场景的适用性
- **中文流畅度**: 中文表达的自然程度

## 🔍 中文优化建议

1. **数据增强**: 增加中文心理学数据
2. **模板优化**: 使用中文友好的对话模板
3. **后处理**: 对中文输出进行质量检查
4. **人工反馈**: 收集中文使用者的反馈

## 📚 参考资料

- [Llama3.1 官方文档](https://github.com/meta-llama/llama3)
- [Meta AI 模型页面](https://ai.meta.com/llama/)
- [ms-swift 训练指南](https://github.com/modelscope/swift)

---

*配置版本: v1.0*  
*最后更新: 2025-08-20*
