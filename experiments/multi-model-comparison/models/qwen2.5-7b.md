# Qwen2.5-7B-Instruct 模型配置

## 📋 基本信息

- **模型名称**: Qwen2.5-7B-Instruct
- **参数量**: 7B
- **许可证**: Apache 2.0
- **发布方**: Alibaba Cloud
- **模型ID**: Qwen/Qwen2.5-7B-Instruct
- **特点**: 原生中文支持，对话能力强，心理学任务表现优秀

## 🚀 模型优势

1. **中文能力**: 原生支持中文，无需额外微调
2. **对话质量**: 在 ShareGPT 格式数据上表现优秀
3. **安全性**: 内置安全对齐，适合心理咨询场景
4. **效率**: 7B 参数，单 H100 训练效率高
5. **社区**: 阿里云维护，更新频繁

## ⚙️ 训练配置

### 基础配置
```bash
# 模型加载
--model Qwen/Qwen2.5-7B-Instruct
--model_type qwen2_5
--template qwen2_5

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
# LoRA 参数（ms-swift 默认）
--lora_rank 8
--lora_alpha 32
--lora_dropout 0.1
--target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

### 训练超参数
```bash
# 批次和优化
--per_device_train_batch_size 8
--gradient_accumulation_steps 4
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
--gradient_accumulation_steps 4

# 梯度检查点
--gradient_checkpointing true

# 混合精度
--bf16 true
```

## 📊 预期性能

### 训练阶段
- **显存使用**: ~45-50GB (H100 80GB 足够)
- **训练速度**: ~2-3 小时/epoch
- **收敛性**: 1-2 epoch 即可收敛

### 推理阶段
- **显存使用**: ~15-20GB
- **推理速度**: 快，适合实时对话
- **质量**: 在心理学任务上表现优秀

## 🔧 部署配置

### LMDeploy 部署
```bash
swift deploy \
  --model Qwen/Qwen2.5-7B-Instruct \
  --adapters ./output/qwen2.5-7b-sft/checkpoint-xxx \
  --infer_backend lmdeploy \
  --served_model_name qwen2.5-7b-sft \
  --tp 1 \
  --cache-max-entry-count 0 \
  --session-len 4096
```

### 模型导出
```bash
swift export \
  --ckpt_dir ./output/qwen2.5-7b-sft/checkpoint-xxx \
  --merge_lora true \
  --safe_serialization true \
  --max_shard_size 2GB \
  --output_dir ./export/qwen2.5-7b-sft
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

## ⚠️ 注意事项

1. **Flash Attention**: 需要正确安装 flash_attn 包
2. **中文编码**: 确保系统 locale 设置正确
3. **显存监控**: 训练过程中监控显存使用情况
4. **模型保存**: 定期保存检查点，避免训练中断

## 📚 参考资料

- [Qwen2.5 官方文档](https://github.com/QwenLM/Qwen2)
- [ModelScope 模型页面](https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct)
- [ms-swift 训练指南](https://github.com/modelscope/swift)

---

*配置版本: v1.0*  
*最后更新: 2025-08-20*
