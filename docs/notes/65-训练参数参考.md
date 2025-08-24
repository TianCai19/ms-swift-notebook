# 训练参数参考指南

本文档提供 `ms-swift` 训练时的常用参数说明，以及针对不同模型大小的可直接使用的配置版本。

## 1. 核心训练参数

### 1.1 模型相关
| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--model` | 基座模型 ID（HuggingFace/ModelScope） | `Qwen/Qwen2.5-7B-Instruct` |
| `--model_type` | 模型类型（影响 tokenizer 和架构） | `qwen2_5`, `llama`, `chatglm3` |
| `--template` | 对话模板（影响输入格式） | `qwen2_5`, `llama3`, `chatglm3` |
| `--train_type` | 训练方式 | `lora`（推荐）, `full`, `qlora` |

### 1.2 数据相关
| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--dataset` | 数据集 ID 或本地路径 | `CodyWhy/mh-sharegpt-20250820` |
| `--max_length` | 最大序列长度 | `3072`（推荐）, `4096`, `2048` |
| `--packing` | 是否启用序列打包（需 flash_attn） | `true`（推荐）, `false` |

### 1.3 训练配置
| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--bf16` | 是否使用 bfloat16 精度 | `true`（推荐）, `false` |
| `--learning_rate` | 学习率 | `2e-4`（推荐）, `1e-4`, `5e-4` |
| `--num_train_epochs` | 训练轮数 | `1`（推荐）, `2`, `3` |
| `--warmup_ratio` | 预热比例 | `0.1`（推荐）, `0.05`, `0.15` |
| `--save_steps` | 保存检查点步数 | `200`（推荐）, `100`, `500` |
| `--save_total_limit` | 最大保存检查点数 | `3`（推荐）, `5`, `10` |
| `--logging_steps` | 日志记录步数 | `20`（推荐）, `10`, `50` |

### 1.4 批次与优化
| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--per_device_train_batch_size` | 每设备批次大小 | 见下方配置表 |
| `--gradient_accumulation_steps` | 梯度累积步数 | 见下方配置表 |
| `--gradient_checkpointing` | 梯度检查点（节省显存） | `true`（推荐） |
| `--max_grad_norm` | 梯度裁剪范数 | `1.0`（推荐）, `0.5`, `2.0` |

### 1.5 硬件配置
| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--attn_impl` | 注意力实现 | `flash_attn`（推荐）, `flash_attention_2`, `sdpa` |
| `--dtype` | 数据类型 | `bf16`（推荐）, `fp16`, `fp32` |

## 2. 可直接使用的配置版本

### 2.1 7B 模型标准配置
```bash
CUDA_VISIBLE_DEVICES=6 \
swift sft \
  --model <MODEL_ID> \
  --model_type <MODEL_TYPE> \
  --template <TEMPLATE> \
  --train_type lora \
  --dataset CodyWhy/mh-sharegpt-20250820 \
  --bf16 true \
  --max_length 3072 \
  --packing true \
  --gradient_checkpointing true \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --warmup_ratio 0.1 \
  --save_steps 200 \
  --save_total_limit 3 \
  --logging_steps 20 \
  --output_dir ./output/<model-name>-sft \
  --report_to swanlab \
  --swanlab_token $SWANLAB_TOKEN \
  --swanlab_project multi-model-psychology \
  --swanlab_mode cloud \
  --swanlab_exp_name <model-name>-sft \
  --attn_impl flash_attn
```

### 2.2 8B 模型配置（显存紧张）
```bash
CUDA_VISIBLE_DEVICES=6 \
swift sft \
  --model <MODEL_ID> \
  --model_type <MODEL_TYPE> \
  --template <TEMPLATE> \
  --train_type lora \
  --dataset CodyWhy/mh-sharegpt-20250820 \
  --bf16 true \
  --max_length 3072 \
  --packing true \
  --gradient_checkpointing true \
  --per_device_train_batch_size 6 \
  --gradient_accumulation_steps 6 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --warmup_ratio 0.1 \
  --save_steps 200 \
  --save_total_limit 3 \
  --logging_steps 20 \
  --output_dir ./output/<model-name>-sft \
  --report_to swanlab \
  --swanlab_token $SWANLAB_TOKEN \
  --swanlab_project multi-model-psychology \
  --swanlab_mode cloud \
  --swanlab_exp_name <model-name>-sft \
  --attn_impl flash_attn
```

### 2.3 20B 模型配置（大模型）
```bash
CUDA_VISIBLE_DEVICES=6 \
swift sft \
  --model <MODEL_ID> \
  --model_type <MODEL_TYPE> \
  --template <TEMPLATE> \
  --train_type lora \
  --dataset CodyWhy/mh-sharegpt-20250820 \
  --bf16 true \
  --max_length 3072 \
  --packing true \
  --gradient_checkpointing true \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --warmup_ratio 0.1 \
  --save_steps 200 \
  --save_total_limit 3 \
  --logging_steps 20 \
  --output_dir ./output/<model-name>-sft \
  --report_to swanlab \
  --swanlab_token $SWANLAB_TOKEN \
  --swanlab_project multi-model-psychology \
  --swanlab_mode cloud \
  --swanlab_exp_name <model-name>-sft \
  --attn_impl flash_attn
```

### 2.4 显存不足时的应急配置
```bash
CUDA_VISIBLE_DEVICES=6 \
swift sft \
  --model <MODEL_ID> \
  --model_type <MODEL_TYPE> \
  --template <TEMPLATE> \
  --train_type lora \
  --dataset CodyWhy/mh-sharegpt-20250820 \
  --bf16 true \
  --max_length 2048 \
  --packing false \
  --gradient_checkpointing true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --learning_rate 1e-4 \
  --num_train_epochs 1 \
  --warmup_ratio 0.1 \
  --save_steps 100 \
  --save_total_limit 2 \
  --logging_steps 50 \
  --output_dir ./output/<model-name>-sft \
  --report_to swanlab \
  --swanlab_token $SWANLAB_TOKEN \
  --swanlab_project multi-model-psychology \
  --swanlab_mode cloud \
  --swanlab_exp_name <model-name>-sft \
  --attn_impl sdpa
```

## 3. 批次大小配置表

| 模型大小 | 推荐配置 | 总批次大小 | 适用场景 |
|---------|----------|-----------|----------|
| 6-7B    | 8×4      | 32        | 标准训练，显存充足 |
| 8B      | 6×6      | 36        | 显存紧张，保持总批次 |
| 20B     | 2×16     | 32        | 大模型，显存优化 |
| 应急     | 1×32     | 32        | 显存严重不足 |

## 4. 参数调优建议

### 4.1 学习率调优
- **标准值**: `2e-4`（大多数情况）
- **显存不足**: `1e-4`（更稳定）
- **快速收敛**: `5e-4`（需监控 loss）

### 4.2 序列长度调优
- **标准值**: `3072`（平衡效果与显存）
- **显存充足**: `4096`（更好的长文本理解）
- **显存紧张**: `2048`（快速训练）

### 4.3 批次大小调优
- **原则**: 保持总批次大小 ≥ 32
- **显存不足**: 降低单批次，提高累积步数
- **显存充足**: 提高单批次，降低累积步数

### 4.4 保存策略调优
- **标准**: `save_steps=200, save_total_limit=3`
- **快速迭代**: `save_steps=100, save_total_limit=5`
- **长期训练**: `save_steps=500, save_total_limit=2`

## 5. 常见问题与解决方案

### 5.1 显存不足 (CUDA OOM)
```bash
# 解决方案 1: 降低批次大小
--per_device_train_batch_size 4
--gradient_accumulation_steps 8

# 解决方案 2: 降低序列长度
--max_length 2048

# 解决方案 3: 关闭打包
--packing false

# 解决方案 4: 使用梯度检查点
--gradient_checkpointing true
```

### 5.2 训练速度慢
```bash
# 解决方案 1: 启用 Flash Attention
--attn_impl flash_attn

# 解决方案 2: 启用序列打包
--packing true

# 解决方案 3: 提高批次大小
--per_device_train_batch_size 16
--gradient_accumulation_steps 2
```

### 5.3 收敛不稳定
```bash
# 解决方案 1: 降低学习率
--learning_rate 1e-4

# 解决方案 2: 调整预热比例
--warmup_ratio 0.15

# 解决方案 3: 启用梯度裁剪
--max_grad_norm 0.5
```

## 6. 环境变量设置

```bash
# 设置代理（避免本地服务被拦截）
export NO_PROXY=127.0.0.1,localhost,::1
export no_proxy=$NO_PROXY

# 设置 SwanLab（可选）
export SWANLAB_TOKEN=<YOUR_TOKEN>
export SWANLAB_PROJECT=multi-model-psychology

# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=6
```

## 7. 快速启动命令

### 7.1 复制粘贴版本（Qwen2.5-7B）
```bash
export NO_PROXY=127.0.0.1,localhost,::1
export no_proxy=$NO_PROXY
export SWANLAB_TOKEN=<YOUR_TOKEN>
export SWANLAB_PROJECT=multi-model-psychology

CUDA_VISIBLE_DEVICES=6 \
swift sft \
  --model Qwen/Qwen2.5-7B-Instruct \
  --model_type qwen2_5 \
  --template qwen2_5 \
  --train_type lora \
  --dataset CodyWhy/mh-sharegpt-20250820 \
  --bf16 true \
  --max_length 3072 \
  --packing true \
  --gradient_checkpointing true \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --warmup_ratio 0.1 \
  --save_steps 200 \
  --save_total_limit 3 \
  --logging_steps 20 \
  --output_dir ./output/qwen2.5-7b-sft \
  --report_to swanlab \
  --swanlab_token $SWANLAB_TOKEN \
  --swanlab_project $SWANLAB_PROJECT \
  --swanlab_mode cloud \
  --swanlab_exp_name qwen2p5-7b-sft \
  --attn_impl flash_attn
```

---

**注意**: 请根据实际硬件配置调整参数，特别是批次大小和序列长度。建议先用小配置测试，确认无问题后再使用标准配置。
