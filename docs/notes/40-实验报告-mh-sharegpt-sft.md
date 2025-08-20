## 实验报告：mh-sharegpt SFT（LoRA）

### 1. 实验概述
- **目标**：基于 ShareGPT 数据集进行对话模型的指令微调（SFT），验证训练链路、监控与导出流程。
- **数据集**：`CodyWhy/mh-sharegpt-20250820`
- **基座模型（示例）**：`Qwen/Qwen2.5-7B-Instruct`（7B），以及 32B 变体
- **训练类型**：LoRA（参数高效微调）
- **监控**：SwanLab（云端），必须显式 `--report_to swanlab`

关联文档：
- 数据与训练流程：`notes/20-数据与SFT训练.md`
- SwanLab 集成：`notes/30-SwanLab集成.md`
- Flash Attention 安装：`notes/50-FlashAttention安装.md`
- 密钥与数据索引（私密）：`notes/90-密钥与数据索引.md`

### 2. 环境与依赖
- ms-swift：建议最新版本（参考 `pip install -U ms-swift`）
- ModelScope：用于登录与数据集访问（可选）
- SwanLab：用于实验记录与可视化
- Flash Attention：支持 `--packing true` 功能（已安装）
- 建议通过 `.env` 管理敏感信息：`SWANLAB_API_KEY`、`MODELSCOPE_API_TOKEN`、`DATASET_ID`

### 3. 数据
- 数据格式：ShareGPT（AutoPreprocessor 自动识别，无需手动转换）
- 数据集 ID：`CodyWhy/mh-sharegpt-20250820`
- 本地数据也可直接训练；若需共享/复用，可上传到 ModelScope

### 4. Flash Attention 状态
- **状态**：✅ 已安装
- **版本**：2.6.3+cu124torch2.6-cp310
- **安装方式**：预编译 wheel（ghproxy.net 代理）
- **验证命令**：`python -c "import flash_attn, flash_attn_cuda; print('OK')"`
- **支持功能**：`--packing true` + `--attn_impl flash_attn`

### 5. 训练配置与命令

#### 5.1 最小可行（7B，单卡，支持 packing）
```bash
CUDA_VISIBLE_DEVICES=6 \
swift sft \
  --model Qwen/Qwen2.5-7B-Instruct \
  --train_type lora \
  --dataset CodyWhy/mh-sharegpt-20250820 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --max_length 3072 \
  --packing true \
  --attn_impl flash_attn \
  --bf16 true \
  --gradient_checkpointing true \
  --save_steps 200 \
  --save_total_limit 3 \
  --logging_steps 20 \
  --output_dir output/mh-sft-7b-lora \
  --report_to swanlab \
  --swanlab_token ${SWANLAB_API_KEY} \
  --swanlab_project ${SWANLAB_PROJECT:-mh-sft} \
  --swanlab_mode ${SWANLAB_MODE:-cloud} \
  --swanlab_exp_name mh-sft-qwen2p5-7b-lora
```

#### 5.2 32B（H100 80GB，示例为 6 号卡）
```bash
CUDA_VISIBLE_DEVICES=6 \
swift sft \
  --model Qwen/Qwen2.5-32B-Instruct \
  --train_type lora \
  --dataset CodyWhy/mh-sharegpt-20250820 \
  --bf16 true \
  --max_length 4096 \
  --packing true \
  --attn_impl flash_attn \
  --gradient_checkpointing true \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --save_steps 200 \
  --save_total_limit 3 \
  --logging_steps 20 \
  --output_dir output/mh-sft-32b-lora \
  --report_to swanlab \
  --swanlab_token ${SWANLAB_API_KEY} \
  --swanlab_project ${SWANLAB_PROJECT:-mh-sft} \
  --swanlab_mode ${SWANLAB_MODE:-cloud} \
  --swanlab_exp_name mh-sft-qwen2p5-32b-lora
```

#### 5.3 使用本地文件（可替代数据集 ID）
```bash
CUDA_VISIBLE_DEVICES=6 \
swift sft \
  --model Qwen/Qwen2.5-7B-Instruct \
  --train_type lora \
  --dataset /abs/path/to/your_sharegpt.jsonl \
  --num_train_epochs 1 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --max_length 3072 \
  --packing true \
  --attn_impl flash_attn \
  --bf16 true \
  --gradient_checkpointing true \
  --output_dir output/local-file-run \
  --report_to swanlab \
  --swanlab_token ${SWANLAB_API_KEY} \
  --swanlab_project ${SWANLAB_PROJECT:-mh-sharegpt} \
  --swanlab_mode ${SWANLAB_MODE:-cloud} \
  --swanlab_exp_name mh-sft-local-file
```

### 6. 监控与验证
- 必须显式加入：`--report_to swanlab`，否则不会推送任何指标
- 运行时预期日志：
```text
[swanlab] experiment created: project=mh-sft, exp_name=...
[swanlab] uploading metrics ...
```
- Python 本地连通性测试示例见：`notes/30-SwanLab集成.md`

### 7. 推理与导出

#### 7.1 推理测试（2025-08-20 完成）

**测试环境**：
- GPU: NVIDIA H100 (卡号 6)
- 模型: Qwen/Qwen2.5-7B-Instruct + LoRA checkpoint-508
- 检查点路径: `./output/mh-sft/v0-20250820-144506/checkpoint-508`

**推理方式对比**：

1) **PyTorch 后端（启动最快）**
```bash
CUDA_VISIBLE_DEVICES=6 \
swift infer \
  --model Qwen/Qwen2.5-7B-Instruct \
  --adapters ./output/mh-sft/v0-20250820-144506/checkpoint-508 \
  --infer_backend pt \
  --attn_impl flash_attn \
  --stream true \
  --max_new_tokens 512
```

2) **SGLang 后端（并发性能好）**
```bash
CUDA_VISIBLE_DEVICES=6 \
swift deploy \
  --model Qwen/Qwen2.5-7B-Instruct \
  --adapters ./output/mh-sft/v0-20250820-144506/checkpoint-508 \
  --infer_backend sglang \
  --served_model_name mh-sft-7b
```

3) **LMDeploy 后端（推荐，启动快+性能好）**
```bash
CUDA_VISIBLE_DEVICES=6 \
swift deploy \
  --model Qwen/Qwen2.5-7B-Instruct \
  --adapters ./output/mh-sft/v0-20250820-144506/checkpoint-508 \
  --infer_backend lmdeploy \
  --served_model_name mh-sft-7b \
  --tp 1 \
  --cache-max-entry-count 0 \
  --session-len 4096
```

**推理测试结果**：
- ✅ 所有后端均成功加载 LoRA 权重
- ✅ 中文输入输出正常（locale 已修复）
- ✅ Flash Attention 正常工作
- ✅ 推理质量符合预期

#### 7.2 模型导出

1) **合并 LoRA 权重**
```bash
swift export \
  --ckpt_dir ./output/mh-sft/v0-20250820-144506/checkpoint-508 \
  --merge_lora true \
  --merge_dtype float16 \
  --output_dir ./export/qwen-sft-7b
```

2) **上传到 ModelScope**
```bash
swift upload \
  --model_dir ./export/qwen-sft-7b \
  --hub modelscope \
  --model_id yourname/qwen-sft-7b
```

3) **上传到 HuggingFace**
```bash
huggingface-cli upload ./export/qwen-sft-7b yourname/qwen-sft-7b
```

### 8. 已知问题与修复
- ✅ 已解决：`--report_to swanlab` 参数缺失导致不推送
- ✅ 已解决：Flash Attention 安装支持 `--packing true`
- GPU 不支持 bf16：切换为 `--fp16 true`
- 显存不足：增大 `--gradient_accumulation_steps`、开启 `--gradient_checkpointing`、或使用 DeepSpeed `--deepspeed zero2/zero3`
- 长度限制：将 `--max_length` 降到 1024/1536 以降低显存

### 9. 结果与结论（占位）
- 指标汇总（SwanLab 截图/链接）：待补
- 样例对话与主观评估：待补
- 对比基线（如未微调 vs 微调）：待补

### 10. 后续计划
- 增加多卡 DeepSpeed 配置与吞吐统计
- 尝试 Flash-Attn（如硬件/驱动允许）
- 数据清洗与模板改进（如需）

### 11. 变更日志
- 初始版本：整理本次实验的全流程并统一命令模板（含 `--report_to swanlab`）
- 更新：加入 Flash Attention 安装状态与支持 packing 的训练命令


