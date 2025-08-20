## ShareGPT → ModelScope 上传与 ms-swift 微调

### 目标
- 将本地 ShareGPT 格式的 `.jsonl` 上传到 ModelScope（可选），或直接用本地文件。
- 使用 ms-swift 启动一次最小可行的 LoRA SFT 训练，并记录常见加速/避坑点。
- 当前数据集 ID：`CodyWhy/mh-sharegpt-20250820`。

### 一、上传 ShareGPT `.jsonl` 到 ModelScope（可选）
> 说明：ms-swift 可以直接使用本地路径训练，不一定要上传到 ModelScope。若需要分享/复用，可使用 Hub。

1) 安装与登录
```bash
pip install -U modelscope
modelscope login --token <你的ModelScope SDK Token>
```

2) 在网页端创建数据集仓库（Datasets -> New），假设 ID 为 `your_name/sharegpt-demo`

3) 上传本地文件到该数据集仓库
```bash
modelscope upload your_name/sharegpt-demo \
  /abs/path/to/your_sharegpt.jsonl \
  data/train.jsonl \
  --repo-type dataset \
  --commit-message "init sharegpt jsonl"
```

### 二、数据格式检查（ShareGPT 可直接使用）
AutoPreprocessor 会自动识别多种格式（messages / sharegpt / alpaca / query-response 等）。ShareGPT 单条样例（行级 JSON）：
```json
{"system":"<可选>","conversation":[{"human":"<q1>","assistant":"<a1>"},{"human":"<q2>","assistant":"<a2>"}]}
```

### 三、最快启动一次 LoRA 微调（使用本地文件）
> 若本地已有 `your_sharegpt.jsonl`，无需先上传即可训练。

```bash
pip install -U ms-swift

CUDA_VISIBLE_DEVICES=0 \
swift sft \
  --model Qwen/Qwen2.5-7B-Instruct \
  --train_type lora \
  --dataset /abs/path/to/your_sharegpt.jsonl \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --max_length 2048 \
  --packing true \
  --save_steps 200 \
  --save_total_limit 3 \
  --logging_steps 20 \
  --bf16 true \
  --output_dir output/your-run
  
  # 若需推送到 SwanLab，需额外添加：
  --report_to swanlab \
  --swanlab_token ${SWANLAB_API_KEY} \
  --swanlab_project ${SWANLAB_PROJECT:-mh-sharegpt} \
  --swanlab_mode ${SWANLAB_MODE:-cloud} \
  --swanlab_exp_name mh-sft-local
```

要点：
- `--dataset` 支持本地路径或 ModelScope 数据集 ID。ShareGPT 会被自动预处理为内部标准格式。
- 若 GPU 不支持 bfloat16，将 `--bf16 true` 换成 `--fp16 true`。
- Base（非对话）模型做 LoRA 时，必要时补充 `--template default`。

### 四、使用 ModelScope 数据集 ID 训练
> 已上传到 Hub 的情况，直接用数据集 ID：

```bash
CUDA_VISIBLE_DEVICES=0 \
swift sft \
  --model Qwen/Qwen2.5-7B-Instruct \
  --train_type lora \
  --dataset CodyWhy/mh-sharegpt-20250820 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --max_length 2048 \
  --packing true \
  --save_steps 200 \
  --save_total_limit 3 \
  --logging_steps 20 \
  --bf16 true \
  --output_dir output/mh-sft
  
  # 若需推送到 SwanLab，需额外添加：
  --report_to swanlab \
  --swanlab_token ${SWANLAB_API_KEY} \
  --swanlab_project ${SWANLAB_PROJECT:-mh-sharegpt} \
  --swanlab_mode ${SWANLAB_MODE:-cloud} \
  --swanlab_exp_name mh-sft
```

### 五、训练后推理 / 合并 LoRA / 回传 Hub
1) 交互式本地推理（LoRA 增量权重）
```bash
CKPT=output/mh-sft/checkpoint-xxxx

CUDA_VISIBLE_DEVICES=0 swift infer \
  --adapters "$CKPT" \
  --stream true \
  --temperature 0.7 \
  --max_new_tokens 512
```

2) 合并 LoRA 并用 vLLM 推理
```bash
CUDA_VISIBLE_DEVICES=0 swift infer \
  --adapters "$CKPT" \
  --merge_lora true \
  --infer_backend vllm \
  --max_new_tokens 512
```

3) 一键推回 ModelScope 模型仓库
```bash
swift export \
  --adapters "$CKPT" \
  --push_to_hub true \
  --hub_model_id "CodyWhy/mh-sft-qwen2.5-7b" \
  --hub_token "<你的SDK Token>" \
  --use_hf false
```

### 六、小贴士（提速与避坑）
- **大数据量**：`--streaming true` 边流式训练，降低内存峰值。
- **显存不够**：保持 LoRA；必要时装 DeepSpeed 并加 `--deepspeed zero2/zero3`；A100 等可装 flash-attn，并加 `--attn_impl flash_attn`。
- **吞吐优化**：开启 `--packing true`（样本拼接）。
- **长度与显存**：显存吃紧可将 `--max_length` 调低到 1024/1536。
- **数据列名**：若列名不一致可用 `--columns` 映射；ShareGPT 通常无需额外配置。


