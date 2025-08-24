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
  --ckpt_dir ./output/qwen2.5-7b-sft/v0-20250824-022332/checkpoint-43 \
  --merge_lora true \
  --safe_serialization true \
  --max_shard_size 2GB \
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

## 8. 模型评测（2025-08-20 完成）

### 8.1 评测环境准备

**环境变量设置**（解决代理问题）：
```bash
# 设置本地地址不走代理
export NO_PROXY=127.0.0.1,localhost,::1
export no_proxy=$NO_PROXY

# 验证设置
echo "NO_PROXY: $NO_PROXY"
echo "no_proxy: $no_proxy"
```

**服务启动**（LMDeploy 后端）：
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

**服务健康检查**：
```bash
# 测试 /models 接口
curl --noproxy '*' http://127.0.0.1:8000/v1/models

# 测试推理接口
curl --noproxy '*' http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mh-sft-7b","messages":[{"role":"user","content":"你好"}]}'
```

### 8.2 评测工具安装与配置

#### 方案 A：lm-eval-harness（推荐，快速测试）

**安装**：
```bash
pip install "lm-eval==0.4.2"
```

**准备评测数据**：
创建 `data/pceb_mcq.jsonl`：
```json
{"id":"q1","question":"求助者因出国与家庭矛盾产生的心理冲突类型是？","choices":["变形","趋避式","常形","双趋式"],"answer":"B"}
{"id":"q2","question":"心理咨询中，共情是指？","choices":["同情","理解","指导","分析"],"answer":"B"}
```

**定义评测任务**：
创建 `tasks/pceb_mcq.yaml`：
```yaml
task: pceb_mcq
dataset_path: data/pceb_mcq.jsonl
output_type: multiple_choice
doc_to_text: "问：{{question}}\n选项：{% for c in choices %}{{ loop.index0 | chr(65) }}. {{ c }} {% endfor %}\n答："
doc_to_target: "{{answer}}"
doc_to_choice: "{{choices}}"
should_decontaminate: false
fewshot_delimiter: "\n\n"
generation_kwargs:
  temperature: 0.0
```

**运行评测**：
```bash
# 使用本地服务
lm_eval \
  --model openai-chat-completions \
  --model_args api_base=http://127.0.0.1:8000/v1,model=mh-sft-7b \
  --tasks pceb_mcq \
  --num_fewshot 0 \
  --batch_size 1

# 或使用合并后的权重
lm_eval \
  --model hf \
  --model_args pretrained=./export/mh-sft-qwen2p5-7b,dtype=bfloat16,attn_implementation=flash_attention_2 \
  --tasks pceb_mcq \
  --num_fewshot 0 \
  --batch_size 1
```

#### 方案 B：EvalScope（完整评测）

**安装**：
```bash
# 基础版本
pip install evalscope

# 带可视化界面的完整版本（推荐）
pip install 'evalscope[app]'
```

**使用导出模型进行评测**：

由于我们已经导出了完整的模型权重，可以直接使用 EvalScope 进行评测，无需启动服务：

```bash
# 1. 直接评测导出模型（推荐）
evalscope eval \
  --model ./export/mh-sft-qwen2p5-7b \
  --eval-type local \
  --datasets ceval cmmlu \
  --limit 100 \
  --output-file ./results/mh-sft-7b-ceval-cmmlu.json

# 2. 评测心理学基准（如果支持）
evalscope eval \
  --model ./export/mh-sft-qwen2p5-7b \
  --eval-type local \
  --datasets pceb_mcq psyqa \
  --limit 100 \
  --output-file ./results/mh-sft-7b-psychology.json

# 3. 评测多个数据集
evalscope eval \
  --model ./export/mh-sft-qwen2p5-7b \
  --eval-type local \
  --datasets ceval cmmlu pceb_mcq psyqa \
  --limit 100 \
  --output-file ./results/mh-sft-7b-all-benchmarks.json
```

**评测结果可视化**：
```bash
# 启动 EvalScope 可视化界面
evalscope app --lang zh

# 在浏览器中打开 http://127.0.0.1:7860
# 上传评测结果文件进行可视化分析
```

**评测参数说明**：
- `--model`: 导出模型的本地路径
- `--eval-type local`: 使用本地模型（而非 API 服务）
- `--datasets`: 要评测的数据集列表
- `--limit`: 每个数据集的评测样本数量
- `--output-file`: 评测结果保存路径

**支持的心理学基准数据集**：
- `ceval`: 中文语言理解评估基准
- `cmmlu`: 中文多任务语言理解
- `pceb_mcq`: 心理学考试选择题
- `psyqa`: 心理学问答数据集

**运行评测**：
```bash
# 设置环境变量后运行
export NO_PROXY=127.0.0.1,localhost,::1
export no_proxy=$NO_PROXY

evalscope eval \
  --model mh-sft-7b \
  --api-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --eval-type service \
  --datasets ceval cmmlu \
  --limit 100
```

**结果可视化**：
```bash
# 启动可视化界面（中文）
evalscope app --lang zh

# 启动可视化界面（英文）
evalscope app --lang en
```

**可视化功能**：
- 📊 **评测指标图表**：准确率、F1、BLEU 等指标的趋势图
- 🔍 **详细输出对比**：查看模型在每个样本上的具体回答
- 📈 **性能分析**：不同数据集、不同维度的性能对比
- 💾 **结果导出**：支持 CSV、JSON 等格式的结果导出
- 🎯 **错误分析**：重点关注错误样本，分析模型弱点

### 8.3 评测结果记录

**选择题评测指标**：
- 标准准确率（exact match）
- 弹性准确率（subset/overlap，多选题部分分）

**主观题评测指标**：
- Rouge-1 / Rouge-L / BLEU
- 要点命中率

**心理学专用维度**：
- 情绪共情 (EmoE)
- 认知共情 (CogE)
- 对话策略 (Con.)
- 态度与状态 (Sta.)
- 安全性 (Saf.)

**结果可视化与分析**：
- 📊 **EvalScope 可视化界面**：`evalscope app --lang zh`
- 📈 **性能趋势图**：不同数据集、不同维度的性能对比
- 🔍 **样本级分析**：查看具体问答对，分析模型表现
- 📋 **错误样本收集**：重点关注错误案例，用于后续改进
- 💾 **结果导出**：CSV、JSON 格式，便于进一步分析

### 8.4 评测数据准备建议

**推荐题库**：
- **PCEB/CPsyExam**：知识/伦理/案例综合评测
- **PsyQA**：同理心、支持性回答质量
- **C-Eval/CMMLU**：心理学子集知识测试

**数据格式**：
```json
# 单选题
{"id":"q1","question":"问题描述","choices":["选项A","选项B","选项C","选项D"],"answer":"B"}

# 多选题
{"id":"q2","question":"问题描述","choices":["选项A","选项B","选项C","选项D"],"answer":"B,C"}
```

## 9. 已知问题与修复
- ✅ 已解决：`--report_to swanlab` 参数缺失导致不推送
- ✅ 已解决：Flash Attention 安装支持 `--packing true`
- ✅ 已解决：网络代理问题（设置 NO_PROXY 环境变量）
- GPU 不支持 bf16：切换为 `--fp16 true`
- 显存不足：增大 `--gradient_accumulation_steps`、开启 `--gradient_checkpointing`、或使用 DeepSpeed `--deepspeed zero2/zero3`
- 长度限制：将 `--max_length` 降到 1024/1536 以降低显存

## 10. 结果与结论（占位）
- 指标汇总（SwanLab 截图/链接）：待补
- 样例对话与主观评估：待补
- 对比基线（如未微调 vs 微调）：待补
- 评测结果（选择题准确率、主观题质量等）：待补

## 11. 后续计划
- 增加多卡 DeepSpeed 配置与吞吐统计
- 尝试 Flash-Attn（如硬件/驱动允许）
- 数据清洗与模板改进（如需）
- 扩展心理学基准评测（PsyQA、CPsyExam 等）
- 自动化评测流程（CI/CD 集成）

## 12. 变更日志
- 初始版本：整理本次实验的全流程并统一命令模板（含 `--report_to swanlab`）
- 更新：加入 Flash Attention 安装状态与支持 packing 的训练命令
- 更新：加入完整的模型评测流程和工具配置
- 更新：加入使用导出模型进行 EvalScope 评测的完整命令和流程

## 13. 导出模型评测完整流程

### 13.1 评测前准备

**确认导出模型路径**：
```bash
# 检查导出目录
ls -la ./export/
# 应该看到：mh-sft-qwen2p5-7b/

# 检查模型文件
ls -la ./export/mh-sft-qwen2p5-7b/
# 应该包含：config.json, model.safetensors, tokenizer.json 等
```

**创建结果目录**：
```bash
mkdir -p ./results
mkdir -p ./results/evalscope
mkdir -p ./results/analysis
```

### 13.2 使用 EvalScope 评测导出模型

#### 基础评测（推荐开始）
```bash
# 评测中文语言理解基准
evalscope eval \
  --model ./export/mh-sft-qwen2p5-7b \
  --eval-type local \
  --datasets ceval \
  --limit 100 \
  --output-file ./results/evalscope/mh-sft-7b-ceval.json

# 评测中文多任务语言理解
evalscope eval \
  --model ./export/mh-sft-qwen2p5-7b \
  --eval-type local \
  --datasets cmmlu \
  --limit 100 \
  --output-file ./results/evalscope/mh-sft-7b-cmmlu.json
```

#### 心理学专项评测
```bash
# 评测心理学考试选择题
evalscope eval \
  --model ./export/mh-sft-qwen2p5-7b \
  --eval-type local \
  --datasets pceb_mcq \
  --limit 100 \
  --output-file ./results/evalscope/mh-sft-7b-pceb.json

# 评测心理学问答质量
evalscope eval \
  --model ./export/mh-sft-qwen2p5-7b \
  --eval-type local \
  --datasets psyqa \
  --limit 100 \
  --output-file ./results/evalscope/mh-sft-7b-psyqa.json
```

#### 综合评测（一次性评测多个数据集）
```bash
# 评测所有支持的基准
evalscope eval \
  --model ./export/mh-sft-qwen2p5-7b \
  --eval-type local \
  --datasets ceval cmmlu pceb_mcq psyqa \
  --limit 100 \
  --output-file ./results/evalscope/mh-sft-7b-all-benchmarks.json
```

### 13.3 评测结果分析

#### 启动可视化界面
```bash
# 启动中文界面
evalscope app --lang zh

# 启动英文界面
evalscope app --lang en

# 默认地址：http://127.0.0.1:7860
```

#### 结果文件分析
```bash
# 查看评测结果
cat ./results/evalscope/mh-sft-7b-ceval.json | jq '.'

# 提取关键指标
cat ./results/evalscope/mh-sft-7b-ceval.json | jq '.results | keys'

# 查看具体分数
cat ./results/evalscope/mh-sft-7b-ceval.json | jq '.results.ceval.metrics'
```

### 13.4 评测参数调优

#### 性能优化
```bash
# 增加评测样本数量（更准确的结果）
evalscope eval \
  --model ./export/mh-sft-qwen2p5-7b \
  --eval-type local \
  --datasets ceval \
  --limit 500 \
  --output-file ./results/evalscope/mh-sft-7b-ceval-500.json

# 使用多进程加速
evalscope eval \
  --model ./export/mh-sft-qwen2p5-7b \
  --eval-type local \
  --datasets ceval \
  --limit 100 \
  --num_workers 4 \
  --output-file ./results/evalscope/mh-sft-7b-ceval-fast.json
```

#### 自定义评测
```bash
# 评测特定子集
evalscope eval \
  --model ./export/mh-sft-qwen2p5-7b \
  --eval-type local \
  --datasets ceval \
  --limit 100 \
  --subset psychology \
  --output-file ./results/evalscope/mh-sft-7b-ceval-psychology.json
```

### 13.5 评测结果记录模板

**评测记录表**：
| 数据集 | 样本数 | 准确率 | F1分数 | 备注 |
|--------|--------|--------|--------|------|
| C-Eval | 100    | -      | -      | 待评测 |
| CMMLU  | 100    | -      | -      | 待评测 |
| PCEB   | 100    | -      | -      | 待评测 |
| PsyQA  | 100    | -      | -      | 待评测 |

**性能对比**：
- 基座模型 vs 微调后模型
- 不同训练轮数的性能变化
- 与其他同类模型的对比

### 13.6 常见问题解决

**评测失败排查**：
```bash
# 1. 检查模型路径
ls -la ./export/mh-sft-qwen2p5-7b/

# 2. 检查模型完整性
python -c "from transformers import AutoModel, AutoTokenizer; model = AutoModel.from_pretrained('./export/mh-sft-qwen2p5-7b'); print('Model loaded successfully')"

# 3. 检查 EvalScope 安装
evalscope --version

# 4. 检查可用数据集
evalscope list-datasets
```

**显存不足**：
```bash
# 减少评测样本数量
--limit 50

# 使用更小的批次
--batch_size 1

# 启用梯度检查点
--gradient_checkpointing true
```

---

**注意**: 使用导出模型进行评测的优势是无需启动服务，直接加载模型权重，评测速度更快，资源占用更少。建议先用小样本测试，确认无问题后再进行完整评测。


