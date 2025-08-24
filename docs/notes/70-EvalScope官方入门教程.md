# EvalScope 官方入门教程

## 概述

EvalScope 是 ModelScope 官方的 LLM 评测框架，支持多种评测方式和数据集。本教程基于官方文档，提供完整的使用指南。

## 1. 基本使用

### 1.1 简单评测

在指定的若干数据集上使用默认配置评测某个模型，本框架支持两种启动评测任务的方式：使用命令行启动或使用Python代码启动评测任务。

#### 方式1. 使用命令行

**使用 `eval` 命令**

在任意路径下执行 `eval` 命令：

```bash
evalscope eval \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --datasets gsm8k arc \
  --limit 5
```

**运行 `run.py`**

在 `evalscope` 根目录下执行：

```bash
python evalscope/run.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --datasets gsm8k arc \
  --limit 5
```

#### 方式2. 使用Python代码

使用python代码进行评测时需要用 `run_task` 函数提交评测任务，传入一个 `TaskConfig` 作为参数，也可以为python字典、yaml文件路径或json文件路径。

**使用Python字典**

```python
from evalscope.run import run_task

task_cfg = {
    'model': 'Qwen/Qwen2.5-0.5B-Instruct',
    'datasets': ['gsm8k', 'arc'],
    'limit': 5
}

run_task(task_cfg=task_cfg)
```

**使用 `TaskConfig`**

```python
from evalscope.run import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='Qwen/Qwen2.5-0.5B-Instruct',
    datasets=['gsm8k', 'arc'],
    limit=5
)

run_task(task_cfg=task_cfg)
```

**使用 `yaml` 文件**

`config.yaml`:
```yaml
model: Qwen/Qwen2.5-0.5B-Instruct
datasets:
  - gsm8k
  - arc
limit: 5
```

```python
from evalscope.run import run_task

run_task(task_cfg="config.yaml")
```

**使用 `json` 文件**

`config.json`:
```json
{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "datasets": ["gsm8k", "arc"],
    "limit": 5
}
```

```python
from evalscope.run import run_task

run_task(task_cfg="config.json")
```

### 1.2 基本参数说明

- **`--model`**: 指定了模型在 [ModelScope](https://modelscope.cn/) 中的 `model_id`，可自动下载，例如 [Qwen/Qwen2.5-0.5B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/summary)；也可使用模型的本地路径，例如 `/path/to/model`

- **`--datasets`**: 数据集名称，支持输入多个数据集，使用空格分开，数据集将自动从modelscope下载，支持的数据集参考 [数据集列表](supported_dataset/index.html)

- **`--limit`**: 每个数据集最大评测数据量，不填写则默认为全部评测，可用于快速验证

### 1.3 输出结果

```
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Model Name            | Dataset Name   | Metric Name     | Category Name   | Subset Name   |   Num |   Score |
+=======================+================+=================+=================+===============+=======+=========+
| Qwen2.5-0.5B-Instruct | gsm8k          | AverageAccuracy | default         | main          |     5 |     0.4 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Qwen2.5-0.5B-Instruct | ai2_arc        | AverageAccuracy | default         | ARC-Easy      |     5 |     0.8 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Qwen2.5-0.5B-Instruct | ai2_arc        | AverageAccuracy | default         | ARC-Challenge |     5 |     0.4 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
```

## 2. 复杂评测

若想进行更加自定义的评测，例如自定义模型参数，或者数据集参数，可以使用以下命令，启动评测方式与简单评测一致，下面展示了使用 `eval` 命令启动评测：

```bash
evalscope eval \
  --model Qwen/Qwen3-0.6B \
  --model-args '{"revision": "master", "precision": "torch.float16", "device_map": "auto"}' \
  --generation-config '{"do_sample":true,"temperature":0.6,"max_new_tokens":512,"chat_template_kwargs":{"enable_thinking": false}}' \
  --dataset-args '{"gsm8k": {"few_shot_num": 0, "few_shot_random": false}}' \
  --datasets gsm8k \
  --limit 10
```

### 2.1 参数说明

- **`--model-args`**: 模型加载参数，以json字符串格式传入：
  - `revision`: 模型版本
  - `precision`: 模型精度
  - `device_map`: 模型分配设备

- **`--generation-config`**: 生成参数，以json字符串格式传入，将解析为字典：
  - `do_sample`: 是否使用采样
  - `temperature`: 生成温度
  - `max_new_tokens`: 生成最大长度
  - `chat_template_kwargs`: 模型推理模板参数

- **`--dataset-args`**: 评测数据集的设置参数，以json字符串格式传入，key为数据集名称，value为参数，注意需要跟 `--datasets` 参数中的值一一对应：
  - `few_shot_num`: few-shot的数量
  - `few_shot_random`: 是否随机采样few-shot数据，如果不设置，则默认为 `true`

### 2.2 输出结果

```
+------------+-----------+-----------------+----------+-------+---------+---------+
| Model      | Dataset   | Metric          | Subset   |   Num |   Score | Cat.0   |
+============+===========+=================+==========+=======+=========+=========+
| Qwen3-0.6B | gsm8k     | AverageAccuracy | main     |    10 |     0.3 | default |
+------------+-----------+-----------------+----------+-------+---------+---------+
```

## 3. 模型API服务评测

指定模型API服务地址(`api_url`)和API Key(`api_key`)，评测部署的模型API服务，**此时 `eval-type` 参数必须指定为 `service`**。

例如使用 [vLLM](https://github.com/vllm-project/vllm) 拉起模型服务：

```bash
export VLLM_USE_MODELSCOPE=True && python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-0.5B-Instruct --served-model-name qwen2.5 --trust_remote_code --port 8801
```

然后使用以下命令评测模型API服务：

```bash
evalscope eval \
  --model qwen2.5 \
  --api-url http://127.0.0.1:8801/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --limit 10
```

## 4. 使用裁判模型

在评测时，可以使用裁判模型对模型的输出进行评估，此外有些数据集需要使用裁判模型进行评测，例如 `simple_qa` 数据集，使用以下命令启动评测：

```python
from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType, JudgeStrategy

task_cfg = TaskConfig(
    model='qwen2.5-7b-instruct',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    eval_type=EvalType.SERVICE,
    datasets=[
        # 'simple_qa',
        'chinese_simpleqa',
    ],
    eval_batch_size=5,
    limit=5,
    judge_strategy=JudgeStrategy.AUTO,
    judge_model_args={
        'model_id': 'qwen2.5-72b-instruct',
        'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'api_key': os.getenv('DASHSCOPE_API_KEY'),
    }
)

run_task(task_cfg=task_cfg)
```

## 5. 离线评测

数据集默认托管在 [ModelScope](https://modelscope.cn/datasets) 上，加载需要联网。如果是无网络环境，可以使用本地数据集，流程如下：

假如当前本地工作路径为 `/path/to/workdir`。

### 5.1 下载数据集到本地

**重要**: 在下载数据集之前请确认你想使用的数据集是存放在 `zip` 中，还是在modelscope中。

#### 下载zip数据集

由于历史原因，部分数据集是通过执行python脚本的方式进行加载的，这部分数据集我们将其整理到了一个 `zip` 文件中，包括如下数据集：

```
.
├── arc
├── bbh
├── ceval
├── cmmlu
├── competition_math
├── general_qa
├── gsm8k
├── hellaswag
├── humaneval
├── mmlu
├── race
├── trivia_qa
└── truthful_qa
```

对于这部分数据集，执行以下命令：

```bash
wget https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/benchmark/data.zip
unzip data.zip
```

解压后的数据集在：`/path/to/workdir/data` 目录下，该目录在后续步骤将会作为 `local_path` 参数的值传入。

#### 下载modelscope数据集

对于不在 `zip` 中的数据集，例如 [mmlu_pro](https://modelscope.cn/datasets/modelscope/MMLU-Pro) 数据集，数据集地址参考 [支持的数据集](supported_dataset/index.html)，执行以下命令：

```bash
git lfs install
git clone https://www.modelscope.cn/datasets/modelscope/MMLU-Pro.git
```

使用目录 `/path/to/MMLU-Pro` 作为 `local_path` 参数的值传入即可。

### 5.2 下载模型到本地

模型文件托管在ModelScope Hub端，需要联网加载，当需要在离线环境创建评测任务时，可提前将模型下载到本地：

例如使用Git下载Qwen2.5-0.5B-Instruct模型到本地：

```bash
git lfs install
git clone https://www.modelscope.cn/Qwen/Qwen2.5-0.5B-Instruct.git
```

### 5.3 执行评测任务

运行下面的命令进行评测，传入本地数据集路径和模型路径，注意 `local_path` 需要跟 `--datasets` 参数中的值一一对应：

```bash
evalscope eval \
  --model /path/to/workdir/Qwen2.5-0.5B-Instruct \
  --datasets arc \
  --dataset-args '{"arc": {"local_path": "/path/to/workdir/data/arc"}}' \
  --limit 10
```

## 6. 实际应用示例

### 6.1 快速测试模型性能

```bash
# 测试数学推理能力
evalscope eval \
  --model Qwen/Qwen2.5-7B-Instruct \
  --datasets gsm8k \
  --limit 10

# 测试中文理解能力
evalscope eval \
  --model Qwen/Qwen2.5-7B-Instruct \
  --datasets ceval cmmlu \
  --limit 50
```

### 6.2 评测本地微调模型

```bash
# 评测本地微调后的模型
evalscope eval \
  --model ./output/qwen2.5-7b-sft/checkpoint-1000 \
  --datasets gsm8k ceval \
  --limit 100
```

### 6.3 评测部署的模型服务

```bash
# 评测本地部署的模型
evalscope eval \
  --model my-model \
  --api-url http://127.0.0.1:8000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k arc \
  --limit 50
```

## 7. 常见问题与解决方案

### 7.1 网络连接问题

**问题**: 无法下载模型或数据集
**解决方案**: 
- 使用国内镜像源
- 配置代理设置
- 使用离线模式

### 7.2 显存不足

**问题**: CUDA out of memory
**解决方案**:
- 减少 `eval_batch_size`
- 使用 `--model-args '{"device_map": "auto"}'`
- 使用更小的模型进行测试

### 7.3 数据集加载失败

**问题**: 数据集无法正确加载
**解决方案**:
- 检查数据集名称是否正确
- 确认网络连接
- 使用本地数据集路径

## 8. 最佳实践

### 8.1 评测流程建议

1. **从小规模开始**: 使用 `--limit 10` 快速验证
2. **逐步扩大**: 确认无误后增加样本数量
3. **多数据集对比**: 使用多个数据集全面评估
4. **结果记录**: 保存评测结果便于对比分析

### 8.2 性能优化

1. **批处理大小**: 根据显存调整 `eval_batch_size`
2. **并行处理**: 使用多进程加速数据处理
3. **缓存利用**: 启用缓存避免重复计算

### 8.3 结果分析

1. **可视化**: 使用 `evalscope app` 查看结果
2. **对比分析**: 与基线模型进行对比
3. **错误分析**: 分析失败案例找出问题

---

**参考资源**:
- [EvalScope 官方文档](https://modelscope.cn/docs/eval-scope)
- [支持的数据集列表](supported_dataset/index.html)
- [ModelScope 模型库](https://modelscope.cn/models)
- [GitHub 仓库](https://github.com/modelscope/eval-scope)
