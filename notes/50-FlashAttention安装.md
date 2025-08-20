## Flash Attention 安装笔记

### 目标
在 H100 环境中安装 Flash Attention，以支持 ms-swift 的 `--packing true` 功能，提升训练吞吐。

### 环境信息
- **GPU**: NVIDIA H100 80GB HBM3
- **PyTorch**: 2.6.0+cu124
- **CUDA Runtime**: 12.4
- **Python**: 3.10
- **系统**: Linux

### 安装方式选择

#### 方式 1：预编译 Wheel（推荐）
> 适用于有网络访问的环境，速度最快

**下载命令**：
```bash
# 使用 ghproxy.net 代理（推荐）
wget "https://ghproxy.net/https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.6.3+cu124torch2.6-cp310-cp310-linux_x86_64.whl"

# 备用代理（如果 ghproxy.net 不可用）
wget "https://ghfast.top/github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.6.3+cu124torch2.6-cp310-cp310-linux_x86_64.whl"
```

**安装命令**：
```bash
pip install ./flash_attn-2.6.3+cu124torch2.6-cp310-cp310-linux_x86_64.whl
```

#### 方式 2：源码编译
> 适用于无预编译包或需要自定义编译选项

```bash
# 安装编译依赖
pip install --no-cache-dir ninja packaging

# 编译安装
pip install --no-cache-dir flash-attn --no-build-isolation
```

### 常见问题与解决方案

#### 问题 1：Invalid wheel filename
**错误信息**：
```
ERROR: Invalid wheel filename (wrong number of parts): 'flash_attl'
```

**原因**：使用 `-O` 重命名时丢失了版本信息，wheel 文件名必须保持完整格式。

**解决**：保留原始文件名下载
```bash
# 错误做法
wget <url> -O flash_attn.whl

# 正确做法
wget "https://ghproxy.net/https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.6.3+cu124torch2.6-cp310-cp310-linux_x86_64.whl"
```

#### 问题 2：SSL 连接失败
**错误信息**：
```
Unable to establish SSL connection.
```

**解决**：尝试不同代理节点
```bash
# 主推：ghproxy.net
wget "https://ghproxy.net/https://github.com/..."

# 备用：ghfast.top
wget "https://ghfast.top/github.com/..."

# 备用：ghproxy.com
wget "https://ghproxy.com/https://github.com/..."
```

#### 问题 3：版本不匹配
**检查要点**：
- Python 版本：`python --version`
- PyTorch 版本：`python -c "import torch; print(torch.__version__)"`
- CUDA 版本：`python -c "import torch; print(torch.version.cuda)"`

**版本对应表**：
| PyTorch | CUDA | Python | Wheel 文件名示例 |
|---------|------|--------|------------------|
| 2.6.x   | 12.4 | 3.10   | flash_attn-2.6.3+cu124torch2.6-cp310-cp310-linux_x86_64.whl |
| 2.5.x   | 12.4 | 3.12   | flash_attn-2.6.3+cu124torch2.5-cp312-cp312-linux_x86_64.whl |

### 安装验证

#### 快速验证
```bash
python - << 'PY'
try:
    import flash_attn, flash_attn_cuda
    print("✅ Flash Attention 安装成功")
    print(f"   - flash_attn: {flash_attn.__version__}")
    print(f"   - flash_attn_cuda: {flash_attn_cuda}")
except Exception as e:
    print("❌ Flash Attention 安装失败:", e)
PY
```

#### 功能验证
```bash
# 在 ms-swift 中测试
CUDA_VISIBLE_DEVICES=6 \
swift sft \
  --model Qwen/Qwen2.5-7B-Instruct \
  --train_type lora \
  --dataset CodyWhy/mh-sharegpt-20250820 \
  --bf16 true \
  --max_length 3072 \
  --packing true \
  --attn_impl flash_attn \
  --gradient_checkpointing true \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --save_steps 200 \
  --save_total_limit 3 \
  --logging_steps 20 \
  --output_dir output/mh-sft-7B-lora \
  --report_to swanlab \
  --swanlab_token ${SWANLAB_API_KEY} \
  --swanlab_project ${SWANLAB_PROJECT:-mh-sft} \
  --swanlab_mode ${SWANLAB_MODE:-cloud} \
  --swanlab_exp_name mh-sft-qwen2p5-7b-lora
```

### 性能提升
- **开启 packing**：显著提升训练吞吐，减少显存碎片
- **Flash Attention**：相比 SDPA 有 20-40% 的速度提升
- **显存优化**：更好的显存利用率，支持更大的 batch size

### 故障排除清单
- [ ] 检查 Python 版本兼容性
- [ ] 确认 PyTorch 和 CUDA 版本匹配
- [ ] 验证 GPU 架构支持（H100 支持 sm90）
- [ ] 测试网络代理连通性
- [ ] 验证 wheel 文件完整性
- [ ] 确认安装后的导入测试

### 参考链接
- [Flash Attention Prebuilt Wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels)
- [ms-swift 文档](https://swift.readthedocs.io/)
- [PyTorch Flash Attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
