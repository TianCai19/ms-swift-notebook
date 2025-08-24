#!/bin/bash

# 多模型批量训练脚本
# 用于对比多个开源模型在心理学任务上的表现

set -e  # 遇到错误立即退出

# 配置变量
export SWANLAB_TOKEN="P2PI8lAMWL1fF90kZAoXj"
export SWANLAB_PROJECT="multi-model-psychology"
export DATASET_ID="CodyWhy/mh-sharegpt-20250820"
export OUTPUT_BASE_DIR="./output"
export LOG_DIR="./logs"

# 创建必要的目录
mkdir -p $OUTPUT_BASE_DIR
mkdir -p $LOG_DIR

# 设置环境变量（解决代理问题）
export NO_PROXY=127.0.0.1,localhost,::1
export no_proxy=$NO_PROXY

# 模型配置数组
declare -a models=(
    "qwen2.5-7b:Qwen/Qwen2.5-7B-Instruct:qwen2_5:qwen2_5"
    "llama3.1-8b:meta-llama/Meta-Llama-3.1-8B-Instruct:llama:llama3"
    "chatglm3-6b:ZhipuAI/chatglm3-6b:chatglm3:chatglm3"
    "internlm2-7b:internlm/internlm2-chat-7b:internlm2:internlm2"
    "baichuan2-7b:baichuan-inc/Baichuan2-7B-Chat:baichuan2:baichuan2"
)

# 训练函数
train_model() {
    local model_name=$1
    local model_id=$2
    local model_type=$3
    local template=$4
    
    echo "🚀 开始训练模型: $model_name"
    echo "📋 模型ID: $model_id"
    echo "🔧 模型类型: $model_type"
    echo "📝 模板: $template"
    
    # 设置输出目录
    local output_dir="$OUTPUT_BASE_DIR/$model_name-sft"
    local log_file="$LOG_DIR/${model_name}_training.log"
    
    # 创建输出目录
    mkdir -p $output_dir
    
    # 根据模型大小调整批次大小
    local batch_size=8
    local grad_accum_steps=4
    
    if [[ $model_name == *"8b"* ]] || [[ $model_name == *"13b"* ]]; then
        batch_size=6
        grad_accum_steps=6
    fi
    
    echo "⚙️ 训练参数: batch_size=$batch_size, grad_accum_steps=$grad_accum_steps"
    
    # 开始训练
    local start_time=$(date +%s)
    
    CUDA_VISIBLE_DEVICES=6 \
    swift sft \
        --model $model_id \
        --model_type $model_type \
        --template $template \
        --train_type lora \
        --dataset $DATASET_ID \
        --bf16 true \
        --max_length 3072 \
        --packing true \
        --gradient_checkpointing true \
        --per_device_train_batch_size $batch_size \
        --gradient_accumulation_steps $grad_accum_steps \
        --learning_rate 2e-4 \
        --num_train_epochs 1 \
        --warmup_ratio 0.1 \
        --save_steps 200 \
        --save_total_limit 3 \
        --logging_steps 20 \
        --output_dir $output_dir \
        --report_to swanlab \
        --swanlab_token $SWANLAB_TOKEN \
        --swanlab_project $SWANLAB_PROJECT \
        --swanlab_mode cloud \
        --swanlab_exp_name "${model_name}-sft-psychology" \
        --attn_impl flash_attn \
        2>&1 | tee $log_file
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo "✅ 模型 $model_name 训练完成！"
    echo "⏱️ 总耗时: $((duration / 60)) 分钟"
    echo "📁 输出目录: $output_dir"
    echo "📝 日志文件: $log_file"
    echo "---"
}

# 主训练循环
main() {
    echo "🎯 多模型心理学任务训练开始"
    echo "📅 开始时间: $(date)"
    echo "🔧 使用 GPU: CUDA_VISIBLE_DEVICES=6"
    echo "📊 SwanLab 项目: $SWANLAB_PROJECT"
    echo "📚 数据集: $DATASET_ID"
    echo "---"
    
    # 记录开始时间
    local total_start_time=$(date +%s)
    
    # 遍历所有模型进行训练
    for model_config in "${models[@]}"; do
        IFS=':' read -r model_name model_id model_type template <<< "$model_config"
        
        echo "🔄 准备训练模型: $model_name"
        
        # 检查是否已经训练过
        if [ -d "$OUTPUT_BASE_DIR/$model_name-sft" ]; then
            echo "⚠️  模型 $model_name 已经存在，跳过训练"
            echo "   如需重新训练，请删除目录: $OUTPUT_BASE_DIR/$model_name-sft"
            echo "---"
            continue
        fi
        
        # 开始训练
        train_model "$model_name" "$model_id" "$model_type" "$template"
        
        # 训练间隔，避免资源冲突
        echo "⏳ 等待 30 秒后继续下一个模型..."
        sleep 30
    done
    
    # 计算总耗时
    local total_end_time=$(date +%s)
    local total_duration=$((total_end_time - total_start_time))
    
    echo "🎉 所有模型训练完成！"
    echo "📅 结束时间: $(date)"
    echo "⏱️ 总耗时: $((total_duration / 3600)) 小时 $(((total_duration % 3600) / 60)) 分钟"
    echo "📊 训练结果保存在: $OUTPUT_BASE_DIR"
    echo "📝 训练日志保存在: $LOG_DIR"
    
    # 生成训练总结
    echo "📋 训练总结:"
    for model_config in "${models[@]}"; do
        IFS=':' read -r model_name model_id model_type template <<< "$model_config"
        local output_dir="$OUTPUT_BASE_DIR/$model_name-sft"
        if [ -d "$output_dir" ]; then
            echo "  ✅ $model_name: $output_dir"
        else
            echo "  ❌ $model_name: 训练失败或未完成"
        fi
    done
}

# 错误处理
error_handler() {
    local exit_code=$?
    echo "❌ 训练过程中发生错误，退出码: $exit_code"
    echo "📝 请检查日志文件: $LOG_DIR"
    exit $exit_code
}

# 设置错误处理
trap error_handler ERR

# 检查依赖
check_dependencies() {
    echo "🔍 检查训练环境..."
    
    # 检查 swift 命令
    if ! command -v swift &> /dev/null; then
        echo "❌ 错误: 未找到 swift 命令"
        echo "   请先安装 ms-swift: pip install 'ms-swift[all]'"
        exit 1
    fi
    
    # 检查 GPU
    if ! nvidia-smi &> /dev/null; then
        echo "❌ 错误: 未找到 NVIDIA GPU"
        echo "   请确保在 GPU 环境中运行此脚本"
        exit 1
    fi
    
    # 检查显存
    local gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ "$gpu_memory" -lt 70000 ]; then
        echo "⚠️  警告: GPU 显存可能不足 (当前: ${gpu_memory}MB, 建议: 80GB+)"
        echo "   建议使用 H100 或 A100 进行训练"
    fi
    
    echo "✅ 环境检查通过"
    echo "---"
}

# 运行主函数
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    check_dependencies
    main "$@"
fi
