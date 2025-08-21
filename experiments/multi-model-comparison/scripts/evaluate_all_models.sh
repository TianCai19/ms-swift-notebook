#!/bin/bash

# 多模型批量评估脚本
# 用于对比多个开源模型在心理学任务上的表现

set -e  # 遇到错误立即退出

# 配置变量
export OUTPUT_BASE_DIR="./output"
export EXPORT_BASE_DIR="./export"
export EVAL_RESULTS_DIR="./evaluation_results"
export LOG_DIR="./logs"

# 创建必要的目录
mkdir -p $EXPORT_BASE_DIR
mkdir -p $EVAL_RESULTS_DIR
mkdir -p $LOG_DIR

# 设置环境变量（解决代理问题）
export NO_PROXY=127.0.0.1,localhost,::1
export no_proxy=$NO_PROXY

# 模型配置数组（与训练脚本保持一致）
declare -a models=(
    "qwen2.5-7b:Qwen/Qwen2.5-7B-Instruct:qwen2_5"
    "llama3.1-8b:meta-llama/Meta-Llama-3.1-8B-Instruct:llama"
    "chatglm3-6b:THUDM/chatglm3-6b:chatglm3"
    "internlm2-7b:internlm/internlm2-chat-7b:internlm2"
    "baichuan2-7b:baichuan-inc/Baichuan2-7B-Chat:baichuan2"
)

# 评估数据集配置
declare -a eval_datasets=(
    "ceval:ceval"
    "cmmlu:cmmlu"
    "pceb:pceb_mcq"
    "psyqa:psyqa"
)

# 模型导出函数
export_model() {
    local model_name=$1
    local base_model_id=$2
    local model_type=$3
    
    echo "📦 开始导出模型: $model_name"
    
    # 查找最新的检查点
    local output_dir="$OUTPUT_BASE_DIR/$model_name-sft"
    local latest_checkpoint=""
    
    if [ -d "$output_dir" ]; then
        # 查找最新的 checkpoint 目录
        latest_checkpoint=$(find "$output_dir" -name "checkpoint-*" -type d | sort -V | tail -1)
        
        if [ -z "$latest_checkpoint" ]; then
            echo "❌ 错误: 未找到检查点目录: $output_dir"
            return 1
        fi
        
        echo "🔍 找到检查点: $latest_checkpoint"
    else
        echo "❌ 错误: 训练输出目录不存在: $output_dir"
        return 1
    fi
    
    # 设置导出目录
    local export_dir="$EXPORT_BASE_DIR/$model_name-sft"
    local log_file="$LOG_DIR/${model_name}_export.log"
    
    echo "📁 导出目录: $export_dir"
    echo "📝 日志文件: $log_file"
    
    # 开始导出
    local start_time=$(date +%s)
    
    CUDA_VISIBLE_DEVICES=6 \
    swift export \
        --ckpt_dir "$latest_checkpoint" \
        --merge_lora true \
        --safe_serialization true \
        --max_shard_size 2GB \
        --output_dir "$export_dir" \
        2>&1 | tee "$log_file"
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo "✅ 模型 $model_name 导出完成！"
    echo "⏱️ 导出耗时: $((duration / 60)) 分钟"
    echo "📁 导出目录: $export_dir"
    echo "---"
}

# 模型部署函数
deploy_model() {
    local model_name=$1
    local base_model_id=$2
    local model_type=$3
    
    echo "🚀 开始部署模型: $model_name"
    
    # 检查导出目录
    local export_dir="$EXPORT_BASE_DIR/$model_name-sft"
    if [ ! -d "$export_dir" ]; then
        echo "❌ 错误: 导出目录不存在: $export_dir"
        echo "   请先运行导出步骤"
        return 1
    fi
    
    # 设置服务名称
    local service_name="${model_name}-sft"
    local log_file="$LOG_DIR/${model_name}_deploy.log"
    
    echo "🔧 服务名称: $service_name"
    echo "📝 日志文件: $log_file"
    
    # 启动部署（后台运行）
    echo "🔄 启动模型服务..."
    
    CUDA_VISIBLE_DEVICES=6 \
    swift deploy \
        --model_dir "$export_dir" \
        --infer_backend lmdeploy \
        --served_model_name "$service_name" \
        --tp 1 \
        --cache-max-entry-count 0 \
        --session-len 4096 \
        2>&1 | tee "$log_file" &
    
    local deploy_pid=$!
    echo "📊 部署进程 PID: $deploy_pid"
    
    # 等待服务启动
    echo "⏳ 等待服务启动..."
    sleep 30
    
    # 检查服务状态
    if curl --noproxy '*' -s "http://127.0.0.1:8000/v1/models" > /dev/null; then
        echo "✅ 模型 $model_name 部署成功！"
        echo "🌐 服务地址: http://127.0.0.1:8000"
        echo "📊 模型名称: $service_name"
    else
        echo "❌ 模型 $model_name 部署失败"
        echo "📝 请检查日志: $log_file"
        return 1
    fi
    
    echo "---"
}

# 模型评估函数
evaluate_model() {
    local model_name=$1
    local base_model_id=$2
    local model_type=$3
    
    echo "📊 开始评估模型: $model_name"
    
    # 设置服务名称
    local service_name="${model_name}-sft"
    local results_dir="$EVAL_RESULTS_DIR/$model_name"
    
    mkdir -p "$results_dir"
    
    # 遍历评估数据集
    for dataset_config in "${eval_datasets[@]}"; do
        IFS=':' read -r dataset_name dataset_id <<< "$dataset_config"
        
        echo "🔍 评估数据集: $dataset_name ($dataset_id)"
        
        local result_file="$results_dir/${dataset_name}_results.json"
        local log_file="$LOG_DIR/${model_name}_${dataset_name}_eval.log"
        
        # 运行 EvalScope 评估
        echo "🔄 运行 EvalScope 评估..."
        
        evalscope eval \
            --model "$service_name" \
            --api-url "http://127.0.0.1:8000/v1" \
            --api-key "EMPTY" \
            --eval-type "service" \
            --datasets "$dataset_id" \
            --limit 100 \
            --output-file "$result_file" \
            2>&1 | tee "$log_file"
        
        if [ $? -eq 0 ]; then
            echo "✅ 数据集 $dataset_name 评估完成"
            echo "📊 结果文件: $result_file"
        else
            echo "❌ 数据集 $dataset_name 评估失败"
            echo "📝 请检查日志: $log_file"
        fi
        
        echo "---"
    done
    
    echo "🎉 模型 $model_name 评估完成！"
    echo "📁 结果目录: $results_dir"
    echo "---"
}

# 批量评估函数
batch_evaluate() {
    echo "🎯 多模型批量评估开始"
    echo "📅 开始时间: $(date)"
    echo "🔧 使用 GPU: CUDA_VISIBLE_DEVICES=6"
    echo "📊 评估结果目录: $EVAL_RESULTS_DIR"
    echo "---"
    
    # 记录开始时间
    local total_start_time=$(date +%s)
    
    # 遍历所有模型进行评估
    for model_config in "${models[@]}"; do
        IFS=':' read -r model_name base_model_id model_type <<< "$model_config"
        
        echo "🔄 准备评估模型: $model_name"
        
        # 检查训练输出
        if [ ! -d "$OUTPUT_BASE_DIR/$model_name-sft" ]; then
            echo "⚠️  模型 $model_name 未训练，跳过评估"
            echo "---"
            continue
        fi
        
        # 步骤 1: 导出模型
        echo "📦 步骤 1: 导出模型"
        export_model "$model_name" "$base_model_id" "$model_type"
        
        # 步骤 2: 部署模型
        echo "🚀 步骤 2: 部署模型"
        deploy_model "$model_name" "$base_model_id" "$model_type"
        
        # 步骤 3: 评估模型
        echo "📊 步骤 3: 评估模型"
        evaluate_model "$model_name" "$base_model_id" "$model_type"
        
        # 停止当前服务，准备下一个模型
        echo "🛑 停止当前服务..."
        pkill -f "swift deploy" || true
        sleep 10
        
        echo "---"
    done
    
    # 计算总耗时
    local total_end_time=$(date +%s)
    local total_duration=$((total_end_time - total_start_time))
    
    echo "🎉 所有模型评估完成！"
    echo "📅 结束时间: $(date)"
    echo "⏱️ 总耗时: $((total_duration / 3600)) 小时 $(((total_duration % 3600) / 60)) 分钟"
    echo "📊 评估结果保存在: $EVAL_RESULTS_DIR"
    echo "📝 评估日志保存在: $LOG_DIR"
    
    # 生成评估总结
    echo "📋 评估总结:"
    for model_config in "${models[@]}"; do
        IFS=':' read -r model_name base_model_id model_type <<< "$model_config"
        local results_dir="$EVAL_RESULTS_DIR/$model_name"
        if [ -d "$results_dir" ]; then
            echo "  ✅ $model_name: $results_dir"
        else
            echo "  ❌ $model_name: 评估失败或未完成"
        fi
    done
}

# 错误处理
error_handler() {
    local exit_code=$?
    echo "❌ 评估过程中发生错误，退出码: $exit_code"
    echo "📝 请检查日志文件: $LOG_DIR"
    
    # 清理进程
    pkill -f "swift deploy" || true
    
    exit $exit_code
}

# 设置错误处理
trap error_handler ERR

# 检查依赖
check_dependencies() {
    echo "🔍 检查评估环境..."
    
    # 检查 swift 命令
    if ! command -v swift &> /dev/null; then
        echo "❌ 错误: 未找到 swift 命令"
        echo "   请先安装 ms-swift: pip install 'ms-swift[all]'"
        exit 1
    fi
    
    # 检查 evalscope 命令
    if ! command -v evalscope &> /dev/null; then
        echo "❌ 错误: 未找到 evalscope 命令"
        echo "   请先安装 EvalScope: pip install evalscope"
        exit 1
    fi
    
    # 检查 GPU
    if ! nvidia-smi &> /dev/null; then
        echo "❌ 错误: 未找到 NVIDIA GPU"
        echo "   请确保在 GPU 环境中运行此脚本"
        exit 1
    fi
    
    # 检查端口占用
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
        echo "⚠️  警告: 端口 8000 已被占用"
        echo "   请确保端口 8000 可用"
    fi
    
    echo "✅ 环境检查通过"
    echo "---"
}

# 运行主函数
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    check_dependencies
    batch_evaluate "$@"
fi
