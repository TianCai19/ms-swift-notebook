#!/bin/bash

# 多模型对比实验启动脚本
# 完整的端到端实验流程

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 配置变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_NAME="multi-model-psychology-comparison"
START_TIME=$(date +%s)

# 创建实验目录
EXPERIMENT_DIR="$SCRIPT_DIR/results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EXPERIMENT_DIR"

print_info "🚀 启动多模型对比实验"
print_info "📅 开始时间: $(date)"
print_info "📁 实验目录: $EXPERIMENT_DIR"
print_info "🔧 脚本目录: $SCRIPT_DIR"

# 检查环境
check_environment() {
    print_info "🔍 检查实验环境..."
    
    # 检查 Python 环境
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未安装"
        exit 1
    fi
    
    # 检查 GPU
    if ! nvidia-smi &> /dev/null; then
        print_error "NVIDIA GPU 未找到"
        exit 1
    fi
    
    # 检查显存
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ "$GPU_MEMORY" -lt 70000 ]; then
        print_warning "GPU 显存可能不足 (当前: ${GPU_MEMORY}MB, 建议: 80GB+)"
    else
        print_success "GPU 显存充足: ${GPU_MEMORY}MB"
    fi
    
    # 检查依赖包
    print_info "检查 Python 依赖..."
    python3 -c "import torch, transformers, swift" 2>/dev/null || {
        print_error "缺少必要的 Python 包"
        print_info "请安装: pip install torch transformers 'ms-swift[all]'"
        exit 1
    }
    
    print_success "环境检查通过"
}

# 准备实验环境
prepare_environment() {
    print_info "🔧 准备实验环境..."
    
    # 设置环境变量
    export NO_PROXY="127.0.0.1,localhost,::1"
    export no_proxy="$NO_PROXY"
    
    # 创建必要的目录
    mkdir -p "$EXPERIMENT_DIR"/{logs,output,export,evaluation_results,human_judge_results}
    
    # 复制配置文件
    cp -r "$SCRIPT_DIR/configs" "$EXPERIMENT_DIR/"
    
    print_success "环境准备完成"
}

# 阶段 1: 模型训练
run_training() {
    print_info "🎯 阶段 1: 开始模型训练..."
    
    cd "$EXPERIMENT_DIR"
    
    # 运行训练脚本
    if bash "$SCRIPT_DIR/scripts/train_all_models.sh"; then
        print_success "所有模型训练完成"
    else
        print_error "模型训练失败"
        return 1
    fi
}

# 阶段 2: 模型评估
run_evaluation() {
    print_info "📊 阶段 2: 开始模型评估..."
    
    cd "$EXPERIMENT_DIR"
    
    # 运行评估脚本
    if bash "$SCRIPT_DIR/scripts/evaluate_all_models.sh"; then
        print_success "所有模型评估完成"
    else
        print_error "模型评估失败"
        return 1
    fi
}

# 阶段 3: Human Judge 测试
run_human_judge() {
    print_info "🧠 阶段 3: 启动 Human Judge 测试..."
    
    cd "$EXPERIMENT_DIR"
    
    # 检查依赖
    python3 -c "import gradio, pandas, requests" 2>/dev/null || {
        print_warning "缺少 Human Judge 依赖包，跳过此阶段"
        print_info "如需运行，请安装: pip install gradio pandas requests"
        return 0
    }
    
    # 启动 Human Judge 系统
    print_info "启动 Human Judge Web 界面..."
    python3 "$SCRIPT_DIR/scripts/human_judge_test.py" &
    HUMAN_JUDGE_PID=$!
    
    print_success "Human Judge 系统已启动 (PID: $HUMAN_JUDGE_PID)"
    print_info "🌐 访问地址: http://localhost:7860"
    print_info "按 Ctrl+C 停止实验"
    
    # 等待用户手动停止
    wait $HUMAN_JUDGE_PID
}

# 生成实验报告
generate_report() {
    print_info "📝 生成实验报告..."
    
    cd "$EXPERIMENT_DIR"
    
    # 创建实验报告
    cat > "experiment_report.md" << EOF
# 多模型心理学任务对比实验报告

## 实验信息
- **实验名称**: $EXPERIMENT_NAME
- **开始时间**: $(date -d @$START_TIME)
- **结束时间**: $(date)
- **实验目录**: $EXPERIMENT_DIR

## 实验流程
1. ✅ 环境检查
2. ✅ 模型训练
3. ✅ 模型评估
4. ✅ Human Judge 测试

## 模型列表
- Qwen2.5-7B-Instruct
- Llama3.1-8B-Instruct
- ChatGLM3-6B
- InternLM2-7B-Chat
- Baichuan2-7B-Chat

## 结果目录
- **训练输出**: \`output/\`
- **模型导出**: \`export/\`
- **评估结果**: \`evaluation_results/\`
- **Human Judge**: \`human_judge_results/\`
- **日志文件**: \`logs/\`

## 下一步
1. 查看 SwanLab 训练曲线
2. 分析评估结果
3. 收集 Human Judge 评分
4. 生成对比分析报告

---
*报告生成时间: $(date)*
EOF
    
    print_success "实验报告已生成: $EXPERIMENT_DIR/experiment_report.md"
}

# 主函数
main() {
    # 错误处理
    trap 'print_error "实验过程中发生错误，退出码: $?"; exit 1' ERR
    
    # 检查环境
    check_environment
    
    # 准备环境
    prepare_environment
    
    # 阶段 1: 模型训练
    if run_training; then
        print_success "阶段 1 完成"
    else
        print_error "阶段 1 失败，停止实验"
        exit 1
    fi
    
    # 阶段 2: 模型评估
    if run_evaluation; then
        print_success "阶段 2 完成"
    else
        print_error "阶段 2 失败，停止实验"
        exit 1
    fi
    
    # 阶段 3: Human Judge 测试
    if run_human_judge; then
        print_success "阶段 3 完成"
    else
        print_warning "阶段 3 跳过或失败"
    fi
    
    # 生成报告
    generate_report
    
    # 计算总耗时
    END_TIME=$(date +%s)
    TOTAL_DURATION=$((END_TIME - START_TIME))
    
    print_success "🎉 实验完成！"
    print_info "⏱️ 总耗时: $((TOTAL_DURATION / 3600)) 小时 $(((TOTAL_DURATION % 3600) / 60)) 分钟"
    print_info "📁 结果保存在: $EXPERIMENT_DIR"
    print_info "📊 查看 SwanLab: https://swanlab.ai"
}

# 显示帮助信息
show_help() {
    cat << EOF
多模型对比实验启动脚本

用法: $0 [选项]

选项:
    -h, --help      显示此帮助信息
    -t, --train     仅运行训练阶段
    -e, --eval      仅运行评估阶段
    -h, --human     仅运行 Human Judge 测试
    -f, --full      运行完整实验流程 (默认)

示例:
    $0                    # 运行完整实验
    $0 --train           # 仅运行训练
    $0 --eval            # 仅运行评估
    $0 --human           # 仅运行 Human Judge

注意事项:
    1. 确保已安装所有必要的依赖包
    2. 确保 GPU 显存充足 (建议 80GB+)
    3. 确保网络连接正常，能够访问 ModelScope
    4. 训练过程可能需要数小时，请耐心等待

EOF
}

# 解析命令行参数
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    -t|--train)
        print_info "仅运行训练阶段"
        check_environment
        prepare_environment
        run_training
        generate_report
        ;;
    -e|--eval)
        print_info "仅运行评估阶段"
        check_environment
        prepare_environment
        run_evaluation
        generate_report
        ;;
    -h|--human)
        print_info "仅运行 Human Judge 测试"
        check_environment
        prepare_environment
        run_human_judge
        ;;
    -f|--full|"")
        main
        ;;
    *)
        print_error "未知选项: $1"
        show_help
        exit 1
        ;;
esac
