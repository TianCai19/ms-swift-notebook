# MS-Swift 笔记库

欢迎来到 MS-Swift 学习与实践笔记库！这里记录了我在使用 ms-swift 进行大语言模型微调过程中的经验、配置和最佳实践。

## 🚀 快速开始

### 环境准备
- [环境与升级](notes/10-环境与工具.md) - 安装和配置 ms-swift
- [Flash Attention 安装](notes/50-FlashAttention安装.md) - 提升训练性能

### 训练指南
- [数据与SFT训练](notes/20-数据与SFT训练.md) - ShareGPT 数据格式和训练流程
- [SwanLab 集成](notes/30-SwanLab集成.md) - 实验监控和可视化

### 实验记录
- [mh-sharegpt SFT 实验](notes/40-实验报告-mh-sharegpt-sft.md) - 完整的实验报告和配置

## 📚 主要内容

### 核心功能
- **LoRA 微调**: 参数高效的模型微调方法
- **数据预处理**: ShareGPT 格式自动识别和处理
- **性能优化**: Flash Attention 和 packing 支持
- **实验追踪**: SwanLab 集成，实时监控训练过程

### 技术栈
- **框架**: ms-swift (ModelScope Swift)
- **模型**: Qwen2.5 系列 (7B/32B)
- **数据**: ShareGPT 格式对话数据
- **监控**: SwanLab 实验管理平台
- **硬件**: NVIDIA H100 GPU

## 🔧 本地开发

### 安装依赖
```bash
pip install -r requirements.txt
```

### 本地预览
```bash
mkdocs serve
```

### 构建文档
```bash
mkdocs build
```

## 📖 文档特性

- **响应式设计**: 支持桌面和移动设备
- **搜索功能**: 全文搜索，快速定位内容
- **代码高亮**: 支持多种编程语言
- **暗色主题**: 护眼的暗色模式
- **导航优化**: 清晰的层级结构

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个笔记库！

## 📄 许可证

本项目采用 MIT 许可证。

---

*最后更新: {{ git_revision_date_localized }}*
