# 部署说明

本文档说明如何将 MS-Swift 笔记库部署为在线文档站点。

## 🚀 自动部署（推荐）

### 1. 配置 GitHub Pages

1. 进入你的 GitHub 仓库
2. 点击 `Settings` → `Pages`
3. 在 `Source` 部分选择 `GitHub Actions`
4. 确保仓库是公开的（或你有 GitHub Pro 计划）

### 2. 推送代码

将代码推送到 GitHub 后，GitHub Actions 会自动：
- 安装依赖
- 构建文档
- 部署到 GitHub Pages

```bash
git add .
git commit -m "Add MkDocs documentation"
git push origin main
```

### 3. 查看部署状态

- 进入 `Actions` 标签页查看构建状态
- 构建成功后，文档会自动部署到 `https://yourusername.github.io/ms-swift-notebook`

## 🔧 本地预览

### 安装依赖
```bash
pip install -r requirements.txt
```

### 启动本地服务器
```bash
mkdocs serve
```

然后在浏览器中访问 `http://127.0.0.1:8000`

### 构建静态文件
```bash
mkdocs build
```

生成的静态文件在 `site/` 目录中。

## 📝 自定义配置

### 修改站点信息

编辑 `mkdocs.yml` 文件：

```yaml
site_name: 你的站点名称
site_description: 你的站点描述
site_url: https://yourusername.github.io/your-repo-name
repo_name: your-repo-name
repo_url: https://github.com/yourusername/your-repo-name
```

### 修改导航结构

在 `mkdocs.yml` 的 `nav` 部分调整文档结构：

```yaml
nav:
  - 首页: index.md
  - 你的分类:
    - 文档1: path/to/doc1.md
    - 文档2: path/to/doc2.md
```

### 添加新文档

1. 在 `notes/` 目录下创建新的 `.md` 文件
2. 在 `mkdocs.yml` 的 `nav` 中添加链接
3. 推送代码，自动部署

## 🎨 主题定制

### 颜色方案

修改 `mkdocs.yml` 中的 `palette` 部分：

```yaml
theme:
  palette:
    - scheme: default
      primary: blue  # 主色调
      accent: green  # 强调色
```

### 功能开关

启用/禁用主题功能：

```yaml
theme:
  features:
    - navigation.tabs      # 导航标签
    - navigation.sections  # 导航分组
    - search.highlight     # 搜索高亮
    - content.code.copy    # 代码复制
```

## 🔍 搜索配置

MkDocs 内置全文搜索功能，无需额外配置。搜索支持：
- 标题搜索
- 内容搜索
- 代码搜索
- 实时高亮

## 📱 移动端支持

文档站点完全响应式，支持：
- 桌面浏览器
- 平板设备
- 手机设备
- 触摸操作

## 🚨 常见问题

### 构建失败
- 检查 `requirements.txt` 中的依赖版本
- 确认 Markdown 文件语法正确
- 查看 GitHub Actions 日志

### 页面不显示
- 确认 GitHub Pages 已启用
- 检查 `mkdocs.yml` 配置是否正确
- 等待几分钟让部署完成

### 样式问题
- 清除浏览器缓存
- 检查主题配置
- 确认 CSS 文件正确加载

## 📚 更多资源

- [MkDocs 官方文档](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [GitHub Pages 文档](https://pages.github.com/)
- [GitHub Actions 文档](https://docs.github.com/en/actions)
