# 医学影像课程智能助教 Agent (CLI MVP)

基于 RAG + 工具调用的命令行智能助教，面向"医学影像 + AI"交叉课程的学生。

## 📖 项目简介

本系统是一个智能助教 Agent，具备以下核心能力：

1. **知识库问答**：解析并索引课程 PDF 课件，支持语义检索
2. **差异化回答**：根据学生背景（CS/BME/零基础）提供个性化解答
3. **上下文对话**：支持连续多轮对话，处理代码、公式、医学概念混合问题
4. **Agent 工具调用**：自动调用 arXiv 论文搜索、网络搜索、结构化推理等工具
5. **问答日志**：记录对话历史，便于分析学生共性困难

## ✨ 功能特性

| 功能 | 说明 |
|------|------|
| 🔍 **知识库检索** | 搜索课程 PPT、教材内容，优先基于课程资料回答 |
| 📄 **arXiv 搜索** | 搜索学术论文，获取最新研究进展 |
| 🌐 **网络搜索** | DuckDuckGo 搜索，补充最新资讯和工具文档 |
| 🤔 **结构化推理** | 分步推理复杂问题，如调试代码、设计实验 |
| 👤 **学生画像** | 根据 CS/BME/零基础背景提供差异化回答 |
| 📝 **问答日志** | JSON Lines 格式记录，支持后续分析 |

## ✅ 环境要求

- **Python**: 3.10+
- **操作系统**: macOS / Linux / Windows
- **API**: 任意 OpenAI 兼容的 LLM 服务（OpenAI、DeepSeek、SiliconFlow 等）

## 🚀 快速开始

以下步骤默认你**已经位于本仓库的根目录**（克隆或解压源码后 `cd` 进入 `medical-imaging-ta-agent`）。若尚未获取源码，请先查看文末 [从 GitHub 获取源码](#从-github-获取源码)。

### 1. 创建虚拟环境（推荐）

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或: venv\Scripts\activate  # Windows
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 API 信息（**请勿将 `.env` 提交到 Git**）：

```bash
# 使用 OpenAI
LLM_API_KEY=sk-xxxxxxxx
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4o-mini

# 或使用 DeepSeek
# LLM_API_KEY=sk-xxxxxxxx
# LLM_BASE_URL=https://api.deepseek.com/v1
# LLM_MODEL_NAME=deepseek-chat

# Embedding 配置（可与 LLM 不同）
EMBEDDING_API_KEY=${LLM_API_KEY}
EMBEDDING_BASE_URL=${LLM_BASE_URL}
EMBEDDING_MODEL_NAME=text-embedding-3-small
```

### 4. 放入课件 PDF

将课程 PDF 课件放入 `data/raw/` 目录：

```bash
mkdir -p data/raw
cp /path/to/your/lectures/*.pdf data/raw/
```

### 5. 构建知识库

```bash
python setup.py
```

首次构建会：
- 解析所有 PDF 提取文本
- 切分为语义块
- 生成向量嵌入
- 构建 FAISS 索引

### 6. 启动使用

```bash
python main.py
```

首次启动会引导你配置学生画像（背景、兴趣等）。

### 从 GitHub 获取源码

```bash
git clone https://github.com/Flexibility06/medical-imaging-ta-agent.git
cd medical-imaging-ta-agent
```

也可在仓库页面下载 Release 中的 Source code 压缩包；该归档默认**不包含**根目录下的 `test_*.py` 等开发自检脚本（完整功能请使用 `git clone` 获取全部文件）。

## 💬 使用方法

### 基本对话

直接输入问题即可与智能助教对话：

```
你: 什么是 U-Net 网络？
助手: U-Net 是一种专门用于生物医学图像分割的卷积神经网络...
```

### 特殊命令

| 命令 | 说明 |
|------|------|
| `/help` | 显示帮助信息 |
| `/clear` | 清空对话历史 |
| `/profile` | 查看/修改学生画像 |
| `/status` | 查看当前状态（会话 ID、背景、工具列表） |
| `/exit` | 退出程序 |

### 启动选项

```bash
python main.py --help

# 重置学生画像
python main.py --reset

# 跳过画像配置（使用默认）
python main.py --no-profile
```

## ⚙️ 配置说明

### 环境变量 (.env)

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_API_KEY` | LLM API 密钥 | 必填 |
| `LLM_BASE_URL` | API 基础 URL | https://api.openai.com/v1 |
| `LLM_MODEL_NAME` | 模型名称 | gpt-4o-mini |
| `EMBEDDING_API_KEY` | Embedding API 密钥（可选） | 使用 LLM_API_KEY |
| `EMBEDDING_BASE_URL` | Embedding API URL（可选） | 使用 LLM_BASE_URL |
| `EMBEDDING_MODEL_NAME` | Embedding 模型 | text-embedding-3-small |
| `EMBEDDING_BATCH_SIZE` | Embedding 批次大小 | 64 |
| `LLM_TIMEOUT` | API 超时时间（秒） | 60 |
| `LLM_MAX_RETRIES` | 最大重试次数 | 3 |

### 工具配置 (config/tools_config.yaml)

```yaml
tools:
  search_course_knowledge_base:
    enabled: true          # 启用知识库检索
    default_top_k: 5       # 默认返回结果数

  search_arxiv:
    enabled: true          # 启用 arXiv 搜索
    default_max_results: 5

  web_search:
    enabled: true          # 启用网络搜索
    default_max_results: 5
    rate_limit_delay: 1.0  # 搜索间隔（秒）

  sequential_thinking:
    enabled: true          # 启用结构化推理
    max_thoughts: 20       # 最大推理步数
```

## 📚 知识库管理

### 添加新课件

1. 将新的 PDF 文件放入 `data/raw/`
2. 重新构建知识库：

```bash
python setup.py --force
```

### 查看知识库状态

```bash
python -c "
from knowledge_base.vector_store import VectorStore
from config.settings import settings
store = VectorStore.load(settings.INDEX_PATH, settings.CHUNKS_PATH)
print(store.get_stats())
"
```

## 🔧 工具配置

各工具的作用和使用场景：

### search_course_knowledge_base
- **作用**：搜索课程知识库
- **使用场景**：学生提问涉及课程内容时**优先**使用

### search_arxiv
- **作用**：搜索学术论文
- **使用场景**：需要最新研究进展、论文参考时
- **提示**：建议使用英文关键词获得更好结果

### web_search
- **作用**：互联网搜索
- **使用场景**：查找最新资讯、工具文档、教程
- **注意**：需要网络连接，有速率限制

### sequential_thinking
- **作用**：结构化多步推理
- **使用场景**：复杂问题分析（调试代码、设计实验方案）

## 🔌 MCP 扩展（可选）

MCP (Model Context Protocol) 是 Anthropic 推出的标准化工具协议。

### 安装 MCP 支持

```bash
pip install mcp
```

### 配置 MCP Server

编辑 `config/tools_config.yaml`：

```yaml
mcp:
  enabled: true
  servers:
    - name: "sequential-thinking"
      command: "npx"
      args: ["-y", "@anthropic/mcp-sequential-thinking"]
      env: {}
```

### 常用 MCP Server

- `@anthropic/mcp-sequential-thinking` - 结构化推理
- `@anthropic/mcp-web-search` - 网络搜索
- 更多参考: [MCP Servers](https://github.com/modelcontextprotocol/servers)

## 🛠️ 开发者指南

### 项目结构

```
project-root/
├── config/               # 配置
│   ├── settings.py       # 全局配置
│   ├── prompts.py        # System Prompt 模板
│   └── tools_config.yaml # 工具配置
├── data/                 # 数据
│   ├── raw/              # PDF 课件
│   ├── processed/        # 解析缓存
│   └── index/            # FAISS 索引
├── knowledge_base/       # 知识库构建
│   ├── pdf_parser.py
│   ├── chunker.py
│   ├── embedder.py
│   └── vector_store.py
├── agent/                # Agent 核心
│   ├── student_profile.py
│   ├── chat_engine.py    # Agent Loop
│   └── response_formatter.py
├── tools/                # 工具系统
│   ├── base.py
│   ├── arxiv_search.py
│   ├── web_search.py
│   ├── sequential_thinking.py
│   ├── knowledge_base_search.py
│   └── mcp_bridge.py
├── utils/                # 工具函数
│   ├── llm_client.py     # LLM API 封装
│   └── logger.py
├── logs/                 # 问答日志
├── main.py               # CLI 入口
├── setup.py              # 知识库构建
├── test_*.py             # 开发用自检脚本（可选）
└── README.md
```

### 开始开发

```bash
# 1. 安装开发依赖
pip install -r requirements.txt

# 2. 验证 LLM 连接
python test_llm.py

# 3. 验证知识库
python test_retrieval.py "你的查询"

# 4. 验证工具系统
python test_tools.py

# 5. 验证对话引擎
python test_chat.py
```

## ❓ 常见问题

### Q: 启动时提示 "配置不完整"
A: 请检查 `.env` 文件是否存在且包含 `LLM_API_KEY` 和 `LLM_BASE_URL`。

### Q: 知识库检索不到内容
A:
1. 检查 `data/raw/` 是否有 PDF 文件
2. 运行 `python setup.py` 构建索引
3. 运行 `python test_retrieval.py "查询"` 验证

### Q: 工具调用失败
A:
1. 检查网络连接
2. 查看日志 `logs/chat_*.jsonl`
3. 检查 `config/tools_config.yaml` 中工具是否启用

### Q: 如何更换 LLM 模型？
A: 修改 `.env` 文件中的 `LLM_MODEL_NAME` 和对应的 API 配置。

### Q: 日志文件在哪里？
A: 日志保存在 `logs/chat_YYYY-MM-DD.jsonl`，按日期命名。
