# LangChain 1.2 Learning Tutorial

## 一、项目概述

本项目是个人系统学习 **LangChain 1.2** 新版架构的实操代码仓库，按照由浅入深的学习路线分章节实践，所有代码均在本地环境验证可运行。

### 创建背景

LangChain 1.x 相比 0.x 版本进行了架构级重构，API 发生了较大变化。为系统掌握新版架构的核心组件与开发模式，作者按照官方学习路线逐章编写实践代码，形成本教程仓库。

### 核心目标

- 跟随官方学习路线，完整实践 LangChain 1.2 的核心功能
- 通过 Jupyter Notebook 交互式演示，实现"即学即练"
- 覆盖从模型调用到 RAG 检索增强生成的完整知识链路

### 目标用户

- 具备 Python 基础、希望系统学习 LangChain 1.2 的开发者
- 已了解 LLM 基本概念，需要上手实操的工程师
- 从 LangChain 0.x 迁移到 1.x 版本的学习者

### 项目性质声明

> 本项目为 **个人学习教程仓库**，非生产级应用，不提供对外 Web 服务或前端界面。代码以 Jupyter Notebook 演示为主，适合按章节阅读和运行练习。

---

## 二、教程覆盖范围

本项目共 10 个章节，涵盖 LangChain 1.2 从基础概念到 RAG 实战的完整学习路线。各章节内容如下：

| 章节 | 目录 | 学习主题 | 状态 |
|------|------|----------|------|
| 01 LangChain 概述 | `chapter01_summary` | LangChain 1.2 整体架构、核心组件、开发流程介绍 | ✅ 已完成 |
| 02 模型的创建与调用 | `chapter02_model` | 在线/本地模型初始化、同步/异步调用、流式输出、批量调用、参数配置 | ✅ 已完成 |
| 03 LangSmith 的使用 | `chapter03_langsmith` | 链路日志监控、调用异常排查、项目区分配置 | ✅ 已完成 |
| 04 Message 与提示词模板 | `chapter04_messages_prompt` | 对话消息结构、ChatPromptTemplate、动态变量、内容块封装 | ✅ 已完成 |
| 05 Tools 工具开发 | `chapter05-tools` | 自定义工具、`@tool` 装饰器、工具调用、tool_choice、实践总结 | ✅ 已完成 |
| 06 结构化输出 | `chapter06-structured_output` | Pydantic / TypedDict / JSON Schema / dataclass 四种模式、结果校验 | ✅ 已完成 |
| 07 智能体 Agent | `chapter07_Agent` | Agent 基本用法、高级用法、ToolStrategy、错误处理、流式输出、实战助手 | ✅ 已完成 |
| 08 中间件 Middleware | `chapter08-Middleware` | 内置中间件、自定义 hook 函数、Node/Wrap-style hooks、执行顺序 | ✅ 已完成 |
| 09 上下文与记忆 Memory | `chapter09-memory` | 短期记忆、记忆治理策略、长期记忆 API 与 Agent、写入时机控制 | ✅ 已完成 |
| 10 RAG 检索增强生成 | `chapter10-RAG` | 文档加载、文本切分、嵌入模型、Milvus 向量库、客服知识库实战 | ✅ 已完成 |

> **说明**：章节状态反映 **教程内容的编写完成度**，即对应目录下已有完整的 Notebook 代码并验证可运行。

---

## 三、技术架构

### 3.1 仓库组织结构

本项目采用 **线性章节式** 组织结构，每个章节为独立目录，互不依赖，可按需单独学习：

```
langchain1.2_tutorial/
├── chapter01_summary/           # 01 LangChain 概述
├── chapter02_model/             # 02 模型的创建与调用
├── chapter03_langsmith/         # 03 LangSmith 的使用
├── chapter04_messages_prompt/   # 04 Message 与提示词模板
├── chapter05-tools/             # 05 Tools 工具开发
├── chapter06-structured_output/ # 06 结构化输出
├── chapter07_Agent/             # 07 智能体 Agent
├── chapter08-Middleware/        # 08 中间件 Middleware
├── chapter09-memory/            # 09 上下文与记忆 Memory
├── chapter10-RAG/               # 10 RAG 检索增强生成
├── asset/                       # 测试数据文件（文档加载、格式转换等）
│   └── load/                    # 各格式测试文件：txt/csv/json/pdf/docx/md/html/py
├── todo_workspace/              # 试验性测试代码（含 pytest 示例）
├── knowledge.txt                # RAG 章节示例知识库（Atguigu 助手客服 KB）
├── requirements.txt             # 核心依赖（前 9 章所需）
├── requirements_full.txt        # 完整依赖（含 RAG 章节的向量库、文档解析等）
├── env.example                  # 环境变量模板（无密钥）
├── .gitignore                   # Git 忽略配置（含 .env 屏蔽规则）
└── README.md                    # 项目说明文档
```

### 3.2 学习路径架构

教程内容按知识深度分为四个层次，建议按顺序学习：

```
入门层（ch01-03）         核心层（ch04-06）         进阶层（ch07-09）        实战层（ch10）
┌─────────────┐         ┌─────────────┐         ┌─────────────┐        ┌─────────────┐
│ LangChain   │         │ Message &   │         │ Agent       │        │ RAG         │
│ 概述        │ ──────►  │ Prompt      │ ──────► │ Middleware  │ ─────► │ 文档加载     │
│ 模型调用     │         │ 结构化输出    │         │ Memory      │        │ 向量检索     │
│ LangSmith   │         │ Tools       │         │             │        │ 问答生成     │
└─────────────┘         └─────────────┘         └─────────────┘        └─────────────┘
    基础概念                核心组件                智能编排                综合应用
```

### 3.3 依赖分层架构

项目依赖按功能分为三层：

| 层级 | 依赖文件 | 覆盖范围 | 说明 |
|------|----------|----------|------|
| 核心依赖层 | `requirements.txt` | 第 1-9 章 | LangChain 生态、模型服务商 SDK、LangGraph、MCP 等 |
| RAG 扩展层 | `requirements_full.txt` | 第 10 章 | Milvus 向量库、PyTorch、Unstructured 文档解析等 |
| 配置层 | `env.example` | 全局 | 多服务商 API Key、LangSmith 追踪配置 |

### 3.4 多模型服务商抽象

项目通过 LangChain 的统一 `ChatModel` 抽象层接入多家大模型服务商，实现代码层面的模型无缝切换：

```
                    ┌─────────────────────────────────┐
                    │     LangChain ChatModel 抽象     │
                    └──────────────┬──────────────────┘
                                   │
          ┌────────────┬───────────┼───────────┬────────────┐
          ▼            ▼           ▼           ▼            ▼
     DeepSeek      智谱 AI     阿里百炼    OpenRouter    本地 Ollama
  (langchain-    (ZhipuAI)  (DashScope)  (OpenRouter)  (langchain-
   deepseek)                                            ollama)
```

---

## 四、技术栈详情

以下技术栈严格对应 `requirements.txt` 与 `requirements_full.txt` 中的实际依赖。

### 4.1 核心依赖（requirements.txt）

#### LangChain 核心框架

| 库名 | 版本 | 用途 |
|------|------|------|
| `langchain` | 1.2.12 | LangChain 主体框架 |
| `langchain-core` | 1.2.18 | 核心抽象（Runnable、ChatModel 等） |
| `langchain-community` | 0.4.1 | 社区集成组件 |
| `langchain-classic` | 1.0.2 | 经典链式调用兼容模块 |
| `langchain-text-splitters` | 1.1.1 | 文本切分器（RAG 章节使用） |
| `langchain-experimental` | 0.4.1 | 实验性组件 |

#### LangGraph / Agent 编排

| 库名 | 版本 | 用途 |
|------|------|------|
| `langgraph` | 1.1.2 | Agent 状态图、多步骤工作流编排 |
| `langgraph-prebuilt` | 1.0.8 | 预置 Agent 工具 |
| `langgraph-checkpoint` | 4.0.1 | 检查点持久化 |
| `langgraph-checkpoint-postgres` | 3.0.5 | PostgreSQL 检查点（Memory 章节） |
| `langgraph-sdk` | 0.3.9 | LangGraph SDK |

#### MCP / FastMCP 集成

| 库名 | 版本 | 用途 |
|------|------|------|
| `mcp` | 1.27.0 | MCP 协议核心 |
| `fastmcp` | 3.2.4 | MCP Server 快速构建 |
| `langchain-mcp-adapters` | 0.2.1 | LangChain 与 MCP 适配器 |

#### 大模型服务商 SDK

| 库名 | 版本 | 用途 |
|------|------|------|
| `langchain-deepseek` | 1.0.1 | DeepSeek 模型接入 |
| `langchain-openai` | 1.1.11 | OpenAI 兼容接口接入 |
| `langchain-anthropic` | 1.3.4 | Anthropic Claude 接入 |
| `langchain-openrouter` | 0.1.0 | OpenRouter 多模型路由 |
| `langchain-ollama` | 1.0.1 | 本地 Ollama 模型接入 |
| `openai` | 2.26.0 | OpenAI SDK |
| `anthropic` | 0.84.0 | Anthropic SDK |
| `dashscope` | 1.25.6 | 阿里云百炼（通义千问）SDK |
| `tencentcloud-sdk-python` | 3.1.86 | 腾讯云 SDK |
| `openrouter` | 0.7.11 | OpenRouter SDK |

#### 工具与搜索集成

| 库名 | 版本 | 用途 |
|------|------|------|
| `langchain-tavily` | 0.2.17 | Tavily 联网搜索工具 |

#### Web 服务 / API

| 库名 | 版本 | 用途 |
|------|------|------|
| `fastapi` | 0.135.1 | API 服务框架（Mock Server 用） |
| `uvicorn` | 0.46.0 | ASGI 服务器 |
| `sse-starlette` | 3.3.4 | SSE 流式通信 |
| `httpx` | 0.28.1 | 异步 HTTP 客户端 |
| `httpx-sse` | 0.4.3 | SSE 客户端支持 |
| `aiohttp` | 3.12.14 | 异步 HTTP |
| `requests` | 2.32.5 | HTTP 客户端 |

#### 配置管理 / 数据校验

| 库名 | 版本 | 用途 |
|------|------|------|
| `python-dotenv` | 1.2.1 | `.env` 文件读取 |
| `pydantic` | 2.12.5 | 数据校验、结构化输出 |
| `pydantic-settings` | 2.12.0 | 配置管理 |
| `orjson` | 3.11.7 | 高性能 JSON 序列化 |
| `jsonschema` | 4.26.0 | JSON Schema 校验 |
| `dataclasses-json` | 0.6.7 | dataclass JSON 序列化 |

#### 日志 / 调试辅助

| 库名 | 版本 | 用途 |
|------|------|------|
| `loguru` | 0.7.3 | 日志输出 |
| `rich` | 14.3.3 | 富文本终端输出 |
| `tenacity` | 9.1.4 | 重试机制 |
| `tqdm` | 4.67.3 | 进度条 |

#### 测试与持久化

| 库名 | 版本 | 用途 |
|------|------|------|
| `pytest` | 9.0.3 | 单元测试 |
| `psycopg[binary]` | 3.3.3 | PostgreSQL 驱动（Memory 章节） |
| `psycopg-pool` | 3.3.0 | PostgreSQL 连接池 |

#### 交互式开发

| 库名 | 版本 | 用途 |
|------|------|------|
| `jupyter` | 最新版 | 交互式 Notebook 环境 |

### 4.2 RAG 扩展依赖（requirements_full.txt 新增部分）

以下依赖仅在第 10 章 RAG 章节中使用，已从 `requirements.txt` 中注释分离：

| 类别 | 库名 | 版本 | 用途 |
|------|------|------|------|
| 向量数据库 | `langchain-milvus` | 0.3.3 | Milvus 向量库集成 |
| 向量数据库 | `pymilvus` | 2.6.12 | Milvus Python 客户端 |
| 向量数据库 | `SQLAlchemy` | 2.0.48 | SQL 数据源 |
| 文本处理 | `tiktoken` | 0.12.0 | Token 计数与切分 |
| 文本处理 | `numpy` | 2.4.4 | 数值计算 |
| 文本处理 | `scikit-learn` | 1.8.0 | 文本相似度计算 |
| 文本处理 | `nltk` | 3.9.4 | 自然语言处理工具 |
| HuggingFace | `transformers` | 5.3.0 | 本地模型与 Tokenizer |
| HuggingFace | `tokenizers` | 0.22.2 | 分词器 |
| 深度学习 | `torch` | 2.11.0 | PyTorch（CPU 版，Unstructured 依赖） |
| 深度学习 | `torchvision` | 0.26.0 | 视觉模型支持 |
| 文档解析 | `unstructured` | 0.20.6 | 多格式文档解析（PDF/Word/PPT/Excel 等） |
| 文档解析 | `pypdf` | 6.10.2 | PDF 解析 |
| 文档解析 | `python-docx` | 1.2.0 | Word 文档解析 |
| 文档解析 | `python-pptx` | 1.0.2 | PPT 解析 |
| 文档解析 | `openpyxl` | 3.1.5 | Excel 解析 |
| 文档解析 | `pandas` | 3.0.2 | 数据处理 |
| 文档解析 | `beautifulsoup4` | 4.14.3 | HTML 解析 |
| 文档解析 | `lxml` | 6.1.0 | XML/HTML 解析 |
| 文档解析 | `Markdown` | 3.10.2 | Markdown 解析 |

> **注意**：`requirements_full.txt` 不包含 `sentence-transformers`。如需加载本地 Embedding 模型，需单独安装匹配 CUDA 版本的 PyTorch。

---

## 五、安装部署步骤

### 5.1 前置条件

- **Python**：3.10 及以上版本
- **操作系统**：Windows / macOS / Linux 均可（本项目在 Windows 环境开发验证）
- **API Key**：至少需要一个 LLM 服务商的 API Key（推荐 DeepSeek，性价比高）
- **可选**：如需学习本地模型章节，需预装 [Ollama](https://ollama.com/)
- **可选**：如需学习 Memory 持久化章节，需预装 PostgreSQL 数据库

### 5.2 克隆仓库

```bash
git clone https://github.com/clfnbnbnb-coder/langchain1.2-learning-tutorial.git
cd langchain1.2-learning-tutorial
```

### 5.3 创建虚拟环境（推荐）

```bash
# 使用 venv
python -m venv venv

# 激活虚拟环境
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# macOS / Linux:
source venv/bin/activate
```

### 5.4 安装依赖

项目提供两套依赖文件，按需安装：

**方案 A：最小安装（前 9 章可用）**

```bash
pip install -r requirements.txt
```

**方案 B：完整安装（含 RAG 章节，依赖体积较大）**

```bash
pip install -r requirements_full.txt
```

> ⚠️ 完整安装包含 PyTorch、Unstructured 等大型依赖，下载体积约 2-3 GB。如仅需学习前 9 章，请使用方案 A。

### 5.5 配置 API 密钥

1. 将根目录 `env.example` 文件复制一份并重命名为 `.env`：

```bash
# Windows PowerShell
Copy-Item env.example .env

# macOS / Linux
cp env.example .env
```

2. 前往各平台申请对应的 API Key，填入 `.env` 文件等号后方：

| 配置项 | 获取来源 | 是否必需 |
|--------|----------|----------|
| `DEEPSEEK_API_KEY` | [DeepSeek 开放平台](https://platform.deepseek.com/) | 推荐 |
| `ZHIPUAI_API_KEY` | [智谱 AI 开放平台](https://open.bigmodel.cn/) | 可选 |
| `DASHSCOPE_API_KEY` | [阿里云百炼平台](https://dashscope.aliyuncs.com/) | 可选 |
| `OPENROUTER_API_KEY` | [OpenRouter](https://openrouter.ai/) | 可选 |
| `CLOSEAI_API_KEY` | OpenAI 代理服务 | 可选 |
| `LANGSMITH_API_KEY` | [LangSmith](https://smith.langchain.com/) | 第 3 章必需 |

3. `.env` 文件已被 `.gitignore` 屏蔽，仅本地保留，不会上传至代码仓库，保障密钥安全。

### 5.6 验证安装

运行以下命令验证 LangChain 安装成功：

```bash
python chapter01_summary/test.py
```

预期输出类似：

```
langchain1.2
1.2.12
```

---

## 六、使用指南

### 6.1 启动 Jupyter Notebook

```bash
# 在项目根目录执行
jupyter notebook
```

浏览器将自动打开 Jupyter 界面，按章节目录进入对应 Notebook。

### 6.2 推荐学习路径

**零基础路径（按顺序学习）**：

1. **第 1-3 章**：建立 LangChain 1.2 整体认知，掌握模型调用与 LangSmith 调试
2. **第 4-6 章**：掌握消息结构、提示词模板、工具开发、结构化输出
3. **第 7-9 章**：学习 Agent 智能体编排、中间件机制、记忆管理
4. **第 10 章**：综合应用 RAG 检索增强生成（需完整安装依赖）

**按主题路径（有基础者可跳读）**：

- 只学 Agent → 第 5、7 章
- 只学 RAG → 第 10 章（需先了解第 4 章提示词模板）
- 只学 Memory → 第 9 章

### 6.3 各章节使用说明

#### 第 1 章：LangChain 概述
运行 `chapter01_summary/test.py`，验证 LangChain 版本与安装。

#### 第 2 章：模型的创建与调用
按编号顺序打开 `chapter02_model/` 下的 Notebook：
- `01-model-init-online.ipynb`：在线模型初始化
- `03-model-init-ollama.ipynb`：本地 Ollama 模型（需预装 Ollama）
- `05-model-stream-batch.ipynb`：流式输出与批量调用

#### 第 3 章：LangSmith 的使用
需先在 `.env` 中配置 `LANGSMITH_API_KEY`，然后运行 `chapter03_langsmith/` 下的 Notebook，在 [LangSmith WebUI](https://smith.langchain.com/) 查看链路追踪。

#### 第 5-9 章：核心组件学习
各章节 Notebook 按编号顺序阅读即可，每章均从基础概念到高级用法递进。

#### 第 10 章：RAG 实战
- 需先完成 `requirements_full.txt` 的完整安装
- 需安装并启动 [Milvus](https://milvus.io/) 向量数据库
- `knowledge.txt` 为示例客服知识库（Atguigu 助手），供 RAG 检索使用

### 6.4 测试代码运行

`todo_workspace/` 目录包含示例测试代码，可使用 pytest 运行：

```bash
pytest todo_workspace/test_my_add.py
```

---

## 七、接口调用说明

> ⚠️ **说明**：本项目为个人学习教程，**不提供生产级 API 服务**。本章节汇总的是教程中调用到的外部 LLM 服务接口与 LangChain 框架核心抽象，用于学习参考。

### 7.1 LLM 服务商接口

教程中通过环境变量配置各服务商接口，所有服务商均兼容 OpenAI API 协议：

| 服务商 | Base URL | 环境变量 | 对应章节 |
|--------|----------|----------|----------|
| DeepSeek | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` | 第 2 章起 |
| 智谱 AI | `https://open.bigmodel.cn/api/paas/v4` | `ZHIPUAI_API_KEY` | 第 2 章起 |
| 阿里云百炼 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` | 第 2 章起 |
| OpenRouter | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | 第 2 章起 |
| CloseAI（OpenAI 代理） | `https://api.openai-proxy.org/v1` | `CLOSEAI_API_KEY` | 第 2 章起 |
| 本地 Ollama | `http://localhost:11434` | 无需 Key | 第 2 章 |

### 7.2 LangChain 核心抽象接口

教程中涉及的主要 LangChain 框架接口：

| 抽象接口 | 说明 | 对应章节 |
|----------|------|----------|
| `ChatModel` / `BaseChatModel` | 聊天模型统一接口，支持 `invoke` / `stream` / `batch` | 第 2 章 |
| `BaseMessage` / `HumanMessage` / `AIMessage` | 消息类型抽象 | 第 4 章 |
| `ChatPromptTemplate` | 聊天提示词模板 | 第 4 章 |
| `BaseTool` / `@tool` 装饰器 | 工具定义与调用抽象 | 第 5 章 |
| `with_structured_output()` | 结构化输出方法 | 第 6 章 |
| `create_agent()` / `StateGraph` | Agent 创建与状态图编排 | 第 7 章 |
| Middleware（中间件） | 请求拦截与预处理 | 第 8 章 |
| `BaseMemory` / Checkpointer | 记忆与持久化抽象 | 第 9 章 |
| `VectorStore` / `Embeddings` / `Retriever` | 向量存储与检索抽象 | 第 10 章 |

### 7.3 LangSmith 追踪接口

| 配置项 | 说明 |
|--------|------|
| `LANGSMITH_TRACING=true` | 启用链路追踪 |
| `LANGSMITH_ENDPOINT` | 追踪数据上报地址 |
| `LANGSMITH_API_KEY` | LangSmith API Key |
| `LANGSMITH_PROJECT` | 项目名称，用于区分不同项目的运行记录 |

### 7.4 教程中的 Mock Server

部分章节使用 Mock Server 模拟 LLM 服务，**仅供教学演示，非生产可用**：

| 文件 | 位置 | 用途 |
|------|------|------|
| `05-fake_server.py` | `chapter06-structured_output/` | 模拟 LLM 返回结构化数据 |
| `fake_deepseek_server.py` | `chapter08-Middleware/` | 模拟 DeepSeek API 响应 |

---

## 八、贡献规范

### 8.1 项目性质

本项目为个人学习记录仓库，欢迎提交 Issue 反馈错误或改进建议。

### 8.2 代码规范

- Python 代码遵循 PEP 8 规范
- Notebook 代码应包含中文注释说明
- 新增依赖需同步更新 `requirements.txt` 或 `requirements_full.txt`
- 禁止将 `.env` 文件或真实 API Key 提交至仓库

### 8.3 提交信息格式

```
<type>: <description>

类型说明：
- feat:   新增章节内容或功能
- fix:    修复代码错误
- docs:   文档更新
- refactor: 代码重构
- chore:  构建/依赖/配置变更
```

### 8.4 分支管理

- `main`：稳定分支，保持代码可运行
- 学习中新增内容建议在独立分支开发，验证通过后合并至 `main`

---

## 九、注意事项

### 9.1 API 费用

- 调用 DeepSeek、智谱 AI、阿里云百炼等 LLM 服务商接口 **会产生实际费用**
- 建议先使用免费额度或低成本模型（如 DeepSeek）进行练习
- 流式输出和批量调用会增加 Token 消耗，注意控制用量

### 9.2 密钥安全

- `.env` 文件已被 `.gitignore` 屏蔽，**切勿**将真实 API Key 提交至仓库
- `env.example` 仅作为模板，不含真实密钥
- 如不慎泄露 Key，请立即前往对应平台重置

### 9.3 依赖安装

- `requirements_full.txt` 包含 PyTorch 等大型依赖，安装体积较大
- Windows 环境下安装 `unstructured` 等文档解析库可能需要额外系统依赖
- 如遇安装失败，建议先使用 `requirements.txt` 最小安装

### 9.4 版本时效性

- 本项目基于 LangChain 1.2.x 版本，该版本迭代较快，部分 API 可能在后续版本中变更
- 如遇 API 不兼容问题，请参考 [LangChain 官方文档](https://python.langchain.com/) 最新说明

### 9.5 学习性质声明

- 本仓库代码为 **学习用途**，未经过生产级测试，不建议直接用于生产环境
- `knowledge.txt` 为示例知识库（Atguigu 助手客服 KB），仅供 RAG 检索练习使用
- `asset/load/` 目录下的测试文件仅供文档加载演示使用

### 9.6 已知限制

- 第 9 章 Memory 章节的持久化功能需要预装 PostgreSQL 数据库
- 第 10 章 RAG 章节需要预装 Milvus 向量数据库
- 第 2 章本地模型章节需要预装 Ollama 并下载对应模型
- Windows 环境下部分路径可能存在编码问题，建议使用英文路径
