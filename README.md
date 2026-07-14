# LangChain 1.2 Learning Tutorial
## 项目简介
本仓库是个人循序渐进系统学习 LangChain 1.2 新版架构的实操代码仓库，严格按照官方学习路线分章节实践，所有代码均可本地运行。
项目采用规范工程化配置，通过 `.gitignore` 隔离本地密钥文件，杜绝API Key泄露，配套无密钥模板 `env.example`。

## 技术栈
- Python 3.10+
- LangChain 1.2（全新架构，区分0.x旧版本API）
- 多厂商大模型：DeepSeek、智谱AI、阿里云百炼、OpenRouter、CloseAI等
- 本地Ollama部署大模型
- LangSmith 链路追踪、可视化调试
- Jupyter Notebook 交互式代码演示
- Chroma 向量数据库、RAG检索增强、Agent智能体开发

## 本地环境部署

### 1. 克隆仓库

```bash
git clone https://github.com/clfnbnbnb-coder/langchain1.2-learning-tutorial.git
cd langchain1.2-learning-tutorial
```

### 2. 安装项目全部依赖

```bash
pip install -r requirements.txt
```

### 3. API 密钥配置

将根目录 `env.example` 文件复制一份，重命名为 `.env`。

前往 DeepSeek、智谱 AI、阿里云百炼、LangSmith 等平台申请对应 API Key，填入等号后方。

`.env` 文件已被 `.gitignore` 屏蔽，仅本地保留，不会上传至代码仓库，保障密钥安全。

## 完整学习路线（仓库章节一一对应）
### 01 LangChain 概述
✅ 已完成
对应目录：`chapter01_summary`
学习内容：LangChain1.2整体架构、核心组件、开发流程介绍

### 02 模型的创建与调用
✅ 已完成
对应目录：`chapter02_model`
学习内容：在线/本地模型初始化、同步/异步调用、流式输出、批量调用、参数配置

### 03 LangSmith的使用
✅ 已完成
对应目录：`chapter03_langsmith`
学习内容：链路日志监控、调用异常排查、项目区分配置

### 04 Message与提示词模板
✅ 已完成
对应目录：`chapter04_messages_prompt`
学习内容：对话消息结构、ChatPromptTemplate、动态变量、内容块封装

### 05 Tools 工具开发
✅ 已完成
对应目录：`chapter05_tools`
学习内容：创建tool、自定义工具、工具参数定义与调用、工具的应用案例、tool_choice的使用

### 06 结构化输出
✅ 已完成
对应目录：`chapter06-structured_output`
学习内容：Pydantic约束LLM输出、强制JSON格式化返回

### 07 智能体 Agent
⏳ 待学习
对应目录：`chapter07_agent`
学习内容：ReAct推理逻辑、自主工具调用、多轮决策智能体

### 08 中间件 Middleware
⏳ 待学习
对应目录：`chapter08_middleware`
学习内容：请求拦截、日志埋点、调用前后预处理

### 09 上下文与记忆 Memory
⏳ 待学习
对应目录：`chapter09_memory`
学习内容：对话持久化、窗口记忆、多轮会话上下文管理

### 10 RAG 检索增强生成
⏳ 待学习
对应目录：`chapter10_rag`
学习内容：文档加载、文本分割、向量入库、相似度检索、问答生成