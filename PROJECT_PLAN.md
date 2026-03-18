# 多工具 ReAct Agent 项目计划

> 一个基于 LangGraph + 手写 ReAct 的多工具智能 Agent，集成 YouTube/播客总结、个性化学习辅导等工具能力，使用本地 Qwen 模型，提供 Web UI 交互界面。

## 一、项目架构

```
┌─────────────────────────────────────────────┐
│                  Web UI (Gradio)             │
├─────────────────────────────────────────────┤
│               FastAPI 后端服务               │
├─────────────────────────────────────────────┤
│          ReAct Agent 核心 (LangGraph)        │
│  ┌─────────┐  ┌──────────┐  ┌────────────┐ │
│  │ 推理引擎 │  │ 工具路由  │  │  记忆管理   │ │
│  └─────────┘  └──────────┘  └────────────┘ │
├─────────────────────────────────────────────┤
│                  工具层 (Tools)              │
│  ┌──────────┐ ┌──────────┐ ┌────────────┐  │
│  │YouTube   │ │学习辅导   │ │ 网页搜索    │  │
│  │内容总结   │ │助手      │ │            │  │
│  └──────────┘ └──────────┘ └────────────┘  │
│  ┌──────────┐ ┌──────────┐ ┌────────────┐  │
│  │代码执行   │ │天气查询   │ │ 文件读写    │  │
│  └──────────┘ └──────────┘ └────────────┘  │
├─────────────────────────────────────────────┤
│           本地 LLM (Qwen via Ollama)        │
└─────────────────────────────────────────────┘
```

## 二、技术栈

| 类别       | 选型                        | 说明                          |
| ---------- | --------------------------- | ----------------------------- |
| LLM        | Qwen2.5 (Ollama 本地部署)   | 通过 OpenAI 兼容 API 调用     |
| Agent 框架 | LangGraph + 手写 ReAct 循环 | 展示对 Agent 原理的深入理解   |
| 后端       | FastAPI                     | 异步高性能，SSE 流式输出      |
| 前端       | Gradio                      | 快速搭建交互 UI               |
| 工具管理   | 自定义 Tool Registry        | 动态注册、统一接口            |
| 数据存储   | SQLite                      | 对话历史、用户偏好            |

## 三、分阶段实施计划

### Phase 1：基础框架搭建（核心骨架）

**目标**：跑通 ReAct 循环，能调用至少一个工具

#### 任务清单

- [ ] 1.1 项目初始化
  - 创建项目目录结构
  - 配置 `pyproject.toml` / `requirements.txt`
  - 配置 Ollama + Qwen 模型

- [ ] 1.2 手写 ReAct 核心逻辑
  - 实现 ReAct Prompt 模板（Thought → Action → Observation → ... → Final Answer）
  - 实现 LLM 调用封装（兼容 Ollama OpenAI API）
  - 实现输出解析器（解析 LLM 输出中的 Action/Action Input）

- [ ] 1.3 LangGraph 状态图
  - 定义 AgentState（messages, tool_calls, intermediate_steps）
  - 构建节点：reason → act → observe → 判断是否结束
  - 实现条件边（继续推理 or 返回结果）

- [ ] 1.4 工具注册机制
  - 定义 Tool 基类（name, description, parameters, execute）
  - 实现 ToolRegistry（注册、查找、列举）
  - 实现一个示例工具（如：计算器）验证流程

```
预期产出：命令行中输入问题 → Agent 思考 → 调用工具 → 返回答案
```

---

### Phase 2：核心工具开发

**目标**：实现 6 个实用工具，覆盖面试展示需要的多样性

#### 任务清单

- [ ] 2.1 YouTube/播客内容总结工具
  - 通过 yt-dlp 提取视频字幕/音频
  - 音频转文字（Whisper 或直接用字幕）
  - LLM 分段总结 + 要点提取
  - 支持输入：YouTube URL、本地音频文件

- [ ] 2.2 个性化学习辅导工具
  - 用户画像管理（学习水平、感兴趣领域）
  - 知识点讲解（根据用户水平调整深度）
  - 生成练习题 + 批改反馈
  - 学习进度追踪

- [ ] 2.3 网页搜索工具
  - 集成 DuckDuckGo 搜索（无需 API Key）
  - 搜索结果摘要提取
  - 返回结构化结果（标题、链接、摘要）

- [ ] 2.4 代码执行工具
  - 沙箱化 Python 代码执行（subprocess + 超时控制）
  - 捕获 stdout/stderr
  - 安全限制（禁止危险操作）

- [ ] 2.5 天气查询工具
  - 调用免费天气 API（wttr.in 或 OpenWeatherMap）
  - 返回结构化天气信息

- [ ] 2.6 文件读写工具
  - 读取本地文件内容
  - 写入/保存文件
  - 路径安全校验（限制在工作目录内）

```
预期产出：Agent 能根据用户意图自动选择合适的工具
```

---

### Phase 3：记忆与上下文管理

**目标**：让 Agent 具备多轮对话记忆和上下文理解能力

#### 任务清单

- [ ] 3.1 短期记忆（对话历史）
  - 滑动窗口管理对话上下文
  - Token 数量控制，避免超出模型上下文窗口

- [ ] 3.2 长期记忆（持久化）
  - SQLite 存储对话历史
  - 关键信息提取与持久化存储
  - 用户偏好记录

- [ ] 3.3 LangGraph 中的状态管理
  - Checkpointer 实现对话状态持久化
  - 支持对话恢复

```
预期产出：Agent 能记住之前的对话内容，提供连贯的多轮交互
```

---

### Phase 4：后端 API 服务

**目标**：将 Agent 封装为 HTTP 服务，支持流式输出

#### 任务清单

- [ ] 4.1 FastAPI 服务搭建
  - POST `/api/chat` — 发送消息，SSE 流式返回
  - GET `/api/history` — 获取对话历史
  - GET `/api/tools` — 获取可用工具列表
  - DELETE `/api/history` — 清空对话

- [ ] 4.2 SSE 流式输出
  - 实时推送 Agent 思考过程（Thought/Action/Observation）
  - 最终答案流式输出

- [ ] 4.3 错误处理与日志
  - 统一异常处理
  - 结构化日志（loguru）
  - 工具调用超时保护

```
预期产出：可通过 HTTP 接口与 Agent 交互，思考过程实时可见
```

---

### Phase 5：Web UI

**目标**：提供直观的用户交互界面

#### 任务清单

- [ ] 5.1 Gradio 聊天界面
  - 对话式交互
  - 显示 Agent 思考链（Thought → Action → Observation 展开/折叠）
  - 工具调用状态指示

- [ ] 5.2 功能面板
  - 可用工具列表展示
  - 对话历史侧边栏
  - 用户设置（学习偏好等）

```
预期产出：完整可演示的 Web 应用
```

---

### Phase 6：优化与面试准备

**目标**：打磨细节，准备面试演示

#### 任务清单

- [ ] 6.1 性能优化
  - Prompt 优化（减少 token 消耗，提高工具选择准确率）
  - 工具调用并行化（无依赖的工具可并行执行）

- [ ] 6.2 可观测性
  - Agent 决策过程可视化
  - 工具调用耗时统计

- [ ] 6.3 测试
  - 核心逻辑单元测试
  - 工具集成测试
  - 端到端场景测试

- [ ] 6.4 文档与演示
  - README.md（项目介绍、架构图、快速开始）
  - 准备 3-5 个演示场景脚本
  - 面试讲解要点准备

---

## 四、目录结构

```
llm_agent/
├── pyproject.toml
├── requirements.txt
├── README.md
├── config/
│   └── settings.py          # 配置管理
├── src/
│   ├── __init__.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── react.py          # 手写 ReAct 核心逻辑
│   │   ├── graph.py          # LangGraph 状态图定义
│   │   ├── prompt.py         # Prompt 模板
│   │   └── parser.py         # 输出解析器
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py           # Tool 基类 + Registry
│   │   ├── youtube_summary.py
│   │   ├── study_tutor.py
│   │   ├── web_search.py
│   │   ├── code_executor.py
│   │   ├── weather.py
│   │   └── file_rw.py
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── short_term.py     # 对话窗口管理
│   │   └── long_term.py      # SQLite 持久化
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py            # FastAPI 应用
│   │   ├── routes.py         # 路由定义
│   │   └── schemas.py        # 请求/响应模型
│   └── ui/
│       ├── __init__.py
│       └── gradio_app.py     # Gradio 界面
├── tests/
│   ├── test_react.py
│   ├── test_tools.py
│   └── test_api.py
├── data/
│   └── agent.db              # SQLite 数据库
└── logs/
    └── agent.log
```

## 五、面试亮点提炼

| 面试考察点         | 本项目对应展示                                      |
| ------------------ | --------------------------------------------------- |
| Agent 原理理解     | 手写 ReAct 循环，不是黑盒调用框架                   |
| 工程架构能力       | 分层设计：Agent → Tools → Memory → API → UI         |
| 框架使用能力       | LangGraph 状态图、条件路由、Checkpointer            |
| 工具设计能力       | 统一 Tool 接口、动态注册、6 种异构工具              |
| 流式交互           | SSE 实时推送思考过程                                 |
| 本地模型部署       | Ollama + Qwen，展示私有化部署能力                   |
| 生产级意识         | 错误处理、超时保护、安全沙箱、结构化日志            |

## 六、关键依赖

```txt
langgraph>=0.2.0
langchain-core>=0.3.0
langchain-ollama>=0.2.0
fastapi>=0.115.0
uvicorn>=0.32.0
gradio>=5.0.0
yt-dlp>=2024.0.0
duckduckgo-search>=6.0.0
loguru>=0.7.0
pydantic>=2.0.0
httpx>=0.27.0
```
