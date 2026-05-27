# my_agent_llms

`my_agent_llms` 是一个轻量级、可演进的 Agent 框架。目标不是再写一层 LLM SDK 封装,而是构建一个可运行、可扩展的 **Agent Runtime**:让大语言模型在提示词、工具、记忆、任务循环和安全边界的协同下,持续完成真实场景中的复杂任务。

模型只是系统中的推理核心,真正的智能体能力来自模型与上下文、工具、记忆、规则和执行流程之间的协同。

---

## 项目特性

- **多形态 Agent**:Simple / FunctionCall / ReAct / Plan-Solve / Reflection 五种执行范式,共享同一套 Runtime
- **分层记忆系统**:L0 Playbook 卡片 / L1 工作记忆 / L2 摘要 / 冷存储 / 向量检索 / 知识图谱,可配置可插拔
- **可插拔工具生态**:文件读写编辑、目录浏览、附件导入导出、计算器、Web 搜索、记忆召回等内置工具
- **现代化 CLI**:基于 `prompt-toolkit` + `rich` 的 Warp/Vercel 风格交互,支持 Slash 命令、Markdown 渲染、主题、状态栏
- **Workspace 沙箱**:文件类工具在受控目录下执行,支持 pending edits 的审阅与一键导出
- **Hook 机制**:在 Agent 响应前后注入自定义逻辑(诚实协议、记忆检索补全等)
- **完整测试**:单元测试 + 集成测试覆盖核心模块

---

## 快速开始

环境要求:Python ≥ 3.13,推荐使用 [uv](https://github.com/astral-sh/uv) 管理依赖。

```bash
# 1. 安装依赖
uv sync

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 LLM_API_KEY / LLM_MODEL_ID / LLM_BASE_URL (可选 SERPAPI_API_KEY)

# 3. 启动 CLI
uv run python chat.py
```

进入 CLI 后可用的 Slash 命令包括 `/help`、`/config`、`/memory`、`/clear` 等。即使没配 API Key 也能进入,通过 `/config key` 现场补上即可。

---

## 架构总览

```
my_agent_llms/
├── core/           # Agent 基类、LLM 客户端、Message、Hook、Config
├── agents/         # 五种 Agent 形态(Simple / FunctionCall / ReAct / Plan-Solve / Reflection)
├── memory/         # 分层记忆系统
│   ├── playbook/   # L0:用户偏好/规则卡片(SQLite 持久化)
│   ├── working.py  # L1:近期对话工作记忆
│   ├── summary.py  # L2:LLM 摘要压缩
│   ├── semantic.py # 长期语义检索
│   ├── cold.py     # 冷存储抽象
│   ├── backends/   # JSONL / SQLite / InMemory 后端
│   ├── embeddings.py
│   ├── conflict.py # 冲突检测(相似度 + LLM 双策略)
│   ├── kg.py       # 知识图谱
│   └── manager.py  # 统一调度器
├── tools/          # 工具注册与执行链
│   ├── builtin/    # 内置工具(file/search/calc/recall/...)
│   ├── registry.py
│   ├── chain.py
│   └── async_executor.py
├── cli/            # CLI 界面(theme/prompt/chat_view/help_view/...)
├── workspace/      # 文件操作沙箱
└── test/           # 单元测试 + 集成测试
chat.py             # CLI 入口
```

### 核心模块说明

#### Agents
所有 Agent 继承自 `core/agent.py` 的基类,统一管理 `system_prompt`、`memory`、`hooks` 与 LLM 调用。

| Agent | 适用场景 |
|---|---|
| `MySimpleAgent` | 单轮问答,最小开销 |
| `MyFunctionCallAgent` | 原生 Tool Calling,适合工具密集型任务 |
| `MyReActAgent` | 思考-行动-观察循环,适合多步推理 |
| `MyPlanSolveAgent` | 先规划再分步执行,适合复杂任务拆解 |
| `MyReflectionAgent` | 输出后自我反思修正,适合质量敏感场景 |

#### Memory
`MemoryManager` 统一调度多层记忆,通过 `MemoryConfig` 自由配置:

- **L0 Playbook**:长期偏好/规则卡片,具备 active / dormant / archived 生命周期与冲突检测
- **L1 Working**:近期对话窗口
- **L2 Summary**:LLM 驱动的滚动摘要
- **冷存储**:JSONL(零依赖)或 SQLite(可恢复)
- **向量检索**:InMemory(快)或 SQLite(持久)
- **冲突检测**:相似度匹配 + LLM 仲裁
- **诚实协议**:通过 `HONESTY_CONTRACT` 约束模型不凭印象回答历史信息,触发 `[NEEDS_RECALL: ...]` 自动召回

#### Tools
- 文件相关:`read_file` / `write_file` / `edit_file` / `list_dir` / `attach_file` / `export_file` / `pending_edits`
- 通用能力:`calculator` / `search`(SerpAPI)/ `recall`(记忆检索)
- 工具通过 `ToolRegistry` 注册,支持同步与异步执行链

#### Workspace
文件类工具不会直接操作磁盘,而是在 `Workspace` 沙箱中暂存修改,通过 `pending_edits` 审阅后由 `export_file` 导出,具备 deny set 防误操作。

---

## 设计原则

- **轻量优先**:核心结构清晰,避免早期堆叠重依赖
- **模块解耦**:LLM / Prompt / Tool / Memory / Loop / Channel 分层
- **可控优先**:每项能力都有明确边界,而不是把所有权力丢给模型
- **可扩展优先**:基础模块为未来替换和增强预留接口
- **面向运行时**:关注"持续执行任务",而不仅是"生成文本"

---

## 开发

```bash
# 运行测试
uv run pytest

# 仅跑某个模块
uv run pytest my_agent_llms/test/test_playbook.py -v
```

详细的设计文档位于 `docs/superpowers/specs/` 与 `docs/superpowers/plans/`。

---

## 演进方向

- 更稳定的 Prompt 编排与上下文管理
- 更规范的工具协议与生命周期
- 更丰富的长期记忆结构与跨会话持久化
- 更可靠的任务循环与异常恢复
- 更清晰的权限控制与沙箱隔离
- 更灵活的多渠道接入(Web / 消息平台 / 自动化任务)
- 面向多 Agent 协作的基础设施

---

如果你关注的不只是"让模型回答问题",而是"让模型在规则、工具和记忆协同下持续完成任务",`my_agent_llms` 就是为这类系统设计的起点。
