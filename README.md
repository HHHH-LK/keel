# Keel

> 让大语言模型不偏航。

**Keel** 是一个轻量级、可演进的 **Agent Runtime**。

名字取自船的「龙骨」——那根贯穿船体、提供稳定性与航向控制的结构。这正是这个项目的内核:不是再写一层 LLM SDK 封装,也不是把所有权力丢给模型,而是用**上下文工程、分层记忆、工具协议、验证闭环、TDD 硬门和审批边界**,把模型框在一条「能持续、可控地完成真实任务」的轨道上。

模型只是系统中的推理核心。真正的智能体能力,来自模型与上下文、记忆、规则、工具和执行流程之间的协同。

---

## 项目特性

- **多形态 Agent**:Simple / FunctionCall / ReAct / Plan-Solve / Reflection 五种执行范式,共享同一套 Runtime
- **上下文工程编排层**:`ContextEngine` 作为编排器,记忆层是"源",通过保底 + 共享池预算策略组装每轮上下文,只丢弃不截断
- **分层记忆系统**:L0 Playbook 卡片 / L1 工作记忆 / L2 摘要 / 冷存储 / 向量检索 / **时态知识图谱**,并支持**用户层 + 项目层双层记忆与跨项目提升**
- **验证-重试 Harness**:`verify/` 提供 spec → 执行 → 检查 → 残差 → 收敛/重规划的在线闭环,让 Agent 不止"生成"还能"自证完成"
- **真 TDD 模式**:`tdd/` 独立出题 + 硬红门(实现前必须先看到测试失败)+ 哈希防篡改,杜绝"先写实现再补测试"
- **离线评测 Bench**:`bench/` 作为开发"方向盘",用例驱动地回归 Agent 行为
- **可插拔工具生态**:文件读写编辑、目录浏览、Glob/Grep、Bash 执行、计算器、Web 搜索、记忆召回/写入等内置工具
- **动态审批**:危险操作(如 Bash 命令)触发三态审批,完整改动落 scrollback、紧凑框常驻选项
- **现代化 CLI**:基于 `prompt-toolkit` + `rich` 的 Claude Code 风交互,Slash 命令、Markdown 渲染、主题、状态栏、钉底 todo 面板
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
my_agent_llms/            # Python 包(品牌名为 Keel)
├── core/           # Agent 基类、LLM 客户端、Message、Hook、Config
├── agents/         # 五种 Agent 形态(Simple / FunctionCall / ReAct / Plan-Solve / Reflection)
├── context/        # 上下文工程编排层(ContextEngine)
├── memory/         # 分层记忆系统
│   ├── playbook/   # L0:用户偏好/规则卡片(SQLite 持久化)
│   ├── working.py  # L1:近期对话工作记忆
│   ├── summary.py  # L2:LLM 摘要压缩
│   ├── semantic.py # 长期语义检索
│   ├── cold.py     # 冷存储抽象
│   ├── kg.py       # 时态知识图谱(扁平时态事实表)
│   ├── kg_reconcile.py / kg_vocab.py  # 冲突消解 + 词表
│   ├── user_layer.py / promotion_ledger.py  # 用户层 + 跨项目提升
│   ├── conflict.py # 冲突检测(相似度 + LLM 双策略)
│   ├── recall_buffer.py / seed_score.py / maintenance.py
│   └── manager.py  # 统一调度器
├── tools/          # 工具注册与执行链
│   └── builtin/    # read/write/edit/list_dir/glob/grep/bash/calculator/search/recall/remember
├── verify/         # 验证-重试 harness(spec/loop/checkers/residual/convergence/replan)
├── tdd/            # 真 TDD 模式(classify/test_author/gates/runner/orchestrator)
├── planning/       # 任务 todo 规划(todo.py)
├── bench/          # 离线评测 harness(case/runner/scorer/report)
├── cli/            # CLI 界面(theme/prompt/chat_view/permission/scrollback_renderer/...)
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

#### Context
`ContextEngine` 是上下文的编排器:把各记忆层当作"源",按**保底 + 共享池预算**策略组装每轮送入模型的上下文,预算不足时**只丢弃、不截断**,保证片段完整性。

#### Memory
`MemoryManager` 统一调度多层记忆,通过 `MemoryConfig` 自由配置:

- **L0 Playbook**:长期偏好/规则卡片,具备 active / dormant / archived 生命周期与冲突检测
- **L1 Working**:近期对话窗口
- **L2 Summary**:LLM 驱动的滚动摘要
- **时态知识图谱**:扁平时态事实表,内核是"时态 + 冲突消解";检索走三路混合 + RRF
- **双层记忆**:用户层 / 项目层分离,支持跨项目提升(promotion ledger)
- **冷存储**:JSONL(零依赖)或 SQLite(可恢复)
- **冲突检测**:相似度匹配 + LLM 仲裁
- **诚实协议**:通过 `HONESTY_CONTRACT` 约束模型不凭印象回答历史信息,触发 `[NEEDS_RECALL: ...]` 自动召回

#### Verify
`verify/` 把"完成"从模型的自我声明变成可检验的闭环:spec 定义目标 → 执行 → checkers 检查 → residual 计算残差 → convergence 判收敛,未达标则 replan 重试。

#### TDD
`tdd/` 实现"真 TDD 模式":由独立出题器先写测试,经**硬红门**确认实现前测试确实失败,并用**哈希防篡改**锁住测试,杜绝"先写实现再补测试"或"边改边标完成"。

#### Tools
- 文件相关:`read_file` / `write_file` / `edit_file` / `list_dir` / `glob` / `grep`
- 执行能力:`bash`(危险命令触发审批)
- 通用能力:`calculator` / `search`(SerpAPI)/ `recall` / `remember`(记忆检索与写入)
- 工具通过 `ToolRegistry` 注册,支持同步与异步执行链

#### Workspace & 审批
文件类工具不直接操作磁盘,而是在 `Workspace` 沙箱中暂存修改,经审阅后导出,具备 deny set 防误操作。危险操作通过 CLI 的**动态三态审批**确认:完整改动写入 scrollback(可上滑查看),紧凑框常驻选项。

---

## 设计原则

- **轻量优先**:核心结构清晰,避免早期堆叠重依赖
- **模块解耦**:LLM / Prompt / Tool / Memory / Context / Loop / Channel 分层
- **可控优先**:每项能力都有明确边界,而不是把所有权力丢给模型
- **可验证优先**:完成靠检验,不靠模型自报
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

如果你关注的不只是"让模型回答问题",而是"让模型在规则、工具和记忆协同下持续、可控地完成任务",**Keel** 就是为这类系统设计的起点。
