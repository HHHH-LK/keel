# Keel

> 一个不会偏航的 AI 伙伴。

<p align="center">
  <img src="https://img.shields.io/badge/python-≥3.13-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/uv-managed-purple" alt="package manager">
</p>

Keel 是**一个 agent**——一个有长期记忆、会自证、懂分寸的 AI 伙伴。

名字取自船的「龙骨」:那根贯穿船体、撑住稳定性与航向的结构。大多数 agent 要么聊完就忘、要么任务没做完就嘴上说"完成了"、要么给把 `rm` 就敢闯祸。Keel 想成为相反的那一个——模型负责聪明,**她负责靠谱**:记得住、做得完、不闯祸。

---

## Table of Contents

- [🧠 她是真的"记得"](#她是真的记得不是把历史硬塞回去)
- [✅ 她会"自证完成"](#她会自证完成而不是嘴上说说)
- [🛡️ 她"先问再动手"](#她先问再动手不会替你闯祸)
- [🎛️ 上下文编排器](#她有个真正懂上下文工程的大脑)
- [🎨 五种思考方式](#她还会换着法子干活)
- [用起来什么效果](#用起来什么效果)
- [实际跑一段给你看](#实际跑一段给你看)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
- [What she's made of](#what-shes-made-of)
- [Her philosophy](#her-philosophy-design-principles)
- [Project Status](#project-status)
- [Contributing](#contributing)
- [License](#license)

## 🧠 她是真的"记得",不是把历史硬塞回去

很多 agent 靠把对话历史一股脑灌进 context 来"假装有记忆",窗口一满就忘、就乱。Keel 给的是一套**真正的分层记忆**:

- **L0 偏好卡片 / L1 工作窗口 / L2 滚动摘要 / 冷存储**,层层有职责,可配置可插拔;
- 一张**时态知识图谱**——内核是"时态 + 冲突消解":同一件事你前后说法变了,她知道哪条**现在**有效,而不是把矛盾的事实一起记着;
- **用户层 + 项目层双层记忆**,好经验还能**跨项目提升**——在 A 项目摸清你的偏好,B 项目直接受用;
- 顶上压着一个**诚实契约**:她敢凭印象瞎答历史信息,就会被拦下来,触发真正的召回,而不是编一个看起来对的答案。

她把"记性"当成一等公民来设计,而不是挂了个 memory 模块。

## ✅ 她会"自证完成",而不是嘴上说说

agent 最大的谎言是"我做完了"。Keel 不吃这一套:

- **验证闭环**:定目标 → 执行 → 检查 → 算残差 → 判收敛,没达标就自动重规划重试。"完成"是**检出来的**;
- **真 TDD 模式**:独立出题器先写测试,经**硬红门**确认"实现前测试确实是红的",再用**哈希防篡改**把测试锁死——从机制上杜绝"先写实现再补测试""边改边标完成";
- **离线评测**:用例驱动地回归她的行为,是开发时手里那只稳稳的"方向盘"。

她把"完成"从一句空话,变成一道道过不去就不放行的门。

## 🛡️ 她"先问再动手",不会替你闯祸

- 危险操作(比如跑一条 `bash` 命令)触发**三态审批**:完整改动落进 scrollback 任你上滑核对,紧凑框常驻"允许 / 仅本次 / 拒绝";
- 文件操作从不直接落盘,先进 **Workspace 沙箱**暂存,审阅通过才导出,还带 deny set 防误删。

## 🎛️ 她有个真正懂"上下文工程"的大脑

记忆只是"源",真正决定每轮往模型嘴里喂什么的,是她的 **上下文编排器**:按**保底 + 共享池预算**精打细算地组装上下文,预算不够时**只丢弃、绝不截断**——保证送进去的每一块都是完整的,而不是被腰斩的半句话。

## 🎨 她还会换着法子干活

需要不同打法,她有**五种思考方式**——直接问答、原生工具调用、ReAct 思考-行动-观察、先规划再执行、输出后自我反思。配上一身 Claude Code 风的 CLI:Slash 命令、Markdown 渲染、主题、状态栏、钉底 todo 面板,体感一个不少。

---

## 用起来什么效果

- **你随口提过的偏好,她下次还记得**,而且不会被你后来改过的说法搞糊涂——不用每次都重新交代一遍;
- **她说"做完了",就是真的过了验证**,不是糊弄你,省下你逐条复查的功夫;
- **她碰你机器上有风险的东西之前,一定先停下来问**,不会自作主张删文件、跑命令;
- **跨项目用得越久,她越懂你**——这个项目学到的习惯,下个项目直接带过去;
- **上下文从不被腰斩**,长对话里她也不会丢掉关键前提、答出前言不搭后语的话。

一句话:**模型负责聪明,Keel 负责靠谱。** 她不是那种"演示惊艳、上手翻车"的 agent,而是越用越让你放心交东西给她的那种。

---

## 实际跑一段给你看

<p align="center">
  <img src="assets/screenshot.png" alt="Keel CLI in action — theme, status bar, Markdown rendering, pinned todo" width="720">
  <br>
  <em>Keel CLI: theme-aware, with status bar, Markdown rendering, and a pinned todo panel.</em>
</p>

下面是一段**真实对话**(逐字未改,仅去掉重复日志行)。注意她怎么**主动把记忆当工具来管理**——存的时候自己调 `remember`,用的时候自己调 `recall`,而不是把历史一股脑塞回 context:

```
你 ▸ 我习惯用 4 空格缩进,别用 tab。

Keel ▸ 🔧 remember ← 用户偏好缩进风格为 4 空格,不要使用 tab
       好的,已经记录:你习惯使用 4 空格缩进,不要用 tab。
       之后的代码或格式化内容,我会一律使用 4 空格。

   …(中间隔了一整轮无关的问题:Python 列表推导式怎么写,她照常作答)…

你 ▸ 我刚说我缩进习惯是什么来着?

Keel ▸ 🔧 recall ← 用户关于缩进习惯的信息
       根据你刚才说的——你习惯用 4 空格缩进,不要用 tab。
       我已经记住了,随后的代码中也会采用 4 空格风格。

你 ▸ 帮我算 15 * 8 + 32

Keel ▸ 🔧 calculator ← 15 * 8 + 32
       15 × 8 + 32 = 152。
```

隔了一轮无关对话,她依然准确取回了你的偏好,还明说"根据你刚才说的"——**Recall real memory, don't make up a plausible answer from vague impressions.** That's the honesty contract + layered memory at work.

---

## Quick Start

她用 [uv](https://github.com/astral-sh/uv) 管理一切,**不用你手动装 Python**——`uv sync` 会自动拉好 Python ≥ 3.13 和全部依赖。下面按系统选一份照抄即可。

<details open>
<summary><b>🍎 macOS</b></summary>

```bash
# 1. 装 uv（已装可跳过）
brew install uv                                   # 或: curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 拉代码
git clone https://github.com/HHHH-LK/keel.git && cd keel

# 3. 装依赖 + 配置
uv sync
cp .env.example .env                              # 然后编辑 .env 填 key/model/base_url

# 4. 启动
uv run keel
```
</details>

<details>
<summary><b>🐧 Ubuntu / Linux</b></summary>

```bash
# 1. 装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env                       # 让当前终端认到 uv(或重开终端)

# 2. 拉代码
git clone https://github.com/HHHH-LK/keel.git && cd keel

# 3. 装依赖 + 配置
uv sync
cp .env.example .env                              # 然后编辑 .env 填 key/model/base_url

# 4. 启动
uv run keel
```
</details>

<details>
<summary><b>🪟 Windows (PowerShell)</b></summary>

```powershell
# 1. 装 uv
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. 拉代码
git clone https://github.com/HHHH-LK/keel.git
cd keel

# 3. 装依赖 + 配置
uv sync
Copy-Item .env.example .env                       # 然后编辑 .env 填 key/model/base_url

# 4. 启动
uv run keel
```
</details>

> No API key? You can still launch and set it inside with `/config key`. The `.env` expects `LLM_API_KEY` / `LLM_MODEL_ID` / `LLM_BASE_URL`. Model providers are swappable: OpenAI / Anthropic / Gemini / local models.

Once inside, try Slash commands: `/help`, `/config`, `/memory`, `/clear`. The legacy entry `uv run python chat.py` is also kept.

---

## Configuration Reference

| Key | Required | Description |
|-----|----------|-------------|
| `LLM_API_KEY` | ✅ | API key for your preferred LLM provider (OpenAI / Anthropic / Gemini / local) |
| `LLM_MODEL_ID` | ✅ | Model name, e.g. `gpt-4o`, `claude-sonnet-4-20250514`, `gemini-2.5-flash` |
| `LLM_BASE_URL` | ❌ | API base URL; defaults to OpenAI. Set to a local endpoint (e.g. `http://localhost:8000/v1`) for self-hosted models. |

All other settings can be configured interactively via `/config` once inside.

---

## What she's made of

```
keel/                     # Python 包(她叫 Keel)
├── core/        # Agent 基类、LLM 客户端、Message、Hook、Config
├── agents/      # 五种思考方式
├── context/     # 上下文编排器
├── memory/      # 分层记忆 + 时态 KG + 双层记忆
├── tools/       # 工具注册与执行链
├── verify/      # 验证-重试闭环
├── tdd/         # 真 TDD 模式
├── planning/    # 任务 todo 规划
├── bench/       # 离线评测
├── cli/         # Claude Code 风 CLI
├── workspace/   # 文件操作沙箱
└── test/        # 单元 + 集成测试
```

设计文档在 `docs/superpowers/specs/` 与 `docs/superpowers/plans/`。

---

## Her philosophy (design principles)

- **轻量优先**:核心结构清晰,不在早期堆重依赖
- **可控优先**:每项能力都有明确边界,而不是把权力全丢给模型
- **可验证优先**:完成靠检验,不靠自报
- **面向运行时**:在意"持续把任务做完",而不只是"生成文本"

---

## Project Status

Keel is currently in **active development (alpha)**. The core architecture is stable, but APIs may evolve as we add features and polish. We'll tag a stable release once the design settles.

What to expect now:
- ✅ Core agent loop, layered memory, context orchestrator, tool execution
- ✅ True TDD & verification-closed loop
- ✅ CLI with themes, status bar, Markdown rendering
- 🚧 More tool integrations (web browsing, file search, etc.)
- 🚧 Plugin system for custom tools & memory backends
- 📝 Comprehensive multi-language docs

---

## Contributing

Contributions are welcome! Here's how:

1. **Open an [Issue](https://github.com/HHHH-LK/keel/issues)** — bug reports, feature requests, design discussions
2. **Submit a PR** — fork → branch → commit → pull request
3. **Keep it clean** — the project uses `uv` for dev; run `uv sync` and make sure tests pass before opening

```bash
# dev setup
git clone https://github.com/your-fork/keel.git && cd keel
uv sync                      # install all deps
cp .env.example .env         # configure
uv run keel                  # try it
```

See the design docs under `docs/superpowers/specs/` and `docs/superpowers/plans/` for architectural context. No strict code style guide yet — just use good judgment and consistent formatting.

---

## License

This project is open-sourced under the MIT License — see [LICENSE](LICENSE).

## Feedback

Questions or ideas? Feel free to open an [Issue](https://github.com/HHHH-LK/keel/issues) or PR.

---

如果你想要的不是"问一句答一句"的工具,而是一个**记得住、做得完、不闯祸**的长期 AI 伙伴——那就是 Keel。
