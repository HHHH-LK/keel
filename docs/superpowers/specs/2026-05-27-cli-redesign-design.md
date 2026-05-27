# CLI 重设计 — 设计文档

**日期**: 2026-05-27
**作者**: lk_hhh + Claude
**状态**: pending review

## 1. 目标

把现有 `chat.py` 的 CLI 界面从「皮卡丘 ASCII + 朴素 ● 前缀 + 单行 ❯」改成**现代渐变滚动式（Warp/Vercel 风）**，并补一个用户主动要的核心功能：**slash 命令补全菜单**（输入 `/` 立即弹出，输入字母实时过滤）。

同时趁这次重设计把 986 行的 `chat.py` 按职责拆开，UI 层独立成 `my_agent_llms/cli/` 子包。

## 2. 非目标

- **不引入 textual / blessed 等新 TUI 框架**：保持 inline 滚动式体验（退出后历史仍在终端里），不接管全屏 alt-screen。
- **不改 Agent / Memory / Tools 任何行为**：纯 UI 层重构。
- **不改命令清单**：所有现有 slash 命令（`/help` `/config` `/memory` `/recall` `/remember` `/forget` `/pin` `/l0` `/restore` `/facts` `/kg` `/clear` `/multiline` `/quit`）名字、参数、副作用保持不变，只换显示。
- **不做子命令补全**：`/config <Tab>` 弹 `provider/model/key/...` 列入阶段 2，MVP 不做。
- **不做鼠标点击交互**：纯键盘。

## 3. 用户需求 & 决策回顾

用户痛点（多选）：banner、配色与层次感、回答与输入区布局、/help 等命令的输出排版，**加一条主动需求：「/ 弹补全菜单 + 字母过滤」**。

7 个核心决策（已对齐）：

| 维度 | 选择 |
|---|---|
| 整体风格 | 现代渐变（Warp/Vercel 风） |
| 输入区 | 上下双行：状态条 + ❯ prompt |
| Slash 菜单 | 双列：命令 + 描述 |
| 对话区 | 左侧彩色竖条 + role label |
| 命令输出 | 轻量表格（rich.Table，无外框） |
| 配色 | magenta→cyan 渐变；user=cyan / 伙伴=magenta / ok=green / warn=yellow / err=red |
| 皮卡丘 ASCII | 完全拿掉 |
| 实现方案 | 拆分模块：新建 `my_agent_llms/cli/` 子包 |

## 4. 模块结构

```
my_agent_llms/cli/
├── __init__.py          # 对外导出 ChatCLI、build_agent
├── theme.py             # 所有 color token（单一改皮入口）
├── banner.py            # 启动 banner（一次性）
├── status_bar.py        # 每轮输入前的状态行 + 分隔线
├── chat_view.py         # 渲染用户输入、AI 回复、错误
├── help_view.py         # /help、/config show、/memory 的 rich.Table
├── completer.py         # SlashCompleter + SLASH_COMMANDS 命令清单（单一来源）
└── prompt.py            # PromptSession 构造（绑定 completer + key bindings）

chat.py                  # 入口、配置加载、命令 dispatch、Agent 装配（~400 行）
```

每个文件目标 < 200 行，单一职责。`chat.py` 不再直接调 `console.print`，全部走 `cli/` 中的 view 函数。

## 5. 设计细则

### 5.1 color tokens（`cli/theme.py`）

集中定义全部颜色，所有其他模块**只能从这里 import**，禁止硬编码颜色字符串。

```python
# Roles
YOU      = "cyan"              # 用户消息左竖条 + role label
AGENT    = "magenta"           # AI 回复左竖条 + role label

# Accents
ACCENT   = "bright_magenta"    # ❯ prompt 箭头、表头、Slash 高亮行 bg
LOGO_L   = "magenta"           # banner 左渐变方块
LOGO_R   = "bright_magenta"    # banner 右渐变方块
TITLE    = "bold cyan"         # banner 标题词

# States
OK       = "green"             # ● ready、✓
WARN     = "yellow"            # ⚠
ERR      = "red"               # ● error

# Neutral
DIM      = "bright_black"      # meta、时间戳、描述
RULE     = "bright_black"      # 分隔线
DEFAULT  = ""                  # 主体文字保持终端默认色
```

未来想换皮（冷色 / mono / 暖色）只改这一个文件。

### 5.2 Banner（`cli/banner.py`）

启动时一次性打印，**inline 滚动、不画外框**：

```
                                                    
  █▒  my·companion                                  
                                                    
  A long-term AI partner with memory                
                                                    
  ●  minimax  /  MiniMax-Text-01                    
  ●  ready  ·  L4 cold: sqlite  ·  4 tools loaded   
  ●  workspace: ~/.my_companion/sandbox             
                                                    
  Type /help for commands  ·  /  for menu           
                                                    
```

- 左缩进 2 列；`█▒` logo 用 LOGO_L → LOGO_R 渐变；`my·companion` 用 TITLE 加粗
- 第一行 bullet `●` = OK（ready）/ ERR（not ready）；后两行 bullet `●` 用 DIM
- `not ready` 时第 2 条 bullet 变 `●  not ready  —  run /config key`，第 3 行 bullet（workspace）省略
- workspace 路径用 `~` 简化 home 前缀
- bullet 中的 model 名用 cyan，其他文本默认色或 DIM

### 5.3 状态条 + Prompt（`cli/status_bar.py` + `cli/prompt.py`）

**每轮 `❯` 之前**，先打一条横线 + 一行 dim 状态：

```
─────────────────────────────────────────────────
minimax / MiniMax-Text-01  ·  turn 7  ·  L1 1.2k/4k tokens
❯ 
```

- 横线宽度跟随终端宽度（`console.width`），颜色 RULE
- 状态字段：`{provider_key} / {model}` (cyan) · `turn {n}` (DIM) · `L1 {tokens}/{max_tokens}` (DIM)
- `not ready` 时改成 `not ready  ·  run /config key`（ERR + DIM）
- `multiline = True` 时追加 ` · multiline`，prompt 换 `❯ multiline ›`
- prompt 箭头 `❯` 用 ACCENT；输入内容默认色

### 5.4 对话区（`cli/chat_view.py`）

每条消息分两层：**header（一行 meta）+ body（多行左竖条内容）**。

**用户输入回显**（输入后立即渲染）：

```
 you  ·  20:34
┃ 上次发你那个甜点配方，改成无坚果版本。
```

- header：`you` (YOU) + `·` + 时间 `HH:MM` (DIM)
- body：每行前缀 `┃ `（YOU 同色）+ 用户原文（默认色）
- 用户输入跨行时，每行都加 `┃ ` 前缀

**AI 回复**：

```
 伙伴  ·  4 tools  ·  2.3s
┃ 好的，我看了你附的 dessert.md：
┃ 
┃ 1. 跳杰果仁换成南瓜子
┃ 2. 避免所有坚果（你对花生过敏）
```

- header：`伙伴` (AGENT) + `·` + `{n} tools` (DIM, 仅当 n>0) + `·` + 耗时 (DIM)
- body：每行前缀 `┃ `（AGENT 同色）+ markdown 渲染结果
- markdown 仍走 `rich.Markdown`，但用一个 helper 把 `rich.Markdown` 渲染产物逐行包装为 `Padding(┃ + 内容)`
- 思考期间：先打 `伙伴  ·  ⠋ thinking…`（spinner 用 `console.status`），spinner 退出后**在下一行追加最终 header + body**（不改写已打印的 header 行 —— 滚动式 inline 无法可靠擦除已落屏内容）。最终展示形如：spinner 行短暂出现后被 status context manager 自动清除，紧接着打印完整 header + body

**错误**：

```
 伙伴  ·  ● error
┃ openai.AuthenticationError: invalid api key
```

- header `伙伴` 整体变红；`● error` 用 ERR
- body 竖条变红（ERR），文字默认色

**Agent not ready 提示**（用户尝试聊天但没配 key 时）：保留现有 4 行多语句提示，但内部统一通过 `chat_view.print_not_ready_hint()` 渲染。

### 5.5 Slash 补全（`cli/completer.py`）

**单一命令清单**（同时供 Completer 和 `/help` 使用）：

```python
SLASH_COMMANDS: list[tuple[str, str, str]] = [
    # (name, description, group)
    ("/help",      "show all commands",                 "Basic"),
    ("/quit",      "exit (also /exit, Ctrl+D)",        "Basic"),
    ("/multiline", "toggle multiline input",            "Basic"),
    ("/config",    "configure provider, model, key",   "Config"),
    ("/clear",     "clear context (keeps long-term)",  "Memory"),
    ("/memory",    "show memory stats",                 "Memory"),
    ("/recall",    "search long-term memory",           "Memory"),
    ("/remember",  "add a memory card",                 "Memory"),
    ("/forget",    "forget a memory card",              "Memory"),
    ("/pin",       "lock a memory card",                "Memory"),
    ("/l0",        "list active L0 cards",              "Memory"),
    ("/restore",   "load recent history from cold",     "Memory"),
    ("/facts",     "query KG facts",                    "Memory"),
    ("/kg",        "export knowledge graph (mermaid)",  "Memory"),
]
```

**Completer 行为**：

- 继承 `prompt_toolkit.completion.Completer`，挂到 `PromptSession(completer=…, complete_while_typing=True)`
- 触发条件：输入行**以 `/` 开头**才返回补全（避免在普通聊天中误弹）
- 过滤算法：`document.text_before_cursor.lower()` 作为查询串，**只匹配 `name.lower()` 子串**（描述不参与匹配）；保持 `SLASH_COMMANDS` 中的声明顺序
- 每条 `Completion(name, start_position=-len(query), display=name, display_meta=description)`
- 没有匹配时菜单消失，不弹「no matches」
- Tab / Enter / ↓→ 在 prompt_toolkit 默认就是接受补全；Esc 退出菜单不接受

**菜单视觉**：

通过 `prompt_toolkit.styles.Style` 改 completion-menu 类的样式：

```python
"completion-menu":             "bg:default",
"completion-menu.completion":  "fg:default",
"completion-menu.completion.current": f"bg:{ACCENT} fg:black",
"completion-menu.meta.completion":         "fg:bright_black",
"completion-menu.meta.completion.current": f"bg:{ACCENT} fg:black",
```

prompt_toolkit 会自动把 `display` 和 `display_meta` 排成双列。最终呈现近似：

```
❯ /re
  /recall       search long-term memory
▶ /remember     add a memory card           ← 当前高亮
  /restore      load recent history from cold
```

### 5.6 命令输出（`cli/help_view.py`）

`/help`、`/config show`、`/memory` 都走 `rich.Table(box=None, show_header=True)`，靠列对齐 + 分组小标题表达层次。

**`/help`**：

```
  COMMANDS
  ────────────────────────────────────────────────
   Basic
     /help          show this help
     /quit          exit (also /exit, Ctrl+D)
     /multiline     toggle multiline input
   
   Config
     /config        configure provider, model, key
   
   Memory
     /clear         clear context (keeps long-term)
     /memory        show memory stats
     /recall <q>    search long-term memory
     ... 
  ────────────────────────────────────────────────
```

实现：`Table(box=None, show_header=False, padding=(0, 2))`，2 列；分组小标题通过插入一行 `Table.add_row(f"[bold cyan]{group}[/]", "")` 实现，组之间 `Table.add_row("", "")` 空行。表头 `COMMANDS` 单独 `console.print("[bold bright_magenta]  COMMANDS[/]")` 然后 `RULE` 横线，表后再来一条横线。

**命令来源**：`help_view` 从 `cli/completer.py` 的 `SLASH_COMMANDS` 读取并按 `group` 字段分组，杜绝 Completer 菜单和 `/help` 列表数据漂移。

**`/config show`**：

同样格式，三组：`LLM`（provider_key / provider / model / base_url / api_key 掩码）、`Memory`（cold/vector/conflict/tick/embedding）、`Meta`（config path / agent ready 状态）。

**`/memory`**：

两列：name (DIM) / value (bright_white)。表头 `MEMORY STATS`。

**统一错误打印**：

```python
def print_error(msg: str) -> None:
    console.print(f"[{ERR}]●[/] [bold {ERR}]error[/] [{ERR}]{msg}[/]")
```

替代散落各处的 `console.print(f"[red]…[/red]")`。同样 `print_warn`、`print_ok`。

### 5.7 总体渲染流程

```
print_banner()                          # 一次性
loop:
  print_status_bar(cfg, agent_state)    # 每轮一次
  text = prompt(❯, completer=SlashCompleter)
  if text.startswith("/"):
      handle_command(text)              # 命令输出走 help_view 等
      continue
  render_user(text)                     # 立即回显
  with thinking_spinner():
      reply = agent.run(text)
  render_agent(reply, meta=…)
```

## 6. 关键技术点

### 6.1 markdown + 左竖条的组合

`rich.Markdown` 直接 print 会铺满整行，不好嵌竖条。MVP 方案：

```python
import io
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

def _prefix_markdown(md_text: str, color: str) -> Text:
    # 用临时 Console 把 markdown 渲染到带 ANSI 的字符串，再按行拆，前面 append ┃
    buf = io.StringIO()
    Console(file=buf, force_terminal=True, color_system="truecolor",
            width=console.width - 4).print(Markdown(md_text))
    raw = buf.getvalue().rstrip("\n")
    out = Text()
    for i, line in enumerate(raw.split("\n")):
        if i: out.append("\n")
        out.append("┃ ", style=color)
        out.append(Text.from_ansi(line))   # 保留 ANSI 高亮
    return out
```

`Text.from_ansi` 是 rich 内置 API，明确支持把带 ANSI 的字符串重新解析回 Text 对象，保留全部样式 —— 不存在「丢色」问题。这是 MVP 方案，不需要 fallback。

### 6.2 prompt_toolkit completion-menu 双列宽度

`display_meta` 列的宽度由 prompt_toolkit 自动计算，但默认 meta 列会贴右边。若菜单看起来太宽（描述被截断），通过 `completion-menu.meta.completion` 的 padding 调整；若太挤，截断描述到 ~40 字。

### 6.3 状态条 token 估算

`L1 {tokens}/{max_tokens}` 字段来自 `agent.memory.stats()["l1_tokens"]` 和 memory_config 里的 `l1_max_tokens` (默认 4000)。`agent` 为 None 时整段省略。

### 6.4 时间戳

用户输入回显的时间戳来自调用 `render_user` 时的本地时间 `datetime.now().strftime("%H:%M")`，不读消息时间，因为消息对象没有时间字段。

## 7. 实施顺序

按依赖顺序，每步独立可测：

1. **`cli/theme.py`** —— 纯常量，无逻辑
2. **`cli/completer.py`** —— 先把 `SLASH_COMMANDS` 列表立起来（help_view 也要依赖），同时实现 `SlashCompleter`
3. **`cli/help_view.py` + 统一 print_error/print_warn/print_ok** —— 从 `SLASH_COMMANDS` 读命令，迁移 `/help` `/config show` `/memory`
4. **`cli/chat_view.py`** —— `render_user` / `render_agent` / `print_not_ready_hint`，替换现有 `render_response`
5. **`cli/banner.py`** —— 替换 `print_banner`
6. **`cli/status_bar.py` + `cli/prompt.py`** —— 在 `get_input` 前插入状态条；`PromptSession` 改用 `prompt.py` 工厂，绑定 `SlashCompleter` 与样式
7. **`chat.py` 收尾** —— 把所有直接 `console.print` 改成 view 调用，删除旧的 banner / render / POSES / 老 cmd_help 代码
8. **手动跑一遍**：启动 → /help → /config → 聊一句 → /clear → /quit，确认无回归

## 8. 测试策略

- **不写自动化 UI 测试**（终端渲染快照很脆弱）
- 给 `cli/completer.py` 的 `SlashCompleter` 写单元测试：输入 `/`、`/re`、`/xxx` 分别返回的 completion 列表
- 给 `cli/theme.py` 写一个 smoke test：所有常量字符串能被 rich 接受（用 `Text("x", style=YOU)` 不抛错）
- 手测 checklist（在 PR 描述里勾）：
  - [ ] 启动：banner 显示，ready/not ready 两种状态各看一次
  - [ ] 输入 `/` → 菜单弹出；输入 `/re` → 过滤到 3 条；按 ↑↓ → 高亮移动；按 Tab → 填充
  - [ ] 跑一轮对话：用户消息有 cyan 竖条；伙伴消息有 magenta 竖条 + meta
  - [ ] `/help` 输出三组对齐
  - [ ] `/config show` 三组对齐，api_key 掩码
  - [ ] `/memory` 表格 stats 显示
  - [ ] not ready 时聊天 → 红色错误 + 配置提示
  - [ ] `/clear` 清屏后 banner 重画
  - [ ] `/multiline` 切换：状态条带 `· multiline` 标记，prompt 变 `❯ multiline ›`

## 9. 风险 & 缓解

| 风险 | 缓解 |
|---|---|
| markdown + 左竖条的 Segment 重组损失色彩 | 优先验证；不行就退化为纯文本竖条 + markdown 内部高亮（接受丢失高亮）|
| prompt_toolkit completion-menu 样式不支持 bg/fg | 已确认支持，参考 ptpython 源码 |
| 终端窗口宽度过窄（< 80 列）导致状态条折行 | 状态条按 console.width 截断；< 60 时只显示 model 名 |
| chat.py 拆分中临时破坏现有功能 | 每步一个小提交，能独立跑 |

## 10. 阶段 2（不做，仅记录）

- `/config <Tab>` 子命令补全
- 历史聊天的 timestamp 来自消息本身而非 `now()`
- 切皮命令 `/theme <warm|cool|mono>`
- 鼠标点击 Slash 菜单选项
