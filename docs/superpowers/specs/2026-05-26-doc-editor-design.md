# Doc Editor 设计文档

- **日期**：2026-05-26
- **作者**：lk_hhh × Claude
- **状态**：Draft（待审阅）

## 1. 背景与目标

`my_agent_llms` 目前的能力栈集中在 LLM 调用、工具系统、记忆系统三条线，缺少"让 Agent 可控地读写用户真实文件"的能力。本设计要在不动核心运行时的前提下，给框架补上一组安全、可控、可扩展的文档编辑工具，使 Agent 能够：

- 接受用户在对话中给出的文件路径
- 把这些文件拉进一个隔离的工作区
- 在工作区内进行精确替换式编辑
- 经过用户明确确认后，把修改写回原位置

设计上参考 Claude Code 的 `Read` / `Edit` / `Write` 工具范式（精确字符串替换、行号 + offset/limit 读取、操作前 diff 确认），并在此之上叠加一层"沙箱 + attach/export"的隔离模型，使原文件在编辑过程中绝对安全。

### 1.1 与 RAG 的关系

本特性**不是** RAG。RAG 解决的是"在大量未知文档里语义检索"的问题；本特性解决的是"对用户已经明确指定的文档进行读写"的问题。两者正交，互不依赖。框架现有的 `memory.semantic` 层不在本设计涉及范围内。

## 2. 非目标（Out of Scope）

为了让 MVP 收敛、避免 over-engineering，以下能力**不在本期设计**：

- 多文件批量替换（如"把 100 份合同里的甲方都改成供应商"）
- 二进制文件 / PDF / Office 文档的解析与编辑（仅支持 UTF-8 文本）
- 文件版本管理（依赖用户的 git 或其它机制）
- 跨进程协作 / 多用户工作区
- 大文件流式编辑（>200KB 文件读取仍要求 offset/limit 分页）
- 编辑历史回滚 / undo 栈（沙箱本身就是隔离层，错了删沙箱重来）
- 与现有 `memory` 系统的任何耦合

## 3. 整体架构

### 3.1 沙箱模型

```
真实文件                  Sandbox（agent 唯一可读写区域）              真实文件
~/Documents/         ─AttachFile─►  ~/.my_agent_llms/workspaces/abc/  ─ExportFile─►  ~/Documents/
  report.md                              ├── report.md  (复制进来)                       report.md
                                          └── MANIFEST.json (记录源路径)                (改完写回)
```

**核心不变量**：

1. **沙箱外的真实文件，在 attach 之前 / export 之前完全只读、永不被改**
2. **沙箱内所有路径都经过 `Workspace.resolve()`，越界直接拒绝**
3. **export 是唯一改用户真实文件的时刻**，必须走两步确认 + diff vs 原文件 + 原文件 hash 校验

### 3.2 模块结构

```
my_agent_llms/
├── workspace/
│   ├── __init__.py
│   └── workspace.py            # Workspace 类、Manifest、异常
│
├── tools/
│   └── builtin/
│       ├── pending_edits.py    # PendingEdit / PendingEditStore
│       ├── read_file.py
│       ├── edit_file.py
│       ├── write_file.py
│       ├── list_dir.py
│       ├── attach_file.py
│       └── export_file.py
│
└── chat.py                     # 改造：启动可选 --workspace；缺省自动建沙箱
```

### 3.3 启动逻辑

- `python chat.py`（不传 `--workspace`）→ 在 `~/.my_agent_llms/workspaces/<timestamp>/` 自动 mkdir 一个空目录作为沙箱。`<timestamp>` 格式：`YYYYMMDD-HHMMSS-<6 位随机后缀>`，例如 `20260526-143002-a8f3c1`，避免同秒启动多实例时碰撞
- `python chat.py --workspace ./mydocs` → 用指定目录作为沙箱（必须已存在）
- 启动时 stderr 打印一行 `Workspace: <abs_path>`，让用户知情

## 4. 组件设计

### 4.1 `workspace/workspace.py`

```python
class WorkspaceViolation(Exception):
    """路径越界 / 命中黑名单。Tool 内捕获后转成字符串返回给 LLM。"""

DEFAULT_DENY_DIRS = {".git", ".env", "node_modules", "__pycache__", ".venv"}
DEFAULT_DENY_SUFFIXES = {".pem", ".key"}

class Workspace:
    """所有文件工具的安全边界。"""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        deny_dirs: Iterable[str] = DEFAULT_DENY_DIRS,
        deny_suffixes: Iterable[str] = DEFAULT_DENY_SUFFIXES,
    ):
        # root=None 时自动创建 ~/.my_agent_llms/workspaces/<timestamp>/
        # root!=None 时要求该目录已存在
        ...
        self.root: Path           # 沙箱绝对路径
        self.manifest_path: Path  # root/MANIFEST.json

    # --- 路径守门 ---
    def resolve(self, user_path: str) -> Path:
        """沙箱内路径解析；越界 / 命中黑名单 → raise WorkspaceViolation。
        允许尚未存在的路径（WriteFile 要建新文件）。
        符号链接跟随后仍需在 root 内。"""

    def relative(self, p: Path) -> str:
        """把绝对路径转成相对 root 的展示形式。"""

    # --- Manifest 管理 ---
    def manifest(self) -> dict[str, str]:
        """读取 sandbox 文件名 → 原始源路径 的映射。"""

    def attach(self, source_path: Path) -> Path:
        """把外部文件复制进 sandbox，更新 manifest，返回 sandbox 内路径。
        - source 命中 deny → raise WorkspaceViolation
        - sandbox 内已有同名 → raise FileExistsError（不静默覆盖）"""

    def origin_of(self, sandbox_path: Path) -> Path | None:
        """根据 manifest 查 sandbox 文件对应的源路径。新建文件返回 None。"""
```

**deny list 在 attach 时的语义**：源路径中任一段命中 `deny_dirs`、或后缀命中 `deny_suffixes`，直接拒绝。这是防止 `.env` / 私钥 / `.git/HEAD` 被复制进沙箱的最后一道闸。

### 4.2 `tools/builtin/pending_edits.py`

```python
EditKind = Literal["edit", "write", "export"]

@dataclass
class PendingEdit:
    id: str
    kind: EditKind
    path: Path                # 目标文件绝对路径（edit/write 是 sandbox 内；export 是真实路径）
    new_content: str          # 完整新文件内容（不是 diff，是整文件）
    diff_preview: str         # 给 LLM/用户看的 unified diff
    source_hash: str | None   # 创建 pending 时目标文件的 SHA-256（新建文件为 None）
    created_at: float

class PendingEditStore:
    """进程级单例。dict + 锁，TTL 7 分钟。
    崩溃即丢失，用户重新发起即可（MVP 不持久化）。"""

    def __init__(self, ttl_seconds: int = 420): ...
    def put(self, pe: PendingEdit) -> None: ...
    def pop(self, pid: str) -> PendingEdit | None: ...
    def discard(self, pid: str) -> bool: ...
```

### 4.3 工具集

所有工具构造时注入 `Workspace`（部分还注入 `PendingEditStore`）。所有 `run()` 返回 `str` —— 错误也以字符串形式返回，包含"为什么 + 该怎么办"。

| 工具 | 必填参数 | 可选参数 | 返回语义 |
|---|---|---|---|
| `ReadFile` | `path` | `offset`(默认 0), `limit`(默认 200) | 带行号文本 + "共 N 行，已显示 a-b 行" |
| `EditFile` | 提案: `path` + `old_string` + `new_string` <br> 执行: `pending_id` + `action`(apply/cancel) | — | 提案返 `[待确认] pending_id=...` + diff；执行返结果 |
| `WriteFile` | 提案: `path` + `content` <br> 执行: `pending_id` + `action` | — | 同上，新建时 diff 为 `(新建文件，xxx 字节)` |
| `ListDir` | — | `path`(默认 root), `pattern`(默认 `*`), `max_depth`(默认 2) | 每行一个 `relative_path  size  mtime  [→ origin]` |
| `AttachFile` | `source_path` | — | `✅ 已 attach: report.md（来自 /Users/.../report.md）` |
| `ExportFile` | 提案: `sandbox_path` + `dest_path?`(新建文件必传) <br> 执行: `pending_id` + `action` | — | 提案返 diff vs 原文件；执行落盘到真实路径。**`dest_path` 同样走 deny list 校验**（防止导出到 `~/.ssh/`、`.env` 等敏感位置） |

### 4.4 工具注册（chat.py 改造）

```python
ws = Workspace(args.workspace)   # None → 自动建沙箱
store = PendingEditStore()
print(f"Workspace: {ws.root}", file=sys.stderr)

agent.register_tool(ReadFile(ws))
agent.register_tool(EditFile(ws, store))
agent.register_tool(WriteFile(ws, store))
agent.register_tool(ListDir(ws))
agent.register_tool(AttachFile(ws))
agent.register_tool(ExportFile(ws, store))
```

### 4.5 System Prompt 增量

LLM 不会自动理解"两步确认 + 沙箱 + attach/export"流程，必须在 system prompt 显式告知。建议加入：

```
你可以操作文件，但必须按以下流程：

1. 用户给的外部路径（如 ./report.md, /Users/.../foo.md），先用 AttachFile 拉进 sandbox
2. 在 sandbox 内用 ReadFile / EditFile / WriteFile / ListDir 操作
3. 修改和写入都是两步：第一次调用返回 pending_id 和 diff，等用户在对话中明确说"确认"后，再用 pending_id + action=apply 落盘
4. 用户希望把修改写回原位置时，用 ExportFile（同样两步确认 + diff vs 原文件）
5. sandbox 内新建的文件，ExportFile 时必须显式给 dest_path
```

## 5. 数据流：一次完整的"编辑 + 写回"

```
用户：把 /Users/lk_hhh/Documents/report.md 里的"2024"全改成"2025"
  │
  ▼
LLM 调 AttachFile(source_path="/Users/lk_hhh/Documents/report.md")
  → Workspace.attach() 复制到 sandbox，更新 MANIFEST
  → 返回 "✅ 已 attach: report.md"
  │
  ▼
LLM 调 ReadFile(path="./report.md")    # 注意此 path 是 sandbox 内相对路径
  → 返回带行号文本
  │
  ▼
LLM 调 EditFile(path="./report.md", old_string="2024", new_string="2025")
  → Workspace.resolve() 校验通过
  → 在文件中精确匹配；唯一 → 生成 new_content（整文件）+ diff
  → PendingEditStore.put(PendingEdit(kind="edit", source_hash=sha256(file)))
  → 返回 "[待确认] pending_id=abc123\ndiff:\n- 第 12 行：发布于 2024 年\n+ 第 12 行：发布于 2025 年\n回复'确认 abc123'落盘"
  │
  ▼
用户：确认 abc123
  │
  ▼
LLM 调 EditFile(pending_id="abc123", action="apply")
  → PendingEditStore.pop()
  → 重算目标文件 hash，与 source_hash 对比
    - 一致 → 写入 new_content（原子写：先写 tmp 再 rename）
    - 不一致 → "❌ 文件在确认期间被外部修改..."
  → 返回 "✅ 已修改 sandbox 内 ./report.md。如需写回原位置 (/Users/lk_hhh/Documents/report.md)，请调 ExportFile"
  │
  ▼
（用户继续别的编辑，或直接说"导出"）
  │
  ▼
LLM 调 ExportFile(sandbox_path="./report.md")
  → 从 MANIFEST 查 dest = /Users/lk_hhh/Documents/report.md
  → 读 sandbox 文件 + 读原文件 → 生成 diff vs 原文件
  → PendingEditStore.put(PendingEdit(kind="export", source_hash=sha256(原文件)))
  → 返回 "[待确认] pending_id=xyz789\n即将把 sandbox 内 report.md 写回 /Users/.../report.md\ndiff: ..."
  │
  ▼
用户：确认 xyz789
  │
  ▼
LLM 调 ExportFile(pending_id="xyz789", action="apply")
  → 重算原文件 hash 对比
  → 一致 → 原子写
  → 返回 "✅ 已写回 /Users/lk_hhh/Documents/report.md"
```

## 6. 错误矩阵

所有失败都返回字符串，必须包含"为什么 + 该怎么办"。

| 场景 | 触发位置 | 返回 |
|---|---|---|
| 路径越界 | `Workspace.resolve()` | `❌ 路径越界：xxx 不在 sandbox 内。只能访问 <root> 下的文件` |
| 命中黑名单 | `Workspace.resolve()` / `attach()` | `❌ 路径/源文件命中黑名单：.git/ 不可访问` |
| 文件不存在（Read/Edit） | tool 内 | `❌ 文件不存在：<rel_path>。可用 ListDir 查看 sandbox 内文件` |
| 路径是目录 | tool 内 | `❌ <rel_path> 是目录。用 ListDir 查看其内容` |
| 文件 >200 行且未传 offset | ReadFile | `⚠️ 文件共 N 行，本次仅显示 1-200。请用 offset/limit 分页继续读` |
| `old_string` 不存在 | EditFile | `❌ 在 <rel_path> 中找不到 old_string。请先 ReadFile 确认实际内容` |
| `old_string` 多处匹配 | EditFile | `❌ old_string 在 <rel_path> 匹配 N 处。请扩大 old_string 上下文使其唯一` |
| `pending_id` 不存在 / 过期 | EditFile / WriteFile / ExportFile | `❌ pending_id <abc> 不存在或已过期（7 分钟 TTL）。请重新发起` |
| WriteFile 新内容与原文件一致 | WriteFile | `⚠️ 新内容与原文件相同，无需修改` |
| Source hash 校验失败（apply 时文件被外部改） | 全部 apply 阶段 | `❌ 文件在确认期间被外部修改，pending 已失效。请重新读取并发起编辑` |
| AttachFile 源不存在 | AttachFile | `❌ 源文件不存在：xxx` |
| AttachFile 时 sandbox 已有同名 | AttachFile | `❌ sandbox 内已有 report.md。请先 export 或改名` |
| ExportFile 新建文件未给 dest | ExportFile | `❌ ./summary.md 是 sandbox 内新建文件，请显式提供 dest_path` |
| ExportFile dest 命中黑名单 / 是禁止目录 | ExportFile | `❌ 导出目标命中黑名单：xxx 不可写入` |
| 落盘 IO 失败 | 全部 apply 阶段 | `❌ 写入失败：<errno 原因>。原文件未被改动` |

## 7. 边界情况与不变量

| 情况 | 处理方式 |
|---|---|
| 同一文件多个 pending edit 堆积 | 允许并存，独立 pending_id。第二个 apply 时按"当时的磁盘内容"重算 hash，不一致就拒 |
| PendingEdit 创建后用户用外部编辑器改了文件 | 同上 —— apply 前比对 hash |
| `old_string` 跨多行 / 含 CRLF | 严格按字节匹配，不做任何 normalize |
| WriteFile / ExportFile 写大内容（>1MB） | 不在 MVP 拦截；内存里存到 TTL 过期 |
| 同会话切换 workspace | 不支持，重启 agent |
| 符号链接 | `resolve()` 跟随后仍需在 sandbox 内（防 symlink 越狱） |
| 进程崩溃丢失 pending | 接受，重启即清空，用户重新发起 |
| sandbox 自动建在 `~/.my_agent_llms/workspaces/<ts>/` | MVP 不主动清理，由用户/外部 cron 决定。文档建议在 README 提一句 |
| 原子写实现 | 全部 apply 都先写 `<dst>.tmp`，再 `os.rename()` 覆盖；中途失败保留原文件 |
| 文件编码 | 仅支持 UTF-8；非 UTF-8 文件 ReadFile 返回 `❌ 非 UTF-8 编码：xxx，本期不支持` |

## 8. 测试策略

### 8.1 单元测试（无 LLM，秒级）

**`tests/test_workspace.py`** —— 安全底座，必须 100% 覆盖

| 用例 | 期望 |
|---|---|
| 相对 / 绝对路径在 root 下 | resolve 成功 |
| `../../etc/passwd` 越狱 | raise `WorkspaceViolation` |
| symlink 指向 root 外 | raise `WorkspaceViolation` |
| `.git/HEAD` / `.env` / `*.pem` | raise `WorkspaceViolation` |
| root=None 自动建沙箱 | 目录被创建，路径在 `~/.my_agent_llms/workspaces/` 下 |
| root 不存在且非 None | raise（构造时） |
| attach 源不存在 | raise |
| attach 源命中 deny | raise |
| attach 时 sandbox 已有同名 | raise `FileExistsError` |
| attach 成功 | 文件被复制，manifest 被更新 |
| origin_of 已 attach 文件 | 返回源路径 |
| origin_of 新建文件 | 返回 None |

**`tests/test_pending_edit_store.py`**

| 用例 | 期望 |
|---|---|
| put → pop | 返回原对象 |
| pop 不存在 id | None |
| TTL 过后 pop | None |
| discard | True，后续 pop 返回 None |
| 多 pending 并存 | 互不影响 |

**`tests/test_<tool>.py`** —— 每个工具一份，用 `tmp_path` 建迷你 workspace

- 每个工具都跑：正常路径 + 错误矩阵里**它涉及的每一行**
- EditFile / WriteFile / ExportFile：完整的提案 → apply 闭环，提案 → cancel 闭环
- **关键回归用例**：apply 时文件被外部改动 → hash 校验失败 → 拒绝写入

### 8.2 集成测试（接 LLM，按需跑）

**`tests/integration/test_doc_edit_flow.py`**

```
场景：临时目录放 hello.md（"Hello, 2024"）
prompt："把 hello.md 里的 2024 改成 2025"
期望 LLM 完整走通 5 步：
  1. AttachFile
  2. ReadFile
  3. EditFile（提案）
  4. EditFile（apply，模拟用户确认后）
  5. ExportFile（提案 + apply）
最终断言：原 hello.md 内容已变为 "Hello, 2025"
```

主要验证 system prompt 描述是否清晰到让 LLM 走通完整流程。

### 8.3 不写的测试（YAGNI）

- 大文件性能 benchmark
- 多线程并发
- 文件系统 mock（`tmp_path` 已经够快够真实）

## 9. 未来扩展（非本期）

- **批量编辑工具**：`EditFiles(pattern, old, new)` 跨多个 sandbox 文件
- **PendingEdit 持久化**：把 store 持久化到 sandbox 内 `pending.json`，进程崩溃可恢复
- **与 memory 系统联动**：把"用户最近 attach / 编辑过的文件"沉淀进 working memory，跨会话保持上下文
- **多用户 / 多会话 workspace**：当前是单进程单 workspace
- **二进制文件支持**：图像、PDF、Office 等
- **编辑撤销栈**：每次 apply 前自动备份到 `<sandbox>/.history/`

## 附录 A：依赖与配置

- 新增依赖：无（hashlib / pathlib / shutil 都在标准库）
- 新增 CLI 参数：`--workspace <path>`（可选）
- 新增环境变量：无
- 新增配置文件：无（MANIFEST.json 是 sandbox 内部状态，不算用户配置）
