# Doc Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a sandbox-isolated document editor for `my_agent_llms` agents — Workspace + PendingEditStore + 6 tools (Read/Edit/Write/List/Attach/Export) with two-step diff confirmation and SHA-256 source verification.

**Architecture:** All file edits happen inside an isolated sandbox dir (`~/.my_agent_llms/workspaces/<ts>/` by default). User files become safe except at two moments: `AttachFile` (copy in) and `ExportFile` (copy back). Edit and Write tools are two-step (propose → user confirms → apply) with diff preview and source hash check. ExportFile uses the same two-step pattern when writing back to the real file.

**Tech Stack:** Python 3.13, stdlib only (pathlib, hashlib, shutil, difflib, json, secrets, threading), pydantic (existing), pytest (existing).

**Spec:** [docs/superpowers/specs/2026-05-26-doc-editor-design.md](../specs/2026-05-26-doc-editor-design.md)

---

## File Plan

**Create:**
- `my_agent_llms/workspace/__init__.py` — re-export Workspace, WorkspaceViolation
- `my_agent_llms/workspace/workspace.py` — Workspace class, DEFAULT_DENY constants, exceptions
- `my_agent_llms/tools/builtin/pending_edits.py` — PendingEdit dataclass + PendingEditStore
- `my_agent_llms/tools/builtin/read_file.py` — ReadFile tool
- `my_agent_llms/tools/builtin/edit_file.py` — EditFile tool
- `my_agent_llms/tools/builtin/write_file.py` — WriteFile tool
- `my_agent_llms/tools/builtin/list_dir.py` — ListDir tool
- `my_agent_llms/tools/builtin/attach_file.py` — AttachFile tool
- `my_agent_llms/tools/builtin/export_file.py` — ExportFile tool
- `my_agent_llms/test/test_workspace.py`
- `my_agent_llms/test/test_pending_edit_store.py`
- `my_agent_llms/test/test_read_file.py`
- `my_agent_llms/test/test_edit_file.py`
- `my_agent_llms/test/test_write_file.py`
- `my_agent_llms/test/test_list_dir.py`
- `my_agent_llms/test/test_attach_file.py`
- `my_agent_llms/test/test_export_file.py`
- `my_agent_llms/test/integration/__init__.py`
- `my_agent_llms/test/integration/test_doc_edit_flow.py`

**Modify:**
- `chat.py` — add `workspace` key to DEFAULT_CONFIG; register 6 new tools in `build_agent`; extend system prompt

---

## Task 1: Workspace foundation (init + auto-sandbox)

**Files:**
- Create: `my_agent_llms/workspace/__init__.py`
- Create: `my_agent_llms/workspace/workspace.py`
- Test: `my_agent_llms/test/test_workspace.py`

- [ ] **Step 1: Write the failing tests**

```python
# my_agent_llms/test/test_workspace.py
from pathlib import Path
import pytest
from my_agent_llms.workspace import Workspace, WorkspaceViolation


def test_explicit_root_is_used_when_exists(tmp_path):
    ws = Workspace(tmp_path)
    assert ws.root == tmp_path.resolve()


def test_explicit_root_must_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        Workspace(tmp_path / "does_not_exist")


def test_none_root_creates_auto_sandbox(monkeypatch, tmp_path):
    # 把 HOME 指到 tmp_path,避免污染用户真实 home
    monkeypatch.setenv("HOME", str(tmp_path))
    ws = Workspace(None)
    assert ws.root.exists()
    # 用 resolve() 抹平 macOS /var → /private/var 的 symlink 差异
    expected_parent = (tmp_path / ".my_agent_llms" / "workspaces").resolve()
    assert ws.root.parent == expected_parent
    # 目录名: YYYYMMDD-HHMMSS-<6 位>
    name = ws.root.name
    assert len(name) == 22
    assert name[8] == "-"
    assert name[15] == "-"


def test_manifest_path_is_under_root(tmp_path):
    ws = Workspace(tmp_path)
    assert ws.manifest_path == tmp_path.resolve() / "MANIFEST.json"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest my_agent_llms/test/test_workspace.py -v`
Expected: ImportError or `ModuleNotFoundError: No module named 'my_agent_llms.workspace'`

- [ ] **Step 3: Write minimal implementation**

```python
# my_agent_llms/workspace/__init__.py
from my_agent_llms.workspace.workspace import (
    Workspace,
    WorkspaceViolation,
    DEFAULT_DENY_DIRS,
    DEFAULT_DENY_SUFFIXES,
)

__all__ = ["Workspace", "WorkspaceViolation", "DEFAULT_DENY_DIRS", "DEFAULT_DENY_SUFFIXES"]
```

```python
# my_agent_llms/workspace/workspace.py
"""Workspace —— Agent 文件工具的安全边界。

所有 file tool 构造时注入同一个 Workspace 实例。Workspace 负责:
- 决定 sandbox 根目录 (显式 root / 自动创建)
- 路径 resolve + 越界拦截 + 黑名单
- 维护 MANIFEST.json (sandbox 文件 → 原始源路径)
- attach / origin_of
"""
from __future__ import annotations

import datetime as _dt
import json
import secrets
from pathlib import Path
from typing import Iterable

DEFAULT_DENY_DIRS = frozenset({".git", ".env", "node_modules", "__pycache__", ".venv"})
DEFAULT_DENY_SUFFIXES = frozenset({".pem", ".key"})


class WorkspaceViolation(Exception):
    """路径越界 / 命中黑名单。Tool 内捕获后转字符串返回给 LLM。"""


def _auto_sandbox_name() -> str:
    """YYYYMMDD-HHMMSS-<6 位 hex>"""
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{secrets.token_hex(3)}"


class Workspace:
    def __init__(
        self,
        root: str | Path | None = None,
        *,
        deny_dirs: Iterable[str] = DEFAULT_DENY_DIRS,
        deny_suffixes: Iterable[str] = DEFAULT_DENY_SUFFIXES,
    ):
        if root is None:
            parent = Path.home() / ".my_agent_llms" / "workspaces"
            parent.mkdir(parents=True, exist_ok=True)
            root_path = parent / _auto_sandbox_name()
            root_path.mkdir()
        else:
            root_path = Path(root).expanduser()
            if not root_path.exists():
                raise FileNotFoundError(f"workspace 根目录不存在: {root_path}")
            if not root_path.is_dir():
                raise NotADirectoryError(f"workspace 根不是目录: {root_path}")

        self.root: Path = root_path.resolve(strict=True)
        self.manifest_path: Path = self.root / "MANIFEST.json"
        self._deny_dirs = frozenset(deny_dirs)
        self._deny_suffixes = frozenset(deny_suffixes)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest my_agent_llms/test/test_workspace.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add my_agent_llms/workspace/ my_agent_llms/test/test_workspace.py
git commit -m "feat(workspace): scaffold Workspace with auto-sandbox creation"
```

---

## Task 2: Workspace.resolve (path safety)

**Files:**
- Modify: `my_agent_llms/workspace/workspace.py` (add `resolve` and `relative` methods)
- Test: `my_agent_llms/test/test_workspace.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `my_agent_llms/test/test_workspace.py`:

```python
def test_resolve_relative_path_in_root(tmp_path):
    ws = Workspace(tmp_path)
    (tmp_path / "report.md").write_text("x")
    p = ws.resolve("report.md")
    assert p == (tmp_path / "report.md").resolve()


def test_resolve_absolute_path_in_root(tmp_path):
    ws = Workspace(tmp_path)
    abs_in = (tmp_path / "sub" / "a.md")
    abs_in.parent.mkdir()
    abs_in.write_text("x")
    p = ws.resolve(str(abs_in))
    assert p == abs_in.resolve()


def test_resolve_allows_not_yet_existing(tmp_path):
    """WriteFile 要写新文件,resolve 必须允许尚未存在的路径。"""
    ws = Workspace(tmp_path)
    p = ws.resolve("new_file.md")
    assert p == (tmp_path / "new_file.md").resolve()
    assert not p.exists()


def test_resolve_blocks_traversal(tmp_path):
    ws = Workspace(tmp_path)
    with pytest.raises(WorkspaceViolation, match="路径越界"):
        ws.resolve("../../etc/passwd")


def test_resolve_blocks_absolute_outside(tmp_path):
    ws = Workspace(tmp_path)
    with pytest.raises(WorkspaceViolation, match="路径越界"):
        ws.resolve("/etc/passwd")


def test_resolve_blocks_symlink_escape(tmp_path):
    ws = Workspace(tmp_path)
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret")
    link = tmp_path / "link.txt"
    link.symlink_to(outside)
    with pytest.raises(WorkspaceViolation, match="路径越界"):
        ws.resolve("link.txt")


def test_resolve_blocks_deny_dir(tmp_path):
    ws = Workspace(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref")
    with pytest.raises(WorkspaceViolation, match="黑名单"):
        ws.resolve(".git/HEAD")


def test_resolve_blocks_deny_suffix(tmp_path):
    ws = Workspace(tmp_path)
    with pytest.raises(WorkspaceViolation, match="黑名单"):
        ws.resolve("server.pem")


def test_relative_strips_root(tmp_path):
    ws = Workspace(tmp_path)
    sub = (tmp_path / "a" / "b.md")
    sub.parent.mkdir()
    sub.write_text("x")
    assert ws.relative(sub.resolve()) == "a/b.md"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest my_agent_llms/test/test_workspace.py -v -k "resolve or relative"`
Expected: 9 failures (AttributeError: 'Workspace' object has no attribute 'resolve' / 'relative')

- [ ] **Step 3: Write minimal implementation**

Add to `my_agent_llms/workspace/workspace.py` inside `Workspace`:

```python
    # ── 路径守门 ────────────────────────────────────────────
    def resolve(self, user_path: str) -> Path:
        """把 user_path 解析到 sandbox 内绝对路径。
        - 相对路径基于 self.root
        - 绝对路径直接用,但必须落在 self.root 下
        - 跟随符号链接,跟随后仍要在 self.root 下
        - 命中 deny_dirs / deny_suffixes → raise
        - 允许尚未存在的路径 (WriteFile 要建新文件)
        """
        up = Path(user_path).expanduser()
        candidate = up if up.is_absolute() else (self.root / up)

        # 父目录存在就解析父目录,再拼回文件名 —— 这样新建文件也能跟随中间链接
        if candidate.exists():
            p = candidate.resolve(strict=True)
        else:
            parent = candidate.parent
            if parent.exists():
                p = parent.resolve(strict=True) / candidate.name
            else:
                # 父也不存在 —— 用 resolve(strict=False) 做尽力解析
                p = candidate.resolve()

        try:
            p.relative_to(self.root)
        except ValueError:
            raise WorkspaceViolation(f"路径越界: {p} 不在 workspace {self.root} 内")

        for part in p.parts:
            if part in self._deny_dirs:
                raise WorkspaceViolation(f"路径命中黑名单目录: {p}")
        if p.suffix in self._deny_suffixes:
            raise WorkspaceViolation(f"文件类型在黑名单: {p.suffix}")
        return p

    def relative(self, p: Path) -> str:
        return str(p.relative_to(self.root))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest my_agent_llms/test/test_workspace.py -v`
Expected: all tests pass (13 total: 4 from Task 1 + 9 new)

- [ ] **Step 5: Commit**

```bash
git add my_agent_llms/workspace/workspace.py my_agent_llms/test/test_workspace.py
git commit -m "feat(workspace): add resolve() with traversal/symlink/deny guards"
```

---

## Task 3: Workspace manifest + attach + origin_of

**Files:**
- Modify: `my_agent_llms/workspace/workspace.py` (add manifest helpers + attach + origin_of)
- Test: `my_agent_llms/test/test_workspace.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `my_agent_llms/test/test_workspace.py`:

```python
def test_manifest_empty_when_no_file(tmp_path):
    ws = Workspace(tmp_path)
    assert ws.manifest() == {}


def _make_ws(tmp_path) -> tuple[Workspace, Path]:
    """统一 fixture: sandbox 在 tmp_path/sandbox,outside 在 tmp_path/outside"""
    sandbox = tmp_path / "sandbox"
    outside = tmp_path / "outside"
    sandbox.mkdir()
    outside.mkdir()
    return Workspace(sandbox), outside


def test_attach_copies_file_and_updates_manifest(tmp_path):
    ws, outside = _make_ws(tmp_path)
    src = outside / "report.md"
    src.write_text("hello 2024")

    dst = ws.attach(src)

    assert dst == ws.root / "report.md"
    assert dst.read_text() == "hello 2024"
    assert ws.manifest() == {"report.md": str(src.resolve())}


def test_attach_rejects_missing_source(tmp_path):
    ws, outside = _make_ws(tmp_path)
    with pytest.raises(FileNotFoundError):
        ws.attach(outside / "no_such_file.md")


def test_attach_rejects_deny_source(tmp_path):
    ws, outside = _make_ws(tmp_path)
    src = outside / "private.pem"
    src.write_text("KEY")
    with pytest.raises(WorkspaceViolation, match="黑名单"):
        ws.attach(src)


def test_attach_refuses_existing_target(tmp_path):
    ws, outside = _make_ws(tmp_path)
    (ws.root / "report.md").write_text("existing")
    src = outside / "report.md"
    src.write_text("incoming")
    with pytest.raises(FileExistsError):
        ws.attach(src)


def test_origin_of_attached_file(tmp_path):
    ws, outside = _make_ws(tmp_path)
    src = outside / "a.md"
    src.write_text("x")
    dst = ws.attach(src)
    assert ws.origin_of(dst) == src.resolve()


def test_origin_of_unmapped_file(tmp_path):
    ws, _ = _make_ws(tmp_path)
    (ws.root / "new.md").write_text("brand new")
    assert ws.origin_of(ws.root / "new.md") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest my_agent_llms/test/test_workspace.py -v -k "manifest or attach or origin"`
Expected: 7 failures (AttributeError on manifest/attach/origin_of)

- [ ] **Step 3: Write minimal implementation**

Add to `my_agent_llms/workspace/workspace.py` (need `shutil` import at top of file):

```python
import shutil  # add to imports
```

Add inside `Workspace`:

```python
    # ── Manifest 管理 ───────────────────────────────────────
    def manifest(self) -> dict[str, str]:
        if not self.manifest_path.exists():
            return {}
        with self.manifest_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write_manifest(self, data: dict[str, str]) -> None:
        tmp = self.manifest_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(self.manifest_path)

    def attach(self, source_path: str | Path) -> Path:
        """把外部文件复制进 sandbox 根。返回 sandbox 内绝对路径。
        - source 不存在 → FileNotFoundError
        - source 命中 deny (按目录段 / 后缀) → WorkspaceViolation
        - sandbox 已有同名文件 → FileExistsError
        """
        src = Path(source_path).expanduser()
        if not src.exists():
            raise FileNotFoundError(f"源文件不存在: {src}")
        if not src.is_file():
            raise IsADirectoryError(f"源不是文件: {src}")

        src_resolved = src.resolve(strict=True)
        for part in src_resolved.parts:
            if part in self._deny_dirs:
                raise WorkspaceViolation(f"源文件命中黑名单目录: {src_resolved}")
        if src_resolved.suffix in self._deny_suffixes:
            raise WorkspaceViolation(f"源文件类型在黑名单: {src_resolved.suffix}")

        dst = self.root / src_resolved.name
        if dst.exists():
            raise FileExistsError(f"sandbox 已有同名文件: {self.relative(dst)}")

        shutil.copy2(src_resolved, dst)
        m = self.manifest()
        m[self.relative(dst)] = str(src_resolved)
        self._write_manifest(m)
        return dst

    def origin_of(self, sandbox_path: str | Path) -> Path | None:
        """查 sandbox 文件对应的原始源路径。未在 manifest 中 → None。"""
        p = Path(sandbox_path).resolve()
        try:
            rel = str(p.relative_to(self.root))
        except ValueError:
            return None
        src = self.manifest().get(rel)
        return Path(src) if src else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest my_agent_llms/test/test_workspace.py -v`
Expected: all 20 tests pass

- [ ] **Step 5: Commit**

```bash
git add my_agent_llms/workspace/workspace.py my_agent_llms/test/test_workspace.py
git commit -m "feat(workspace): add MANIFEST.json + attach() + origin_of()"
```

---

## Task 4: PendingEditStore

**Files:**
- Create: `my_agent_llms/tools/builtin/pending_edits.py`
- Test: `my_agent_llms/test/test_pending_edit_store.py`

- [ ] **Step 1: Write the failing tests**

```python
# my_agent_llms/test/test_pending_edit_store.py
import time
from pathlib import Path
import pytest
from my_agent_llms.tools.builtin.pending_edits import (
    PendingEdit,
    PendingEditStore,
)


def _make_pe(pid: str = "abc") -> PendingEdit:
    return PendingEdit(
        id=pid,
        kind="edit",
        path=Path("/tmp/x.md"),
        new_content="hello",
        diff_preview="(no diff)",
        source_hash="dummyhash",
        created_at=time.time(),
    )


def test_put_then_pop_returns_original():
    store = PendingEditStore()
    pe = _make_pe()
    store.put(pe)
    got = store.pop("abc")
    assert got is pe


def test_pop_unknown_id_returns_none():
    assert PendingEditStore().pop("nope") is None


def test_pop_is_destructive():
    """同一 pending_id 只能 apply 一次,pop 后再 pop 必须 None。"""
    store = PendingEditStore()
    store.put(_make_pe())
    assert store.pop("abc") is not None
    assert store.pop("abc") is None


def test_discard_removes_entry():
    store = PendingEditStore()
    store.put(_make_pe())
    assert store.discard("abc") is True
    assert store.pop("abc") is None
    assert store.discard("abc") is False  # 已被 discard


def test_ttl_expiry():
    store = PendingEditStore(ttl_seconds=1)
    store.put(_make_pe())
    time.sleep(1.1)
    assert store.pop("abc") is None


def test_multiple_pendings_independent():
    store = PendingEditStore()
    store.put(_make_pe("a"))
    store.put(_make_pe("b"))
    assert store.pop("a") is not None
    assert store.pop("b") is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest my_agent_llms/test/test_pending_edit_store.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

```python
# my_agent_llms/tools/builtin/pending_edits.py
"""PendingEdit + PendingEditStore —— 两步确认机制的状态载体。

EditFile / WriteFile / ExportFile 第一次调用时不真写,而是构造一个
PendingEdit 放进 store 并返回 pending_id; 用户在对话中明确确认后,
LLM 再用 pending_id + action=apply 触发真正落盘。
"""
from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

EditKind = Literal["edit", "write", "export"]


@dataclass
class PendingEdit:
    id: str
    kind: EditKind
    path: Path              # 目标绝对路径 (edit/write: sandbox 内; export: 真实路径)
    new_content: str        # 整文件新内容
    diff_preview: str
    source_hash: Optional[str]   # 目标文件当前 SHA-256; 新建文件为 None
    created_at: float = field(default_factory=time.time)


class PendingEditStore:
    """进程级单例。MVP 用 dict + 锁,TTL 过期就静默丢弃。"""

    def __init__(self, ttl_seconds: int = 420):
        self._items: dict[str, PendingEdit] = {}
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    @staticmethod
    def new_id() -> str:
        return secrets.token_hex(4)  # 8 字符,够用且短

    def put(self, pe: PendingEdit) -> None:
        with self._lock:
            self._items[pe.id] = pe

    def pop(self, pid: str) -> Optional[PendingEdit]:
        with self._lock:
            self._evict_expired_locked()
            return self._items.pop(pid, None)

    def discard(self, pid: str) -> bool:
        with self._lock:
            return self._items.pop(pid, None) is not None

    def _evict_expired_locked(self) -> None:
        now = time.time()
        expired = [pid for pid, pe in self._items.items() if now - pe.created_at > self._ttl]
        for pid in expired:
            self._items.pop(pid, None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest my_agent_llms/test/test_pending_edit_store.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add my_agent_llms/tools/builtin/pending_edits.py my_agent_llms/test/test_pending_edit_store.py
git commit -m "feat(tools): add PendingEdit + PendingEditStore with TTL"
```

---

## Task 5: ReadFile tool

**Files:**
- Create: `my_agent_llms/tools/builtin/read_file.py`
- Test: `my_agent_llms/test/test_read_file.py`

- [ ] **Step 1: Write the failing tests**

```python
# my_agent_llms/test/test_read_file.py
from pathlib import Path
import pytest
from my_agent_llms.workspace import Workspace
from my_agent_llms.tools.builtin.read_file import ReadFile


def _run(ws: Workspace, **params) -> str:
    return ReadFile(ws).run(params)


def test_read_small_file_returns_numbered_lines(tmp_path):
    ws = Workspace(tmp_path)
    (tmp_path / "a.md").write_text("hello\nworld\n")
    out = _run(ws, path="a.md")
    assert "1\thello" in out
    assert "2\tworld" in out
    assert "共 2 行" in out


def test_read_large_file_pages_by_default(tmp_path):
    ws = Workspace(tmp_path)
    big = "\n".join(str(i) for i in range(1, 501)) + "\n"
    (tmp_path / "big.txt").write_text(big)
    out = _run(ws, path="big.txt")
    assert "1\t1" in out
    assert "200\t200" in out
    assert "201\t201" not in out
    assert "共 500 行" in out and "已显示 1-200" in out


def test_read_with_offset_and_limit(tmp_path):
    ws = Workspace(tmp_path)
    big = "\n".join(str(i) for i in range(1, 501)) + "\n"
    (tmp_path / "big.txt").write_text(big)
    out = _run(ws, path="big.txt", offset=100, limit=5)
    assert "101\t101" in out
    assert "105\t105" in out
    assert "106\t106" not in out


def test_read_missing_file(tmp_path):
    ws = Workspace(tmp_path)
    out = _run(ws, path="nope.md")
    assert out.startswith("❌")
    assert "不存在" in out


def test_read_directory_rejected(tmp_path):
    ws = Workspace(tmp_path)
    (tmp_path / "sub").mkdir()
    out = _run(ws, path="sub")
    assert out.startswith("❌")
    assert "目录" in out


def test_read_path_traversal_rejected(tmp_path):
    ws = Workspace(tmp_path)
    out = _run(ws, path="../../etc/passwd")
    assert out.startswith("❌")
    assert "越界" in out


def test_read_non_utf8_rejected(tmp_path):
    ws = Workspace(tmp_path)
    (tmp_path / "bin.dat").write_bytes(b"\xff\xfe\x00\x01")
    out = _run(ws, path="bin.dat")
    assert out.startswith("❌")
    assert "UTF-8" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest my_agent_llms/test/test_read_file.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

```python
# my_agent_llms/tools/builtin/read_file.py
"""ReadFile —— 读 sandbox 内文本文件,带行号,支持 offset/limit 分页。"""
from __future__ import annotations

from typing import Any, Dict, List

from my_agent_llms.tools.base import Tool, ToolParameter
from my_agent_llms.workspace import Workspace, WorkspaceViolation

DEFAULT_LIMIT = 200


class ReadFile(Tool):
    def __init__(self, workspace: Workspace):
        super().__init__(
            name="ReadFile",
            description=(
                "读 sandbox 内文本文件,返回带行号的内容。"
                "大文件默认仅显示前 200 行,需要看后续内容请传 offset/limit 分页。"
            ),
        )
        self.ws = workspace

    def run(self, parameters: Dict[str, Any]) -> str:
        path = str(parameters.get("path") or "").strip()
        if not path:
            return "❌ 缺少 path 参数"

        try:
            p = self.ws.resolve(path)
        except WorkspaceViolation as e:
            return f"❌ {e}"

        if not p.exists():
            return f"❌ 文件不存在: {self._safe_rel(p)}。可用 ListDir 查看 sandbox 内文件"
        if p.is_dir():
            return f"❌ {self._safe_rel(p)} 是目录,不是文件。用 ListDir 查看其内容"

        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"❌ 非 UTF-8 编码: {self._safe_rel(p)},本期不支持"

        lines = text.splitlines()
        total = len(lines)
        offset = int(parameters.get("offset") or 0)
        limit = int(parameters.get("limit") or DEFAULT_LIMIT)
        if offset < 0:
            offset = 0
        if limit <= 0:
            limit = DEFAULT_LIMIT

        chunk = lines[offset : offset + limit]
        numbered = "\n".join(f"{offset + i + 1}\t{line}" for i, line in enumerate(chunk))

        header = f"# {self._safe_rel(p)} (共 {total} 行,已显示 {offset + 1}-{offset + len(chunk)})\n"
        return header + numbered

    def _safe_rel(self, p) -> str:
        try:
            return self.ws.relative(p)
        except Exception:
            return str(p)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="path", type="string", description="sandbox 内文件路径", required=True),
            ToolParameter(name="offset", type="integer", description="从第几行开始(0-based)", required=False, default=0),
            ToolParameter(name="limit", type="integer", description="最多读多少行", required=False, default=DEFAULT_LIMIT),
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest my_agent_llms/test/test_read_file.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add my_agent_llms/tools/builtin/read_file.py my_agent_llms/test/test_read_file.py
git commit -m "feat(tools): add ReadFile with pagination + safety"
```

---

## Task 6: ListDir tool

**Files:**
- Create: `my_agent_llms/tools/builtin/list_dir.py`
- Test: `my_agent_llms/test/test_list_dir.py`

- [ ] **Step 1: Write the failing tests**

```python
# my_agent_llms/test/test_list_dir.py
from pathlib import Path
from my_agent_llms.workspace import Workspace
from my_agent_llms.tools.builtin.list_dir import ListDir


def _run(ws: Workspace, **params) -> str:
    return ListDir(ws).run(params)


def test_list_empty_sandbox(tmp_path):
    ws = Workspace(tmp_path)
    out = _run(ws)
    assert "(空)" in out


def test_list_lists_files_with_size(tmp_path):
    ws = Workspace(tmp_path)
    (tmp_path / "a.md").write_text("12345")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("hi")
    out = _run(ws)
    assert "a.md" in out
    assert "5" in out  # size
    assert "sub/b.md" in out


def test_list_filters_by_pattern(tmp_path):
    ws = Workspace(tmp_path)
    (tmp_path / "a.md").write_text("x")
    (tmp_path / "b.txt").write_text("x")
    out = _run(ws, pattern="*.md")
    assert "a.md" in out
    assert "b.txt" not in out


def test_list_shows_origin_for_attached(tmp_path):
    sandbox = tmp_path / "sandbox"
    outside = tmp_path / "outside"
    sandbox.mkdir()
    outside.mkdir()
    ws = Workspace(sandbox)
    src = outside / "origin.md"
    src.write_text("from outside")
    ws.attach(src)
    out = _run(ws)
    assert "origin.md" in out
    assert "←" in out or "→" in out  # origin marker
    assert "outside" in out  # 源路径片段


def test_list_excludes_manifest(tmp_path):
    """MANIFEST.json 是 sandbox 内部状态,不展示给 LLM。"""
    sandbox = tmp_path / "sandbox"
    outside = tmp_path / "outside"
    sandbox.mkdir()
    outside.mkdir()
    ws = Workspace(sandbox)
    src = outside / "x.md"
    src.write_text("x")
    ws.attach(src)  # 这会创建 MANIFEST.json
    out = _run(ws)
    assert "MANIFEST.json" not in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest my_agent_llms/test/test_list_dir.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

```python
# my_agent_llms/tools/builtin/list_dir.py
"""ListDir —— 列 sandbox 内文件,带大小、mtime,attached 文件附原路径。"""
from __future__ import annotations

import datetime as _dt
import fnmatch
from pathlib import Path
from typing import Any, Dict, List

from my_agent_llms.tools.base import Tool, ToolParameter
from my_agent_llms.workspace import Workspace, WorkspaceViolation

MANIFEST_NAME = "MANIFEST.json"


class ListDir(Tool):
    def __init__(self, workspace: Workspace):
        super().__init__(
            name="ListDir",
            description="列 sandbox 内文件,默认递归 2 层。attached 文件会显示来源路径。",
        )
        self.ws = workspace

    def run(self, parameters: Dict[str, Any]) -> str:
        path = str(parameters.get("path") or "").strip() or "."
        pattern = str(parameters.get("pattern") or "*")
        max_depth = int(parameters.get("max_depth") or 2)

        try:
            base = self.ws.resolve(path)
        except WorkspaceViolation as e:
            return f"❌ {e}"

        if not base.exists() or not base.is_dir():
            return f"❌ {path} 不是目录"

        manifest = self.ws.manifest()
        lines: List[str] = []
        for p in self._walk(base, max_depth):
            rel = self.ws.relative(p)
            if rel == MANIFEST_NAME:
                continue
            if not fnmatch.fnmatch(p.name, pattern):
                continue
            size = p.stat().st_size
            mtime = _dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            origin = manifest.get(rel)
            origin_str = f"  ← {origin}" if origin else ""
            lines.append(f"{rel}\t{size}\t{mtime}{origin_str}")

        if not lines:
            return "(空)"
        return "\n".join(lines)

    def _walk(self, base: Path, max_depth: int):
        base_depth = len(base.parts)
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            if len(p.parts) - base_depth > max_depth:
                continue
            yield p

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="path", type="string", description="起点(默认 sandbox 根)", required=False, default="."),
            ToolParameter(name="pattern", type="string", description="glob 模式,如 *.md", required=False, default="*"),
            ToolParameter(name="max_depth", type="integer", description="最大递归深度", required=False, default=2),
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest my_agent_llms/test/test_list_dir.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add my_agent_llms/tools/builtin/list_dir.py my_agent_llms/test/test_list_dir.py
git commit -m "feat(tools): add ListDir with origin annotation"
```

---

## Task 7: AttachFile tool

**Files:**
- Create: `my_agent_llms/tools/builtin/attach_file.py`
- Test: `my_agent_llms/test/test_attach_file.py`

- [ ] **Step 1: Write the failing tests**

```python
# my_agent_llms/test/test_attach_file.py
from pathlib import Path
from my_agent_llms.workspace import Workspace
from my_agent_llms.tools.builtin.attach_file import AttachFile


def _run(ws: Workspace, **params) -> str:
    return AttachFile(ws).run(params)


def _setup(tmp_path) -> tuple[Workspace, Path]:
    sandbox = tmp_path / "sandbox"
    outside = tmp_path / "outside"
    sandbox.mkdir()
    outside.mkdir()
    return Workspace(sandbox), outside


def test_attach_happy_path(tmp_path):
    ws, outside = _setup(tmp_path)
    src = outside / "report.md"
    src.write_text("hello")
    out = _run(ws, source_path=str(src))
    assert out.startswith("✅")
    assert "report.md" in out
    assert (ws.root / "report.md").read_text() == "hello"
    assert ws.manifest()["report.md"] == str(src.resolve())


def test_attach_missing_source(tmp_path):
    ws, outside = _setup(tmp_path)
    out = _run(ws, source_path=str(outside / "no_such.md"))
    assert out.startswith("❌")
    assert "不存在" in out


def test_attach_deny_source(tmp_path):
    ws, outside = _setup(tmp_path)
    bad = outside / "key.pem"
    bad.write_text("KEY")
    out = _run(ws, source_path=str(bad))
    assert out.startswith("❌")
    assert "黑名单" in out


def test_attach_already_exists(tmp_path):
    ws, outside = _setup(tmp_path)
    (ws.root / "report.md").write_text("x")
    src = outside / "report.md"
    src.write_text("y")
    out = _run(ws, source_path=str(src))
    assert out.startswith("❌")
    assert "已有" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest my_agent_llms/test/test_attach_file.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

```python
# my_agent_llms/tools/builtin/attach_file.py
"""AttachFile —— 把外部文件复制进 sandbox。"""
from __future__ import annotations

from typing import Any, Dict, List

from my_agent_llms.tools.base import Tool, ToolParameter
from my_agent_llms.workspace import Workspace, WorkspaceViolation


class AttachFile(Tool):
    def __init__(self, workspace: Workspace):
        super().__init__(
            name="AttachFile",
            description=(
                "把外部文件复制进 sandbox,后续 Read/Edit/Write 才能操作。"
                "用户提到任何外部路径时,先用本工具拉进来。"
            ),
        )
        self.ws = workspace

    def run(self, parameters: Dict[str, Any]) -> str:
        src = str(parameters.get("source_path") or "").strip()
        if not src:
            return "❌ 缺少 source_path 参数"
        try:
            dst = self.ws.attach(src)
        except FileNotFoundError as e:
            return f"❌ 源文件不存在: {src}"
        except IsADirectoryError as e:
            return f"❌ 源是目录,不是文件: {src}"
        except WorkspaceViolation as e:
            return f"❌ {e}"
        except FileExistsError as e:
            return f"❌ {e}。请先 ExportFile 或改名"
        return f"✅ 已 attach: {self.ws.relative(dst)} (来源: {src})"

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="source_path", type="string", description="外部文件的绝对或相对路径", required=True),
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest my_agent_llms/test/test_attach_file.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add my_agent_llms/tools/builtin/attach_file.py my_agent_llms/test/test_attach_file.py
git commit -m "feat(tools): add AttachFile to import external files into sandbox"
```

---

## Task 8: EditFile tool (propose mode)

**Files:**
- Create: `my_agent_llms/tools/builtin/edit_file.py` (initial — propose half only)
- Test: `my_agent_llms/test/test_edit_file.py`

- [ ] **Step 1: Write the failing tests (propose mode only)**

```python
# my_agent_llms/test/test_edit_file.py
import re
from pathlib import Path
import pytest
from my_agent_llms.workspace import Workspace
from my_agent_llms.tools.builtin.pending_edits import PendingEditStore
from my_agent_llms.tools.builtin.edit_file import EditFile


def _make_tool(tmp_path) -> tuple[EditFile, Workspace, PendingEditStore]:
    ws = Workspace(tmp_path)
    store = PendingEditStore()
    return EditFile(ws, store), ws, store


def _pending_id(text: str) -> str:
    m = re.search(r"pending_id=([0-9a-f]+)", text)
    assert m, f"no pending_id in: {text}"
    return m.group(1)


def test_propose_creates_pending_with_diff(tmp_path):
    tool, ws, store = _make_tool(tmp_path)
    (tmp_path / "a.md").write_text("hello 2024\nbye\n")
    out = tool.run({"path": "a.md", "old_string": "2024", "new_string": "2025"})
    assert "[待确认]" in out
    assert "2024" in out and "2025" in out
    pid = _pending_id(out)
    pe = store.pop(pid)
    assert pe is not None
    assert pe.kind == "edit"
    assert "2025" in pe.new_content
    assert pe.source_hash is not None  # 原文件存在,有 hash


def test_propose_rejects_missing_match(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    (tmp_path / "a.md").write_text("hello\n")
    out = tool.run({"path": "a.md", "old_string": "xyz", "new_string": "abc"})
    assert out.startswith("❌")
    assert "找不到" in out


def test_propose_rejects_multiple_matches(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    (tmp_path / "a.md").write_text("hi\nhi\n")
    out = tool.run({"path": "a.md", "old_string": "hi", "new_string": "yo"})
    assert out.startswith("❌")
    assert "多处" in out or "匹配" in out


def test_propose_rejects_missing_file(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    out = tool.run({"path": "no.md", "old_string": "x", "new_string": "y"})
    assert out.startswith("❌")
    assert "不存在" in out


def test_propose_rejects_traversal(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    out = tool.run({"path": "../etc", "old_string": "x", "new_string": "y"})
    assert out.startswith("❌")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest my_agent_llms/test/test_edit_file.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation (propose half)**

```python
# my_agent_llms/tools/builtin/edit_file.py
"""EditFile —— 精确替换,两步确认。

提案模式: path + old_string + new_string → 校验唯一匹配 → 生成 diff → 存 pending
执行模式: pending_id + action(apply/cancel)        → 校验 hash → 落盘 / 丢弃
"""
from __future__ import annotations

import difflib
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from my_agent_llms.tools.base import Tool, ToolParameter
from my_agent_llms.tools.builtin.pending_edits import (
    PendingEdit,
    PendingEditStore,
)
from my_agent_llms.workspace import Workspace, WorkspaceViolation


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_diff(rel_path: str, old: str, new: str) -> str:
    a = old.splitlines(keepends=True)
    b = new.splitlines(keepends=True)
    diff = difflib.unified_diff(a, b, fromfile=rel_path, tofile=rel_path, n=3)
    return "".join(diff) or "(无文本差异)"


class EditFile(Tool):
    def __init__(self, workspace: Workspace, store: PendingEditStore):
        super().__init__(
            name="EditFile",
            description=(
                "精确替换 sandbox 内文件的某段文字。两步确认: "
                "第一次传 path + old_string + new_string,返回 pending_id 和 diff; "
                "用户在对话中明确确认后,再传 pending_id + action=apply 落盘。"
            ),
        )
        self.ws = workspace
        self.store = store

    def run(self, parameters: Dict[str, Any]) -> str:
        pid = parameters.get("pending_id")
        if pid:
            return self._handle_action(str(pid), str(parameters.get("action") or ""))
        return self._handle_propose(parameters)

    # ── 提案模式 ────────────────────────────────────────────
    def _handle_propose(self, parameters: Dict[str, Any]) -> str:
        path = str(parameters.get("path") or "").strip()
        old = parameters.get("old_string")
        new = parameters.get("new_string")
        if not path or old is None or new is None:
            return "❌ 缺少参数 path / old_string / new_string"

        try:
            p = self.ws.resolve(path)
        except WorkspaceViolation as e:
            return f"❌ {e}"

        if not p.exists():
            return f"❌ 文件不存在: {self.ws.relative(p)}。可用 ListDir 查看 sandbox 内文件"
        if p.is_dir():
            return f"❌ {self.ws.relative(p)} 是目录"

        try:
            content = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"❌ 非 UTF-8 编码: {self.ws.relative(p)},本期不支持"

        count = content.count(old)
        if count == 0:
            return f"❌ 在 {self.ws.relative(p)} 中找不到 old_string。请先 ReadFile 确认实际内容"
        if count > 1:
            return (
                f"❌ old_string 在 {self.ws.relative(p)} 匹配 {count} 处。"
                "请扩大 old_string 的上下文使其唯一"
            )

        new_content = content.replace(old, new, 1)
        if new_content == content:
            return "⚠️ 新内容与原文件相同,无需修改"

        pid = self.store.new_id()
        pe = PendingEdit(
            id=pid,
            kind="edit",
            path=p,
            new_content=new_content,
            diff_preview=_make_diff(self.ws.relative(p), content, new_content),
            source_hash=_sha256(content),
        )
        self.store.put(pe)
        return (
            f"[待确认] pending_id={pid}\n"
            f"即将修改 {self.ws.relative(p)}:\n"
            f"{pe.diff_preview}\n"
            f"请用户回复确认后,再次调用 EditFile,传入 pending_id={pid}, action=apply (或 action=cancel 丢弃)"
        )

    # ── 执行模式 ────────────────────────────────────────────
    def _handle_action(self, pid: str, action: str) -> str:
        # 占位,Task 9 实现
        return "❌ Task 9 will implement this"

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="path", type="string", description="sandbox 内文件路径(提案模式)", required=False),
            ToolParameter(name="old_string", type="string", description="要被替换的原文本(必须在文件中唯一)", required=False),
            ToolParameter(name="new_string", type="string", description="替换后的文本", required=False),
            ToolParameter(name="pending_id", type="string", description="提案模式返回的 id(执行模式用)", required=False),
            ToolParameter(name="action", type="string", description="apply 或 cancel(执行模式用)", required=False),
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest my_agent_llms/test/test_edit_file.py -v`
Expected: 5 passed (the propose-mode tests)

- [ ] **Step 5: Commit**

```bash
git add my_agent_llms/tools/builtin/edit_file.py my_agent_llms/test/test_edit_file.py
git commit -m "feat(tools): add EditFile propose mode with diff + pending_id"
```

---

## Task 9: EditFile apply/cancel + hash guard

**Files:**
- Modify: `my_agent_llms/tools/builtin/edit_file.py` (replace `_handle_action` body)
- Test: `my_agent_llms/test/test_edit_file.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `my_agent_llms/test/test_edit_file.py`:

```python
def test_apply_writes_to_disk(tmp_path):
    tool, ws, store = _make_tool(tmp_path)
    f = tmp_path / "a.md"
    f.write_text("hello 2024\n")
    propose = tool.run({"path": "a.md", "old_string": "2024", "new_string": "2025"})
    pid = _pending_id(propose)
    # 注意 pop is destructive; 重新 put 以模拟 store 状态
    # 真实流程里 propose 完后 pid 还在 store 里,apply 时才 pop
    # _pending_id 只解析字符串,没动 store —— 这里其实没问题
    out = tool.run({"pending_id": pid, "action": "apply"})
    assert out.startswith("✅")
    assert "已修改" in out
    assert f.read_text() == "hello 2025\n"


def test_cancel_discards(tmp_path):
    tool, ws, store = _make_tool(tmp_path)
    f = tmp_path / "a.md"
    f.write_text("hello 2024\n")
    propose = tool.run({"path": "a.md", "old_string": "2024", "new_string": "2025"})
    pid = _pending_id(propose)
    out = tool.run({"pending_id": pid, "action": "cancel"})
    assert "已取消" in out or "丢弃" in out.lower()
    assert f.read_text() == "hello 2024\n"
    # 再次 apply 同一 pid 应失败
    out2 = tool.run({"pending_id": pid, "action": "apply"})
    assert out2.startswith("❌")


def test_apply_unknown_pid(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    out = tool.run({"pending_id": "deadbeef", "action": "apply"})
    assert out.startswith("❌")
    assert "过期" in out or "不存在" in out


def test_apply_rejects_if_file_changed_externally(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    f = tmp_path / "a.md"
    f.write_text("hello 2024\n")
    propose = tool.run({"path": "a.md", "old_string": "2024", "new_string": "2025"})
    pid = _pending_id(propose)
    # 模拟外部编辑
    f.write_text("hello 2024 (touched)\n")
    out = tool.run({"pending_id": pid, "action": "apply"})
    assert out.startswith("❌")
    assert "外部修改" in out or "hash" in out.lower()
    # 文件保持外部修改后的内容,没被覆盖
    assert "(touched)" in f.read_text()


def test_apply_uses_atomic_write(tmp_path):
    """apply 落盘后,sandbox 根下不应残留 .tmp 文件。"""
    tool, ws, _ = _make_tool(tmp_path)
    f = tmp_path / "a.md"
    f.write_text("x\n")
    propose = tool.run({"path": "a.md", "old_string": "x", "new_string": "y"})
    pid = _pending_id(propose)
    tool.run({"pending_id": pid, "action": "apply"})
    leftover = list(tmp_path.glob("*.tmp")) + list(tmp_path.glob(".*.tmp"))
    assert leftover == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest my_agent_llms/test/test_edit_file.py -v -k "apply or cancel"`
Expected: 5 failures (current `_handle_action` returns placeholder string)

- [ ] **Step 3: Replace `_handle_action` and add helpers**

In `my_agent_llms/tools/builtin/edit_file.py`, **replace** the `_handle_action` method body with:

```python
    def _handle_action(self, pid: str, action: str) -> str:
        if action not in ("apply", "cancel"):
            return "❌ action 必须是 apply 或 cancel"

        if action == "cancel":
            if self.store.discard(pid):
                return f"✅ 已取消 pending {pid},文件未改动"
            return f"❌ pending_id {pid} 不存在或已过期"

        # action == apply
        pe = self.store.pop(pid)
        if pe is None:
            return f"❌ pending_id {pid} 不存在或已过期(7 分钟 TTL)。请重新发起编辑"

        # hash 校验
        if pe.source_hash is not None:
            try:
                current = pe.path.read_text(encoding="utf-8")
            except FileNotFoundError:
                return f"❌ 目标文件已被删除: {pe.path}"
            if _sha256(current) != pe.source_hash:
                return (
                    f"❌ 文件在确认期间被外部修改,pending 已失效。"
                    f"请重新读取并发起编辑"
                )

        # 原子写: 先写 tmp 再 rename
        tmp_path = pe.path.with_name(f".{pe.path.name}.tmp")
        try:
            tmp_path.write_text(pe.new_content, encoding="utf-8")
            tmp_path.replace(pe.path)
        except OSError as e:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            return f"❌ 写入失败: {e}。原文件未被改动"

        return f"✅ 已修改 {self.ws.relative(pe.path)}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest my_agent_llms/test/test_edit_file.py -v`
Expected: 10 passed (5 propose + 5 apply/cancel)

- [ ] **Step 5: Commit**

```bash
git add my_agent_llms/tools/builtin/edit_file.py my_agent_llms/test/test_edit_file.py
git commit -m "feat(tools): add EditFile apply/cancel with hash guard + atomic write"
```

---

## Task 10: WriteFile tool (overwrite + new file, both two-step)

**Files:**
- Create: `my_agent_llms/tools/builtin/write_file.py`
- Test: `my_agent_llms/test/test_write_file.py`

- [ ] **Step 1: Write the failing tests**

```python
# my_agent_llms/test/test_write_file.py
import re
from pathlib import Path
import pytest
from my_agent_llms.workspace import Workspace
from my_agent_llms.tools.builtin.pending_edits import PendingEditStore
from my_agent_llms.tools.builtin.write_file import WriteFile


def _make_tool(tmp_path):
    ws = Workspace(tmp_path)
    store = PendingEditStore()
    return WriteFile(ws, store), ws, store


def _pending_id(text: str) -> str:
    m = re.search(r"pending_id=([0-9a-f]+)", text)
    assert m, f"no pending_id in: {text}"
    return m.group(1)


def test_write_new_file_proposes_with_no_diff(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    out = tool.run({"path": "new.md", "content": "hello"})
    assert "[待确认]" in out
    assert "新建文件" in out


def test_write_new_file_applies(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    propose = tool.run({"path": "new.md", "content": "hello"})
    pid = _pending_id(propose)
    out = tool.run({"pending_id": pid, "action": "apply"})
    assert out.startswith("✅")
    assert (tmp_path / "new.md").read_text() == "hello"


def test_write_overwrite_shows_diff(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    (tmp_path / "a.md").write_text("old\n")
    out = tool.run({"path": "a.md", "content": "new\n"})
    assert "[待确认]" in out
    assert "old" in out and "new" in out


def test_write_overwrite_same_content_noop(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    (tmp_path / "a.md").write_text("same\n")
    out = tool.run({"path": "a.md", "content": "same\n"})
    assert out.startswith("⚠️")
    assert "相同" in out


def test_write_rejects_traversal(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    out = tool.run({"path": "../bad.md", "content": "x"})
    assert out.startswith("❌")


def test_write_apply_hash_guard(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    (tmp_path / "a.md").write_text("v1\n")
    propose = tool.run({"path": "a.md", "content": "v2\n"})
    pid = _pending_id(propose)
    (tmp_path / "a.md").write_text("v1.5\n")  # 外部改
    out = tool.run({"pending_id": pid, "action": "apply"})
    assert out.startswith("❌")
    assert (tmp_path / "a.md").read_text() == "v1.5\n"  # 未被覆盖
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest my_agent_llms/test/test_write_file.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

```python
# my_agent_llms/tools/builtin/write_file.py
"""WriteFile —— 写整文件(覆盖或新建),两步确认。"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List

from my_agent_llms.tools.base import Tool, ToolParameter
from my_agent_llms.tools.builtin.pending_edits import (
    PendingEdit,
    PendingEditStore,
)
from my_agent_llms.tools.builtin.edit_file import _make_diff, _sha256
from my_agent_llms.workspace import Workspace, WorkspaceViolation


class WriteFile(Tool):
    def __init__(self, workspace: Workspace, store: PendingEditStore):
        super().__init__(
            name="WriteFile",
            description=(
                "写整个文件内容到 sandbox 路径(覆盖已有或新建)。两步确认: "
                "第一次传 path + content,返回 pending_id 和 diff(新建文件无 diff); "
                "用户确认后再传 pending_id + action=apply。"
            ),
        )
        self.ws = workspace
        self.store = store

    def run(self, parameters: Dict[str, Any]) -> str:
        pid = parameters.get("pending_id")
        if pid:
            return self._handle_action(str(pid), str(parameters.get("action") or ""))
        return self._handle_propose(parameters)

    def _handle_propose(self, parameters: Dict[str, Any]) -> str:
        path = str(parameters.get("path") or "").strip()
        content = parameters.get("content")
        if not path or content is None:
            return "❌ 缺少参数 path / content"

        try:
            p = self.ws.resolve(path)
        except WorkspaceViolation as e:
            return f"❌ {e}"

        if p.exists() and p.is_dir():
            return f"❌ {self.ws.relative(p)} 是目录"

        new_content = str(content)
        rel = self.ws.relative(p)

        if p.exists():
            try:
                old = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return f"❌ 非 UTF-8 编码: {rel},本期不支持覆盖"
            if old == new_content:
                return "⚠️ 新内容与原文件相同,无需修改"
            diff = _make_diff(rel, old, new_content)
            source_hash = _sha256(old)
        else:
            diff = f"(新建文件 {rel},{len(new_content.encode('utf-8'))} 字节)"
            source_hash = None

        pid = self.store.new_id()
        pe = PendingEdit(
            id=pid,
            kind="write",
            path=p,
            new_content=new_content,
            diff_preview=diff,
            source_hash=source_hash,
        )
        self.store.put(pe)
        return (
            f"[待确认] pending_id={pid}\n"
            f"{'即将覆盖' if source_hash else '即将新建'} {rel}:\n"
            f"{diff}\n"
            f"请用户回复确认后,再次调用 WriteFile,传入 pending_id={pid}, action=apply"
        )

    def _handle_action(self, pid: str, action: str) -> str:
        if action not in ("apply", "cancel"):
            return "❌ action 必须是 apply 或 cancel"
        if action == "cancel":
            return f"✅ 已取消 pending {pid},文件未改动" if self.store.discard(pid) \
                else f"❌ pending_id {pid} 不存在或已过期"

        pe = self.store.pop(pid)
        if pe is None:
            return f"❌ pending_id {pid} 不存在或已过期(7 分钟 TTL)"

        if pe.source_hash is not None:
            try:
                current = pe.path.read_text(encoding="utf-8")
            except FileNotFoundError:
                return f"❌ 目标文件已被删除: {pe.path}"
            if _sha256(current) != pe.source_hash:
                return "❌ 文件在确认期间被外部修改,pending 已失效。请重新发起"

        tmp = pe.path.with_name(f".{pe.path.name}.tmp")
        try:
            pe.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(pe.new_content, encoding="utf-8")
            tmp.replace(pe.path)
        except OSError as e:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            return f"❌ 写入失败: {e}。原文件未被改动"

        action_word = "已写入" if pe.source_hash is None else "已覆盖"
        return f"✅ {action_word} {self.ws.relative(pe.path)}"

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="path", type="string", description="sandbox 内文件路径(提案模式)", required=False),
            ToolParameter(name="content", type="string", description="完整新文件内容(提案模式)", required=False),
            ToolParameter(name="pending_id", type="string", description="提案返回的 id(执行模式)", required=False),
            ToolParameter(name="action", type="string", description="apply 或 cancel", required=False),
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest my_agent_llms/test/test_write_file.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add my_agent_llms/tools/builtin/write_file.py my_agent_llms/test/test_write_file.py
git commit -m "feat(tools): add WriteFile with two-step confirm + hash guard"
```

---

## Task 11: ExportFile tool (write back to original)

**Files:**
- Create: `my_agent_llms/tools/builtin/export_file.py`
- Test: `my_agent_llms/test/test_export_file.py`

- [ ] **Step 1: Write the failing tests**

```python
# my_agent_llms/test/test_export_file.py
import re
from pathlib import Path
import pytest
from my_agent_llms.workspace import Workspace
from my_agent_llms.tools.builtin.pending_edits import PendingEditStore
from my_agent_llms.tools.builtin.export_file import ExportFile


def _make_tool(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    ws = Workspace(sandbox)
    store = PendingEditStore()
    return ExportFile(ws, store), ws, store


def _pending_id(text: str) -> str:
    m = re.search(r"pending_id=([0-9a-f]+)", text)
    assert m
    return m.group(1)


def test_export_attached_uses_manifest_dest(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    src = tmp_path / "report.md"
    src.write_text("v1\n")
    ws.attach(src)
    (ws.root / "report.md").write_text("v2\n")  # 在 sandbox 内改了

    propose = tool.run({"sandbox_path": "report.md"})
    assert "[待确认]" in propose
    assert str(src) in propose  # diff 里能看到目标路径
    pid = _pending_id(propose)
    out = tool.run({"pending_id": pid, "action": "apply"})
    assert out.startswith("✅")
    assert src.read_text() == "v2\n"


def test_export_new_file_requires_dest(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    (ws.root / "summary.md").write_text("brand new\n")
    out = tool.run({"sandbox_path": "summary.md"})
    assert out.startswith("❌")
    assert "dest" in out.lower() or "目标" in out


def test_export_new_file_with_explicit_dest(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    (ws.root / "summary.md").write_text("brand new\n")
    dest = tmp_path / "out" / "summary.md"
    dest.parent.mkdir()
    propose = tool.run({"sandbox_path": "summary.md", "dest_path": str(dest)})
    pid = _pending_id(propose)
    out = tool.run({"pending_id": pid, "action": "apply"})
    assert out.startswith("✅")
    assert dest.read_text() == "brand new\n"


def test_export_rejects_deny_dest(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    (ws.root / "k.md").write_text("x\n")
    out = tool.run({"sandbox_path": "k.md", "dest_path": str(tmp_path / "secret.pem")})
    assert out.startswith("❌")
    assert "黑名单" in out


def test_export_apply_hash_guard(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    src = tmp_path / "a.md"
    src.write_text("v1\n")
    ws.attach(src)
    (ws.root / "a.md").write_text("v2\n")

    propose = tool.run({"sandbox_path": "a.md"})
    pid = _pending_id(propose)
    src.write_text("v1.5\n")  # 外部改原文件
    out = tool.run({"pending_id": pid, "action": "apply"})
    assert out.startswith("❌")
    assert src.read_text() == "v1.5\n"


def test_export_missing_sandbox_file(tmp_path):
    tool, ws, _ = _make_tool(tmp_path)
    out = tool.run({"sandbox_path": "nope.md"})
    assert out.startswith("❌")
    assert "不存在" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest my_agent_llms/test/test_export_file.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

```python
# my_agent_llms/tools/builtin/export_file.py
"""ExportFile —— 把 sandbox 内文件写回外部真实路径,两步确认。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from my_agent_llms.tools.base import Tool, ToolParameter
from my_agent_llms.tools.builtin.edit_file import _make_diff, _sha256
from my_agent_llms.tools.builtin.pending_edits import (
    PendingEdit,
    PendingEditStore,
)
from my_agent_llms.workspace import (
    DEFAULT_DENY_DIRS,
    DEFAULT_DENY_SUFFIXES,
    Workspace,
    WorkspaceViolation,
)


def _check_dest_deny(dest: Path) -> None:
    """对外部 dest 路径做黑名单校验(与 Workspace.resolve 内的规则一致)。"""
    for part in dest.parts:
        if part in DEFAULT_DENY_DIRS:
            raise WorkspaceViolation(f"导出目标命中黑名单目录: {dest}")
    if dest.suffix in DEFAULT_DENY_SUFFIXES:
        raise WorkspaceViolation(f"导出目标文件类型在黑名单: {dest.suffix}")


class ExportFile(Tool):
    def __init__(self, workspace: Workspace, store: PendingEditStore):
        super().__init__(
            name="ExportFile",
            description=(
                "把 sandbox 内文件写回外部真实路径。两步确认: "
                "第一次传 sandbox_path (+ dest_path 若为新建文件),返回 pending_id 和 diff vs 原文件; "
                "用户确认后再传 pending_id + action=apply。"
            ),
        )
        self.ws = workspace
        self.store = store

    def run(self, parameters: Dict[str, Any]) -> str:
        pid = parameters.get("pending_id")
        if pid:
            return self._handle_action(str(pid), str(parameters.get("action") or ""))
        return self._handle_propose(parameters)

    def _handle_propose(self, parameters: Dict[str, Any]) -> str:
        sb_path = str(parameters.get("sandbox_path") or "").strip()
        dest_arg = parameters.get("dest_path")
        if not sb_path:
            return "❌ 缺少参数 sandbox_path"

        try:
            sb = self.ws.resolve(sb_path)
        except WorkspaceViolation as e:
            return f"❌ {e}"
        if not sb.exists():
            return f"❌ sandbox 文件不存在: {self.ws.relative(sb)}"
        if sb.is_dir():
            return f"❌ {self.ws.relative(sb)} 是目录"

        # 确定目标
        if dest_arg:
            dest = Path(str(dest_arg)).expanduser().resolve()
        else:
            origin = self.ws.origin_of(sb)
            if origin is None:
                return (
                    f"❌ {self.ws.relative(sb)} 是 sandbox 内新建文件,"
                    "MANIFEST 中无对应源路径。请显式提供 dest_path"
                )
            dest = origin

        try:
            _check_dest_deny(dest)
        except WorkspaceViolation as e:
            return f"❌ {e}"

        try:
            new_content = sb.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"❌ 非 UTF-8 编码: {self.ws.relative(sb)},本期不支持"

        if dest.exists():
            try:
                old = dest.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return f"❌ 原目标文件非 UTF-8: {dest},本期不支持覆盖"
            if old == new_content:
                return "⚠️ sandbox 内容与原目标完全一致,无需导出"
            diff = _make_diff(str(dest), old, new_content)
            source_hash = _sha256(old)
        else:
            diff = f"(新建文件 {dest},{len(new_content.encode('utf-8'))} 字节)"
            source_hash = None

        pid = self.store.new_id()
        pe = PendingEdit(
            id=pid,
            kind="export",
            path=dest,
            new_content=new_content,
            diff_preview=diff,
            source_hash=source_hash,
        )
        self.store.put(pe)
        return (
            f"[待确认] pending_id={pid}\n"
            f"即将把 sandbox {self.ws.relative(sb)} 写回 {dest}:\n"
            f"{diff}\n"
            f"请用户回复确认后,再次调用 ExportFile,传入 pending_id={pid}, action=apply"
        )

    def _handle_action(self, pid: str, action: str) -> str:
        if action not in ("apply", "cancel"):
            return "❌ action 必须是 apply 或 cancel"
        if action == "cancel":
            return f"✅ 已取消 pending {pid}" if self.store.discard(pid) \
                else f"❌ pending_id {pid} 不存在或已过期"

        pe = self.store.pop(pid)
        if pe is None:
            return f"❌ pending_id {pid} 不存在或已过期(7 分钟 TTL)"

        if pe.source_hash is not None:
            try:
                current = pe.path.read_text(encoding="utf-8")
            except FileNotFoundError:
                return f"❌ 原目标文件已被删除: {pe.path}"
            if _sha256(current) != pe.source_hash:
                return "❌ 原文件 hash 变化,导出会覆盖外部修改。请重新发起"

        tmp = pe.path.with_name(f".{pe.path.name}.tmp")
        try:
            pe.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(pe.new_content, encoding="utf-8")
            tmp.replace(pe.path)
        except OSError as e:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            return f"❌ 写入失败: {e}。原文件未被改动"

        return f"✅ 已写回 {pe.path}"

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="sandbox_path", type="string", description="sandbox 内的文件路径", required=False),
            ToolParameter(name="dest_path", type="string", description="外部目标路径(新建文件必传)", required=False),
            ToolParameter(name="pending_id", type="string", description="提案返回的 id", required=False),
            ToolParameter(name="action", type="string", description="apply 或 cancel", required=False),
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest my_agent_llms/test/test_export_file.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add my_agent_llms/tools/builtin/export_file.py my_agent_llms/test/test_export_file.py
git commit -m "feat(tools): add ExportFile with diff vs origin + hash guard"
```

---

## Task 12: chat.py integration

**Files:**
- Modify: `chat.py` (add `workspace` config key, register 6 tools, extend system prompt)

`chat.py` 使用 config dict + slash 命令,不是 argparse。本任务把 `workspace` 加进 `DEFAULT_CONFIG`,在 `build_agent` 里构造 `Workspace` + `PendingEditStore` 并注册 6 个工具,同时扩展 system_prompt 教会 LLM 走两步流程。

- [ ] **Step 1: Read current chat.py to find exact insertion points**

Run: `grep -n "DEFAULT_CONFIG\|register_tool(CalculatorTool" chat.py`

Expected: see line numbers for `DEFAULT_CONFIG = {` (around line 76) and `registry.register_tool(CalculatorTool())` (line 160).

- [ ] **Step 2: Add `workspace` key to DEFAULT_CONFIG**

In `chat.py`, locate the `DEFAULT_CONFIG: Dict = {` block. Add a new top-level key `"workspace": None` near the bottom of that dict (after the existing keys, before the closing `}`). Use Edit to make this change — show the exact change as a single `replace_all=false` edit:

Find the closing `}` of `DEFAULT_CONFIG` and insert `"workspace": None,  # None → 自动建沙箱; 字符串 → 用该绝对/相对路径` on a new line before it. Verify by:

Run: `python -c "import json; from chat import DEFAULT_CONFIG; print(json.dumps(DEFAULT_CONFIG, indent=2))" | grep workspace`
Expected: `"workspace": null,`

- [ ] **Step 3: Wire Workspace + tools into build_agent**

In `chat.py`, locate this block:

```python
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
```

Replace it with:

```python
    # ── Doc Editor: 沙箱 + 6 个文件工具 ────────────────────
    from my_agent_llms.workspace import Workspace
    from my_agent_llms.tools.builtin.pending_edits import PendingEditStore
    from my_agent_llms.tools.builtin.read_file import ReadFile
    from my_agent_llms.tools.builtin.edit_file import EditFile
    from my_agent_llms.tools.builtin.write_file import WriteFile
    from my_agent_llms.tools.builtin.list_dir import ListDir
    from my_agent_llms.tools.builtin.attach_file import AttachFile
    from my_agent_llms.tools.builtin.export_file import ExportFile

    try:
        ws = Workspace(cfg.get("workspace"))
    except (FileNotFoundError, NotADirectoryError) as exc:
        console.print(f"[red]Workspace 初始化失败: {exc}[/red]")
        return None
    console.print(f"[dim]Workspace: {ws.root}[/dim]")
    pe_store = PendingEditStore()

    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
    registry.register_tool(ReadFile(ws))
    registry.register_tool(EditFile(ws, pe_store))
    registry.register_tool(WriteFile(ws, pe_store))
    registry.register_tool(ListDir(ws))
    registry.register_tool(AttachFile(ws))
    registry.register_tool(ExportFile(ws, pe_store))
```

- [ ] **Step 4: Extend system_prompt with file-editing protocol**

In `chat.py`, locate this block inside `build_agent`:

```python
            system_prompt=(
                "你是 lk_hhh 的长期 AI 伙伴。你会记住所有重要对话,"
                "并在用户的偏好/事实变化时主动更新记忆。"
                "用自然、温暖但不啰嗦的语气。"
            ),
```

Replace with:

```python
            system_prompt=(
                "你是 lk_hhh 的长期 AI 伙伴。你会记住所有重要对话,"
                "并在用户的偏好/事实变化时主动更新记忆。"
                "用自然、温暖但不啰嗦的语气。\n\n"
                "## 文件操作协议\n"
                "你拥有 sandbox 文件工具。流程必须严格遵守:\n"
                "1. 用户提到任何外部路径(如 ./report.md, /Users/.../foo.md),先用 AttachFile 把文件复制进 sandbox\n"
                "2. 在 sandbox 内用 ReadFile / EditFile / WriteFile / ListDir 操作\n"
                "3. EditFile 和 WriteFile 是两步: 第一次调用返回 [待确认] pending_id 和 diff; "
                "   等用户在对话里明确说'确认'后,再用 pending_id + action=apply 调一次落盘\n"
                "4. 用户希望把修改写回原位置时,用 ExportFile(同样两步确认 + diff vs 原文件)\n"
                "5. sandbox 内新建的文件,ExportFile 时必须显式给 dest_path\n"
            ),
```

- [ ] **Step 5: Manual smoke test**

Start chat.py and run a smoke test:

```bash
python chat.py
```

Expected first line: `Workspace: /Users/lk_hhh/.my_agent_llms/workspaces/<ts>-<hex>`

In the chat, type: `/quit` to exit cleanly. Then re-run pytest to ensure nothing regressed:

```bash
pytest my_agent_llms/test/ -v --ignore=my_agent_llms/test/integration
```

Expected: all unit tests pass.

- [ ] **Step 6: Commit**

```bash
git add chat.py
git commit -m "feat(chat): wire Workspace + 6 doc-editor tools into build_agent"
```

---

## Task 13: Integration smoke test (end-to-end with real LLM)

**Files:**
- Create: `my_agent_llms/test/integration/__init__.py` (empty)
- Create: `my_agent_llms/test/integration/test_doc_edit_flow.py`

This test runs against a real LLM and validates the full `attach → edit → apply → export → apply` flow. Skipped unless `MY_LLM_API_KEY` is set (avoid breaking CI of contributors without keys).

- [ ] **Step 1: Write the test**

```python
# my_agent_llms/test/integration/__init__.py
```

```python
# my_agent_llms/test/integration/test_doc_edit_flow.py
"""端到端:让真实 LLM 走完 attach → edit → apply → export → apply 完整流程。

跳过条件: 未设置 MY_LLM_API_KEY 环境变量。
"""
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("MY_LLM_API_KEY"),
    reason="需要 MY_LLM_API_KEY 环境变量才能跑 (调用真实 LLM)",
)


def test_full_attach_edit_export_flow(tmp_path):
    from my_agent_llms.agents.function_call_agent import MyFunctionCallAgent
    from my_agent_llms.core.llm import MyLLM
    from my_agent_llms.workspace import Workspace
    from my_agent_llms.tools.registry import ToolRegistry
    from my_agent_llms.tools.builtin.pending_edits import PendingEditStore
    from my_agent_llms.tools.builtin.read_file import ReadFile
    from my_agent_llms.tools.builtin.edit_file import EditFile
    from my_agent_llms.tools.builtin.write_file import WriteFile
    from my_agent_llms.tools.builtin.list_dir import ListDir
    from my_agent_llms.tools.builtin.attach_file import AttachFile
    from my_agent_llms.tools.builtin.export_file import ExportFile

    # 准备一个 sandbox 和一份"用户的真实文件"
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    user_file = tmp_path / "user_docs" / "hello.md"
    user_file.parent.mkdir()
    user_file.write_text("Hello, 2024\n")

    ws = Workspace(sandbox)
    store = PendingEditStore()
    registry = ToolRegistry()
    for t in [
        ReadFile(ws),
        EditFile(ws, store),
        WriteFile(ws, store),
        ListDir(ws),
        AttachFile(ws),
        ExportFile(ws, store),
    ]:
        registry.register_tool(t)

    llm = MyLLM(
        api_key=os.environ["MY_LLM_API_KEY"],
        base_url=os.getenv("MY_LLM_BASE_URL"),
        model=os.getenv("MY_LLM_MODEL", "deepseek-chat"),
    )
    agent = MyFunctionCallAgent(
        name="tester",
        llm=llm,
        tool_registry=registry,
        system_prompt=(
            "你有 sandbox 文件工具。流程:\n"
            "1. 外部路径先 AttachFile\n"
            "2. EditFile / WriteFile 是两步: 提案返回 pending_id+diff,用户说'确认'后再 action=apply\n"
            "3. 想写回原位置用 ExportFile (同样两步)\n"
        ),
        max_steps=15,
    )

    # 模拟一次完整对话: 用户连续 2 条消息
    agent.chat(f"把 {user_file} 里的 2024 改成 2025")
    # 上一步 LLM 应该 attach + read + propose edit; 我们模拟"确认"
    agent.chat("确认。然后写回原位置,并再次确认。")

    # 最终断言: 原文件内容已被更新
    assert user_file.read_text() == "Hello, 2025\n", (
        f"原文件未被正确更新,实际: {user_file.read_text()!r}"
    )
```

- [ ] **Step 2: Run the test (skipped if no API key)**

Run: `pytest my_agent_llms/test/integration/ -v`
Expected (no API key): `1 skipped`
Expected (with API key set): `1 passed` (允许重跑 1-2 次,因为 LLM 行为有随机性)

- [ ] **Step 3: Commit**

```bash
git add my_agent_llms/test/integration/
git commit -m "test(integration): end-to-end attach-edit-export flow with real LLM"
```

---

## Final verification

After all tasks:

```bash
pytest my_agent_llms/test/ -v --ignore=my_agent_llms/test/integration
```

Expected counts:
- test_workspace.py: 20 tests
- test_pending_edit_store.py: 6 tests
- test_read_file.py: 7 tests
- test_list_dir.py: 5 tests
- test_attach_file.py: 4 tests
- test_edit_file.py: 10 tests
- test_write_file.py: 6 tests
- test_export_file.py: 6 tests

Total: **64 passing unit tests** + 1 integration test (skipped unless API key).

Smoke test the CLI once more:

```bash
python chat.py
```

Confirm `Workspace: ...` is printed and all 7 tools (Calculator + 6 new) are listed in `/tools` (or whichever command shows registered tools).
