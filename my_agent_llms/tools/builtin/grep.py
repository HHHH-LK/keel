"""Grep —— 按内容搜文件(rg 优先,纯 Python 兜底),锁项目根子树。"""
from __future__ import annotations

import fnmatch
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from my_agent_llms.tools.base import Tool, ToolParameter
from my_agent_llms.workspace import Workspace, WorkspaceViolation

_MAX_CONTENT_LINES = 100
_MAX_FILES = 80


class GrepTool(Tool):
    side_effect_free = True  # 纯读,可并行,无需审批

    def __init__(self, workspace: Workspace):
        super().__init__(
            name="Grep",
            description=(
                "按正则搜文件内容,用于定位代码后再 Read 精读(别整文件全读)。"
                "output_mode=content 返 路径:行号:内容; files 返匹配文件列表。"
                "可加 glob(如 *.py)过滤、context 上下文行数、ignore_case。只搜当前工作区。"
            ),
        )
        self.ws = workspace

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="pattern", type="string", description="正则表达式", required=True),
            ToolParameter(name="path", type="string", description="搜索起点(默认工作区根),限工作区内", required=False, default="."),
            ToolParameter(name="glob", type="string", description="文件名过滤,如 *.py", required=False),
            ToolParameter(name="output_mode", type="string", description="content(路径:行号:内容) 或 files(匹配文件列表)", required=False, default="content"),
            ToolParameter(name="context", type="integer", description="content 模式上下文行数(-C)", required=False, default=0),
            ToolParameter(name="ignore_case", type="boolean", description="忽略大小写", required=False, default=False),
        ]

    def run(self, parameters: Dict[str, Any]) -> str:
        pattern = str(parameters.get("pattern") or "").strip()
        if not pattern:
            return "❌ 缺少 pattern 参数"
        path = str(parameters.get("path") or ".").strip() or "."
        glob = str(parameters.get("glob") or "").strip()
        mode = str(parameters.get("output_mode") or "content").strip().lower()
        if mode not in ("content", "files"):
            mode = "content"
        ignore_case = bool(parameters.get("ignore_case"))
        try:
            ctx = max(0, int(parameters.get("context") or 0))
        except (ValueError, TypeError):
            ctx = 0

        try:
            base = self.ws.resolve(path)
        except WorkspaceViolation as e:
            return f"❌ {e}"
        if not base.exists():
            return f"❌ 路径不存在: {path}"
        try:
            regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as e:
            return f"❌ 正则错误: {e}"

        if shutil.which("rg"):
            return self._run_rg(pattern, base, glob, mode, ctx, ignore_case)
        return self._run_python(regex, base, glob, mode, ctx)

    def _run_rg(self, pattern: str, base: Path, glob: str, mode: str,
                ctx: int, ignore_case: bool) -> str:
        search_dir = base if base.is_dir() else base.parent
        target = "." if base.is_dir() else base.name
        cmd = ["rg", "--color", "never"]
        if mode == "files":
            cmd.append("--files-with-matches")
        else:
            cmd += ["--line-number", "--no-heading"]
            if ctx:
                cmd += ["-C", str(ctx)]
        if ignore_case:
            cmd.append("--ignore-case")
        if glob:
            cmd += ["--glob", glob]
        cmd += ["-e", pattern, "--", target]
        try:
            proc = subprocess.run(cmd, cwd=str(search_dir), capture_output=True,
                                  text=True, timeout=20)
        except Exception:
            return self._run_python(
                re.compile(pattern, re.IGNORECASE if ignore_case else 0),
                base, glob, mode, ctx)
        if proc.returncode == 1:
            return "(无匹配)"
        if proc.returncode >= 2:
            return f"❌ rg 错误: {proc.stderr.strip()[:200]}"
        lines = proc.stdout.rstrip("\n").split("\n") if proc.stdout.strip() else []
        if not lines:
            return "(无匹配)"
        cap = _MAX_FILES if mode == "files" else _MAX_CONTENT_LINES
        return self._cap(lines, cap)

    def _run_python(self, regex: "re.Pattern", base: Path, glob: str,
                    mode: str, ctx: int) -> str:
        files = [base] if base.is_file() else list(self._walk_files(base, glob))
        if base.is_file() and glob and not fnmatch.fnmatch(base.name, glob):
            files = []
        content: List[str] = []
        match_files: List[str] = []
        for f in files:
            try:
                raw = f.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw[:4096]:
                continue
            try:
                flines = raw.decode("utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            hit = False
            for idx, ln in enumerate(flines):
                if regex.search(ln):
                    hit = True
                    if mode == "content":
                        rel = self.ws.relative(f)
                        lo, hi = max(0, idx - ctx), min(len(flines), idx + ctx + 1)
                        for j in range(lo, hi):
                            content.append(f"{rel}:{j + 1}: {flines[j]}")
            if hit and mode == "files":
                match_files.append(self.ws.relative(f))
            if mode == "content" and len(content) >= _MAX_CONTENT_LINES:
                break
            if mode == "files" and len(match_files) >= _MAX_FILES:
                break
        out = match_files if mode == "files" else content
        if not out:
            return "(无匹配)"
        cap = _MAX_FILES if mode == "files" else _MAX_CONTENT_LINES
        return self._cap(out, cap)

    def _walk_files(self, base: Path, glob: str):
        deny = self.ws._deny_dirs
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            if any(part in deny for part in p.relative_to(base).parts):
                continue
            if glob and not fnmatch.fnmatch(p.name, glob):
                continue
            yield p

    @staticmethod
    def _cap(items: List[str], cap: int) -> str:
        if len(items) > cap:
            return "\n".join(items[:cap]) + f"\n… +{len(items) - cap} more(请缩小 pattern 或 path)"
        return "\n".join(items)
