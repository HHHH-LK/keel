"""MY AI 伙伴 —— CLI 聊天界面。

启动即用,没配置 API Key 也能进:
- 进入后用 /config 查看当前配置
- 用 /config key 输入 API Key
- 用 /config model <id> 改模型
- 用 /config conflict extreme 等改 memory 行为

特性:
- Markdown 渲染回答(代码块语法高亮)
- 持久化输入历史(↑↓ 翻)
- Slash 命令完整体系
- 配置实时改、实时生效(自动重建 agent)
- 优雅降级(没 key 不崩,提示用 /config)
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import radiolist_dialog
from prompt_toolkit.styles import Style
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.text import Text

from my_agent_llms.agents.function_call_agent import MyFunctionCallAgent
from my_agent_llms.core.llm import MyLLM
from my_agent_llms.memory import (
    LLMSummarizer,
    MemoryConfig,
    MemoryManager,
    OpenAIEmbedding,
)
from my_agent_llms.tools.registry import ToolRegistry
from my_agent_llms.tools.builtin.calculator import CalculatorTool


load_dotenv()
console = Console()


# ────────────────────────────────────────────────────────────
# 厂商 preset
# ────────────────────────────────────────────────────────────

PROVIDER_PRESETS = {
    "openai":     ("OpenAI 官方",         "openai",     "https://api.openai.com/v1",                          "gpt-4o"),
    "aliyun":     ("阿里云 (通义千问)",   "aliyun",     "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
    "zhipu":      ("智谱 GLM",           "zhipu",      "https://open.bigmodel.cn/api/paas/v4/",             "glm-4-flash"),
    "deepseek":   ("DeepSeek",           "openai",     "https://api.deepseek.com/v1",                        "deepseek-chat"),
    "minimax":    ("MiniMax",            "openai",     "https://api.minimaxi.com/v1",                        "MiniMax-Text-01"),
    "modelscope": ("魔搭 ModelScope",    "modelscope", "https://api-inference.modelscope.cn/v1/",            "Qwen/Qwen2.5-72B-Instruct"),
    "ollama":     ("本地 Ollama",        "ollama",     "http://localhost:11434/v1",                          "qwen2.5"),
    "custom":     ("自定义",             "openai",     None,                                                  None),
}


# ────────────────────────────────────────────────────────────
# 配置加载 / 持久化
# ────────────────────────────────────────────────────────────

DEFAULT_STORAGE_DIR = Path.home() / ".my_companion"
CONFIG_PATH = DEFAULT_STORAGE_DIR / "config.json"

DEFAULT_CONFIG: Dict = {
    # LLM
    "provider_key": "openai",
    "provider":     "openai",
    "model":        "",
    "api_key":      "",
    "base_url":     None,
    # Memory
    "memory": {
        "cold_backend":      "sqlite",
        "vector_backend":    "sqlite",
        "conflict_strength": "fast",     # 默认 fast,不依赖 embedding API
        "tick_mode":         "async",
        "use_embedding":     False,      # 默认关,避免没 embedding API 时报错
    },
    # Doc Editor
    "workspace": None,  # None → 自动建沙箱; 字符串 → 用该绝对/相对路径
}


def load_config() -> Dict:
    """加载配置;文件不存在或损坏 → 返回默认。永不报错,永不阻塞启动。"""
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return _apply_env_fallback(_merge_defaults(data))
        except Exception as exc:
            console.print(f"[yellow]⚠️ 配置文件损坏 ({exc}),用默认配置[/yellow]")
    return _apply_env_fallback(_merge_defaults({}))


def _merge_defaults(data: Dict) -> Dict:
    out = {**DEFAULT_CONFIG, **data}
    out["memory"] = {**DEFAULT_CONFIG["memory"], **(data.get("memory") or {})}
    return out


def _apply_env_fallback(cfg: Dict) -> Dict:
    """config.json 字段为空时,从 .env 兜底 (LLM_API_KEY / LLM_MODEL_ID / LLM_BASE_URL)。"""
    for cfg_key, env_key in (("api_key", "LLM_API_KEY"),
                              ("model",   "LLM_MODEL_ID"),
                              ("base_url","LLM_BASE_URL")):
        if not cfg.get(cfg_key):
            env_val = (os.getenv(env_key) or "").strip().strip('"').strip("'")
            if env_val:
                cfg[cfg_key] = env_val
    return cfg


def save_config(cfg: Dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        CONFIG_PATH.chmod(0o600)
    except Exception:
        pass


def is_ready(cfg: Dict) -> bool:
    return bool(cfg.get("api_key") and cfg.get("model") and cfg.get("provider"))


# ────────────────────────────────────────────────────────────
# Agent 装配 (失败时返回 None)
# ────────────────────────────────────────────────────────────

def build_agent(cfg: Dict) -> Optional[MyFunctionCallAgent]:
    """根据 cfg 构建 agent。配置不全或构造失败 → 返回 None(不抛异常)。"""
    if not is_ready(cfg):
        return None

    try:
        llm = MyLLM(
            provider=cfg["provider"],
            model=cfg["model"],
            api_key=cfg["api_key"],
            base_url=cfg.get("base_url"),
        )
    except Exception as exc:
        console.print(f"[red]LLM 初始化失败: {exc}[/red]")
        return None

    storage_dir = Path(os.getenv("MY_CHAT_STORAGE", str(DEFAULT_STORAGE_DIR)))
    mem_cfg_data = cfg["memory"]
    memory_config = MemoryConfig(
        storage_dir=storage_dir,
        cold_backend=mem_cfg_data["cold_backend"],
        vector_backend=mem_cfg_data["vector_backend"],
        l1_max_tokens=4000,
        l1_recent_turns=6,
        promote_threshold=0.6,
        promote_min_hits=3,
        tick_mode=mem_cfg_data["tick_mode"],
        tick_every_n_turns=3,
        conflict_strength=mem_cfg_data["conflict_strength"],
    )

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

    try:
        agent = MyFunctionCallAgent(
            name="伙伴",
            llm=llm,
            tool_registry=registry,
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
            memory_config=memory_config,
            max_steps=5,
        )
    except Exception as exc:
        console.print(f"[red]Agent 构造失败: {exc}[/red]")
        return None

    # 升级 memory(embedding + summarizer + KG 等高级能力)
    if mem_cfg_data["use_embedding"]:
        try:
            agent.memory = MemoryManager(
                memory_config,
                embedding=OpenAIEmbedding.from_llm(llm),
                summarizer=LLMSummarizer(llm, max_tokens=400),
                llm=llm,
            )
        except Exception as exc:
            console.print(f"[yellow]⚠️ embedding 升级失败 ({exc}),用 TF-IDF[/yellow]")
            agent.memory = MemoryManager(
                memory_config,
                summarizer=LLMSummarizer(llm, max_tokens=400),
                llm=llm,
            )
    else:
        agent.memory = MemoryManager(
            memory_config,
            summarizer=LLMSummarizer(llm, max_tokens=400),
            llm=llm,
        )

    return agent


# ────────────────────────────────────────────────────────────
# Slash 命令
# ────────────────────────────────────────────────────────────

def cmd_help(_cli) -> None:
    sections = [
        ("Basic", [
            ("/help",                          "show this help"),
            ("/quit, /exit, Ctrl+D",           "exit"),
            ("/multiline",                     "toggle multiline input (Esc+Enter)"),
        ]),
        ("Config", [
            ("/config",                        "interactive wizard (↑↓ pick provider · enter model · paste key)"),
            ("/config show",                   "show current config"),
            ("/config provider <name>",        "openai/aliyun/zhipu/deepseek/minimax/..."),
            ("/config model <id>",             "set model id"),
            ("/config key",                    "set API key (hidden input)"),
            ("/config base_url <url>",         "set base url"),
            ("/config cold <jsonl|sqlite>",    "cold storage backend"),
            ("/config vector <memory|sqlite>", "vector store backend"),
            ("/config conflict <strength>",    "off|fast|accurate|extreme"),
            ("/config tick <mode>",            "sync|async|off"),
            ("/config embedding <on|off>",     "toggle OpenAI embedding"),
            ("/config reset",                  "reset all config"),
        ]),
        ("Memory", [
            ("/clear",                         "clear conversation context (L1+L2, keeps L4/L5)"),
            ("/memory",                        "show memory stats"),
            ("/kg",                            "export knowledge graph (mermaid)"),
            ("/recall <query>",                "search long-term memory"),
            ("/facts <query>",                 "query KG facts"),
        ]),
    ]
    console.print()
    for title, rows in sections:
        console.print(f"  [bold]{title}[/bold]")
        for cmd, desc in rows:
            console.print(f"    [cyan]{cmd:<34}[/cyan] [dim]{desc}[/dim]")
        console.print()


def cmd_clear(cli) -> None:
    """清屏 + 清 L1 工作内存,L4/L5 持久化数据保留。"""
    if cli.agent is not None:
        cli.agent.memory.clear()
    # 清屏并重画 banner —— 类似 Claude Code 的 /clear
    console.clear()
    cli.print_banner()
    console.print("[dim]  context cleared · L4/L5 长期记忆已保留[/dim]")


def cmd_memory(cli) -> None:
    if cli.agent is None:
        console.print("[yellow]Agent 未构建。先用 /config key 配置好 API Key。[/yellow]")
        return
    stats = cli.agent.memory.stats()
    labels = {
        "l1_items":  "L1 items",
        "l1_tokens": "L1 tokens",
        "l2_tokens": "L2 summary tokens",
        "l4_items":  "L4 cold items",
        "l5_items":  "L5 vector items",
    }
    console.print()
    console.print("  [bold]Memory stats[/bold]")
    for k, v in stats.items():
        console.print(f"    [dim]{labels.get(k, k):<22}[/dim] [bright_white]{v}[/bright_white]")
    console.print()


def cmd_kg(cli) -> None:
    if cli.agent is None:
        console.print("  [dim](agent not ready)[/dim]")
        return
    try:
        mermaid = cli.agent.memory.export_kg_graph(format="mermaid", include_inactive=False)
        console.print()
        console.print(mermaid)
        console.print()
        console.print("  [dim]paste into https://mermaid.live to render[/dim]")
    except Exception as exc:
        console.print(f"  [red]export failed: {exc}[/red]")


def cmd_recall(cli, query: str) -> None:
    if cli.agent is None:
        console.print("  [dim](agent not ready)[/dim]")
        return
    if not query:
        console.print("  [dim]usage: /recall <query>[/dim]")
        return
    hits = cli.agent.memory.recall(query, k=5)
    if not hits:
        console.print("  [dim]no matches[/dim]")
        return
    console.print()
    for item, score in hits:
        console.print(
            f"  [yellow]{score:5.2f}[/yellow]  "
            f"[cyan]{item.role:<10}[/cyan] "
            f"{item.content[:100]}"
        )
    console.print()


def cmd_facts(cli, query: str) -> None:
    if cli.agent is None:
        console.print("[yellow]Agent 未构建。[/yellow]")
        return
    if not query:
        console.print("[yellow]用法: /facts <查询词>[/yellow]")
        return
    facts = cli.agent.memory.recall_facts(query)
    if not facts:
        console.print("[dim]KG 中无相关事实(conflict_strength 需为 extreme 且有积累)[/dim]")
        return
    for f in facts:
        console.print(f"  [green]•[/green] {f}")


# ────────────────────────────────────────────────────────────
# /config 子命令路由
# ────────────────────────────────────────────────────────────

def cmd_show_config(cli) -> None:
    cfg = cli.cfg
    masked_key = (cfg["api_key"][:6] + "…" + cfg["api_key"][-4:]) if cfg.get("api_key") else "[dim](not set)[/dim]"

    def row(k: str, v: str) -> None:
        console.print(f"    [dim]{k:<20}[/dim] {v}")

    console.print()
    console.print("  [bold]LLM[/bold]")
    row("provider_key", f"[cyan]{cfg.get('provider_key', '?')}[/cyan]")
    row("provider",     cfg["provider"])
    row("model",        cfg["model"] or "[dim](not set)[/dim]")
    row("base_url",     str(cfg.get("base_url", "") or "[dim]—[/dim]"))
    row("api_key",      masked_key)
    console.print()
    console.print("  [bold]Memory[/bold]")
    mem = cfg["memory"]
    row("cold_backend",      mem["cold_backend"])
    row("vector_backend",    mem["vector_backend"])
    row("conflict_strength", mem["conflict_strength"])
    row("tick_mode",         mem["tick_mode"])
    row("use_embedding",     "on" if mem["use_embedding"] else "[dim]off[/dim]")
    console.print()
    console.print("  [bold]Meta[/bold]")
    row("config path", f"[dim]{CONFIG_PATH}[/dim]")
    row("agent",       "[green]ready[/green]" if cli.agent else "[red]not ready[/red]")
    console.print()


def cmd_setup_wizard(cli) -> None:
    """3 步交互式向导:↑↓ 选 provider → 输 model → 输 api key。"""
    cur = cli.cfg
    cur_key_mask = (
        cur["api_key"][:6] + "…" + cur["api_key"][-4:]
        if cur.get("api_key") else "(not set)"
    )
    console.print()
    console.print(
        f"  [dim]current:[/dim] [cyan]{cur.get('provider_key', '?')}[/cyan]  "
        f"[dim]/[/dim]  [white]{cur.get('model') or '(no model)'}[/white]  "
        f"[dim]/[/dim]  [white]{cur_key_mask}[/white]"
    )
    console.print()

    # ─ Step 1: 选 provider (radiolist 全屏对话框,↑↓ 选,Enter 确认,Esc 取消) ─
    values = [
        (k, f"{k:<10}  {label}")
        for k, (label, _provider, _url, _model) in PROVIDER_PRESETS.items()
    ]
    chosen = radiolist_dialog(
        title=" 选择模型厂家 ",
        text="↑↓ 移动 · Enter 确认 · Esc 取消",
        values=values,
        default=cur.get("provider_key"),
    ).run()

    if chosen is None:
        console.print("  [yellow]已取消[/yellow]")
        return

    label, provider, base_url, default_model = PROVIDER_PRESETS[chosen]
    console.print(f"  [green]✓[/green] provider [cyan]{chosen}[/cyan]  [dim]({label})[/dim]")

    # ─ Step 2: 输 model ─
    # 沿用旧 model 仅当 provider 没变,否则用该家默认
    suggested = (
        cur.get("model")
        if cur.get("provider_key") == chosen and cur.get("model")
        else (default_model or "")
    )
    try:
        new_model = Prompt.ask(
            "  [cyan]model id[/cyan]",
            default=suggested,
        ).strip()
    except (KeyboardInterrupt, EOFError):
        console.print("\n  [yellow]已取消[/yellow]")
        return

    # ─ Step 3: 输 api key (回车保留当前) ─
    has_existing = bool(cur.get("api_key"))
    hint = "回车保留当前 key" if has_existing else "粘贴 key 后回车"
    console.print(f"  [dim]API key ({hint},不回显)[/dim]")
    try:
        new_key = Prompt.ask(
            "  [cyan]api key[/cyan]",
            password=True,
            default=cur.get("api_key", ""),
            show_default=False,
        ).strip()
    except (KeyboardInterrupt, EOFError):
        console.print("\n  [yellow]已取消[/yellow]")
        return

    # ─ Step 4: 保存 + 重建 ─
    cli.cfg["provider_key"] = chosen
    cli.cfg["provider"] = provider
    if base_url:
        cli.cfg["base_url"] = base_url
    if new_model:
        cli.cfg["model"] = new_model
    if new_key:
        cli.cfg["api_key"] = new_key
    save_config(cli.cfg)
    console.print()
    console.print("  [green]✓ 配置已保存[/green]")
    cli.rebuild_agent()
    cmd_show_config(cli)


def cmd_set_provider(cli, value: str) -> None:
    value = value.strip()
    if not value:
        console.print(f"[yellow]可选: {', '.join(PROVIDER_PRESETS.keys())}[/yellow]")
        return
    if value not in PROVIDER_PRESETS:
        console.print(f"[red]未知厂商: {value}。可选: {', '.join(PROVIDER_PRESETS.keys())}[/red]")
        return
    _label, provider, base_url, default_model = PROVIDER_PRESETS[value]
    cli.cfg["provider_key"] = value
    cli.cfg["provider"] = provider
    if base_url:
        cli.cfg["base_url"] = base_url
    if default_model and not cli.cfg["model"]:
        cli.cfg["model"] = default_model
    save_config(cli.cfg)
    console.print(f"[green]✓ provider 改为 {value} (provider={provider}, base_url={base_url})[/green]")
    cli.rebuild_agent()


def cmd_set_simple(cli, field: str, value: str) -> None:
    value = value.strip()
    if not value:
        console.print(f"[yellow]用法: /config {field} <值>[/yellow]")
        return
    cli.cfg[field] = value
    save_config(cli.cfg)
    console.print(f"[green]✓ {field} = {value}[/green]")
    cli.rebuild_agent()


def cmd_set_key(cli) -> None:
    console.print("[dim]输入 API Key (粘贴后按 Enter,不回显):[/dim]")
    try:
        key = Prompt.ask("API Key", password=True, default=cli.cfg.get("api_key", ""), show_default=False).strip()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]已取消[/yellow]")
        return
    if not key:
        console.print("[yellow]空值,未修改[/yellow]")
        return
    cli.cfg["api_key"] = key
    save_config(cli.cfg)
    console.print("[green]✓ API Key 已更新[/green]")
    cli.rebuild_agent()


def cmd_set_memory(cli, mem_field: str, value: str, valid: list) -> None:
    value = value.strip()
    if value not in valid:
        console.print(f"[yellow]可选: {', '.join(valid)}[/yellow]")
        return
    cli.cfg["memory"][mem_field] = value
    save_config(cli.cfg)
    console.print(f"[green]✓ memory.{mem_field} = {value}[/green]")
    cli.rebuild_agent()


def cmd_set_embedding(cli, value: str) -> None:
    value = value.strip().lower()
    if value not in ("on", "off", "true", "false"):
        console.print("[yellow]用法: /config embedding on|off[/yellow]")
        return
    on = value in ("on", "true")
    cli.cfg["memory"]["use_embedding"] = on
    save_config(cli.cfg)
    console.print(f"[green]✓ embedding = {'启用' if on else '关闭'}[/green]")
    cli.rebuild_agent()


def cmd_config_reset(cli) -> None:
    cli.cfg = _merge_defaults({})
    save_config(cli.cfg)
    console.print("[green]✓ 配置已重置为默认 (api_key 也被清空)[/green]")
    cli.rebuild_agent()


CONFIG_DISPATCH = {
    "provider":  ("provider 子命令", lambda cli, v: cmd_set_provider(cli, v)),
    "model":     ("model 子命令",    lambda cli, v: cmd_set_simple(cli, "model", v)),
    "key":       ("交互式输入 key", lambda cli, _v: cmd_set_key(cli)),
    "base_url":  ("base_url 子命令", lambda cli, v: cmd_set_simple(cli, "base_url", v)),
    "cold":      ("冷存储",         lambda cli, v: cmd_set_memory(cli, "cold_backend", v, ["jsonl", "sqlite", "none"])),
    "vector":    ("向量库",         lambda cli, v: cmd_set_memory(cli, "vector_backend", v, ["memory", "sqlite"])),
    "conflict":  ("冲突强度",       lambda cli, v: cmd_set_memory(cli, "conflict_strength", v, ["off", "fast", "accurate", "extreme"])),
    "tick":      ("tick 模式",      lambda cli, v: cmd_set_memory(cli, "tick_mode", v, ["sync", "async", "off"])),
    "embedding": ("embedding 开关", lambda cli, v: cmd_set_embedding(cli, v)),
    "reset":     ("重置配置",       lambda cli, _v: cmd_config_reset(cli)),
}


def cmd_config(cli, args: str) -> None:
    args = args.strip()
    if not args:
        cmd_setup_wizard(cli)
        return
    parts = args.split(maxsplit=1)
    sub = parts[0]
    value = parts[1] if len(parts) > 1 else ""
    if sub == "show":
        cmd_show_config(cli)
        return
    if sub not in CONFIG_DISPATCH:
        console.print(f"[yellow]未知子命令: {sub}[/yellow]")
        console.print(f"[dim]可用: show, {', '.join(CONFIG_DISPATCH.keys())} (或直接 /config 进入向导)[/dim]")
        return
    _label, handler = CONFIG_DISPATCH[sub]
    handler(cli, value)


# ────────────────────────────────────────────────────────────
# 皮卡丘姿势池 —— banner 启动随机抽
# eye_l/eye_r 必须 1-char(对齐),mouth 必须 4-char(居中槽位)
# ────────────────────────────────────────────────────────────

POSES: Dict[str, List[Dict[str, str]]] = {
    "happy": [
        {"eye_l": "o", "eye_r": "o", "mouth": "\\__/", "line": "你好呀~"},
        {"eye_l": "-", "eye_r": "-", "mouth": "\\__/", "line": "⚡!"},      # 蓄电
        {"eye_l": "^", "eye_r": "^", "mouth": "\\oo/", "line": "♪ ♬"},      # 哼歌
        {"eye_l": "♥", "eye_r": "♥", "mouth": "\\__/", "line": "今天也加油~"},
        {"eye_l": ">", "eye_r": "<", "mouth": "\\__/", "line": "嘿嘿"},
        {"eye_l": "o", "eye_r": "o", "mouth": " ?? ", "line": "...?"},
        {"eye_l": "-", "eye_r": "o", "mouth": "\\__/", "line": "miss you~"},  # 眨眼
    ],
    "sleepy": [
        {"eye_l": "-", "eye_r": "-", "mouth": "\\zz/", "line": "··· zzz"},
    ],
}


# ────────────────────────────────────────────────────────────
# 主类
# ────────────────────────────────────────────────────────────

class ChatCLI:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.agent: Optional[MyFunctionCallAgent] = build_agent(cfg)
        self.multiline = False
        self.history_path = Path.home() / ".my_chat_history"

        kb = KeyBindings()

        @kb.add("c-l")
        def _(event):
            console.clear()

        prompt_style = Style.from_dict({
            "prompt.ready":    "ansibrightblack",
            "prompt.notready": "ansired",
            "prompt.arrow":    "ansicyan",
        })

        self.session = PromptSession(
            history=FileHistory(str(self.history_path)),
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=kb,
            style=prompt_style,
        )

    def rebuild_agent(self) -> None:
        """配置变更后调,优雅关闭旧的、用新 cfg 重建。"""
        if self.agent is not None:
            try:
                self.agent.memory.close()
            except Exception:
                pass
        self.agent = build_agent(self.cfg)
        if self.agent:
            console.print("[dim]Agent 已就绪[/dim]")

    def print_banner(self) -> None:
        """6 行中等皮卡丘 + 闪电尾巴 + 上下闪电分隔条。ready/not ready 抽不同 pool。

        图案结构(对称,以视觉中线为轴):
              /\\__      __/\\        ← 耳朵尖
             /   \\____/   \\          ← 头顶弧
            |  o        o  |         ← 眼睛 (随机表情)
            |     \\__/     |         ← 嘴 (随机表情)
             \\____________/          ← 下颌
                  \\__/⚡             ← 闪电尾巴
        """
        ready = self.agent is not None
        pose = random.choice(POSES["happy" if ready else "sleepy"])

        Y     = "yellow"
        BOLT  = "bold bright_yellow"
        LINE  = "bright_cyan"
        CHEEK = "bright_red"

        PIKA = "   "   # 皮卡丘整体左缩进 (3 chars,让最左 / 字符落在 col 4)
        LEAD = "    "  # 标题/状态行左缩进 (4 chars,跟皮卡丘左缘对齐)

        # 上/下闪电分隔条 ——  ⚡━━━…━━━⚡
        # 60 个 ━ + 两端 ⚡(East Asian Wide,2 cols) ≈ 64 visible cols
        bolt_bar = Text("  ")
        bolt_bar.append("⚡", style=BOLT)
        bolt_bar.append("━" * 60, style=BOLT)
        bolt_bar.append("⚡", style=BOLT)

        # 皮卡丘 6 行
        row_ears  = Text(f"{PIKA}  /\\__      __/\\", style=Y)
        row_top   = Text(f"{PIKA} /   \\____/   \\", style=Y)

        # 眼睛行 + 右侧旁白
        row_eyes = Text(PIKA)
        row_eyes.append("|  ", style=Y)
        row_eyes.append(pose["eye_l"], style="bold")
        row_eyes.append("        ", style=Y)
        row_eyes.append(pose["eye_r"], style="bold")
        row_eyes.append("  |", style=Y)
        row_eyes.append(f"    {pose['line']}", style=LINE)

        # 嘴行
        row_mouth = Text(PIKA)
        row_mouth.append("|     ", style=Y)
        row_mouth.append(pose["mouth"], style=CHEEK if ready else Y)
        row_mouth.append("     |", style=Y)

        row_chin = Text(f"{PIKA} \\____________/", style=Y)

        # 闪电尾巴
        row_tail = Text(PIKA)
        row_tail.append("      \\__/", style=Y)
        row_tail.append("⚡", style=BOLT)

        # 标题
        title = Text(LEAD)
        title.append("⚡ ", style=BOLT)
        title.append("MY AI 伙伴", style="bold bright_yellow")
        title.append("  ·  ", style="dim")
        title.append("v0.1", style="dim")

        # 状态 + 命令提示
        status = Text(LEAD)
        if ready:
            status.append("● ", style="green bold")
            status.append("ready", style="green")
            status.append("  ·  ", style="dim")
            status.append(f"{self.cfg['provider_key']} / {self.cfg['model']}", style="cyan")
            cmds = Text(f"{LEAD}/help · /config · /memory · /clear", style="dim")
        else:
            status.append("○ ", style="red bold")
            status.append("not ready", style="red")
            status.append("  —  ", style="dim")
            status.append("run ", style="dim")
            status.append("/config key", style="cyan bold")
            cmds = Text(f"{LEAD}/help · /config", style="dim")

        console.print()
        console.print(Group(
            bolt_bar,
            Text(""),
            row_ears, row_top, row_eyes, row_mouth, row_chin, row_tail,
            Text(""),
            title,
            status,
            cmds,
            Text(""),
            bolt_bar,
        ))
        console.print()

    def get_input(self) -> str:
        # 简洁的 ❯ prompt,跟 Claude Code 风格一致
        # 不在前后加横线 —— prompt_toolkit 在宽终端下会让横线看起来被拉伸
        console.print()
        if self.multiline:
            prompt_html = HTML(
                "<prompt.arrow>❯</prompt.arrow> "
                "<prompt.ready>multiline</prompt.ready> "
                "<prompt.arrow>›</prompt.arrow> "
            )
        else:
            prompt_html = HTML("<prompt.arrow>❯</prompt.arrow> ")
        return self.session.prompt(prompt_html, multiline=self.multiline).strip()

    def handle_command(self, line: str) -> bool:
        if not line.startswith("/"):
            return False
        parts = line.split(maxsplit=1)
        cmd = parts[0]
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("/quit", "/exit"):
            raise EOFError()
        if cmd == "/help":
            cmd_help(self); return True
        if cmd == "/config":
            cmd_config(self, arg); return True
        if cmd == "/clear":
            cmd_clear(self); return True
        if cmd == "/memory":
            cmd_memory(self); return True
        if cmd == "/kg":
            cmd_kg(self); return True
        if cmd == "/recall":
            cmd_recall(self, arg); return True
        if cmd == "/facts":
            cmd_facts(self, arg); return True
        if cmd == "/multiline":
            self.multiline = not self.multiline
            console.print(f"[dim]多行输入: {'开启 (Esc+Enter 提交)' if self.multiline else '关闭'}[/dim]")
            return True

        console.print(f"[yellow]未知命令: {cmd}  (用 /help)[/yellow]")
        return True

    def render_response(self, response: str) -> None:
        console.print()
        console.print("[bright_green]●[/bright_green] ", end="")
        console.print(Markdown(response))

    def chat_once(self, user_input: str) -> None:
        if self.agent is None:
            console.print()
            console.print("[red]●[/red] [bold]Agent not ready[/bold] [dim]— missing API key[/dim]")
            console.print()
            console.print("  [dim]Run these to set up:[/dim]")
            console.print("    [cyan]/config provider <name>[/cyan]  [dim](e.g. minimax)[/dim]")
            console.print("    [cyan]/config model <id>[/cyan]")
            console.print("    [cyan]/config key[/cyan]")
            return

        try:
            with console.status(
                "[bright_black]thinking…[/bright_black]",
                spinner="dots",
                spinner_style="bright_black",
            ):
                response = self.agent.run(user_input)
        except Exception as exc:
            console.print()
            console.print(f"[red]●[/red] [bold red]error:[/bold red] [red]{exc}[/red]")
            return
        self.render_response(response)

    def run(self) -> None:
        self.print_banner()
        while True:
            try:
                line = self.get_input()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]再见 👋[/yellow]")
                break

            if not line:
                continue

            if self.handle_command(line):
                continue

            self.chat_once(line)

        if self.agent is not None:
            try:
                self.agent.memory.close()
            except Exception:
                pass


def main() -> None:
    cfg = load_config()
    cli = ChatCLI(cfg)
    cli.run()


if __name__ == "__main__":
    main()
