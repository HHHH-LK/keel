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
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "WARNING").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
from prompt_toolkit.shortcuts import radiolist_dialog
from rich.prompt import Prompt

from my_agent_llms.cli import banner, chat_view, help_view, status_bar, theme
from my_agent_llms.cli.permission import prompt_permission, TerminalNotInteractiveError
from my_agent_llms.cli.console import console
from my_agent_llms.cli.prompt import build_session, prompt_html

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
            help_view.print_warn(console, f"配置文件损坏 ({exc}),用默认配置")
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


def _preview_tool_args(args: Dict, *, max_total: int = 60, max_value: int = 30) -> str:
    """格式化工具参数给 UI 提示用,过长的值截断。"""
    items = []
    for k, v in args.items():
        v_repr = repr(v)
        if len(v_repr) > max_value:
            v_repr = v_repr[:max_value - 3] + "..."
        items.append(f"{k}={v_repr}")
    full = ", ".join(items)
    if len(full) > max_total:
        full = full[:max_total - 3] + "..."
    return full


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
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "8192")),
        )
    except Exception as exc:
        help_view.print_error(console, f"LLM 初始化失败: {exc}")
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
    from my_agent_llms.tools.builtin.read_file import ReadFile
    from my_agent_llms.tools.builtin.edit_file import EditFile
    from my_agent_llms.tools.builtin.write_file import WriteFile
    from my_agent_llms.tools.builtin.list_dir import ListDir
    from my_agent_llms.tools.builtin.attach_file import AttachFile
    from my_agent_llms.tools.builtin.attach_dir import AttachDir
    from my_agent_llms.tools.builtin.export_file import ExportFile

    try:
        ws = Workspace(cfg.get("workspace"))
    except (FileNotFoundError, NotADirectoryError) as exc:
        help_view.print_error(console, f"Workspace 初始化失败: {exc}")
        return None
    console.print(f"[dim]Workspace: {ws.root}[/dim]")

    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
    registry.register_tool(ReadFile(ws))
    registry.register_tool(EditFile(ws))
    registry.register_tool(WriteFile(ws))
    registry.register_tool(ListDir(ws))
    registry.register_tool(AttachFile(ws))
    registry.register_tool(AttachDir(ws))
    registry.register_tool(ExportFile(ws))

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
                "你拥有 sandbox 文件工具。规则:\n"
                "1. 用户提到外部路径:\n"
                "   - 是文件 → AttachFile 拉进 sandbox\n"
                "   - 是目录 → AttachDir 整个目录递归拉进 sandbox\n"
                "     (自动跳过 .git/.venv/node_modules/__pycache__/二进制等,"
                "      返回值会附带文件清单,无需再额外 ListDir)\n"
                "2. sandbox 内用 ReadFile / EditFile / WriteFile / ListDir 操作\n"
                "3. 写类工具(EditFile / WriteFile / ExportFile)调用时框架会同步弹"
                "审批框给用户。你不用再追问'要不要确认',直接调即可。"
                "用户若拒绝,你会收到 \"用户拒绝了对 X 的调用\",请结合上下文"
                "道歉/改方案/继续聊。\n"
                "4. ExportFile 写回原位置;sandbox 内新建的文件 ExportFile 时必须给 dest_path\n"
            ),
            memory_config=memory_config,
            max_steps=1000,
        )
    except Exception as exc:
        help_view.print_error(console, f"Agent 构造失败: {exc}")
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
            help_view.print_warn(console, f"embedding 升级失败 ({exc}),用 TF-IDF")
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
    help_view.render_help(console)


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
        help_view.print_warn(console, "Agent not ready. Run /config key first.")
        return
    help_view.render_memory_stats(console, cli.agent.memory.stats())


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
        help_view.print_error(console, f"export failed: {exc}")


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


def cmd_remember(cli, arg: str) -> None:
    """显式添加一张 L0 卡片(跨会话保留)。用法: /remember <内容>"""
    if cli.agent is None:
        help_view.print_warn(console, "Agent 未构建。先用 /config key 配置好 API Key。")
        return
    arg = arg.strip()
    if not arg:
        help_view.print_warn(console, "用法: /remember <内容>  例如: /remember 我对花生过敏")
        return
    card = cli.agent.memory.remember(arg)
    help_view.print_ok(
        console,
        f"记下了 [{theme.DIM}](type={card.type.value}, id={card.id})[/]",
    )


def cmd_forget(cli, arg: str) -> None:
    """忘记一张 L0 卡片。用法: /forget <id 或 id 前缀>"""
    if cli.agent is None:
        help_view.print_warn(console, "Agent 未构建")
        return
    arg = arg.strip()
    if not arg:
        help_view.print_warn(console, "用法: /forget <id 或 id 前缀>  (先用 /l0 看 id)")
        return
    cards = cli.agent.memory.list_l0()
    matches = [c for c in cards if c.id.startswith(arg)]
    if not matches:
        console.print("  [dim]没找到匹配的卡片[/dim]")
        return
    if len(matches) > 1:
        help_view.print_warn(console, "匹配多张,请输入更长的 id 前缀:")
        for c in matches:
            console.print(f"    {c.id}  {c.content[:60]}")
        return
    card = matches[0]
    cli.agent.memory.forget(card.id)
    help_view.print_ok(console, f"已忘记: [{theme.DIM}]{card.content[:60]}[/]")


def cmd_pin(cli, arg: str) -> None:
    """锁定一张 L0 卡片,永不衰减。用法: /pin <id 或 id 前缀>"""
    if cli.agent is None:
        help_view.print_warn(console, "Agent 未构建")
        return
    arg = arg.strip()
    if not arg:
        help_view.print_warn(console, "用法: /pin <id 或 id 前缀>")
        return
    cards = cli.agent.memory.list_l0()
    matches = [c for c in cards if c.id.startswith(arg)]
    if not matches:
        console.print("  [dim]没找到匹配的卡片[/dim]")
        return
    if len(matches) > 1:
        help_view.print_warn(console, "匹配多张,请输入更长的 id 前缀")
        return
    card = matches[0]
    cli.agent.memory.pin_card(card.id)
    help_view.print_ok(console, f"已锁定: [{theme.DIM}]{card.content[:60]}[/]")


def cmd_l0(cli, _arg: str) -> None:
    """列出当前 active 的 L0 卡片。"""
    if cli.agent is None:
        help_view.print_warn(console, "Agent 未构建")
        return
    cards = cli.agent.memory.list_l0()
    if not cards:
        console.print("  [dim]L0 为空。用 /remember <内容> 添加,或聊天产生重要内容时自动晋升。[/dim]")
        return
    type_emoji = {
        "hard_constraint": "🔒",
        "identity": "👤",
        "preference": "❤️",
        "state": "📌",
    }
    console.print()
    console.print(f"  [bold]L0 核心卡片[/bold]  [dim]({len(cards)} 条)[/dim]")
    for c in cards:
        emoji = type_emoji.get(c.type.value, "·")
        pinned = " [red]📌[/red]" if c.user_pinned else ""
        console.print(
            f"  {emoji}  [cyan]{c.id[:8]}[/cyan]  "
            f"[dim]c={c.confidence:.2f}[/dim]  "
            f"{c.content[:80]}{pinned}"
        )
    console.print()


def cmd_restore(cli, arg: str) -> None:
    """从 L4 把最近 N 条历史拉回 L1。用法: /restore [N]  默认 N=10。"""
    if cli.agent is None:
        help_view.print_warn(console, "Agent 未构建。先用 /config key 配置好 API Key。")
        return
    arg = arg.strip()
    n = 10
    if arg:
        try:
            n = int(arg)
            if n <= 0:
                raise ValueError
        except ValueError:
            help_view.print_warn(console, "用法: /restore [N]  (N 为正整数,默认 10)")
            return
    loaded = cli.agent.memory.restore_from_cold(n)
    if loaded == 0:
        console.print("  [dim]没有可恢复的历史 (L4 为空,或冷存储未启用)[/dim]")
    else:
        help_view.print_ok(
            console,
            f"从 L4 恢复了 [bold]{loaded}[/] 条历史回 L1",
        )


def cmd_facts(cli, query: str) -> None:
    if cli.agent is None:
        help_view.print_warn(console, "Agent 未构建。")
        return
    if not query:
        help_view.print_warn(console, "用法: /facts <查询词>")
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
    help_view.render_config_show(
        console,
        cli.cfg,
        agent_ready=cli.agent is not None,
        config_path=str(CONFIG_PATH),
    )


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
        help_view.print_warn(console, f"可选: {', '.join(PROVIDER_PRESETS.keys())}")
        return
    if value not in PROVIDER_PRESETS:
        help_view.print_error(console, f"未知厂商: {value}。可选: {', '.join(PROVIDER_PRESETS.keys())}")
        return
    _label, provider, base_url, default_model = PROVIDER_PRESETS[value]
    cli.cfg["provider_key"] = value
    cli.cfg["provider"] = provider
    if base_url:
        cli.cfg["base_url"] = base_url
    if default_model and not cli.cfg["model"]:
        cli.cfg["model"] = default_model
    save_config(cli.cfg)
    help_view.print_ok(console, f"provider 改为 {value} (provider={provider}, base_url={base_url})")
    cli.rebuild_agent()


def cmd_set_simple(cli, field: str, value: str) -> None:
    value = value.strip()
    if not value:
        help_view.print_warn(console, f"用法: /config {field} <值>")
        return
    cli.cfg[field] = value
    save_config(cli.cfg)
    help_view.print_ok(console, f"{field} = {value}")
    cli.rebuild_agent()


def cmd_set_key(cli) -> None:
    console.print("[dim]输入 API Key (粘贴后按 Enter,不回显):[/dim]")
    try:
        key = Prompt.ask("API Key", password=True, default=cli.cfg.get("api_key", ""), show_default=False).strip()
    except (KeyboardInterrupt, EOFError):
        help_view.print_warn(console, "已取消")
        return
    if not key:
        help_view.print_warn(console, "空值,未修改")
        return
    cli.cfg["api_key"] = key
    save_config(cli.cfg)
    help_view.print_ok(console, "API Key 已更新")
    cli.rebuild_agent()


def cmd_set_memory(cli, mem_field: str, value: str, valid: list) -> None:
    value = value.strip()
    if value not in valid:
        help_view.print_warn(console, f"可选: {', '.join(valid)}")
        return
    cli.cfg["memory"][mem_field] = value
    save_config(cli.cfg)
    help_view.print_ok(console, f"memory.{mem_field} = {value}")
    cli.rebuild_agent()


def cmd_set_embedding(cli, value: str) -> None:
    value = value.strip().lower()
    if value not in ("on", "off", "true", "false"):
        help_view.print_warn(console, "用法: /config embedding on|off")
        return
    on = value in ("on", "true")
    cli.cfg["memory"]["use_embedding"] = on
    save_config(cli.cfg)
    help_view.print_ok(console, f"embedding = {'启用' if on else '关闭'}")
    cli.rebuild_agent()


def cmd_config_reset(cli) -> None:
    cli.cfg = _merge_defaults({})
    save_config(cli.cfg)
    help_view.print_ok(console, "配置已重置为默认 (api_key 也被清空)")
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
        help_view.print_warn(console, f"未知子命令: {sub}")
        console.print(f"[dim]可用: show, {', '.join(CONFIG_DISPATCH.keys())} (或直接 /config 进入向导)[/dim]")
        return
    _label, handler = CONFIG_DISPATCH[sub]
    handler(cli, value)


# ────────────────────────────────────────────────────────────
# 主类
# ────────────────────────────────────────────────────────────

class ChatCLI:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.agent: Optional[MyFunctionCallAgent] = build_agent(cfg)
        self.multiline = False
        self.history_path = Path.home() / ".my_chat_history"

        self.session = build_session(
            history_path=self.history_path,
            clear_screen=console.clear,
        )
        self.turn = 0  # for status bar

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
        ready = self.agent is not None
        ws_path = None
        tool_count = 0
        backend_label = ""
        if self.agent is not None:
            mem = self.cfg.get("memory", {})
            backend_label = f"L4 cold: {mem.get('cold_backend', '?')}"
            # Count tools + locate workspace from any workspace-bearing tool
            try:
                tools = list(self.agent.tool_registry._tools.values())
            except AttributeError:
                tools = []
            tool_count = len(tools)
            for tool in tools:
                ws = getattr(tool, "workspace", None)
                if ws is not None:
                    ws_path = ws.root
                    break
        banner.render(
            console,
            ready=ready,
            provider_key=self.cfg.get("provider_key", "?"),
            model=self.cfg.get("model", ""),
            backend_label=backend_label,
            tool_count=tool_count,
            workspace=ws_path,
        )

    def get_input(self) -> str:
        # Per-turn status line then ❯ prompt
        l1_tokens = 0
        if self.agent is not None:
            try:
                l1_tokens = self.agent.memory.stats().get("l1_tokens", 0)
            except Exception:
                pass
        status_bar.render(
            console,
            ready=self.agent is not None,
            provider_key=self.cfg.get("provider_key", "?"),
            model=self.cfg.get("model", ""),
            turn=self.turn,
            l1_tokens=l1_tokens,
            l1_max_tokens=4000,
            multiline=self.multiline,
        )
        return self.session.prompt(
            prompt_html(self.multiline),
            multiline=self.multiline,
        ).strip()

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
        if cmd == "/restore":
            cmd_restore(self, arg); return True
        if cmd == "/facts":
            cmd_facts(self, arg); return True
        if cmd == "/remember":
            cmd_remember(self, arg); return True
        if cmd == "/forget":
            cmd_forget(self, arg); return True
        if cmd == "/pin":
            cmd_pin(self, arg); return True
        if cmd == "/l0":
            cmd_l0(self, arg); return True
        if cmd == "/multiline":
            self.multiline = not self.multiline
            console.print(f"[dim]多行输入: {'开启 (Esc+Enter 提交)' if self.multiline else '关闭'}[/dim]")
            return True

        help_view.print_warn(console, f"未知命令: {cmd}  (用 /help)")
        return True

    def chat_once(self, user_input: str) -> None:
        chat_view.render_user(console, user_input)

        if self.agent is None:
            chat_view.print_not_ready_hint(console)
            return

        start = time.monotonic()
        # 用 slot 装 renderer 让 callback 可以重置它
        renderer_slot = {"current": chat_view.StreamingAgentRenderer(console)}

        # spinner 用 slot 装,审批后 / 工具完成后可重启,让"按 y 立刻看到转圈"
        def _make_status():
            return console.status(
                f"[{theme.AGENT}]伙伴[/] [{theme.DIM}]·  thinking…[/]",
                spinner="dots",
                spinner_style=theme.AGENT,
            )
        status_slot = {"current": _make_status(), "active": True}
        status_slot["current"].start()

        def _stop_status() -> None:
            if status_slot["active"]:
                status_slot["active"] = False
                status_slot["current"].stop()

        def _restart_status() -> None:
            # 已经在转就不重复;之前停了就开一个新的
            if status_slot["active"]:
                return
            status_slot["current"] = _make_status()
            status_slot["current"].start()
            status_slot["active"] = True

        def _on_chunk(text: str) -> None:
            _stop_status()
            renderer_slot["current"].text_chunk(text)

        def _on_tool(name: str, args: Dict) -> None:
            _stop_status()
            renderer_slot["current"].tool_notice(name, _preview_tool_args(args))

        def _on_permission_request(name: str, args: Dict, preview: str) -> bool:
            _stop_status()
            cur = renderer_slot["current"]
            if cur.has_output:
                cur.close()
            try:
                ok = prompt_permission(name, args, preview)
            except TerminalNotInteractiveError:
                console.print("[yellow]⚠ 非交互终端,自动拒绝[/yellow]")
                ok = False
            # 重置 renderer 让后续 chunk 进新 region
            renderer_slot["current"] = chat_view.StreamingAgentRenderer(console)
            # 审批一结束就把 spinner 转起来,用户看到"我正在干活",
            # 直到工具结果或模型下一段 chunk 把它停掉
            _restart_status()
            return ok

        def _on_tool_result(name: str, result: str) -> None:
            # 工具刚跑完,立刻把结果首行 echo 到 renderer,
            # 不用等模型再 invoke 一次"复述"一遍
            _stop_status()
            renderer_slot["current"].tool_result(result)
            # 工具结果出完,接下来还得等模型再说点啥,把 spinner 再转起来
            _restart_status()

        try:
            response = self.agent.run(
                user_input,
                on_text_chunk=_on_chunk,
                on_tool_call=_on_tool,
                on_permission_request=_on_permission_request,
                on_tool_result=_on_tool_result,
            )
        except Exception as exc:
            _stop_status()
            cur = renderer_slot["current"]
            if cur.has_output:
                cur.close()
            chat_view.render_agent_error(console, str(exc))
            return
        finally:
            _stop_status()

        elapsed = time.monotonic() - start
        tools_used = getattr(self.agent, "last_tool_call_count", 0)

        final_renderer = renderer_slot["current"]
        if final_renderer.has_output:
            final_renderer.close(tools_used=tools_used, elapsed_seconds=elapsed)
        else:
            # 流式期间一字未出 (比如 _extract_message_content 走 reasoning 兜底
            # 返回字符串,但 stream 阶段 content 通道全空) → 回退到全文 markdown 渲染
            chat_view.render_agent(console, response,
                                   tools_used=tools_used, elapsed_seconds=elapsed)
        self.turn += 1

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
