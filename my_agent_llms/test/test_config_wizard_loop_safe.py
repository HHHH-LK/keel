"""live UI(运行中事件循环)内 /config 无参向导不能崩 asyncio.run 嵌套,要降级指引子命令。"""
import asyncio
from types import SimpleNamespace

from my_agent_llms.cli.app import cmd_setup_wizard


def test_setup_wizard_degrades_inside_running_loop(capsys):
    cli = SimpleNamespace(cfg={"api_key": "", "provider_key": "openai", "model": ""})

    async def _inside_loop():
        cmd_setup_wizard(cli)        # 不能抛 'asyncio.run() cannot be called...'

    asyncio.run(_inside_loop())
    out = capsys.readouterr().out
    assert "/config" in out          # 给出子命令指引
