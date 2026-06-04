"""入口迁移:app.py 可作为模块 import,且暴露 main/build_agent。"""
import importlib


def test_app_module_exposes_main_and_build_agent():
    app = importlib.import_module("my_agent_llms.cli.app")
    assert callable(app.main)
    assert callable(app.build_agent)


def test_root_chat_shim_reexports_main():
    chat = importlib.import_module("chat")
    from my_agent_llms.cli.app import main as app_main
    assert chat.main is app_main
