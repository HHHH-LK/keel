"""根目录兼容入口 —— 实际逻辑在 my_agent_llms/cli/app.py。
保留它是为了不破坏 `python chat.py` 的旧习惯;新用法是全局命令 `keel`。"""
from my_agent_llms.cli.app import main

if __name__ == "__main__":
    main()
