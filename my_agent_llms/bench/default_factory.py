"""把 build_agent 包成 agent_factory(workspace_root)。需 LLM key。"""
from __future__ import annotations


def build_factory():
    from my_agent_llms.cli.app import load_config, build_agent

    def factory(ws_root):
        cfg = load_config()
        cfg["workspace"] = ws_root
        agent = build_agent(cfg)
        if agent is None:
            raise RuntimeError("build_agent 返回 None(缺 LLM key?)")
        return agent
    return factory
