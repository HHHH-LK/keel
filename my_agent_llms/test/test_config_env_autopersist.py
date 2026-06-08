"""env(.env)优先 + 自动落盘到全局 config.json:配一次 env,任何目录都就绪。"""
import json

from my_agent_llms.cli import app


def test_env_overrides_and_persists_when_absent(monkeypatch, tmp_path):
    p = tmp_path / "config.json"
    monkeypatch.setattr(app, "CONFIG_PATH", p)
    monkeypatch.setenv("LLM_API_KEY", "sk-env")
    monkeypatch.setenv("LLM_MODEL_ID", "envmodel")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    cfg = app.load_config(persist=True)
    assert cfg["api_key"] == "sk-env" and cfg["model"] == "envmodel"
    assert p.exists()                                    # 自动落盘
    assert json.loads(p.read_text())["api_key"] == "sk-env"


def test_env_wins_over_existing_config(monkeypatch, tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"api_key": "old", "model": "oldm", "provider": "openai"}))
    monkeypatch.setattr(app, "CONFIG_PATH", p)
    monkeypatch.setenv("LLM_API_KEY", "new")
    monkeypatch.delenv("LLM_MODEL_ID", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    cfg = app.load_config(persist=True)
    assert cfg["api_key"] == "new"                       # env 优先
    assert cfg["model"] == "oldm"                        # 没 env 的字段保留
    assert json.loads(p.read_text())["api_key"] == "new"  # 落盘


def test_no_env_does_not_rewrite(monkeypatch, tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"api_key": "keep", "model": "m", "provider": "openai"}))
    monkeypatch.setattr(app, "CONFIG_PATH", p)
    for k in ("LLM_API_KEY", "LLM_MODEL_ID", "LLM_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    mtime_before = p.stat().st_mtime_ns
    cfg = app.load_config(persist=True)
    assert cfg["api_key"] == "keep"
    assert p.stat().st_mtime_ns == mtime_before          # 没 env → 不重写


def test_load_config_default_does_not_persist(monkeypatch, tmp_path):
    """默认 persist=False(给测试/非入口用):即便 env 有值也不写盘。"""
    p = tmp_path / "config.json"
    monkeypatch.setattr(app, "CONFIG_PATH", p)
    monkeypatch.setenv("LLM_API_KEY", "sk-env")
    app.load_config()
    assert not p.exists()
