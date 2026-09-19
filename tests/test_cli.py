"""终端版 cli.py 全路径测试（mock 网关，不联网）。"""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile

import core.agent.agent as agent_mod
import cli


def _fake_gateway(tmp):
    state = {"n": 0}

    def F(name="", arguments="{}"):
        r = type("R", (), {})()
        r.name = name
        r.arguments = arguments
        return r

    def chat(messages, provider_model=None, tools=None):
        state["n"] += 1
        if state["n"] == 1:
            tc = type("TC", (), {})
            tc.type = "function"
            tc.id = "1"
            tc.function = F("project_write",
                            '{"relative_path": "main.py", "content": "print(1)"}')
            return {"content": "写文件", "tool_calls": [tc]}
        return {"content": "完成"}

    agent_mod.gateway.chat = chat


def test_cli_one_shot_task(monkeypatch, tmp_path):
    _fake_gateway(tmp_path)
    buf = io.StringIO()
    prev_argv = sys.argv
    sys.argv = ["cli.py", "-d", str(tmp_path), "写一个 main.py"]
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = cli.main()
    finally:
        sys.argv = prev_argv
    out = buf.getvalue()
    assert rc == 0
    assert (tmp_path / "main.py").read_text() == "print(1)"
    assert "项目写入目录" in out
    assert "main.py" in out


def test_cli_interactive_quit(monkeypatch, tmp_path, capsys):
    _fake_gateway(tmp_path)
    monkeypatch.setattr("builtins.input", lambda _p="": "q")
    prev_argv = sys.argv
    sys.argv = ["cli.py", "-d", str(tmp_path)]
    try:
        rc = cli.main()
    finally:
        sys.argv = prev_argv
    assert rc == 0