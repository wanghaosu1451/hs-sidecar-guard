"""数据外发防护 + 生成代码自检/验证 + 全仓上下文 + 写盘 测试。"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from core.agent import tools
from core.agent.agent import Agent
from core.project.workspace import Workspace, build_repo_context


# ---------- ① 数据外发防护：脱敏与敏感路径阻断 ----------
def test_redact_hides_api_key_value():
    out = tools.redact("the api_key is sk-abcdef123456789, keep safe")
    assert "sk-abcdef123456789" not in out
    assert "REDACTED" in out


def test_redact_hides_pem_block():
    pem = ("-----BEGIN RSA PRIVATE KEY-----\n"
           "MIIEpAIBAAKNCAQEAphGksecretdataABC123EfGyJQ8JcA0K\n"
           "-----END RSA PRIVATE KEY-----")
    out = tools.redact(pem)
    assert "MIIEpAIBAAKNCAQEAphGksecretdata" not in out
    assert "REDACTED" in out


def test_sensitive_path_detected():
    assert tools.is_sensitive_path("/proj/.env")
    assert tools.is_sensitive_path("/proj/config/secrets.json")
    assert tools.is_sensitive_path("/home/u/.aws/credentials")
    assert tools.is_sensitive_path("/keys/aws-key.pem")


def test_read_file_blocks_sensitive(tmp_path):
    env = tmp_path / ".env"
    env.write_text("TELEGRAM_BOT_TOKEN=sk-1234567890")
    out = tools.invoke("read_file", {"path": str(env)})
    assert "拒绝读取敏感文件" in out
    assert "1234567890" not in out


def test_run_shell_output_redacted():
    # 命中令牌形状的输出会被脱敏
    out = tools.invoke("run_shell", {"command": "echo sk-xGhsJ23kLmj9F1aZcXv8bQ"})
    assert "sk-xGhsJ23kLmj9F1aZcXv8bQ" not in out


# ---------- ② 生成代码自检 / 隔离验证 ----------
def test_extract_python_fenced():
    text = "解释：\n```python\nprint('hi')\n```"
    assert tools._extract_python(text) == "print('hi')"


def test_verify_code_good():
    r = tools.invoke("verify_code", {"code": "print(1 + 1)"})
    assert "✅" in r and "2" in r


def test_verify_code_bad_syntax():
    r = tools.invoke("verify_code", {"code": "def (\n"})
    assert "❌" in r or "失败" in r


def test_verify_code_undefined_warning():
    r = tools.invoke("verify_code", {"code": "print(nonexistent_xyz)"})
    assert "未定义" in r


# ---------- ③ 全仓上下文注入 ----------
def test_build_repo_context(tmp_path):
    (tmp_path / "mod.py").write_text("def foo(a, b):\n    return a\n\nclass Bar:\n    pass\n")
    (tmp_path / "readme.md").write_text("# hi")
    ws = Workspace(str(tmp_path))
    ctx = build_repo_context(ws)
    assert "# 目录结构" in ctx
    assert "mod.py" in ctx
    assert "def foo" in ctx or "def foo(a, b)" in ctx


def test_agent_injects_workspace_context(tmp_path):
    (tmp_path / "a.py").write_text("def helper(x):\n    return x\n")
    ws = Workspace(str(tmp_path))
    ag = Agent(workspace=ws, provider_model="openai/gpt-4o-mini")
    msgs = ag._messages()
    assert msgs[0]["role"] == "system"
    assert "工作区" in msgs[0]["content"] and "helper" in msgs[0]["content"]


def test_agent_system_mentions_verify():
    ag = Agent(provider_model="openai/gpt-4o-mini")
    msgs = ag._messages()
    assert "verify_code" in msgs[0]["content"]


# ---------- ④ AI 写项目到指定本地文件夹 ----------
def test_project_write_and_root(tmp_path):
    tools.set_project_root(str(tmp_path))
    out = tools.invoke("project_write", {"relative_path": "src/main.py",
                                         "content": "print('hi')"})
    assert "src/main.py" in out
    assert (tmp_path / "src" / "main.py").read_text() == "print('hi')"


def test_project_write_blocks_escape(tmp_path):
    tools.set_project_root(str(tmp_path))
    out = tools.invoke("project_write", {"relative_path": "../../evil.py",
                                         "content": "x"})
    assert "越界" in out
    assert not (tmp_path.parent / "evil.py").exists()


def test_list_project_files(tmp_path):
    tools.set_project_root(str(tmp_path))
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "sub" / "b.txt").parent.mkdir()
    (tmp_path / "sub" / "b.txt").write_text("y")
    files = tools.invoke("list_project_files", {})
    assert "a.py" in files
    assert f"sub{os.sep}b.txt" in files


def test_run_task_autonomous_loop_writes(monkeypatch, tmp_path):
    """自主执行环：AI 调 project_write 落盘，再给最终答案（mock 网关，不联网）。"""
    import core.agent.agent as agent_mod
    steps = []
    state = {"n": 0}

    def fake_chat(messages, provider_model=None, tools=None):
        state["n"] += 1
        if state["n"] == 1:
            def F(name="", arguments="{}"):
                r = type("R", (), {})()
                r.name = name
                r.arguments = arguments
                return r
            tc = type("TC", (), {})
            tc.type = "function"
            tc.id = "1"
            tc.function = F("project_write",
                            '{"relative_path": "app/main.py", "content": "print(123)"}')
            return {"content": "准备写文件", "tool_calls": [tc]}
        return {"content": "已完成，生成 app/main.py"}

    monkeypatch.setattr(agent_mod.gateway, "chat", fake_chat)
    tools.set_project_root(str(tmp_path))
    agent = agent_mod.Agent(provider_model="openai/gpt-4o-mini")
    res = agent.run_task("做个python项目", on_step=lambda n, r: steps.append(n))
    assert res["answer"] == "已完成，生成 app/main.py"
    assert res["tools"] == ["project_write"]
    assert (tmp_path / "app" / "main.py").read_text() == "print(123)"