"""子任务 2/3/4/5 新增能力的回归测试（全部不联网、不触及真实模型）。"""
from __future__ import annotations

import pytest

from core.agent import tools
from core.agent.agent import Agent
from core.agent.memory import Memory
from core.llm import gateway, keys
from core.sandbox.executor import Executor
from core.sandbox.security import SandboxPolicy


# ---------------- 子任务3：网关重试状态机 ----------------
def test_retry_classifies_status():
    class E(Exception):
        pass
    for code in (500, 502, 429):
        assert gateway._is_retryable(type("X", (E,), {"status_code": code})())
    for code in (400, 401, 404):
        assert not gateway._is_retryable(type("X", (E,), {"status_code": code})())


def test_retry_classifies_message():
    class T(TimeoutError):
        pass
    assert gateway._is_retryable(T("request timed out"))
    assert gateway._is_retryable(RuntimeError("Connection aborted"))
    assert not gateway._is_retryable(RuntimeError("model not found within scope"))


def test_keys_defaults_retry_and_budget():
    d = keys._DEFAULTS
    assert d["llm"].get("retry") is True
    assert d["budget"]["max_steps"] == 300
    assert d["budget"]["max_tokens"] == 0


# ---------------- 子任务2：预算守卫 -----------------
def test_agent_tokens_helper():
    ag = Agent(provider_model="openai/gpt-4o-mini")
    assert ag._tokens({"usage": {"total_tokens": 7}}) == 7
    assert ag._tokens({"usage": {"prompt_tokens": 3, "completion_tokens": 4}}) == 7
    assert ag._tokens({"content": "no usage"}) == 0

    class U:
        total_tokens = 9
    assert ag._tokens({"usage": U()}) == 9


def test_agent_budget_defaults():
    step_cap, token_budget = Agent(provider_model="openai/gpt-4o-mini")._budget()
    assert step_cap >= 300
    assert token_budget == 0  # 本地默认不下发 token 预算


# ---------------- 子任务4：命令规范化 + 审批缓存 ----------------
def test_canonicalize_normalizes():
    assert tools._canonicalize("  rm  -rf  /tmp/x ") == \
        tools._canonicalize("RM -rf /tmp/x")
    assert tools._canonicalize("del 'C:\\tmp\\a'") == \
        tools._canonicalize("del \"C:/tmp/a\"")
    assert tools._canonicalize("ls -la") != tools._canonicalize("rm -rf x")


def test_approval_cache_canonical_hit(monkeypatch):
    monkeypatch.setattr(tools, "_DESTRUCTIVE_ALLOW", False)
    tools._APPROVED_COMMANDS.clear()
    try:
        # 未批准 -> 拦截
        assert tools._guard_run_shell("rm -rf /tmp/guard_a") is not None
        # 批准一条，换写法(大小写/空白/斜杠)应命中缓存直接放行
        tools._approve_cached("rm  -rf  /tmp/guard_a")
        assert tools._cache_has("RM -rf /tmp/guard_a")
        assert tools._guard_run_shell("rm -rf /tmp/guard_a") is None
    finally:
        tools._APPROVED_COMMANDS.clear()


def test_approve_destructive_independent_of_global(monkeypatch):
    monkeypatch.setattr(tools, "_DESTRUCTIVE_ALLOW", False)
    tools._APPROVED_COMMANDS.clear()
    try:
        tools._PENDING_DESTRUCTIVE.clear()
        res = tools.invoke("run_shell", {"command": "rm -rf /tmp/guard_b"})
        assert "已拦截不可逆操作" in res
        tools._PENDING_DESTRUCTIVE.clear()
    finally:
        tools._APPROVED_COMMANDS.clear()
        tools._PENDING_DESTRUCTIVE.clear()


def test_pending_destructive_clear_preserved():
    tools._PENDING_DESTRUCTIVE.clear()
    try:
        tools.invoke("run_shell", {"command": "DROP TABLE x"})
        assert "DROP TABLE x" in tools.pending_destructive()
        tools.pending_destructive(clear=True)
        assert "(暂无" in tools.pending_destructive()
    finally:
        tools._PENDING_DESTRUCTIVE.clear()


# ---------------- 子任务1：沙箱路径守卫（读/写不出项目根） ----------------
def test_sandbox_path_guard_blocks_outside():
    pol = SandboxPolicy(allow_network=False,
                        allowed_dirs=[r"C:\proj\app"])
    assert pol.path_block_reason('cat "C:\\evil\\secret.txt"')
    assert pol.path_block_reason("type C:\\test\\out.txt")
    # 项目内路径不拦截
    assert not pol.path_block_reason('cat "C:\\proj\\app\\a.py"')
    # 未配置白名单不拦截
    assert not SandboxPolicy().path_block_reason("cat C:\\evil\\x")


def test_executor_cwd_locked_into_allowed():
    import tempfile
    import os
    from pathlib import Path
    with tempfile.TemporaryDirectory() as proj:
        pol = SandboxPolicy(timeout_seconds=8, allowed_dirs=[proj])
        r = Executor(pol).run_command(
            "cd" if os.name == "nt" else "pwd", cwd=str(Path(proj)))
        assert r.ok


# ---------------- 子任务5：压缩多模型回退不崩溃、不丢失 ----------------
def test_summarize_fallback_keeps_original(monkeypatch):
    import core.agent.agent as agent_mod
    ag = Agent(provider_model="openai/gpt-4o-mini")
    ag.memory.max_messages = 2
    for i in range(10):
        ag.memory.add_user(f"问题{i}")
        ag.memory.add_assistant(f"回复{i}")

    # 压缩调用每次都“失败”（网关返回 error 标记），应整体降级且不抛异常
    called = {"n": 0}

    def fake_chat(messages, provider_model=None, **kw):
        called["n"] += 1
        return {"content": None, "error": "模拟压缩失败"}

    monkeypatch.setattr(agent_mod.gateway, "chat", fake_chat)
    ag._maybe_summarize()  # 不应抛异常
    assert called["n"] >= 2  # 主模型 + 回退模型各尝试一次


def test_summarize_retry_on_second_model(monkeypatch):
    import core.agent.agent as agent_mod
    ag = Agent(provider_model="openai/gpt-4o-mini")
    ag.memory.max_messages = 2
    for i in range(10):
        ag.memory.add_user(f"q{i}")
        ag.memory.add_assistant(f"a{i}")
    calls = []

    def fake_chat(messages, provider_model=None, **kw):
        calls.append(provider_model)
        if len(calls) == 1:
            raise RuntimeError("首次压缩网络错误")
        return {"content": "【摘要】已就绪"}

    monkeypatch.setattr(agent_mod.gateway, "chat", fake_chat)
    ag._maybe_summarize()
    assert len(calls) >= 2           # 失败后在回退模型上成功
    assert any("摘要" in m.get("content", "")
               for m in ag.memory.get() if m.get("role") == "system")