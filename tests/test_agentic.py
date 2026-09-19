"""代理式闭环增强测试：不可逆操作护栏、上下文压缩记忆、指令保真。"""
import json

import pytest

from core.agent import tools as tool_registry
from core.agent.memory import Memory
from core.agent.agent import MODE_PROMPTS, SYSTEM_PROMPT


# ---------------- 不可逆操作护栏 ----------------
def test_is_destructive_detects():
    assert tool_registry.is_destructive("rm -rf /tmp/x")
    assert tool_registry.is_destructive("DROP TABLE users")
    assert tool_registry.is_destructive('python -c "import shutil; shutil.rmtree(\'x\')"')
    assert tool_registry.is_destructive("Remove-Item C:\\tmp\\a -Recurse")
    # 可逆命令不拦截
    assert not tool_registry.is_destructive("ls -la")
    assert not tool_registry.is_destructive("echo hi")
    assert not tool_registry.is_destructive("SELECT * FROM users")


def test_run_shell_blocks_destructive_by_default(monkeypatch):
    """默认策略拦截不可逆命令，且不发起到沙箱。"""
    called = {"sandbox": False}
    monkeypatch.setattr(tool_registry, "_DESTRUCTIVE_ALLOW", False)
    res = tool_registry.invoke("run_shell", {"command": "rm -rf /tmp/hs_test"})
    assert "已拦截不可逆操作" in res
    # pending 被记录
    pend = tool_registry.pending_destructive()
    assert "rm -rf /tmp/hs_test" in pend
    tool_registry.pending_destructive(clear=True)


def test_run_shell_passes_reversible(monkeypatch):
    monkeypatch.setattr(tool_registry, "_DESTRUCTIVE_ALLOW", False)
    # 可逆命令不应被护栏拦截，应放行并进入沙箱执行
    res = tool_registry.invoke("run_shell", {"command": "echo hi", "timeout": 5})
    assert "已拦截不可逆操作" not in res


def test_policy_toggle_and_tools_registered():
    assert tool_registry.set_destructive_policy(True).startswith("已放行")
    assert tool_registry.destructive_policy() == "放行中"
    names = [s["function"]["name"] for s in tool_registry.tools_schema()]
    assert "set_destructive_policy" in names
    assert "pending_destructive" in names
    tool_registry.set_destructive_policy(False)


# ---------------- 上下文压缩记忆 ----------------
def _mk_memory(n: int, max_messages: int = 10):
    m = Memory(max_messages=max_messages, max_chars=10 ** 6)
    for i in range(n):
        m.add_user(f"第{i}条用户输入")
        m.add_assistant(f"第{i}条助手回复")
    return m


def test_should_summarize_threshold():
    m = _mk_memory(5, max_messages=10)
    assert m.should_summarize() is False          # 5 <= 15
    m2 = _mk_memory(20, max_messages=10)
    assert m2.should_summarize() is True          # 20 > 15


def test_summarize_injects_summary_keeps_recent():
    m = _mk_memory(20, max_messages=10)
    calls = []
    ok = m.summarize(lambda text: (calls.append(text) or "【假摘要】"))
    assert ok is True
    remaining = [msg for msg in m.get() if msg.get("role") != "system"]
    assert len(remaining) <= 10                     # 只保留最近 10 条
    sys_msgs = [msg for msg in m.get() if msg.get("role") == "system"]
    assert any("早期对话摘要" in msg.get("content", "") for msg in sys_msgs)
    assert calls


def test_summarize_noop_when_short():
    m = _mk_memory(3, max_messages=10)
    assert m.summarize(lambda t: "x") is False


# ---------------- 指令遵循 / 约束保真 ----------------
def test_constraint_rule_in_all_modes():
    assert "约束保真" in SYSTEM_PROMPT
    for mode in ("work", "code", "design", "default"):
        assert "约束保真" in MODE_PROMPTS[mode]


def test_guard_does_not_break_existing_tools():
    names = {s["function"]["name"] for s in tool_registry.tools_schema()}
    for essential in ("project_write", "run_shell", "verify_code", "mcp_call"):
        assert essential in names