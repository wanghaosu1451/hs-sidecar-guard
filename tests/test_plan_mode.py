"""Plan 只读模式 / Build 模式 / 本地项目配置注入的回归测试（不联网、不真实执行）。"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from core.agent import tools
from core.agent.agent import Agent, MODE_PROMPTS, _is_failure


# ---------------- Plan 只读：模式与白名单 ----------------
def test_modes_include_plan_and_build():
    assert "plan" in MODE_PROMPTS
    assert "build" in MODE_PROMPTS


def test_readonly_whitelist():
    assert tools.is_readonly_tool("read_file")
    assert tools.is_readonly_tool("list_project_files")
    assert tools.is_readonly_tool("graph_query")
    assert not tools.is_readonly_tool("project_write")
    assert not tools.is_readonly_tool("run_shell")
    assert not tools.is_readonly_tool("verify_code")
    assert not tools.is_readonly_tool("patch_file")


# ---------------- Plan 只读：门控不落地 ----------------
def _make_agent(tmp_path, mode):
    ag = Agent(provider_model="openai/gpt-4o-mini", mode=mode)
    tools.set_project_root(str(tmp_path))
    return ag


def test_plan_mode_blocks_write_tool(tmp_path, monkeypatch):
    import core.agent.agent as agent_mod
    ag = _make_agent(tmp_path, mode="plan")
    # 伪造一次工具调用：模型请求 project_write（写文件）
    fn_name = "project_write"
    write_args = json.dumps({"path": "x.txt", "content": "BOOM"}, ensure_ascii=False)

    class Fn:
        name = fn_name
        arguments = write_args

    class Tc:
        type = "function"
        id = "call_1"
        function = Fn

    blocked = ag._invoke_tool(fn_name, write_args)
    assert "只读(Plan)模式" in blocked
    assert not (tmp_path / "x.txt").exists()  # 确认未写任何文件

    class Resp:
        content = None
        tool_calls = None

    resp = {
        "content": None,
        "tool_calls": [Tc()],
        "usage": {"total_tokens": 3},
    }
    monkeypatch.setattr(agent_mod.gateway, "chat", lambda *a, **k: resp)
    res = ag.run_task("分析项目并给出方案", max_steps=2)
    assert (tmp_path / "x.txt").exists() is False


def test_plan_mode_allows_read_tool():
    ag = Agent(provider_model="openai/gpt-4o-mini", mode="plan")
    out = ag._invoke_tool("read_file", "{}")
    # 只读工具应放行（read_file 无入参会给出友好提示，而非“只读拒绝”）
    assert "只读(Plan)模式" not in out


def test_build_mode_not_readonly():
    ag = Agent(provider_model="openai/gpt-4o-mini", mode="build")
    # build 模式不拦截写工具（返回真正的执行/错误提示，而非“只读拒绝”）
    out = ag._invoke_tool("run_shell", "{}")
    assert out is not None
    assert "只读(Plan)模式" not in out


# ---------------- 本地项目配置注入 ----------------
def test_load_project_config_json(tmp_path):
    (tmp_path / "hside.json").write_text(
        json.dumps({"architecture": "pipeline", "commands": {"run": "python app.py"},
                    "verbose": {"deep": 1}}), encoding="utf-8")
    ag = Agent(provider_model="openai/gpt-4o-mini")
    cfg = ag._load_project_config(str(tmp_path))
    assert "pipeline" in cfg
    assert "commands" in cfg
    # 深层无关字段被裁剪，避免把大 JSON 灌入上下文
    assert "verbose" not in cfg


def test_load_project_config_none(tmp_path):
    ag = Agent(provider_model="openai/gpt-4o-mini")
    assert ag._load_project_config(str(tmp_path)) == ""


def test_project_config_injected_into_system(tmp_path):
    (tmp_path / "hside.json").write_text(
        json.dumps({"architecture": "micro-ctl"}), encoding="utf-8")
    ag = Agent(provider_model="openai/gpt-4o-mini")
    ag.workspace = str(tmp_path)
    sys = ag._system()
    assert "micro-ctl" in sys
    assert "项目配置" in sys


def test_is_failure_still_works():
    assert _is_failure("错误：xxx")
    assert not _is_failure("(exit=0)ok")


# ---------------- ① 沙箱资源限制 ----------------
def test_resexec_resource_flag_flat():
    from core.sandbox.executor import ExecResult
    r = ExecResult(-1, "xx", "", resource="内存 5000MB 超过上限 4096MB")
    assert not r.ok
    assert "[资源受限]" in r.text
    assert "内存" in r.text


def test_resexec_timeout_flag_unchanged():
    from core.sandbox.executor import ExecResult
    r = ExecResult(-1, "", "执行超时", timed_out=True)
    assert not r.ok
    assert "[超时]" in r.text


def test_policy_has_max_processes():
    from core.sandbox.security import SandboxPolicy
    p = SandboxPolicy()
    assert p.max_processes == 32
    assert p.to_dict()["max_processes"] == 32


def test_policy_posix_rlimits_smoke():
    from core.sandbox import executor
    p = __import__("core.sandbox.security", fromlist=["SandboxPolicy"]).SandboxPolicy(
        max_memory_mb=1024, max_cpu_percent=10)
    fn = executor._posix_rlimits(p)
    # POSIX/无 resource 环境下返回 None 或可调用，均不应抛错
    assert fn is None or callable(fn)
    if fn:
        fn()  # 设置 rlimit 本身不应抛错（当前平台若无权限也由内部吞掉）


# ---------------- ② 规则文件子目录层级 ----------------
def test_load_agents_rules_subdirs(tmp_path):
    (tmp_path / "AGENTS.md").write_text("根规则", encoding="utf-8")
    sub = tmp_path / "src" / "core"
    sub.mkdir(parents=True)
    (sub / "AGENTS.md").write_text("core 目录规则", encoding="utf-8")
    ag = Agent(provider_model="openai/gpt-4o-mini")
    rules = ag._load_agents_rules(str(tmp_path))
    assert "根目录 AGENTS.md" in rules
    assert "根规则" in rules
    assert "src/core/AGENTS.md" in rules
    assert "子目录规则" in rules
    assert "core 目录规则" in rules


def test_load_agents_rules_skips_hidden_dirs(tmp_path):
    (tmp_path / "AGENTS.md").write_text("根", encoding="utf-8")
    hide = tmp_path / ".git"
    hide.mkdir()
    (hide / "AGENTS.md").write_text("不应出现", encoding="utf-8")
    ag = Agent(provider_model="openai/gpt-4o-mini")
    rules = ag._load_agents_rules(str(tmp_path))
    assert "不应出现" not in rules
    assert "根" in rules


# ---------------- ③ 全局自动上下文压缩（重入安全） ----------------
def test_auto_compress_reentrant_safe(tmp_path):
    # 短会话下 should_summarize 为假，_messages 应安全返回且不触发网关
    ag = Agent(provider_model="openai/gpt-4o-mini")
    ag.memory.add_user("1+1")
    ag.memory.add_assistant("2", None)
    msgs = ag._messages()
    assert msgs and msgs[-1]["content"] in ("2",)
    assert msgs[-1]["content"] == "2"


def test_auto_compress_guard_flag(monkeypatch):
    ag = Agent(provider_model="openai/gpt-4o-mini")
    called = {"n": 0}

    def fake_inner():
        called["n"] += 1
        # 重入场景：压缩内部再调 _messages 不应再度触发压缩
        ag._auto_compress()

    monkeypatch.setattr(ag, "_maybe_summarize", fake_inner)
    # 强制命中限频条件
    ag._compress_counter = 3
    import types
    ag.memory.should_summarize = types.MethodType(
        lambda self: True, ag.memory)
    ag._auto_compress()
    # 计数器 3→4 未到 %5，应不触发；再调一次到 5 触发
    assert called["n"] == 0
    ag._compress_counter = 4
    ag._auto_compress()
    assert called["n"] >= 1