"""生命周期钩子测试：JSON 规则阻断/注入、脚本隔离、异常静默。

全程使用本地临时目录作为项目根，不联网、不调用远端模型。
"""
from __future__ import annotations

import json

import pytest

from core.agent import hooks as hooks_mod
from core.agent import tools as tool_registry


@pytest.fixture
def proj(tmp_path, monkeypatch):
    """把项目写入目录临时改为 tmp 目录，测完还原为默认；同时清空钩子缓存。"""
    monkeypatch.setattr(tool_registry, "_PROJECT_ROOT", str(tmp_path))
    hooks_mod.reload(str(tmp_path))
    yield tmp_path
    hooks_mod.reload(".")
    monkeypatch.setattr(tool_registry, "_PROJECT_ROOT", None, raising=False)


def _write_rules(proj, rules):
    (proj / ".hs_hooks.json").write_text(
        json.dumps({"enabled": True, "pre_tool_use": rules}), encoding="utf-8")
    hooks_mod.reload(str(proj))


def test_pre_block_hard_banned_tool_never_executes(proj):
    """PreToolUse 按名称永久禁止某工具：返回阻断提示，真实工具不被执行。"""
    _write_rules(proj, [
        {"name": "project_write", "action": "block",
         "message": "project_write 已被钩子禁用"},
    ])
    res = tool_registry.invoke("project_write",
                               {"relative_path": "a.txt", "content": "x"})
    assert "已被钩子禁用" in res
    assert not (proj / "a.txt").exists()  # 真实写文件未发生


def test_pre_block_argument_keyword(proj):
    """参数关键字命中即阻断：仅命中指定参数才拦，其它正常执行。"""
    _write_rules(proj, [
        {"name": "run_*", "arguments_contain": "sudo",
         "action": "block", "message": "禁止使用 sudo"},
    ])
    res = tool_registry.invoke("run_shell", {"command": "sudo whoami", "timeout": 5})
    assert "禁止使用 sudo" in res
    # 未命中关键字：不应被钩子拦（执行结果由沙箱/护栏决定，而非钩子提示）
    res2 = tool_registry.invoke("run_shell", {"command": "echo hi", "timeout": 5})
    assert "禁止使用 sudo" not in res2


def test_pre_inject_appends_to_result(proj):
    """inject 规则：非阻断，注入修正词随结果返回，模型可见。"""
    _write_rules(proj, [
        {"name": "project_write", "action": "inject",
         "inject": "写文件前请先确认父目录已存在"},
    ])
    res = tool_registry.invoke("project_write",
                               {"relative_path": "a.txt", "content": "x"})
    assert "已写入" in res
    assert "已确认父目录已存在" in res or "父目录已存在" in res
    assert (proj / "a.txt").exists()  # 非阻断，正常写入


def test_unknown_tool_still_blocked_by_name_rule(proj):
    """名称规则阻断在“未知工具”判断之前：命中则返回钩子提示而非未知工具报错。"""
    _write_rules(proj, [{"name": "nope_tool", "action": "block",
                         "message": "永远禁用该工具"}])
    res = tool_registry.invoke("nope_tool", {})
    assert "永远禁用该工具" in res


def test_broken_rules_file_silent(proj):
    """破损/非法 JSON 规则文件：静默降级，不影响工具正常执行。"""
    (proj / ".hs_hooks.json").write_text("{ not valid json ", encoding="utf-8")
    hooks_mod.reload(str(proj))
    res = tool_registry.invoke("project_write",
                               {"relative_path": "a.txt", "content": "x"})
    assert "已写入" in res or (proj / "a.txt").exists()


def test_hook_exception_never_crashes(proj, monkeypatch):
    """钩子自身抛异常：invoke 仍正常返回，主流程不炸。"""
    def boom(*a, **k):
        raise RuntimeError("hook exploded")
    monkeypatch.setattr(hooks_mod, "pre_tool_use", boom)
    monkeypatch.setattr(hooks_mod, "post_tool_use", boom)
    hooks_mod.reload(str(proj))
    res = tool_registry.invoke("list_project_files", {})
    # 不因钩子异常抛错；返回的是真实工具结果
    assert isinstance(res, str)


def test_script_block_optin(proj):
    """可选脚本默认关闭；显式 script_enabled=true 且脚本返回 block 才生效。"""
    (proj / "hooks.py").write_text(
        "def pre_tool_use(name, arguments):\n"
        "    if name == 'project_write':\n"
        "        return {'block': True, 'message': '脚本拦截写文件'}\n"
        "    return {}\n",
        encoding="utf-8")
    # 默认 script_enabled=false：脚本不被执行，写文件正常
    (proj / ".hs_hooks.json").write_text(
        json.dumps({"hooks_module": {"path": "hooks.py", "timeout": 5},
                    "script_enabled": False}), encoding="utf-8")
    hooks_mod.reload(str(proj))
    res = tool_registry.invoke("project_write", {"relative_path": "a.txt", "content": "x"})
    assert (proj / "a.txt").exists()
    # 显式开启：脚本 pre_tool_use 拦截生效
    (proj / ".hs_hooks.json").write_text(
        json.dumps({"hooks_module": {"path": "hooks.py", "timeout": 5},
                    "script_enabled": True}), encoding="utf-8")
    hooks_mod.reload(str(proj))
    res2 = tool_registry.invoke("project_write", {"relative_path": "b.txt", "content": "x"})
    assert "脚本拦截写文件" in res2
    assert not (proj / "b.txt").exists()


def test_stop_hook_runs_via_agent(proj):
    """run_task 结束路径触发 Stop：脚本 stop 被执行（用副作用文件验证）。"""
    marker = proj / "stop_fired.flag"
    path_repr = repr(str(marker))
    (proj / "hooks.py").write_text(
        "import pathlib\n"
        "def stop():\n"
        "    pathlib.Path(" + path_repr + ").write_text('1', encoding='utf-8')\n",
        encoding="utf-8")
    (proj / ".hs_hooks.json").write_text(
        json.dumps({"script_enabled": True,
                    "hooks_module": {"path": "hooks.py", "timeout": 5}}),
        encoding="utf-8")
    hooks_mod.reload(str(proj))
    from core.agent.agent import Agent
    from core.agent import memory as mem
    from core.llm import gateway
    import core.agent.agent as agent_mod
    # 记录 stop 前已由 Agent 走 run_task 结束路径
    old = gateway.chat
    state = {"n": 0}
    def chat(messages, provider_model=None, tools=None):
        state["n"] += 1
        return {"content": "done", "tool_calls": None}
    agent_mod.gateway.chat = chat
    try:
        ag = Agent(provider_model="openai/gpt-4o-mini")
        ag.run_task("做个简单任务", max_steps=1)
    finally:
        agent_mod.gateway.chat = old
    assert marker.exists()