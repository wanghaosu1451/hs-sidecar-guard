"""后台并行子智能体编排的测试。

关键：绝不能在测试里真调远端模型（settings 里是远端 qwen3，会慢/断网）。
一律用 monkeypatch 把 core.llm.gateway.chat 替换成固定返回，或把子任务执行
流程替换成可控闭包，保证不碰网络、可重现、很快结束。
"""
from __future__ import annotations

import re
import threading
import time

import pytest

from core.llm import gateway
from core.agent import tools as tool_registry
from core.agent import subagent as sub


# ---------------- 工具注册 / CORE_TOOL_NAMES ----------------

def test_sub_tools_registered_in_core():
    assert {"sub_spawn", "sub_list", "sub_result"} <= set(tool_registry.CORE_TOOL_NAMES)
    names = {s["function"]["name"] for s in tool_registry.tools_schema()}
    assert {"sub_spawn", "sub_list", "sub_result"} <= names
    core_names = {s["function"]["name"] for s in tool_registry.core_tools_schema()}
    assert {"sub_spawn", "sub_list", "sub_result"} <= core_names


def _tid(msg: str) -> str:
    m = re.search(r"(sub-\d+-\w+)", msg)
    assert m, f"未从返回中解析到 task_id: {msg}"
    return m.group(1)


# ---------------- (1) spawn → sub_list 能看到 running ----------------

def test_spawn_shows_running(monkeypatch):
    """把 _run_sub 换成可控闭包（等待事件再置 done），保证 observe 到 running 态。"""
    gate = threading.Event()
    def block(tid, prompt, role, model):
        gate.wait(10)
        with sub._LOCK:
            sub._TASKS[tid]["status"] = "done"
            sub._TASKS[tid]["result"] = "ok"
    monkeypatch.setattr(sub, "_run_sub", block)
    msg = sub.spawn_task("hello")
    listing = sub.sub_list()
    assert "running" in listing
    assert _tid(msg) in listing
    gate.set()  # 放行子线程收尾，避免后期环境残留


# ---------------- (2)(3) 完成后 sub_result 取回，取走后表里无该条 ----------------

def _patch_chat(monkeypatch):
    monkeypatch.setattr(gateway, "chat",
                        lambda *a, **k: {"content": "ok", "tool_calls": None})


def _poll_done(tid: str, timeout: float = 10.0) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with sub._LOCK:
            item = sub._TASKS.get(tid)
            st = item.get("status") if item else None
        if st and st != "running":
            return st
        time.sleep(0.02)
    return "timeout"


def test_collect_result_and_removed(monkeypatch):
    sub._TASKS.clear()
    _patch_chat(monkeypatch)
    msg = sub.spawn_task("请给出 1+1 等于几")
    tid = _tid(msg)
    st = _poll_done(tid)
    assert st == "done"
    res = sub.sub_result(tid)
    assert "ok" in res
    # 一次性取走：表中不再有该条
    assert tid not in sub.sub_list()


def test_collect_keep_keeps_entry(monkeypatch):
    sub._TASKS.clear()
    _patch_chat(monkeypatch)
    tid = _tid(sub.spawn_task("给我一段计划"))
    _poll_done(tid)
    sub.sub_result(tid, keep=True)
    assert tid in sub.sub_list()


def test_sub_result_unknown_and_running(monkeypatch):
    # 不存在的 task_id
    assert "未找到子任务" in sub.sub_result("sub-none")
    # 仍在运行（未放行前 status=running）
    gate = threading.Event()
    def block(tid, prompt, role, model):
        gate.wait(5)
        with sub._LOCK:
            sub._TASKS[tid]["status"] = "done"
            sub._TASKS[tid]["result"] = "ok"
    monkeypatch.setattr(sub, "_run_sub", block)
    tid = _tid(sub.spawn_task("x"))
    try:
        assert "仍在运行" in sub.sub_result(tid)
    finally:
        gate.set()


# ---------------- (4) 并发上限 ----------------

def test_concurrency_cap_rejects(monkeypatch):
    """并发达到上限时再委派应被拒绝；用可控闭包把任务卡在 running 态，确定性可重现。"""
    monkeypatch.setattr(sub, "_MAX_CONCURRENT", 2)
    gate = threading.Event()
    def block(tid, prompt, role, model):
        gate.wait(10)
        with sub._LOCK:
            sub._TASKS[tid]["status"] = "done"
            sub._TASKS[tid]["result"] = "ok"
    monkeypatch.setattr(sub, "_run_sub", block)
    try:
        r1 = sub.spawn_task("a")
        r2 = sub.spawn_task("b")
        assert "已提交" in r1 and "已提交" in r2
        r3 = sub.spawn_task("c")
        assert "并发已达上限" in r3
    finally:
        gate.set()


# ---------------- (5) 递归禁止 ----------------

def test_recursion_rejected_in_sub_thread():
    """子线程内再调 spawn_task 应返回不支持。"""
    captured = {}

    def worker():
        sub._thread_local.in_sub = True
        try:
            captured["ret"] = sub.spawn_task("再次下发")
        finally:
            sub._thread_local.in_sub = False

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=10)
    assert "不支持再次下发" in captured["ret"]


# ---------------- 空表友好提示 ----------------

def test_sub_list_empty_message():
    sub._TASKS.clear()
    assert "暂无子任务" in sub.sub_list()