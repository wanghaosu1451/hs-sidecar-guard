"""Work/Code/Design 三模式系统提示词语义化测试。

覆盖：
- core.agent.agent.MODE_PROMPTS 含 work/code/design 且互不相同、含默认 work；
- cli 的 --mode 参数能选出对应 system_prompt。
"""
from __future__ import annotations

import sys

import core.agent.agent as agent_mod
import cli


def test_modes_present_distinct_with_default():
    p = agent_mod.MODE_PROMPTS
    assert {"work", "code", "design"} <= set(p)
    assert p["work"] != p["code"] != p["design"]
    assert p["default"] == p["work"]
    assert "工作台" in p["work"]
    assert "代码" in p["code"]
    assert "设计" in p["design"]


class _FakeAgent:
    """替换 cli.Agent：记录构造时的 system_prompt，不联网执行。"""

    instances: list["_FakeAgent"] = []

    def __init__(self, system_prompt=..., provider_model=None, mode="work"):
        self.system_prompt = system_prompt
        self.provider_model = provider_model
        self.mode = mode
        self.calls = []
        _FakeAgent.instances.append(self)

    def set_provider(self, m):
        pass

    def attach_workspace(self, w):
        pass

    def reset(self, keep_system=True):
        pass

    def run_task(self, task, max_steps=30, on_step=None, should_stop=None):
        self.calls.append(task)
        if on_step:
            on_step("[project_write]", "ok")
        return {"answer": "完成", "steps": 0, "tools": []}


def test_cli_mode_selects_corresponding_prompt(monkeypatch, tmp_path):
    _FakeAgent.instances = []
    monkeypatch.setattr(cli, "Agent", _FakeAgent)

    prev_argv = sys.argv
    sys.argv = ["cli.py", "--mode", "design", "-d", str(tmp_path), "做一个登录页"]
    try:
        rc = cli.main()
    finally:
        sys.argv = prev_argv

    assert rc == 0
    assert len(_FakeAgent.instances) == 1
    fake = _FakeAgent.instances[0]
    assert fake.calls == ["做一个登录页"]
    assert fake.system_prompt == agent_mod.MODE_PROMPTS["design"]


def test_cli_default_mode_is_work(monkeypatch, tmp_path):
    _FakeAgent.instances = []
    monkeypatch.setattr(cli, "Agent", _FakeAgent)

    prev_argv = sys.argv
    sys.argv = ["cli.py", "-d", str(tmp_path), "写一份报告"]
    try:
        rc = cli.main()
    finally:
        sys.argv = prev_argv

    assert rc == 0
    assert len(_FakeAgent.instances) == 1
    assert _FakeAgent.instances[0].system_prompt == agent_mod.MODE_PROMPTS["work"]