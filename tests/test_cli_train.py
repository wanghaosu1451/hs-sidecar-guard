"""终端版训练/微调命令离线测试（不联网）。

覆盖交互式训练命令的核心分发 _cmd_train：环境探测、研究创新纯计算、CLI 参数入口。
依赖 torch/transformers 的重路径（lora/full）不在此触发。
"""
from __future__ import annotations

import sys

from core.agent import tools as tool_registry
import cli


def test_cmd_train_fp4_generates_plan(tmp_path, capsys):
    tool_registry.set_project_root(str(tmp_path))
    cli._cmd_train("fp4")
    out = capsys.readouterr().out
    assert "FP4" in out or "量化" in out
    assert (tmp_path / "artifacts" / "research" / "fp4" / "fp4_qat_plan.json").is_file()


def test_cmd_train_status(tmp_path, capsys):
    tool_registry.set_project_root(str(tmp_path))
    cli._cmd_train("status")
    out = capsys.readouterr().out
    assert "训练依赖" in out and "torch" in out


def test_cmd_train_unknown(tmp_path, capsys):
    tool_registry.set_project_root(str(tmp_path))
    cli._cmd_train("bogus")
    assert "未知操作" in capsys.readouterr().out


def test_cli_train_flag(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["cli.py", "--train", "ced", "--dir", str(tmp_path)])
    assert cli.main() == 0
    assert (tmp_path / "artifacts" / "research" / "ced" / "ced_moe_plan.json").is_file()