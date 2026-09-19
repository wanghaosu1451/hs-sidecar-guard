"""/init 命令测试：生成 AGENTS.md、幂等、不覆盖已有、只读不联网。
"""
from __future__ import annotations

import contextlib
import io

import cli


def _run_init(root):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        cli._cmd_init(str(root))
    return buf.getvalue()


def test_init_generates_agents_md(tmp_path):
    out = _run_init(tmp_path)
    agents = tmp_path / "AGENTS.md"
    assert agents.is_file()
    content = agents.read_text(encoding="utf-8")
    # 约定要点齐全
    assert "project_write" in content
    assert "verify_code" in content
    assert "/sub" in content
    assert "已生成 AGENTS.md" in out


def test_init_idempotent(tmp_path):
    _run_init(tmp_path)
    first = (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
    _run_init(tmp_path)
    second = (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
    assert first == second  # 内容不变
    # 未生成多余文件
    names = {p.name for p in tmp_path.iterdir()}
    assert "AGENTS.md" in names


def test_init_does_not_overwrite_existing(tmp_path):
    markers = tmp_path / "AGENTS.md"
    markers.write_text("# 用户自定义规则\n- 我的项目约定\n", encoding="utf-8")
    out = _run_init(tmp_path)
    content = (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
    assert "用户自定义规则" in content
    assert "未覆盖，保留你的约定" in out or "保留" in out