"""智能体核心冒烟测试：工具注册表 + 记忆裁剪。"""
from __future__ import annotations

import pytest

from core.agent import tools
from core.agent.memory import Memory
from core.sandbox.security import SandboxPolicy
from core.sandbox.executor import Executor


def test_tools_registered():
    schema = tools.tools_schema()
    names = {f["function"]["name"] for f in schema}
    assert {"read_file", "write_file", "list_dir", "run_shell"} <= names
    assert "patch_file" in names
    assert "computer_screenshot" in names


def test_tool_invoke_hello():
    out = tools.invoke("hello", {"name": "张三"})
    assert "张三" in out


def test_tool_unknown_returns_error():
    out = tools.invoke("no_such_tool", {})
    assert "未知工具" in out


def test_memory_add_and_compact():
    m = Memory(max_messages=4, max_chars=10000)
    m.add_user("你好")
    m.add_assistant("你好！")
    assert len(m.get()) == 2
    m.compact()
    assert len(m.get()) == 2


def test_memory_compact_drops_system():
    m = Memory(max_messages=2)
    m.messages.append({"role": "system", "content": "sys"})
    for i in range(6):
        m.add_user(f"u{i}")
        m.add_assistant(f"a{i}")
    m.compact()
    roles = [x["role"] for x in m.get()]
    assert roles[0] == "system"
    assert roles.count("user") <= 2


def test_sandbox_policy_paths():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        allow = Path(d)
        (allow / "ok.txt").write_text("x")
        policy = SandboxPolicy(allowed_dirs=[str(allow)])
        ok = policy.validate_paths([str(allow / "ok.txt")])
        assert ok == [str(allow / "ok.txt")]


def test_executor_run_code_isolated():
    policy = SandboxPolicy(timeout_seconds=10)
    ex = Executor(policy)
    r = ex.run_python("print(1+1)")
    assert r.ok
    assert r.stdout.strip() == "2"


def test_executor_timeout():
    policy = SandboxPolicy(timeout_seconds=1)
    ex = Executor(policy)
    r = ex.run_python("import time; time.sleep(5)")
    assert r.timed_out


# ---- 边缘 shell 命令 / 小众文件 / Git 复杂操作兼容性 ----

def test_decode_bytes_encodings():
    # UTF-8
    assert tools._decode_bytes("中文".encode("utf-8")) == "中文"
    # GBK 小众编码
    assert tools._decode_bytes("中文".encode("gb18030")) == "中文"
    # UTF-16 带 BOM
    assert tools._decode_bytes("中文".encode("utf-16")) == "中文"
    # 二进制（含 NUL）判定为 None
    assert tools._decode_bytes(b"\x00\x01\x02") is None


def test_read_file_binary_returns_preview(tmp_path):
    bin_f = tmp_path / "logo.bin"
    bin_f.write_bytes(b"\x89PNG\x00\x01\x02\x03")
    out = tools._read_file(str(bin_f))
    assert "二进制文件" in out
    assert "89 50 4e 47" in out  # hex 预览


def test_read_file_gbk_and_utf16(tmp_path):
    gbk = tmp_path / "note_gbk.txt"
    gbk.write_bytes("中文标题".encode("gbk"))
    assert "中文标题" in tools._read_file(str(gbk))
    u16 = tmp_path / "note_u16.txt"
    u16.write_bytes("UTF16内容".encode("utf-16"))
    assert "UTF16内容" in tools._read_file(str(u16))


def test_executor_decodes_non_utf8_stdout():
    policy = SandboxPolicy(timeout_seconds=10)
    r = Executor(policy).run_python(
        "import sys; sys.stdout.buffer.write('中文'.encode('gbk')); "
        "sys.stdout.buffer.flush()"
    )
    assert "中文" in r.stdout


def test_executor_env_git_pager_hardened():
    policy = SandboxPolicy(allow_network=False)
    env = Executor(policy)._env()
    assert env.get("GIT_PAGER") == "cat"
    assert env.get("PAGER") == "cat"
    assert env.get("GIT_TERMINAL_PROMPT") == "0"
    assert "http_proxy" not in env  # 断网脱代理


def test_run_shell_edge_command(tmp_path):
    tools.set_project_root(str(tmp_path))
    try:
        out = tools.invoke("run_shell", {"command": "echo hi-边缘"})
        assert "exit=0" in out
        assert "hi-边缘" in out
    finally:
        tools.set_project_root(None)


# ---- patch_file：结构化 diff 精确编辑（借鉴 codex apply_patch） ----

def test_apply_unified_diff_roundtrip():
    orig = "def a():\n    return 1\n\ndef b():\n    return 2\n"
    new = "def a():\n    return 1\n\ndef b():\n    return 2\n\ndef c():\n    return 3\n"
    import difflib
    diff = "".join(difflib.unified_diff(orig.splitlines(keepends=True),
                                        new.splitlines(keepends=True), n=2))
    out, note = tools._apply_unified_diff(orig, diff)
    assert out == new.rstrip("\n") or out == new
    assert "hunk" in note


def test_patch_file_end_to_end(tmp_path):
    tools.set_project_root(str(tmp_path))
    try:
        (tmp_path / "x.py").write_text("v = 1\n", encoding="utf-8")
        diff = "@@ -1,1 +1,2 @@\n v = 1\n+v = 2\n"
        res = tools.invoke("patch_file", {"relative_path": "x.py", "diff": diff})
        assert "已写入" in res
        assert (tmp_path / "x.py").read_text(encoding="utf-8") == "v = 1\nv = 2\n"
    finally:
        tools.set_project_root(None)


def test_patch_file_unlocatable_fails(tmp_path):
    tools.set_project_root(str(tmp_path))
    try:
        (tmp_path / "x.py").write_text("v = 1\n", encoding="utf-8")
        res = tools.invoke("patch_file",
                           {"relative_path": "x.py",
                            "diff": "@@ -99,3 +99,3 @@\n ghost\n line\n"})
        assert "应用失败" in res
        assert (tmp_path / "x.py").read_text(encoding="utf-8") == "v = 1\n"
    finally:
        tools.set_project_root(None)


def test_patch_file_path_traversal_rejected(tmp_path):
    tools.set_project_root(str(tmp_path))
    try:
        res = tools.invoke("patch_file",
                           {"relative_path": "../evil.py",
                            "diff": "@@ -1 +1 @@\n+x"})
        assert "越界" in res
    finally:
        tools.set_project_root(None)