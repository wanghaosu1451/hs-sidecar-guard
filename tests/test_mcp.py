"""MCP 市场 / 生态测试：注册表 CRUD、stdio 客户端健壮性、工具路由、mcp_call 工具。"""
import json
import sys
import tempfile
from pathlib import Path

import pytest

from core import mcp as mcp_mod
from core.mcp import registry as mcp_registry
from core.mcp import client as mcp_client
from core.agent import tools as tool_registry


@pytest.fixture(autouse=True)
def _isolate_registry(monkeypatch, tmp_path):
    """把注册表落到临时文件，避免污染仓库根的真实 mcp_servers.json。"""
    f = tmp_path / "mcp_servers.json"
    monkeypatch.setattr(mcp_registry, "_REGISTRY_FILE", f)
    monkeypatch.setattr(mcp_registry, "_CUSTOM", {})
    yield f
    mcp_registry._CUSTOM = {}


# 一个“真 stdio JSON-RPC 服务器”，用当前 Python 解释器起子进程。
_FAKE_SERVER = """
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
    except Exception:
        continue
    m = req.get("method")
    rid = req.get("id")
    if m == "initialize":
        r = {"jsonrpc":"2.0","id":rid,"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"fake","version":"1.0"}}}
    elif m == "tools/list":
        r = {"jsonrpc":"2.0","id":rid,"result":{"tools":[{"name":"echo","description":"echo text"}]}}
    elif m == "tools/call":
        r = {"jsonrpc":"2.0","id":rid,"result":{"content":[{"type":"text","text":"hello-from-mcp"}]}}
    else:
        continue
    sys.stdout.write(json.dumps(r) + "\\n")
    sys.stdout.flush()
"""

_FAKE_CMD = sys.executable
_FAKE_ARGS = ["-c", _FAKE_SERVER]


def _add_fake(name: str = "fakesrv") -> dict:
    return mcp_registry.add_server(name, _FAKE_CMD, _FAKE_ARGS, "fake test server")


# ---------------- 注册表 ----------------
def test_registry_add_and_list():
    srv = _add_fake("mybox")
    assert srv["name"] == "mybox" and srv["command"] == _FAKE_CMD
    names = [s["name"] for s in mcp_registry.list_servers()]
    assert "mybox" in names
    # 内置目录三条都在
    for builtin in ("filesystem", "sqlite", "webkit"):
        assert builtin in names


def test_registry_remove():
    _add_fake("gone")
    assert mcp_registry.remove_server("gone") is True
    assert mcp_registry.remove_server("gone") is False
    assert mcp_registry.remove_server("sqlite") is False  # 内置不可删


def test_registry_add_requires_fields():
    with pytest.raises(ValueError):
        mcp_registry.add_server("", "cmd")
    with pytest.raises(ValueError):
        mcp_registry.add_server("x", "  ")


def test_builtin_catalog_excludes_installed():
    _add_fake("filesystem")  # 覆盖同名内置
    names = [s["name"] for s in mcp_registry.builtin_catalog()]
    assert "filesystem" not in names
    assert any(s["name"] == "sqlite" for s in mcp_registry.builtin_catalog())


def test_get_server():
    _add_fake("findme")
    assert mcp_registry.get_server("findme")["command"] == _FAKE_CMD
    assert mcp_registry.get_server("nope") is None


# ---------------- 客户端健壮性 ----------------
def test_client_list_tools():
    srv = _add_fake()
    tools = mcp_client.list_tools(srv)
    assert tools == ["echo"]


def test_client_call_tool_ok():
    srv = _add_fake()
    out = mcp_client.call_tool(srv, "echo", {"text": "hi"})
    assert out == "hello-from-mcp"


def test_client_missing_command_does_not_crash():
    srv = {"name": "bad", "command": "no_such_cmd_hs_xyz_123", "args": []}
    out = mcp_client.call_tool(srv, "t", {})
    assert isinstance(out, str) and "MCP" in out


def test_client_no_output_graceful():
    """假服务器不返回任何东西时，返回可读报错而非崩溃。"""
    server_code = "import sys; sys.stdin.readline()"
    srv = mcp_registry.add_server("silent", sys.executable, ["-c", server_code])
    out = mcp_client.call_tool(srv, "t", {})
    assert isinstance(out, str)


# ---------------- 工具路由 ----------------
def test_tools_schema_contains_mcp_call():
    names = [t["function"]["name"] for t in tool_registry.tools_schema()]
    assert "mcp_call" in names


def test_invoke_mcp_route_via_meta():
    _add_fake("echo_srv")
    res = tool_registry.invoke("mcp_call",
                               {"server": "echo_srv", "tool": "echo", "arguments": {"text": "x"}})
    assert "hello-from-mcp" in res


def test_invoke_mcp_dynamic_route():
    _add_fake("dyn")
    res = tool_registry.invoke("mcp::dyn::echo", {"text": "x"})
    assert "hello-from-mcp" in res


def test_invoke_unknown_mcp_server_errors():
    res = tool_registry.invoke("mcp::nope::t", {})
    assert "错误：" in res and "nope" in res


def test_invoke_mcp_route_redacts_and_caps():
    # 返回超大 + 密钥内容仍不会外泄/膨胀
    big_srv = """
import sys, json
for line in sys.stdin:
    try: req=json.loads(line)
    except Exception: continue
    rid=req.get("id"); m=req.get("method")
    if m=="initialize":
        r={"jsonrpc":"2.0","id":rid,"result":{"capabilities":{}}}
    elif m=="tools/call":
        r={"jsonrpc":"2.0","id":rid,"result":{"content":[{"type":"text","text":"secret api_key = sk-abcdefgh12345678 " + "x"*9000}]}}
    else:
        continue
    sys.stdout.write(json.dumps(r)+"\\n"); sys.stdout.flush()
"""
    _add_fake("leaky")
    # 覆写该服务器的命令为构造的巨型服务器
    mcp_registry.add_server("leaky", sys.executable, ["-c", big_srv])
    res = tool_registry.invoke("mcp_call", {"server": "leaky", "tool": "x", "arguments": {}})
    assert "[REDACTED" in res          # 密钥被脱敏
    assert len(res) < 5000             # 长度被截断