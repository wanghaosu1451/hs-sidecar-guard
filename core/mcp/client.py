"""轻量 stdio MCP 客户端：通过子进程 JSON-RPC 与本机 MCP server 通信。

设计目标：对任何缺失命令 / 超时 / 进程启动失败 / JSON-RPC 错误都返回可读报错，
绝不向上抛出并让 Agent 崩溃。每次调用启动一次进程，简单、安全、可超时。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from typing import Any

_DEFAULT_TIMEOUT = 15


def _is_missing_cmd(e: OSError) -> bool:
    s = str(e).lower()
    return any(k in s for k in ("not found", "no such file", "无法识别的命令",
                                "cannot find", "不是内部或外部"))


def _rpc_request(server: dict, method: str, params: dict, timeout: int) -> Any:
    """向单个 stdio server 发起一个 JSON-RPC 请求，返回 result；失败抛 ValueError。"""
    command = (server.get("command") or "").strip()
    if not command:
        raise ValueError(f"MCP 服务器 {server.get('name')} 未配置 command")
    args = [str(a) for a in (server.get("args") or [])]
    try:
        proc = subprocess.Popen(
            [command, *args],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )
    except OSError as e:
        if _is_missing_cmd(e):
            raise ValueError(
                f"无法启动 MCP 服务器 '{server.get('name')}'：缺少命令 {command!r}"
                f"（请安装或修正 command）。") from e
        raise ValueError(f"启动 MCP 服务器失败: {e}") from e

    request = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    try:
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            err = (proc.stderr.read(2000).strip() or "无输出")
            raise ValueError(f"MCP 服务器 {server.get('name')} 无响应：{err[:400]}")
        resp = json.loads(line) if line.strip() else {}
        if "error" in resp and resp["error"]:
            msg = (resp["error"].get("message") if isinstance(resp["error"], dict)
                   else str(resp["error"]))
            raise ValueError(f"MCP 调用错误: {msg or '未知错误'}")
        return resp.get("result", {})
    except json.JSONDecodeError as e:
        raise ValueError(f"MCP 返回非 JSON：{e}") from e
    finally:
        try:
            proc.stdin.close()
            proc.stdout.close()
            proc.stderr.close()
        except Exception:  # noqa: BLE001
            pass
        _terminate(proc)


def _terminate(proc: subprocess.Popen) -> None:
    try:
        proc.kill()
    except Exception:  # noqa: BLE001
        pass
    try:
        proc.wait(timeout=2)
    except Exception:  # noqa: BLE001
        pass


def _handshake(server: dict, timeout: int) -> None:
    _rpc_request(server, "initialize",
                 {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {
                     "name": "hs-agent", "version": "1.0"}}, timeout)


def list_tools(server: dict) -> list[str]:
    """返回该 server 暴露的工具名列表；失败返回空列表（不抛错）。"""
    try:
        _handshake(server, _DEFAULT_TIMEOUT)
        res = _rpc_request(server, "tools/list", {}, _DEFAULT_TIMEOUT)
        return [(t.get("name") if isinstance(t, dict) else str(t))
                for t in res.get("tools", [])]
    except ValueError:
        return []


def call_tool(server: dict, tool_name: str, args: Any = None,
              timeout: int = _DEFAULT_TIMEOUT) -> str:
    """调用 server 上的某个工具，返回可读文本结果；绝不让调用方崩溃。"""
    name = server.get("name", "?")
    try:
        _handshake(server, timeout)
        params = {"name": tool_name}
        arguments = args or {}
        if not isinstance(arguments, dict):
            arguments = {"input": arguments}
        params["arguments"] = arguments
        res = _rpc_request(server, "tools/call", params, timeout)
        text_parts: list[str] = []
        for c in res.get("content", []) if isinstance(res, dict) else []:
            if isinstance(c, dict):
                t = c.get("text") or c.get("type") or ""
                text_parts.append(str(t))
            else:
                text_parts.append(str(c))
        body = "\n".join(text_parts).strip()
        if isinstance(res, dict) and res.get("isError") and not body:
            return f"(MCP::调用失败:{name}/{tool_name})"
        return body or f"(MCP::{name}::{tool_name} 返回空)"
    except ValueError as e:
        return f"(MCP 错误: {e})"
    except Exception as e:  # noqa: BLE001
        return f"(MCP 调用异常: {e})"