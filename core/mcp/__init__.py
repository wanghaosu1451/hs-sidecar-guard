"""MCP 市场与 stdio 客户端：内置目录 + 用户自定义注册表 + 轻量 JSON-RPC 调用。"""
from .registry import (list_servers, get_server, add_server, remove_server,
                       builtin_catalog, registry_path)
from .client import call_tool, list_tools

__all__ = ["list_servers", "get_server", "add_server", "remove_server",
           "builtin_catalog", "registry_path", "call_tool", "list_tools"]