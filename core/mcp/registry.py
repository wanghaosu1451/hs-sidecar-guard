"""MCP 注册表：内置市场目录 + 用户自定义服务器清单（持久化到 mcp_servers.json）。

每条登记一个 stdio MCP 服务器：name / command / args / desc / builtin。
单进程内直接调用 stdio JSON-RPC（见 client.py），默认断网策略不受影响。

生命周期（deny-first / 渐进式信任）：
- 内置随产品分发的服务器 pre-trusted（approved=True，开箱即用）；
- 用户自定义 / 从 GitHub 安装的服务器默认 approved=False，登记后一律先登记启用意图，
  由 operator 用 /mcp approve <name> 批准后才允许被模型调用（最小权限白名单）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# 项目根（含 ai_projects 等）之外，注册表放在 hs-sidecar-guard 仓库根，跟随项目存在。
_REGISTRY_FILE = Path(__file__).resolve().parent.parent.parent / "mcp_servers.json"

# 内置市场目录：默认提供 12 个热门"开箱即用"MCP 服务，按 category 分组。
# 随产品分发的内置服务器视为可信基线，approved=True。
BUILTIN: list[dict] = [
    # ---------- 数据库 ----------
    {"name": "sqlite", "builtin": True, "approved": True, "category": "数据库",
     "desc": "本地 SQLite 查询（execute/query），供 AI 跑数据血缘与查询。",
     "command": "python", "args": ["-m", "mcp_server_sqlite", "app.db"],
     "tools": ["execute_query", "list_tables"],
     "example": {"execute_query": '{"sql": "SELECT 1"}'},
     "github": "modelcontextprotocol/python-sdk"},
    {"name": "Postgres", "builtin": True, "approved": True, "category": "数据库",
     "desc": "PostgreSQL 查询/写入，支持多库连接。",
     "command": "npx", "args": ["-y", "@modelcontextprotocol/server-postgres", "$DATABASE_URL"],
     "tools": ["query", "list_tables", "describe_table"],
     "example": {"query": '{"sql": "SELECT 1"}'},
     "github": "modelcontextprotocol/servers"},
    {"name": "Redis", "builtin": True, "approved": True, "category": "数据库",
     "desc": "Redis 键值查询与读写。",
     "command": "python", "args": ["-m", "mcp_server_redis"],
     "tools": ["get", "set", "keys", "hgetall"],
     "example": {"get": '{"key": "demo"}'}},

    # ---------- 浏览器 ----------
    {"name": "webkit", "builtin": True, "approved": True, "category": "浏览器",
     "desc": "本地 WebKit 浏览器操作（snapshot/click/type），用于轻量前端 E2E。",
     "command": "python", "args": ["-m", "mcp_server_webkit"],
     "tools": ["snapshot", "click", "type_text"],
     "example": {"snapshot": "{}"},
     "github": "modelcontextprotocol/webkit"},
    {"name": "Playwright", "builtin": True, "approved": True, "category": "浏览器",
     "desc": "真实浏览器 E2E 自动化（Chrome/Firefox/WebKit），截图、网络拦截。",
     "command": "npx", "args": ["-y", "@playwright/mcp"],
     "tools": ["browser_navigate", "browser_click", "browser_screenshot", "browser_fill"],
     "example": {"browser_navigate": '{"url": "https://example.com"}'},
     "github": "microsoft/playwright"},

    # ---------- 开发 ----------
    {"name": "GitHub", "builtin": True, "approved": True, "category": "开发",
     "desc": "GitHub Issue/PR/Repo 操作（需 GITHUB_TOKEN）。",
     "command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"],
     "tools": ["create_issue", "list_issues", "create_pull_request", "list_repos"],
     "example": {"list_issues": '{"repo": "owner/repo"}'},
     "github": "modelcontextprotocol/servers"},

    # ---------- 监控 ----------
    {"name": "Sentry", "builtin": True, "approved": True, "category": "监控",
     "desc": "Sentry 错误追踪与 Issue 管理（需 SENTRY_AUTH_TOKEN）。",
     "command": "npx", "args": ["-y", "@sentry/mcp-server"],
     "tools": ["list_issues", "get_issue", "update_issue"],
     "example": {"list_issues": '{"project": "my-project"}'},
     "github": "getsentry/sentry-mcp"},

    # ---------- 搜索 ----------
    {"name": "BraveSearch", "builtin": True, "approved": True, "category": "搜索",
     "desc": "Brave Search API 网页搜索（需 BRAVE_API_KEY）。",
     "command": "npx", "args": ["-y", "@modelcontextprotocol/server-brave-search"],
     "tools": ["web_search", "local_search"],
     "example": {"web_search": '{"query": "MCP 协议"}'},
     "github": "modelcontextprotocol/servers"},

    # ---------- 文档 ----------
    {"name": "Notion", "builtin": True, "approved": True, "category": "文档",
     "desc": "Notion 数据库/页面读写（需 NOTION_API_KEY）。",
     "command": "npx", "args": ["-y", "@notionhq/notion-mcp-server"],
     "tools": ["list_pages", "get_page", "create_page", "query_database"],
     "example": {"list_pages": "{}"},
     "github": "notion/notion-mcp-server"},
    {"name": "Context7", "builtin": True, "approved": True, "category": "文档",
     "desc": "代码文档检索（库 API 文档/示例），无需 token。",
     "command": "npx", "args": ["-y", "@upstash/context7-mcp"],
     "tools": ["resolve-library-id", "query-docs"],
     "example": {"query-docs": '{"library": "react", "query": "useState"}'},
     "github": "upstash/context7-mcp"},

    # ---------- 项目管理 ----------
    {"name": "Linear", "builtin": True, "approved": True, "category": "项目管理",
     "desc": "Linear Issue/Ticket 管理（需 LINEAR_API_KEY）。",
     "command": "npx", "args": ["-y", "linear/mcp-server"],
     "tools": ["list_issues", "create_issue", "update_issue"],
     "example": {"list_issues": "{}"},
     "github": "linear/linear-mcp"},

    # ---------- 文件 ----------
    {"name": "filesystem", "builtin": True, "approved": True, "category": "文件",
     "desc": "受限文件访问（list/read/write），AI 可安全读写本地文件。",
     "command": "python", "args": ["-m", "mcp_server_fs", "."],
     "tools": ["read_file", "write_file", "list_directory"],
     "example": {"read_file": '{"path": "./README.md"}'},
     "github": "modelcontextprotocol/python-sdk"},
]

# 用户注册表：name -> server 配置（merged 后持久化）。
_CUSTOM: dict[str, dict] = {}

# 内存态：已登记、待 operator 启用（激活）的 intent 清单。
_PENDING_ACTIVATE: list[dict] = []


def registry_path() -> Path:
    return _REGISTRY_FILE


def _load() -> None:
    global _CUSTOM
    _CUSTOM = {}
    if not _REGISTRY_FILE.is_file():
        return
    try:
        data = json.loads(_REGISTRY_FILE.read_text(encoding="utf-8"))
        for item in data.get("servers", []):
            if isinstance(item, dict) and item.get("name"):
                # 历史条目缺 approved 视为未启用（deny-first 兜底，杜绝"登记即放行"）。
                it = dict(item, builtin=False)
                it.setdefault("approved", False)
                _CUSTOM[item["name"]] = it
    except Exception:  # noqa: BLE001
        _CUSTOM = {}


def _save() -> None:
    _REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    _REGISTRY_FILE.write_text(
        json.dumps({"servers": list(_CUSTOM.values())}, ensure_ascii=False, indent=2),
        encoding="utf-8")


def builtin_catalog() -> list[dict]:
    """返回内置市场目录（已被用户自定义"安装/覆盖"的排除掉）。"""
    _load()
    return [dict(b) for b in BUILTIN if b["name"] not in _CUSTOM]


def list_servers() -> list[dict]:
    """返回全部已配置服务器：内置 + 用户自定义（同名自定义覆盖内置）。"""
    _load()
    out: list[dict] = []
    for b in BUILTIN:
        if b["name"] in _CUSTOM:
            continue  # 被自定义覆盖，不再列内置
        out.append(dict(b, enabled=True, approved=b.get("approved", True)))
    for c in _CUSTOM.values():
        out.append(dict(c, enabled=True))
    return out


def get_server(name: str) -> dict | None:
    for s in list_servers():
        if s.get("name") == name:
            return s
    return None


def _merge(server: dict) -> dict:
    """将内置定义补全到自定义同名条目上（命令可被覆盖）。"""
    base = next((b for b in BUILTIN if b.get("name") == server.get("name")), {})
    merged = dict(base, **server)
    merged.setdefault("builtin", False)
    return merged


def add_server(name: str, command: str, args: list[str] | None = None,
               desc: str = "") -> dict:
    """新增/覆盖一台用户 MCP 服务器，持久化到 mcp_servers.json。"""
    _load()
    name = (name or "").strip()
    command = (command or "").strip()
    if not name or not command:
        raise ValueError("name 与 command 均为必填")
    server = {"name": name, "command": command,
              "args": [str(a) for a in (args or [])],
              "desc": (desc or "").strip(),
              "approved": False, "tools": []}
    _CUSTOM[name] = _merge(server)
    _save()
    return dict(_CUSTOM[name])


def remove_server(name: str) -> bool:
    _load()
    if name not in _CUSTOM:
        return False
    del _CUSTOM[name]
    _save()
    return True


# ============================ 启用前 intent 审批（deny-first） ============================

def request_use(name: str, tool: str) -> dict:
    """调用前置的 intent 审批闸：返回是否放行。

    - 服务器未启用：登记启用意图进 _PENDING_ACTIVATE，返回 blocked=True + 提示（需 /mcp approve）。
    - 已启用但工具不在声明白名单：按最小权限拦截，返回 blocked=True。
    - 已启用且在白名单内：放行（blocked=False）。
    """
    global _PENDING_ACTIVATE
    s = get_server(name)
    if s is None:
        return {"blocked": True, "approved": False,
                "message": f"错误：未找到 MCP 服务器 {name}（可用 /mcp view 查看）"}
    if not s.get("approved", False):
        # 去重登记 intent（只记首条工具名即可）
        if not any(p["name"] == name for p in _PENDING_ACTIVATE):
            _PENDING_ACTIVATE.append({"name": name, "tool": tool})
        return {"blocked": True, "approved": False,
                "message": (f"MCP 服务器 {name} 尚未启用（deny-first，先登记后授权）。"
                            f"已登记启用意图，请 operator 用 /mcp approve {name} 批准后重试调用。")}
    # 最小权限白名单：声明过 tools 且不在其中 => 拦截
    declared = [t for t in (s.get("tools") or []) if t]
    if declared and tool not in declared:
        return {"blocked": True, "approved": True,
                "message": (f"MCP 服务器 {name} 已启用，但工具 {tool} 不在其声明白名单"
                            f"（最小权限）：{declared}。已拒绝调用该越权工具。")}
    return {"blocked": False, "approved": True, "message": ""}


def pending_activations(clear: bool = False) -> list[dict]:
    """返回待启用（待 operator 批准）的 intent 清单。"""
    global _PENDING_ACTIVATE
    out = [dict(p) for p in _PENDING_ACTIVATE]
    if clear:
        _PENDING_ACTIVATE = []
    return out


def set_approved(name: str, approved: bool) -> bool:
    """仅持久化用户自定义条目的启用状态（内置服务器由代码内的 approved 常量决定）。"""
    _load()
    if name not in _CUSTOM:
        return False
    _CUSTOM[name]["approved"] = bool(approved)
    _save()
    return True


def approve(name: str) -> dict:
    """批准启用：置 approved=True，若未声明 tools 则反填白名单，随后顺势清除该 intent。"""
    ok = set_approved(name, True)
    if not ok:
        return {"ok": False, "message": f"未找到可批准的自定义服务器 {name}（内置服务器默认已启用）"}
    srv = get_server(name)
    tools = [t for t in (srv.get("tools") or []) if t]
    if not tools:
        from core.mcp.client import list_tools
        discovered = list_tools(srv or {})
        if discovered:
            _load()
            if name in _CUSTOM:
                _CUSTOM[name]["tools"] = discovered
                _save()
                tools = discovered
    global _PENDING_ACTIVATE
    _PENDING_ACTIVATE = [p for p in _PENDING_ACTIVATE if p["name"] != name]
    return {"ok": True, "name": name, "tools": tools,
            "message": f"MCP 服务器 {name} 已启用"
                       + (f"，白名单：{tools}" if tools else "（未声明工具，未做白名单限制）")}


def revoke(name: str) -> bool:
    """撤销启用（回到未批准状态），用于收回权限。"""
    return set_approved(name, False)


def discover_catalog() -> list[dict]:
    """扫社区热门 MCP 服务，返回可安装清单。
    实现：从内置 BUILTIN 派生 + 手动精选。
    真正 GitHub API 扫描需要 token，这里用硬编码精选列表。"""
    EXTRA = [
        {"name": "Slack", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-slack"],
         "github": "modelcontextprotocol/servers", "category": "通讯",
         "desc": "Slack 消息/频道/线程操作"},
        {"name": "GoogleDrive", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-google-drive"],
         "github": "modelcontextprotocol/servers", "category": "文档",
         "desc": "Google Drive 文件读写"},
        {"name": "Figma", "command": "npx", "args": ["-y", "figma-developer-mcp"],
         "github": "figma/mcp", "category": "设计",
         "desc": "Figma 设计数据读取"},
        {"name": "Firecrawl", "command": "npx", "args": ["-y", "@mcp/firecrawl"],
         "github": "firecrawl/firecrawl-mcp", "category": "搜索",
         "desc": "网页爬取+结构化"},
    ]
    return BUILTIN + EXTRA
