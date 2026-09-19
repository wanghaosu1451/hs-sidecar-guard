"""工具注册表：Agent 可调用工具的注册与分发（function calling）。

注册表模式：每个工具一个函数 + 一个 JSON schema。新增工具=注册一个函数。
"""
from __future__ import annotations

import ast
import fnmatch
import os
import re
import subprocess
import json
import tempfile
import urllib.request
import urllib.error
from urllib.parse import urlparse
from pathlib import Path
from typing import Any, Callable

# name -> (handler, schema)
_REGISTRY: dict[str, tuple[Callable, dict]] = {}
# 允许的字符串长度上限，防止工具结果撑爆上下文
_RESULT_CAP = 4000

# 核心工具：完整 schema 常驻注入模型（数量少，小模型更易收敛、少死循环）。
# 其余工具仅以「名称+一句话」轻量目录注入提示词，模型知晓可调用、需用时按名调用
#（invoke 按名查表即可执行，不必预先下发完整 schema，实现“按需加载”）。
CORE_TOOL_NAMES = ("project_write", "delete_file", "verify_code", "run_shell",
                   "read_file", "list_project_files",
                   "computer_screenshot", "computer_click", "computer_type",
                   "computer_key", "computer_open",
                   "sub_spawn", "sub_list", "sub_result", "route_chain")

# Plan 只读模式允许的工具白名单：仅分析与查询，不含任何写文件/执行/破坏性操作。
READONLY_TOOLS = frozenset({
    "read_file", "list_dir", "list_project_files", "read_history",
    "graph_query", "graph_query_file", "repo_read",
    "code_review", "vuln_scan", "audit_read",
    "snapshot_list", "diff_file",
    "skill_list", "sub_list", "sub_result", "getenv", "hello",
})


def is_readonly_tool(name: str) -> bool:
    """判断工具是否为只读分析型（Plan 模式下唯一允许调用的一类）。"""
    return name in READONLY_TOOLS

# ============================ 项目写入目录（AI 落盘目标） ============================
# 用户指定 AI 生成项目写入哪个本地文件夹；未设置时默认当前工作目录。
_PROJECT_ROOT: str | None = None


def get_project_root() -> str:
    return _PROJECT_ROOT or "."


def set_project_root(path: str | None) -> str:
    """把 AI 生成项目的写入根目录设为指定本地文件夹（可任意指定）。"""
    global _PROJECT_ROOT
    if path:
        p = Path(path).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        _PROJECT_ROOT = str(p)
        return f"项目写入目录已设为: {p}"
    _PROJECT_ROOT = None
    return "已清除项目写入目录（回退到当前工作目录）"


# ============================ 自动 git 提交 ============================
# 写文件成功后若项目是 git 仓库，顺手 add + commit 一次，让每次改动都有版本可查；
# 非 git 仓库则静默忽略，绝不影响原始返回结果。
_GIT_ENV = {
    "GIT_PAGER": "cat",          # 避免 git 输出进分页器卡住
    "GIT_TERMINAL_PROMPT": "0",  # 非交互环境禁止 git 弹提示/等输入
    "NO_COLOR": "1",             # 关闭 git 彩色输出，便于解析
    "PYTHONIOENCODING": "utf-8", # 强制 utf-8 输出，避免中文乱码
}


def _autocommit(root: str | Path, relative_path: str) -> None:
    """对刚写入的文件执行 git add + git commit 自动提交；非 git 仓库则静默忽略。"""
    try:
        root_s = str(Path(root).resolve())
        env = dict(os.environ)
        env.update(_GIT_ENV)
        # add 之后再 commit；任一失败(非 git 仓库/无改动)都静默跳过
        subprocess.run(["git", "add", "--", relative_path], cwd=root_s, env=env,
                       check=False, capture_output=True, text=True)
        subprocess.run(["git", "commit", "-m", f"autocommit: {relative_path}"],
                       cwd=root_s, env=env, check=False,
                       capture_output=True, text=True)
    except Exception:  # noqa: BLE001
        pass


def _project_write(relative_path: str, content: str) -> str:
    """把文件写入项目目录（只允许相对路径，杜绝越界写盘）。"""
    root = Path(_PROJECT_ROOT).resolve() if _PROJECT_ROOT else Path.cwd().resolve()
    target = (root / relative_path).resolve()
    if not target.is_relative_to(root):
        return "错误：路径越界，仅允许写入项目目录内（相对路径，禁止 ../ 或绝对路径）"
    target.parent.mkdir(parents=True, exist_ok=True)
    # 覆盖写入前自动快照上一版本，供 diff/回滚
    try:
        from core.project.snapshot import backup
        backup(root, relative_path, content)
    except Exception:  # noqa: BLE001
        pass
    target.write_text(content, encoding="utf-8")
    # 备份成功后顺手自动提交（git 仓库时）
    _autocommit(root, relative_path)
    return f"已写入 {target}（{len(content)} 字符，相对项目根: {relative_path}）"


def _list_project_files() -> str:
    root = Path(_PROJECT_ROOT) if _PROJECT_ROOT else Path.cwd()
    if not root.is_dir():
        return "(项目目录不存在)"
    files = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
    return "\n".join(files) if files else "(暂无文件)"


# ============================ 数据外发防护（去敏感化） ============================
# 命中即脱敏的关键字（覆盖 key/token/密码 等）
_SECRET_RE = re.compile(
    r"(api[_-]?key|api[_-]?secret|auth[_-]?token|access[_-]?token|"
    r"secret|password|passwd|private[_-]?key)",
    re.IGNORECASE,
)
# PEM 私钥整块
_PEM_RE = re.compile(r"-----BEGIN [\w ]+-----.*?-----END [\w ]+-----", re.S)
# 高敏感文件名（命中即拒绝读取，避免进入云端 prompt）
_SENSITIVE_NAMES = (".env", ".env.*", "id_rsa", "id_ed25519", "id_dsa",
                    "*.pem", "*.p12", "*.pfx", "keystore", ".npmrc",
                    ".pypirc", ".aws", "credentials", "secrets.json",
                    "credentials.json")


def is_sensitive_name(name: str) -> bool:
    n = name.lower()
    return any(fnmatch.fnmatch(n, p.lower()) for p in _SENSITIVE_NAMES)


def is_sensitive_path(path: str) -> bool:
    parts = Path(path).parts
    return any(is_sensitive_name(p) for p in parts)


def redact(text: str) -> str:
    """把疑似密钥内容替换为 [REDACTED]，防止外发到云端模型。"""
    if not text:
        return text
    # PEM 私钥整块优先隐藏
    text = _PEM_RE.sub("[REDACTED-PRIVATE-KEY]", text)
    # 关键字命中的单词本身也隐藏
    text = _SECRET_RE.sub("[REDACTED-SECRET]", text)
    # 常见 key 形如 = sk-xxxx 或 JWT(eyJ...)
    text = re.sub(r"=?(sk-|eyJ)[A-Za-z0-9_.\-]{8,}\b", "[REDACTED-KEY]", text)
    # 独立成行的 base64（可能是密钥正文）
    text = re.sub(r"(?m)^[A-Za-z0-9+/]{40,}={0,2}\s*$", "[REDACTED-BLOCK]", text)
    return text


def _cap(text: str) -> str:
    return text[:_RESULT_CAP] + ("\n...[截断]" if len(text) > _RESULT_CAP else "")


# ============================ 不可逆操作护栏（操作边界设计） ============================
# 可逆操作（读文件/跑验证）直接做；不可逆操作（删文件/改生产配置/格式化等）默认拦截，
# 需 operator 显式授权后才放行，避免“每步都问”或“完全不管”两个极端。
_DESTRUCTIVE_ALLOW = False   # 全局是否放行不可逆操作
_OPERATOR_SECRET: str | None = None  # operator 可选的二次校验口令，None 表示本地信任模式
_PENDING_DESTRUCTIVE: list[dict] = []  # 被拦截待确认的命令

# 命中即视为不可逆的命令关键字（含 sql/命令行删除、格式化、重建库等）
_DESTRUCTIVE_RE = re.compile(
    r"\b(rm\s+-[a-z]*r|del\s+[\\/]|del\s+/s|Remove-Item|rmtree|shutil\.rmtree|"
    r"os\.remove|os\.unlink|DROP\s+TABLE|DROP\s+DATABASE|DELETE\s+FROM|truncate|"
    r"dropdb|dropdb|format\s+[a-z]:|mkfs|git\s+reset\s+--hard|"
    r"rd\s+[\\/]\.?\.?|rm\s+-rf)\b",
    re.IGNORECASE,
)

# 灾难级作用域：覆盖系统根/家目录/存储介质/叉炸弹。纵深防御——即便 operator 已在同一
# 会话缓存/批准过、甚至有意 approve，也会被硬阻断，只有显式 set_destructive_policy(True)
# 全托管才放行。理由：人类批准只解除"该不该做"，不解除"能不能做"（Adversa 疲劳场景下水管）。
_CRITICAL_RE = re.compile(
    r"rm\s+-rf\s+[~/]|rm\s+-rf\s+\s*$|mkfs|format\s+[a-z]\s*:|"
    r":\(\s*:\|:\s*&|&&\s*rm\s+-rf\s+[~/]|rd\s+/\s*[a-z]:",
    re.IGNORECASE,
)

# 安全⇄性能：命令超过该长度视为"超长/超多子命令"，不做逐段深扫，统一降级为单条审批，
# 避免静态检查在响应线程上冻结（对应 Adversa"50+子命令导致界面冻结"的退化决策，改为显式降级）。
_MAX_SCAN_LEN = 512


def is_destructive(command: str) -> bool:
    """判断一条 shell 命令是否为不可逆操作。"""
    return bool(_DESTRUCTIVE_RE.search(command or ""))


def set_destructive_policy(enable: bool, secret: str | None = None) -> str:
    """设置是否放行不可逆操作；本地开发者信任模式不需要口令。"""
    global _DESTRUCTIVE_ALLOW
    if enable and _OPERATOR_SECRET and secret != _OPERATOR_SECRET:
        return "授权失败：口令不匹配，不可逆操作仍保持拦截。"
    _DESTRUCTIVE_ALLOW = bool(enable)
    return ("已放行不可逆操作（谨慎执行）。" if _DESTRUCTIVE_ALLOW
            else "已恢复拦截不可逆操作。")


def destructive_policy() -> str:
    return ("放行中" if _DESTRUCTIVE_ALLOW else "拦截中")


def pending_destructive(clear: bool = False) -> str:
    global _PENDING_DESTRUCTIVE
    if not _PENDING_DESTRUCTIVE:
        return "(暂无待确认的不可逆操作)"
    lines = [f"[{i}] {p['cmd']}" for i, p in enumerate(_PENDING_DESTRUCTIVE)]
    text = "\n".join(lines)
    if clear:
        _PENDING_DESTRUCTIVE = []
    return text


# ============================ 命令规范化 + 同会话审批缓存 ============================
# 同一会话内，operator 批准过的「不可逆命令」按规范化签名缓存放行，避免重复拦截引起节奏拖沓；
# 但缓存只对「同一种写法的命令」生效，写法变了(参数/拼写不同)仍需重新审批，兼顾安全与效率。
_APPROVED_COMMANDS: set[str] = set()  # 本会话内已被批准并缓存的命令规范化签名


def _canonicalize(cmd: str) -> str:
    """命令规范化签名（仿 codex command_canonicalization）。

    去空行、统一 \\ 为 /、归一化引号与周围空白、压缩连续空白、
    关键字整体小写——使“同一命令不同写法”归一到同一签名。
    仅用于比对/缓存，不影响真实执行。
    """
    if not cmd:
        return ""
    text = cmd.replace("\\", "/")
    # 去空行
    text = "\n".join(l for l in text.splitlines() if l.strip())
    # 归一化引号：单/反引号统一为双引号，并去掉引号两旁空白，让 "a b" 与 'a b' 同签名
    text = re.sub(r"[`'\"]", '"', text)
    text = re.sub(r"\s*\"\s*", " ", text).strip()
    # 压缩连续空白为单空格
    text = re.sub(r"\s+", " ", text)
    # 关键字不区分大小写，整体小写，匹配同一命令的不同大小写写法
    return text.lower()


def _approve_cached(cmd: str) -> bool:
    """把命令的规范化签名写入同会话缓存并返回 True（标记为已放行）。"""
    _APPROVED_COMMANDS.add(_canonicalize(cmd))
    return True


def _cache_has(cmd: str) -> bool:
    """判断某命令是否已在同会话缓存中被批准（规范化签名命中即放行）。"""
    return _canonicalize(cmd) in _APPROVED_COMMANDS


def approve_destructive(index: int) -> str:
    """批准并缓存 _PENDING_DESTRUCTIVE[index] 的命令：写入同会话审批缓存，本会话内不再拦截。"""
    if not _PENDING_DESTRUCTIVE:
        return "(当前没有待确认的不可逆操作。)"
    if index < 0 or index >= len(_PENDING_DESTRUCTIVE):
        return (f"错误：索引 {index} 越界，当前待确认的不可逆操作共 "
                f"{len(_PENDING_DESTRUCTIVE)} 条（从 0 开始）。")
    item = _PENDING_DESTRUCTIVE[index]
    _approve_cached(item["cmd"])
    return (f"已批准并缓存（本会话内不再重复拦截）：「{_canonicalize(item['cmd'])}」")


def _guard_run_shell(command: str) -> str | None:
    """不可逆命令的分层护栏：全托放行 → 灾难级硬阻断 → 渐进信任缓存 → 超长降级单条审批。
    返回拦截提示则不许执行；返回 None 可执行。"""
    if not is_destructive(command):
        return None
    canonical = _canonicalize(command)
    # 第0层：显式 full托管（operator set_destructive_policy(True)）→ 放行
    if _DESTRUCTIVE_ALLOW:
        return None
    # 第1层（纵深防御）：灾难级作用域即使"已批准/已缓存"也强制拒死，审批不可绕过。
    # 灾难级命令不进待确认清单，也就无法被 approve_destructive 缓存放行——系统自兜底。
    if _CRITICAL_RE.search(command or ""):
        return ("⛔ 已硬阻断灾难级操作（纵深防御，审批不可绕过）："
                f"「{command}」作用域覆盖系统根/家目录/存储介质，存在覆盖式损毁风险。\n"
                "即使走 approve_destructive 授权也拒绝执行；如确实需要，仅可由 operator "
                "set_destructive_policy(True) 切换为显式托管模式。")
    # 第2层（渐进式信任）：缓存命中 → 直接放行，不去重复询问（治审批疲劳）。
    if canonical in _APPROVED_COMMANDS:
        return None
    # 第3层（安全⇄性能）：超长/超多子命令不做逐段深扫，降级为单条审批（显式退化，非冻结）。
    if len(canonical) > _MAX_SCAN_LEN:
        _PENDING_DESTRUCTIVE.append({"cmd": command, "risk": "overlong"})
        return ("⚠️ 已拦截不可逆操作（安全⇄性能护栏，命令过长降级为单条审批）："
                f"「{(command or '')[:120]}…」省略逐子命令深扫，需 operator 显式授权后才执行。")
    _PENDING_DESTRUCTIVE.append({"cmd": command, "risk": "normal"})
    return ("⚠️ 已拦截不可逆操作（操作边界护栏）："
            f"「{command}」涉及删除/覆盖/重建，需 operator 显式授权后才执行。\n"
            "请改用 `run_shell` 之外的可逆方案（如先备份再改、新建文件而非删除），"
            "或告知 operator 通过 set_destructive_policy 授权。")


def _delete_file(relative_path: str) -> str:
    """删除文件（不可逆操作，受护栏保护）。相对路径优先，越界/绝对路径也允许——
    但一律默认进入「确认模式」：请求被拦截并加入待确认清单，operator 用 approve_destructive
    （或 CLI 的 /approve）或托管模式 set_destructive_policy(true) 放行后才真正删除。
    """
    root = Path(_PROJECT_ROOT).resolve() if _PROJECT_ROOT else Path.cwd().resolve()
    target = (root / relative_path).resolve() if relative_path else root.resolve()
    # 越界不再硬性拒绝，而是作为「需显式确认的高风险删除」处理；敏感密钥仍强制拒删
    out_of_root = not target.is_relative_to(root)
    if is_sensitive_path(str(target)):
        return "错误：涉及敏感文件（密钥/凭据），一律拒绝删除（即使已授权）。"
    if not target.exists():
        return f"未删除：{relative_path} 不存在。"
    if target.is_dir():
        return "错误：delete_file 仅删除单个文件（不可递归删目录，避免 rm -rf 后果），" \
               "删除整个目录请用托管模式 + 精确命令。"

    # 会话审批签名：绝对路径归一化，approve 后本会话不再重复拦截；越界路径带越界标记
    action = "delete_file:" + _canonicalize(str(target))
    scope = ("⚠️ 越界路径" if out_of_root else "⚠️ 项目内文件")
    if not (_DESTRUCTIVE_ALLOW or _cache_has(action)):
        _PENDING_DESTRUCTIVE.append({"cmd": action, "desc": f"删除 {target}（越界={out_of_root}）"})
        return f"{scope}（不可逆）：「{relative_path}」需确认。\n" \
               f"保持确认模式请运行 approve_destructive 批准该索引；\n" \
               f"如完全信任我可直接操作，请 set_destructive_policy(True) 切换为托管模式（全自动删除）。"

    try:
        os.remove(target)
        return f"✅ 已删除文件 {target}（相对项目根: {relative_path}）"
    except OSError as e:
        return f"删除失败：{e}"


def register(name: str, schema: dict, handler: Callable) -> None:
    _REGISTRY[name] = (handler, schema)


def tools_schema() -> list[dict]:
    """返回所有工具 schema 供注入到模型调用。"""
    return [info[1] for info in _REGISTRY.values()]


def core_tools_schema() -> list[dict]:
    """只返回核心工具（数量少、高价值）的完整 schema，供常驻注入模型。"""
    return [_REGISTRY[n][1] for n in CORE_TOOL_NAMES
            if n in _REGISTRY and n not in ("mcp::",)]


def extended_tools_manifest() -> str:
    """其余工具的轻量目录：名称 + 一句话描述。只让模型“知道存在、可调用”，
    不硬塞完整参数 schema；真正调用时 invoke 按名查表即能执行（按需加载）。"""
    lines = []
    for n, (_, schema) in _REGISTRY.items():
        if n in CORE_TOOL_NAMES:
            continue
        fn = schema.get("function", {})
        desc = (fn.get("description") or "").strip()
        # 只取首句，避免把过长参数说明塞进提示词
        first = desc.split("。")[0].split(".")[0] if desc else ""
        lines.append(f"- {n}：{first}" if first else f"- {n}")
    return "另有以下工具可按需调用（需要时直接调用其名称即可，无需先申请）：\n" + "\n".join(lines)


def has(name: str) -> bool:
    return name in _REGISTRY


def list_all_tools() -> list[dict]:
    """返回全部已注册工具的结构化清单：[{name, description, core}...]，供 /toollist 展示。"""
    out: list[dict] = []
    for n, (_, schema) in _REGISTRY.items():
        fn = schema.get("function", {})
        desc = (fn.get("description") or "").strip()
        # 取首句，避免展示过长
        if len(desc) > 120:
            desc = desc[:120]
        out.append({"name": n, "description": desc, "core": n in CORE_TOOL_NAMES})
    out.sort(key=lambda x: (not x["core"], x["name"]))
    return out


def tools_manifest() -> str:
    """系统提示词中的工具指引：简短提示，不整份塞 48 个工具（省 token、避免小模型自乱）。

    核心工具的完整参数 schema 仍随每次 chat 的 tools 字段注入；这里只告诉模型
    "你有工具可用 + 完整清单可用 /toollist 查看"。
    """
    core = "、".join(list_all_core_ids())
    return (f"你有多种工具可调用完成任务（核心：{core} 等）。"
            "核心工具的确切调用格式已随每次请求的 tools 字段给出，直接用其名称调用即可；"
            "完整工具清单可在终端运行 /toollist 查看。")


def list_all_core_ids() -> list[str]:
    """返回核心工具名（保持注册顺序）。"""
    return [n for n in CORE_TOOL_NAMES if n in _REGISTRY]


def _fuzzy_repair_args(name: str, arguments: Any) -> dict | None:
    """工具参数幻觉修复：小模型(qwen 等)常把参数名写错或夹带多余辅助字段，
    导致 handler(**args) 抛 `unexpected keyword argument`。这里对参数做一次保守归一：
      - schema 标准名命中 → 保留；
      - 与唯一标准名相似度足够高 → 改名为标准名；
      - 非标准名又无唯一模糊命中 → 视为小模型多余的辅助字段，丢弃。
    仅在“确实有改动”时才返回修复结果，供调用层重试；否则返回 None 走原报错。
    """
    if not isinstance(arguments, dict):
        return None
    entry = _REGISTRY.get(name)
    if not entry:
        return None
    props = entry[1].get("function", {}).get("parameters", {}).get("properties") or {}
    std = [k for k in props.keys()]
    if not std:
        # schema 无参（如 list_project_files）：模型夹带的任何参数都是多余，全部丢弃
        return {} if arguments else None
    import difflib
    repaired: dict = {}
    changed = False
    for k, v in arguments.items():
        if k in std:                     # 标准名，原样保留
            repaired[k] = v
            continue
        if isinstance(k, str):
            match = difflib.get_close_matches(k, std, n=1, cutoff=0.62)
            if match:                    # 唯一模糊命中 → 归一为标准名
                repaired[match[0]] = v
                changed = True
                continue
        # 非标准名且无唯一命中 → 视为多余辅助字段丢弃
        changed = True
    return repaired if changed else None


def invoke(name: str, arguments: Any) -> str:
    """执行工具并返回字符串结果。arguments 是模型给的字典。

    在此统一挂接生命周期钩子：PreToolUse 可阻断或在返回结果里注入实时修正提示，
    PostToolUse 在工具执行后触发。钩子自身出错一律静默降级，不影响本流程。
    """
    from . import hooks as _hooks
    try:
        _pre = _hooks.pre_tool_use(name, arguments)
    except Exception:  # noqa: BLE001 - 钩子出错静默放行，不炸主流程
        _pre = None
    injected = bool(_pre and _pre.inject)
    if _pre is not None and _pre.blocked:
        return _pre.message or f"钩子已阻断工具 {name}"
    result = _invoke_inner(name, arguments)
    # 注入非阻断的修正提示：追加到工具结果里让模型看到（实时 prompt 修正）
    if injected:
        result = result + "\n\n[钩子提醒] " + "；".join(_pre.inject)
    try:
        _hooks.post_tool_use(name, arguments, result)
    except Exception:  # noqa: BLE001 - 钩子出错不炸主流程
        pass
    return result


def _invoke_inner(name: str, arguments: Any) -> str:
    """invoke 的原执行主体：动态 MCP 路由 + 本地工具分发 + 脱敏/审计/护栏。"""
    # 动态 MCP 工具路由：mcp::<server>::<tool>
    if name.startswith("mcp::"):
        try:
            from core.mcp.registry import get_server, request_use
            from core.mcp.client import call_tool
            _p = name.split("::")
            server_name, tool_name = _p[1], _p[2]
            server = get_server(server_name)
            if server is None:
                return f"错误：未找到 MCP 服务器 {server_name}（可用 /mcp view 查看）"
            # 启用前 intent 审批闸 + 最小权限白名单（deny-first）
            gate = request_use(server_name, tool_name)
            if gate.get("blocked"):
                return gate["message"]
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            if not isinstance(arguments, dict):
                arguments = {"input": arguments}
            return redact(_cap(call_tool(server, tool_name, arguments)))
        except Exception as e:  # noqa: BLE001
            return f"错误：MCP 调用失败 - {e}"
    if name not in _REGISTRY:
        return f"错误：未知工具 {name}"
    handler, _ = _REGISTRY[name]
    # 操作审计：仅当已设置项目写入目录时记录写类操作，满足审计/等保要求
    if _PROJECT_ROOT and name not in ("audit_read", "repo_read", "graph_query",
                                      "graph_query_file", "pending_destructive",
                                      "list_project_files"):
        try:
            from core.project import audit
            audit.record(_PROJECT_ROOT, "tool_" + name, str(arguments)[:200])
        except Exception:  # noqa: BLE001
            pass
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"text": arguments}
    try:
        result = handler(**(arguments or {}))
    except TypeError as e:
        # 小模型参数幻觉：先用模糊归一修一遍再重试；修复后仍失败才报原错
        repaired = _fuzzy_repair_args(name, arguments)
        if repaired is not None:
            try:
                result = handler(**repaired)
            except Exception:  # noqa: BLE001
                return f"错误：工具参数不匹配 - {e}"
            return redact(_cap(str(result)))
        return f"错误：工具参数不匹配 - {e}"
    except Exception as e:  # noqa: BLE001
        return f"错误：工具执行失败 - {e}"
    # 数据外发防线：所有工具结果统一脱敏后再进入上下文/云端
    return redact(_cap(str(result)))


# ============================ 内置工具 ============================

def _read_file(path: str, start: int | None = None, end: int | None = None) -> str:
    # 敏感文件直接拒绝读取，防止密钥/凭据进入云端 prompt
    if is_sensitive_path(path):
        return f"拒绝读取敏感文件：{path}（为防止数据外发，该路径已被屏蔽）"
    p = Path(path)
    if not p.is_file():
        return f"文件不存在: {path}"
    raw = p.read_bytes()
    text = _decode_bytes(raw, p.name)
    if text is None:
        # 二进制文件：给出摘要 + Hex 预览，避免把乱码灌进上下文
        head = raw[:64].hex(" ")
        return (f"二进制文件（{len(raw)} 字节），已跳过逐字读取，避免上下文污染。\n"
                f"预览: {head}")
    if start is not None or end is not None:
        lines = text.splitlines()
        text = "\n".join(lines[(start or 0):end])
    return text


_BOM_MAP = (
    (b"\xef\xbb\xbf", "utf-8-sig"),   # UTF-8 BOM
    (b"\xff\xfe\x00\x00", "utf-32"),  # UTF-32 LE（utf-32 自动识别字节序并去 BOM）
    (b"\x00\x00\xfe\xff", "utf-32"),
    (b"\xff\xfe", "utf-16"),          # UTF-16 LE（utf-16 自动识别字节序并去 BOM）
    (b"\xfe\xff", "utf-16"),
)


def _decode_bytes(raw: bytes, filename: str = "") -> str | None:
    """按 BOM/编码回退链解码文本文件；二进制返回 None。支持各类小众文本文件。"""
    if not raw:
        return ""
    # 二进制判定：含 NUL 控制字节且不是 UTF-16/32（有 BOM 时先解）
    for _bom, enc in _BOM_MAP:
        if raw.startswith(_bom):
            try:
                return raw.decode(enc)
            except (UnicodeDecodeError, LookupError):
                break
    if b"\x00" in raw:
        return None
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk", "cp1252",
                "latin-1", "utf-16", "utf-32"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def _write_file(path: str, content: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"已写入 {len(content)} 字符 -> {path}"


def _list_dir(path: str = ".") -> str:
    p = Path(path)
    if not p.is_dir():
        return f"目录不存在: {path}"
    entries = []
    for child in sorted(p.iterdir()):
        tag = "D" if child.is_dir() else "F"
        entries.append(f"[{tag}] {child.name}")
    return "\n".join(entries) or "(空目录)"


_EXTERNALIZE_THRESHOLD = 1500  # 超过该长度的工具输出将外置到磁盘，只回指纹，省 token


def _externalize(text: str, label: str = "output") -> str:
    """把超长输出写入项目根下 .trae_outputs/，返回短指纹引用（可寻址回读）。

    创新点（File-Backed Memory）：prompt 只载入文件的路径+大小+首行，不载全文。
    模型需要全文时用 read_file 按路径回读。既省 token 又不丢信息。
    """
    if len(text) <= _EXTERNALIZE_THRESHOLD:
        return text
    try:
        import time as _t
        root = _project_root_for_tools()
        outdir = os.path.join(root, ".trae_outputs")
        os.makedirs(outdir, exist_ok=True)
        name = f"{label}_{int(_t.time())}.txt"
        path = os.path.join(outdir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        rel = os.path.relpath(path, root)
        first = text.splitlines()[0][:80] if text.splitlines() else ""
        return (f"[long-{label}] 完整内容已外置到 `{rel}`（{len(text)} 字符），"
                f"需要全文时用 read_file 读取该文件。\n首行摘要: {first}")
    except Exception:  # noqa: BLE001  外置失败则退回原文，不影响功能
        return text


def _run_shell(command: str, timeout: int = 15) -> str:
    """在隔离沙箱中执行命令：受限时、尽量断网，防 AI 命令失控/泄露。"""
    # 操作边界护栏：不可逆命令默认拦截
    blocked = _guard_run_shell(command)
    if blocked is not None:
        return blocked
    try:
        from core.sandbox.security import SandboxPolicy
        from core.sandbox.executor import Executor
        # 让命令默认在用户指定的项目目录里执行，并把工作目录/读写锁定在项目根内
        root = get_project_root()
        allowed = [root] if root and root != "." else None
        ex = Executor(SandboxPolicy(timeout_seconds=timeout, allow_network=False,
                                    allowed_dirs=allowed))
        r = ex.run_command(command, cwd=root if root and root != "." else None)
        out = (r.stdout or "") + (("\n[stderr] " + r.stderr) if r.stderr else "")
        return (f"(exit={r.exit_code})\n" +
                _externalize(out, "shell_output"))
    except Exception as e:  # noqa: BLE001
        return f"命令执行失败: {e}"


def _getenv(key: str) -> str:
    return os.environ.get(key, "")


def _hello(name: str = "friend") -> str:
    return f"你好，{name}！这是 HS 的内置工具示例。"


# ============================ 生成代码自检 / 执行验证 ============================

def _extract_python(text: str) -> str:
    """从模型回复中抽取第一段 python 代码块；无围栏则返回原文。"""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    return text.strip()


def _undefined_names(code: str) -> list[str]:
    """粗略的“引用了但未定义”检测（NameError 前置预警）。"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    defined: set[str] = set()
    used: set[str] = set()

    def _collect_targets(node_) -> None:
        t = getattr(node_, "targets", None)
        ns = []
        if t is not None:
            ns.extend(t)
        elif hasattr(node_, "target"):
            ns.append(node_.target)
        for x in ns:
            if isinstance(x, ast.Name):
                defined.add(x.id)
            elif isinstance(x, (ast.Tuple, ast.List)):
                for e in x.elts:
                    if isinstance(e, ast.Name):
                        defined.add(e.id)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
            defined.update({a.arg for a in node.args.args})
            defined.update({a.arg for a in node.args.kwonlyargs})
            if node.args.vararg:
                defined.add(node.args.vararg.arg)
            if node.args.kwarg:
                defined.add(node.args.kwarg.arg)
        elif isinstance(node, ast.ClassDef):
            defined.add(node.name)
        elif isinstance(node, (ast.ImportFrom, ast.Import)):
            defined.update(a.asname or a.name for a in node.names)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            _collect_targets(node)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)

    # 自检内建名白名单：dir("__builtins__") 在 exec 沙箱里拿到的是字符串的 dir，极不完整，
    # 会导致 print/len 等常用内建被误报为"未定义"——那正是给模型的错误事实信号。
    _BUILTINS = {
        "__builtins__", "__name__", "__file__", "__package__", "__doc__", "self",
        "print", "len", "range", "str", "int", "float", "bool", "list", "tuple",
        "dict", "set", "frozenset", "bytes", "bytearray", "type", "isinstance",
        "issubclass", "getattr", "setattr", "hasattr", "delattr", "dir", "vars",
        "repr", "abs", "all", "any", "sum", "min", "max", "sorted", "reversed",
        "enumerate", "zip", "map", "filter", "next", "iter", "open", "input",
        "id", "hash", "object", "None", "True", "False", "slice", "format",
        "bin", "oct", "hex", "ord", "chr", "divmod", "round", "pow", "complex",
        "staticmethod", "classmethod", "property", "super", "Exception", "ValueError",
        "TypeError", "KeyError", "IndexError", "AttributeError", "StopIteration",
        "ArithmeticError", "ZeroDivisionError", "NotImplemented", "callable", "compile",
        "eval", "exec", "globals", "locals",
    }
    try:  # 动态收集运行时真实内建（read-only，覆盖第三方注入的内建）
        import builtins as _blt
        _start_blt = set(dir(_blt))
    except Exception:  # noqa: BLE001
        _start_blt = set()
    builtins = _start_blt | _BUILTINS
    undefined = sorted(n for n in used - defined - builtins if n not in _UPERR)
    return undefined


_UPERR = ("pip",)  # 容错白名单


def _assert_stats(code: str) -> tuple[int, int]:
    """统计代码里的断言与测试函数，用于识别「弱验证」（跑通但没断言）。

    - 返回 (assert 语句数, test_ 开头的测试函数数)；
    - 一段代码若能确认「执行到末尾且无 AssertionError」，断言数越多，验证置信度越高。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0, 0
    n_assert, n_test = 0, 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            n_assert += 1
        elif isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            n_test += 1
    return n_assert + n_test, n_test  # assert 语句 + test_ 函数计为双重断言信号


def _verify_code(code: str, timeout: int = 15) -> str:
    """在隔离子进程执行被测代码，返回执行输出 + 未定义引用预警 + 断言强度标注。

    - 断言式验证：代码里含 assert / test_* 时，跑到末尾无 AssertionError 才算"通过"；
    - 无断言时明确标出「弱验证」，引导模型补充断言而非满足于"能跑通"。
    """
    warnings = _undefined_names(code)
    warn_txt = ""
    if warnings:
        warn_txt = f"\n[自检] 疑似引用了未定义名称: {', '.join(warnings)}"
    n_assert, n_test = _assert_stats(code)
    # 长驻/服务器类代码（app.run / uvicorn / serve_forever）会永不自退，
    # 若直接等 timeout 必然超时，让模型误以为"失败"并陷入重试循环。
    # 这类代码改为"启动检测"：确认进程能起来并保持运行即视为通过。
    if "app.run(" in code or "uvicorn.run(" in code or "serve_forever(" in code:
        return _verify_server(code, timeout) + warn_txt
    # 语法前置门：先用 ast 做静态语法校验，语法错误直接判失败返回，
    # 避免“带语法错误的代码”白跑一次沙箱后被运行时报错误导成业务错误。
    try:
        ast.parse(code)
    except SyntaxError as e:
        msg = e.msg or "语法错误"
        line = getattr(e, "lineno", None)
        col = getattr(e, "offset", None)
        loc = f"line {line}" + (f" col {col}" if col else "")
        return f"❌ 语法错误（前置校验）：{msg} @ {loc}{warn_txt}"
    # 用沙箱执行器在隔离 python 进程跑，避免污染主进程
    from core.sandbox.security import SandboxPolicy
    from core.sandbox.executor import Executor
    ex = Executor(SandboxPolicy(timeout_seconds=timeout))
    fd, tmp = tempfile.mkstemp(suffix=".py")
    os.close(fd)
    try:
        Path(tmp).write_text(code, encoding="utf-8")
        r = ex.run_command(
            f'"{__import__("sys").executable}" -c "exec(open(r\'{tmp}\').read())"')
        body = r.stdout + (("\n[stderr] " + r.stderr) if r.stderr else "")
        out_text = _externalize(body, "verify_output")
        low = out_text.lower()
        if r.ok:
            if n_assert:
                # 能跑到这里说明所有执行到的断言都没有触发 AssertionError
                head = (f"✅ 运行正常（含 {n_assert} 个断言信号"
                        + (f"，{n_test} 个 test_ 函数" if n_test else "")
                        + "，执行全程未触发任何 AssertionError）")
            else:
                # 弱验证：跑通了但没有断言，只能证明"能运行"，不能证明"结果正确"
                head = ("✅ 运行正常（⚠️弱验证__NOASSERT__：未检测到断言，仅确认能跑通；"
                        "对关键逻辑请补充断言后再次 verify_code 以获得可靠通过）")
        else:
            head = f"❌ 断言失败（AssertionError）" if "assertionerror" in low \
                else f"❌ 运行失败 (exit={r.exit_code})"
        return f"{head}{warn_txt}\n{out_text}" if out_text else f"{head}{warn_txt}"
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _verify_server(code: str, timeout: int = 15) -> str:
    """对长驻服务器类代码做启动检测：后台拉起进程，能保持运行 1.5s 即判定可正常启动。

    期间不阻塞，防止 app.run 永驻导致测试函数卡死；确认能启动即 kill 子进程返回。
    """
    try:
        compile(code, "<code>", "exec")  # 先保证语法合法
    except SyntaxError as e:
        return f"❌ 语法错误: {e}"
    exe = __import__("sys").executable
    fd, tmp = tempfile.mkstemp(suffix=".py")
    os.close(fd)
    try:
        Path(tmp).write_text(code, encoding="utf-8")
        proc = subprocess.Popen(
            [exe, "-c", f"exec(open(r'{tmp}').read())"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=os.getcwd(),
        )
        try:
            try:
                proc.wait(timeout=1.5)
                # 提前退出说明初始化即失败
                out = proc.stdout.read().decode("utf-8", "replace") if proc.stdout else ""
                err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
                return f"❌ 启动失败 (exit={proc.returncode})\n{out}\n{err}".strip()
            except subprocess.TimeoutExpired:
                # 1.5s 后仍存活 → 服务器成功启动并驻留
                return "✅ 服务启动正常（长驻进程，已确认可启动并保持运行，已自动停止验证的副本）。" \
                       "如需真机运行，请在项目内执行启动命令而非 verify_code。"
        finally:
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


register(
    "verify_code",
    {
        "type": "function",
        "function": {
            "name": "verify_code",
            "description": "在隔离子进程执行一段 Python 代码并返回运行结果，附带未定义引用自检"
                           "与断言强度标注。用于验证你生成/修改的代码能否跑通；建议代码内带 "
                           "assert 断言或 test_* 测试函数做强验证（无断言会标为弱验证）。要求参数 code 自带断言式自测。",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "要验证的 Python 代码"},
                    "timeout": {"type": "integer", "default": 15},
                },
                "required": ["code"],
            },
        },
    },
    _verify_code,
)

# ============================ 注册 ============================

register(
    "read_file",
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取本地文件内容，可指定行范围。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件绝对路径"},
                    "start": {"type": "integer", "description": "起始行(从0)"},
                    "end": {"type": "integer", "description": "结束行(不含)"},
                },
                "required": ["path"],
            },
        },
    },
    _read_file,
)

register(
    "write_file",
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "写入/覆盖本地文件内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    _write_file,
)

register(
    "list_dir",
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "列出目录下的文件与子目录。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                },
                "required": [],
            },
        },
    },
    _list_dir,
)

register(
    "run_shell",
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "在本地执行一条 shell 命令并返回输出。",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "default": 60},
                },
                "required": ["command"],
            },
        },
    },
    _run_shell,
)

register(
    "getenv",
    {
        "type": "function",
        "function": {
            "name": "getenv",
            "description": "读取环境变量。",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    },
    _getenv,
)

register(
    "hello",
    {
        "type": "function",
        "function": {
            "name": "hello",
            "description": "内置打招呼示例工具。",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": [],
            },
        },
    },
    _hello,
)

register(
    "project_write",
    {
        "type": "function",
        "function": {
            "name": "project_write",
            "description": "把生成的项目文件写入用户指定的本地项目目录。"
                           "relative_path 为相对项目根的路径（如 main.py、src/utils.py）。"
                           "生成/改项目文件时优先用此工具，而不要用 write_file。",
            "parameters": {
                "type": "object",
                "properties": {
                    "relative_path": {"type": "string",
                                      "description": "相对项目根的路径"},
                    "content": {"type": "string", "description": "文件内容"},
                },
                "required": ["relative_path", "content"],
            },
        },
    },
    _project_write,
)

register(
    "list_project_files",
    {
        "type": "function",
        "function": {
            "name": "list_project_files",
            "description": "列出用户项目目录下已生成的所有文件（相对路径）。",
            "parameters": {"type": "object", "properties": {}},
            "required": [],
        },
    },
    _list_project_files,
)


# ====================== 任意类型文件上传（拖拽/路径带入） ======================
_TEXT_EXT = {".md", ".txt", ".py", ".js", ".ts", ".json", ".yaml", ".yml",
             ".csv", ".html", ".css", ".xml", ".log", ".sh", ".toml", ".ini"}


def _file_upload(local_path: str) -> str:
    """把用户拖拽/指定的本地任意类型文件拷入项目根下的 uploads/ 并登记。
    文本类返回路径+可寻址内容摘要；二进制返回路径+类型+大小，供脚本解析。
    """
    import hashlib
    lp = str(local_path).strip().strip('"').strip("'")
    if not os.path.isfile(lp):
        return f"✗ 未找到文件: {local_path}"
    root = os.path.abspath(_project_root_for_tools())
    dest_dir = os.path.join(root, "uploads")
    os.makedirs(dest_dir, exist_ok=True)
    fname = os.path.basename(lp)
    dest = os.path.join(dest_dir, fname)
    i = 1
    while os.path.exists(dest):
        stem, ext = os.path.splitext(fname)
        dest = os.path.join(dest_dir, f"{stem}_{i}{ext}")
        i += 1
    with open(lp, "rb") as f:
        data = f.read()
    with open(dest, "wb") as f:
        f.write(data)
    rel = os.path.relpath(dest, root)
    size = len(data)
    digest = hashlib.md5(data).hexdigest()[:10]
    ext = os.path.splitext(fname)[1].lower()
    try:
        preview = data.decode("utf-8", "replace")
        is_text = (ext in _TEXT_EXT) or ("\ufffd" not in preview[:4000])
    except Exception:  # noqa: BLE001
        is_text = False
    if is_text and size < 200_000:
        head = preview.splitlines()[:6]
        preview_txt = "\n".join(head)
        return (f"✅ 已上传 `{rel}`（{size} 字节, md5={digest}）\n"
                f"文本文件，可 read_file 读取。预览前几行:\n{preview_txt[:400]}")
    return (f"✅ 已上传 `{rel}`（{size} 字节, md5={digest}, 类型={ext or '未知'}）。"
            f"为二进制/大文件，未注入 prompt；需要时用 run_shell 写脚本解析它。")


register(
    "file_upload",
    {
        "type": "function",
        "function": {
            "name": "file_upload",
            "description": "把用户本地任意类型文件（文本/图片/pdf/表格/压缩包等）拷入项目 uploads/ 并登记，供后续读取或脚本解析。",
            "parameters": {
                "type": "object",
                "properties": {
                    "local_path": {"type": "string",
                                   "description": "本地文件的绝对路径"},
                },
                "required": ["local_path"],
            },
        },
    },
    _file_upload,
)

# ============================ 结构化精确编辑（apply_patch，借鉴 codex） ============================
# codex 用 apply_patch 解析 unified diff 精确改文件，而不是整文件覆盖重写：
# 更省 token、不误伤无关内容、可结构化校验后才落盘。这里用一个轻量的纯 Python 实现。

_DIFF_HEAD = re.compile(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@.*")


def _fits(lines: list, i: int, old_block: list, fuzz: int) -> bool:
    """判断 old_block 从 lines[i] 起是否匹配（允许前 fuzz 行模糊不匹配，定位更稳）。"""
    for k in range(fuzz, len(old_block)):
        if i + k >= len(lines) or lines[i + k] != old_block[k]:
            return False
    return True


def _apply_unified_diff(text: str, patch: str):
    """把 unified diff 应用到 text，返回 (新文本, 说明)。无法定位/格式错返回 (None, 原因)。"""
    src = text.splitlines()
    hunks = []
    cur = None
    for ln in patch.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        if not ln:
            continue
        if ln.startswith(("---", "+++")):
            continue
        if ln.startswith("@@"):
            m = _DIFF_HEAD.match(ln)
            if not m:
                return None, f"patch 头无法解析: {ln}"
            cur = {"old_start": int(m.group(1)) - 1, "body": []}
            hunks.append(cur)
        elif cur is not None:
            cur["body"].append(ln)
    if not hunks:
        return None, "patch 为空或没有 @@ hunk"

    lines = src
    offset = 0
    for h in hunks:
        old_block = [ln[1:] for ln in h["body"] if ln[:1] in (" ", "-")]
        new_block = [ln[1:] for ln in h["body"] if ln[:1] in (" ", "+")]
        hint = h["old_start"] + offset
        pos = None
        # 优先在提示行附近精确匹配；找不到再小范围滑动
        for cand in (hint, hint + 1, hint - 1, hint + 2):
            if 0 <= cand <= len(lines) - len(old_block) and _fits(lines, cand, old_block, 0):
                pos = cand
                break
        if pos is None:
            # 带 fuzz 模糊定位（前 2 行上下文可略过），增强对模型生成 diff 的鲁棒性
            for cand in range(max(0, hint - 40), min(len(lines) - len(old_block) + 1, hint + 200)):
                if _fits(lines, cand, old_block, 2):
                    pos = cand
                    break
        if pos is None:
            return None, (f"无法在文件中定位 hunk（起始行约 {h['old_start'] + 1}）: "
                          f"{old_block[:3]}")
        lines = lines[:pos] + new_block + lines[pos + len(old_block):]
        offset += len(new_block) - len(old_block)
    out = "\n".join(lines)
    # 原文件以换行结尾时补回末尾换行，避免 diff 应用后丢失末尾换行
    if text.endswith("\n") and out and not out.endswith("\n"):
        out += "\n"
    return out, (
        f"已应用 {len(hunks)} 个 hunk，文件被精确修改（未改动无关内容）。")


def _patch_file(relative_path: str, diff: str) -> str:
    """用 unified diff 对项目内文件做精确编辑（可逆、省 token、不误伤无关内容）。"""
    root = Path(_PROJECT_ROOT) if _PROJECT_ROOT else Path.cwd()
    target = (Path(root) / relative_path).resolve()
    if not target.is_relative_to(root.resolve()):
        return "错误：路径越界，仅允许项目目录内（相对路径）"
    if is_sensitive_path(str(target)):
        return f"拒绝读取敏感文件：{relative_path}"
    if not target.is_file():
        return f"文件不存在: {relative_path}"
    text = _decode_bytes(target.read_bytes()) or ""
    new_text, note = _apply_unified_diff(text, diff)
    if new_text is None:
        return f"❌ patch 应用失败：{note}"
    if new_text == text:
        return "patch 应用后无变化（内容相同）"
    try:
        from core.project.snapshot import backup
        backup(Path(root), relative_path, new_text)
    except Exception:  # noqa: BLE001
        pass
    target.write_text(new_text, encoding="utf-8")
    # 补丁文件落盘成功后顺手自动提交（git 仓库时）
    _autocommit(Path(root), relative_path)
    return f"{note}\n已写入 {target}（相对项目根: {relative_path}）"


register(
    "patch_file",
    {
        "type": "function",
        "function": {
            "name": "patch_file",
            "description": "用 unified diff 精确编辑项目内已有文件；比整文件覆盖更省 token、"
                           "不误伤无关内容。diff 为标准 unified/diff 格式，含 @@ -行,数 +行,数 @@ 头，"
                           "上下文行以空格开头、删除行以-开头、新增行以+开头。能自动模糊定位。",
            "parameters": {
                "type": "object",
                "properties": {
                    "relative_path": {"type": "string", "description": "相对项目根的路径"},
                    "diff": {"type": "string",
                             "description": "unified diff 文本（-- 头可选）"},
                },
                "required": ["relative_path", "diff"],
            },
        },
    },
    _patch_file,
)

# ============================ Computer Use 工具 ============================

def _computer_screenshot(component: str = "text", path: str | None = None) -> str:
    """返回当前前台窗口的文本快照（无视觉模型'看屏'的桥），或把全屏存为图片。"""
    from core.computer import screen
    if component == "image":
        return redact(_cap(screen.save_screenshot(path)))
    return redact(_cap(screen.screenshot_text()))


def _computer_click(x: int | None = None, y: int | None = None,
                    name: str | None = None, button: str = "left") -> str:
    """点击屏幕：给 x,y 坐标，或给元素 name 自动定位后点击。"""
    from core.computer import input as ci
    return ci.click(x, y, name=name, button=button)


def _computer_type(text: str) -> str:
    from core.computer import input as ci
    return ci.type_text(text)


def _computer_key(combo: str) -> str:
    from core.computer import input as ci
    return ci.key(combo)


def _computer_open(target: str) -> str:
    """打开应用/文件/网址（用系统默认程序）。"""
    from core.computer import windows
    return windows.open_app(target)


def _computer_windows() -> str:
    from core.computer import windows
    return redact(_cap(windows.list_windows()))


def _computer_focus(name: str) -> str:
    from core.computer import windows
    return windows.focus_window(name)


register(
    "computer_screenshot",
    {
        "type": "function",
        "function": {
            "name": "computer_screenshot",
            "description": "查看/感知当前屏幕：默认返回前台窗口的文本快照（窗口标题、"
                           "可交互控件的名称/类型/中心坐标），供你据此规划下一步点击；"
                           "component='image' 时把全屏截图存为图片返回路径。",
            "parameters": {
                "type": "object",
                "properties": {
                    "component": {"type": "string", "default": "text",
                                  "description": "text(文本快照) 或 image(保存截图)"},
                    "path": {"type": "string", "description": "截图保存路径(可选)"},
                },
                "required": [],
            },
        },
    },
    _computer_screenshot,
)

register(
    "computer_click",
    {
        "type": "function",
        "function": {
            "name": "computer_click",
            "description": "在屏幕上点击：通过 x,y 坐标点击指定位置，或通过 name 控件名"
                           "自动定位最近匹配的可点击元素后点击。",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "目标 x 坐标"},
                    "y": {"type": "integer", "description": "目标 y 坐标"},
                    "name": {"type": "string", "description": "控件名(模糊包含)，用于定位"},
                    "button": {"type": "string", "default": "left",
                               "description": "left/right/middle"},
                },
                "required": [],
            },
        },
    },
    _computer_click,
)

register(
    "computer_type",
    {
        "type": "function",
        "function": {
            "name": "computer_type",
            "description": "向当前焦点输入框键入一段文本（模拟键盘，逐个字符）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "要键入的文本"},
                },
                "required": ["text"],
            },
        },
    },
    _computer_type,
)

register(
    "computer_key",
    {
        "type": "function",
        "function": {
            "name": "computer_key",
            "description": "发送键盘快捷键/按键，如 'ctrl+s'、'enter'、'alt+tab'、"
                           "'ctrl+shift+esc'。用于确认、切换任务、快捷键操作。",
            "parameters": {
                "type": "object",
                "properties": {
                    "combo": {"type": "string", "description": "按键或组合键"},
                },
                "required": ["combo"],
            },
        },
    },
    _computer_key,
)

register(
    "computer_open",
    {
        "type": "function",
        "function": {
            "name": "computer_open",
            "description": "用系统默认程序打开应用/文件/网址，如 'notepad'、'calc'、"
                           "'https://example.com'、某个 .txt 文件路径。",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "应用名/文件路径/网址"},
                },
                "required": ["target"],
            },
        },
    },
    _computer_open,
)

register(
    "computer_windows",
    {
        "type": "function",
        "function": {
            "name": "computer_windows",
            "description": "列出当前登录会话所有可见窗口的标题与矩形，用于定位要操作的目标窗口。",
            "parameters": {"type": "object", "properties": {}},
            "required": [],
        },
    },
    _computer_windows,
)

register(
    "computer_focus",
    {
        "type": "function",
        "function": {
            "name": "computer_focus",
            "description": "按标题模糊匹配把窗口切换到前台（如 '记事本'、'Chrome'）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "窗口标题关键字"},
                },
                "required": ["name"],
            },
        },
    },
    _computer_focus,
)

# ============================ MCP 生态工具 ============================

def _mcp_call(server: str, tool: str, arguments: dict | None = None) -> str:
    """调用已配置 MCP 服务器上的一个工具（stdio，隔离子进程，返回文本）。"""
    from core.mcp.registry import get_server, request_use
    from core.mcp.client import call_tool
    srv = get_server(server)
    if srv is None:
        return f"错误：未找到 MCP 服务器 {server}（可用 /mcp view 查看已配置项）"
    # 启用前 intent 审批闸 + 最小权限白名单（deny-first）
    gate = request_use(server, tool)
    if gate.get("blocked"):
        return gate["message"]
    return call_tool(srv, tool, arguments or {})


register(
    "mcp_call",
    {
        "type": "function",
        "function": {
            "name": "mcp_call",
            "description": "调用已配置 MCP（Model Context Protocol）服务器上的外部工具，"
                           "扩展生态能力（如 filesystem/sqlite/webkit/用户自定义服务）。"
                           "server 为服务器名，tool 为该服务器暴露的工具名，arguments 传参数对象。",
            "parameters": {
                "type": "object",
                "properties": {
                    "server": {"type": "string", "description": "MCP 服务器名"},
                    "tool": {"type": "string", "description": "该服务器上的工具名"},
                    "arguments": {"type": "object", "description": "工具参数（json 对象）"},
                },
                "required": ["server", "tool"],
            },
        },
    },
    _mcp_call,
)


def _install_plugin(url: str) -> str:
    """安装 GitHub 开源 MCP 插件：clone 并按 MCP 服务器登记（deny-first，默认未启用）。"""
    from core.mcp import installer
    return installer.install_from_github(url)


register(
    "install_plugin",
    {
        "type": "function",
        "function": {
            "name": "install_plugin",
            "description": "安装用户指定的 GitHub 开源 MCP 插件。当 operator 要求“安装 github 中的某个开源插件/仓库”时调用，"
                           "url 可为 owner/repo、https://github.com/owner/repo 或 git@…。"
                           "会 git clone 到本机并按 MCP 服务器登记；为遵循 deny-first 安全时序，"
                           "新插件默认【未启用】，需 operator 用 /mcp approve <name> 批准后才可调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "GitHub 仓库地址，如 owner/repo"},
                },
                "required": ["url"],
            },
        },
    },
    _install_plugin,
)


def _strip_html(raw: str) -> str:
    """把 HTML 粗略转成可读纯文本：去 script/style、标签、合并空白。"""
    s = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = re.sub(r"[ \t\xa0]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n", s)
    return s.strip()


def _webfetch(url: str = "", max_chars: int = 3000, proxy: str = "") -> str:
    """抓取一个 http/https 网页/文本 URL 并返回可读文本；限长返回，防止撑爆上下文。

    proxy 可选：显式传 http://host:port 或 http://user:pass@host:port；为空则回退到
    环境变量 WEBFETCH_PROXY。仅本工具进程内生效，不影响其它沙箱（它们仍断网脱代理）。
    """
    url = (url or "").strip()
    if not url:
        return "错误：缺少 url 参数。"
    scheme = urlparse(url).scheme.lower()
    if scheme not in ("http", "https"):
        return f"错误：仅允许抓取 http/https 地址，收到 {scheme!r}。"
    cap = max(500, min(int(max_chars or 3000), _RESULT_CAP))
    # 代理解析：显式参数优先，其次 WEBFETCH_PROXY，再回退到 HTTP(S)_PROXY。
    proxy = (proxy or "").strip() or os.environ.get("WEBFETCH_PROXY", "")
    if not proxy:
        proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY") \
            or os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY", "")
    opener = None
    if proxy:
        h = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        opener = urllib.request.build_opener(h)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (hs-agent) python-urllib",
                 "Accept": "text/html,text/plain,*/*"})
    try:
        fetcher = opener.open if opener else urllib.request.urlopen
        with fetcher(req, timeout=15) as resp:  # noqa: S310 - 已限定 http/https
            charset = resp.headers.get_content_charset() or "utf-8"
            body = resp.read(200 * 1024).decode(charset, errors="replace")
    except urllib.error.HTTPError as e:
        return f"错误：HTTP {e.code} {e.reason}"
    except urllib.error.URLError as e:
        return f"错误：无法访问（{e.reason}）。如为本地/内网地址可能不可达。"
    except OSError as e:
        return f"错误：网络异常 - {e}"
    text = _strip_html(body)
    if not text:
        text = body[:cap]
    if len(text) > cap:
        text = text[:cap] + f"\n…(已截断，共 {len(text)} 字，仅显示前 {cap} 字)"
    return text or "（页面无可见文本内容）"


register(
    "webfetch",
    {
        "type": "function",
        "function": {
            "name": "webfetch",
            "description": "抓取一个公网 http/https 网页（或 .txt/.md/json 等文本链接）并把正文转成可读文本返回。"
                           "用于查询网页内容、API 返回、文档。仅支持 http/https，结果会限制长度（默认 3000 字符）避免撑爆上下文。"
                           "网络不可达或返回非 2xx 时返回明确错误，不会崩溃。"
                           "当默认直连不可达时，可传 proxy（如 http://127.0.0.1:7890）经代理出网；也可设 WEBFETCH_PROXY 环境变量。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "要抓取的 http/https 完整 URL"},
                    "max_chars": {"type": "integer", "description": "返回正文的最大字符数（默认 3000，上限 4000）"},
                    "proxy": {"type": "string", "description": "可选代理 http://host:port（优先于环境变量 WEBFETCH_PROXY）"},
                },
                "required": ["url"],
            },
        },
    },
    _webfetch,
)


# ============================ 人类决策权威：向 operator 提问 ============================

_ask_handler: Callable[[str], str] | None = None


def set_ask_handler(fn: Callable[[str], str] | None) -> None:
    """注册“向 operator 提问”的处理器（由 CLI/接入层注入，负责终端交互读取回答）。"""
    global _ask_handler
    _ask_handler = fn


def _ask_user(question: str = "") -> str:
    """把一个问题抛给 operator，并返回其回答。无处理器/非交互时返回友好提示，不崩溃。"""
    q = (question or "").strip()
    if not q:
        return "错误：缺少 question 参数。"
    if _ask_handler is None:
        return f"（当前无交互提问处理器）需向 operator 确认：{q}"
    try:
        ans = _ask_handler(q)
        return (ans or "").strip()[:1000] or "（operator 未作答）"
    except Exception as e:  # noqa: BLE001
        return f"错误：提问/等待回答失败 - {e}"


register(
    "ask_user",
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "向 operator（操作者/使用本工具的人）提一个必须由人决定的问题并等待回答。"
                           "当任务出现歧义、需要人类拍板的方向/参数/许可、或涉及关键取舍时调用，体现“人类决策权威”。"
                           "question 为向 operator 展示的问题文本；返回 operator 的回答字符串。"
                           "不要在非必要时调用（会打断任务节奏），能用公开信息或合理默认解决的不要问。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "要向 operator 提出并等待回答的问题"},
                },
                "required": ["question"],
            },
        },
    },
    _ask_user,
)

# ============================ 操作边界护栏管理工具 ============================

register(
    "set_destructive_policy",
    {
        "type": "function",
        "function": {
            "name": "set_destructive_policy",
            "description": "放行或拦截不可逆操作（删文件/删库/格式化等）。默认拦截。"
                           "仅当 operator 已明确同意时才调用；本地信任模式无需口令。",
            "parameters": {
                "type": "object",
                "properties": {
                    "enable": {"type": "boolean", "description": "是否放行不可逆操作"},
                    "secret": {"type": "string", "description": "operator 口令（一般不需要）"},
                },
                "required": ["enable"],
            },
        },
    },
    set_destructive_policy,
)

register(
    "pending_destructive",
    {
        "type": "function",
        "function": {
            "name": "pending_destructive",
            "description": "查看被护栏拦截、待 operator 确认的不可逆操作清单。",
            "parameters": {
                "type": "object",
                "properties": {"clear": {"type": "boolean", "description": "查看后是否清空"}},
                "required": [],
            },
        },
    },
    pending_destructive,
)

register(
    "approve_destructive",
    {
        "type": "function",
        "function": {
            "name": "approve_destructive",
            "description": "按索引批准某条被护栏拦截的不可逆操作，并把其命令写入同会话审批缓存，"
                           "本会话内再次出现同一条命令时不再拦截。索引见 pending_destructive 的 [i]。",
            "parameters": {
                "type": "object",
                "properties": {"index": {"type": "integer", "description": "待确认清单中的索引(从0开始)"}},
                "required": ["index"],
            },
        },
    },
    approve_destructive,
)

register(
    "delete_file",
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "删除项目目录内的一个文件（不可逆，受护栏保护）。默认进入确认模式：请求被拦截并加入待确认清单，"
                           "需 operator 用 approve_destructive 批准；若已 set_destructive_policy(True) 托管模式则直接删除。"
                           "仅删单个文件，不做递归目录删除。",
            "parameters": {
                "type": "object",
                "properties": {
                    "relative_path": {"type": "string", "description": "要删除的文件在项目根下的相对路径（如 docs/old.md）"},
                },
                "required": ["relative_path"],
            },
        },
    },
    _delete_file,
)

# ============================ 文件提交历史（read_history） ============================

def _read_history(relative_path: str) -> str:
    """在项目根下用 git log --oneline -n 10 -- <relative_path> 返回该文件的最近提交记录。"""
    root = str(Path(get_project_root()).resolve())
    env = dict(os.environ)
    env.update(_GIT_ENV)
    try:
        r = subprocess.run(
            ["git", "log", "--oneline", "-n", "10", "--", relative_path],
            cwd=root, env=env, check=False, capture_output=True, text=True,
            timeout=15)
    except Exception:  # noqa: BLE001
        return f"（{relative_path} 无法读取提交历史：可能不是 git 仓库。）"
    if r.returncode != 0:
        return f"（{relative_path} 无可用 git 历史：非 git 仓库或文件不在版本控制中）"
    out = (r.stdout or "").strip()
    if not out:
        return f"（{relative_path} 尚无任何提交记录，可先 project_write/patch_file 落盘后再次查看）"
    return out


register(
    "read_history",
    {
        "type": "function",
        "function": {
            "name": "read_history",
            "description": "查看项目内某文件的 git 提交历史（最近 10 条，git log --oneline）。"
                           "用于了解某文件近期改过什么。",
            "parameters": {
                "type": "object",
                "properties": {
                    "relative_path": {"type": "string", "description": "项目内相对路径，如 src/main.py"},
                },
                "required": ["relative_path"],
            },
        },
    },
    _read_history,
)

# ============================ 项目图谱 / 持久记忆 / 审计 ============================

def _project_root_for_tools() -> str:
    return _PROJECT_ROOT if _PROJECT_ROOT else "."


def _graph_query(symbol: str, reverse: bool = False) -> str:
    from core.project.graph import build_graph, query_impact
    return query_impact(build_graph(_project_root_for_tools()), symbol, reverse)


def _graph_query_file(changed_file: str) -> str:
    from core.project.graph import analyze_dependency_impact
    return analyze_dependency_impact(_project_root_for_tools(), changed_file)


def _repo_remember(topic: str, body: str, kind: str = "note") -> str:
    from core.project.repo_memory import remember
    return remember(_project_root_for_tools(), topic, body, kind)


def _repo_read(topic: str = "", limit: int = 10) -> str:
    from core.project.repo_memory import read
    return read(_project_root_for_tools(), topic, limit)


def _code_review(path: str = "") -> str:
    from core.project.review import code_review
    return code_review(_project_root_for_tools(), path or None)


def _vuln_scan(path: str = "") -> str:
    from core.project.review import vuln_scan
    return vuln_scan(_project_root_for_tools(), path or None)


def _audit_read(limit: int = 50) -> str:
    from core.project.audit import read
    return read(_project_root_for_tools(), limit)


def _snapshot_list(path: str = "") -> str:
    from core.project.snapshot import list_snapshots
    return list_snapshots(_project_root_for_tools(), path or "")


def _snapshot_rollback(name: str) -> str:
    from core.project.snapshot import rollback
    return rollback(_project_root_for_tools(), name)


def _diff_file(rel_path: str) -> str:
    from core.project.snapshot import diff
    return diff(_project_root_for_tools(), rel_path)


def _gen_tests(module: str, out: str = "") -> str:
    from core.project.generators import gen_tests
    return gen_tests(_project_root_for_tools(), module, out)


def _gen_docker() -> str:
    from core.project.generators import gen_docker
    return gen_docker(_project_root_for_tools())


def _gen_doc() -> str:
    from core.project.generators import gen_doc
    return gen_doc(_project_root_for_tools())


register(
    "graph_query",
    {
        "type": "function",
        "function": {
            "name": "graph_query",
            "description": "查项目语义图谱中的某一符号（函数/类），返回其定义位置与全链路影响（谁调用它）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "要查询的符号名（如函数/类名）"},
                    "reverse": {"type": "boolean", "description": "True 改查它调用了谁"},
                },
                "required": ["symbol"],
            },
        },
    },
    _graph_query,
)

register(
    "graph_query_file",
    {
        "type": "function",
        "function": {
            "name": "graph_query_file",
            "description": "给定被修改的文件（相对路径），列出所有受其改动影响的文件——改造前先算影响面。",
            "parameters": {
                "type": "object",
                "properties": {"changed_file": {"type": "string", "description": "被修改文件的相对路径"}},
                "required": ["changed_file"],
            },
        },
    },
    _graph_query_file,
)

register(
    "repo_remember",
    {
        "type": "function",
        "function": {
            "name": "repo_remember",
            "description": "把架构决策/编码规范/历史修改记录持久化到项目记忆，跨会话保留。",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "主题"},
                    "body": {"type": "string", "description": "内容"},
                    "kind": {"type": "string", "description": "种类：note/convention/decision"},
                },
                "required": ["topic", "body"],
            },
        },
    },
    _repo_remember,
)

register(
    "repo_read",
    {
        "type": "function",
        "function": {
            "name": "repo_read",
            "description": "读取项目持久记忆（跨会话共享的架构决策/规范/历史），可按主题过滤。",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string"}, "limit": {"type": "integer"}},
                "required": [],
            },
        },
    },
    _repo_read,
)

register(
    "code_review",
    {
        "type": "function",
        "function": {
            "name": "code_review",
            "description": "对项目或单个文件做静态 Code Review（命名/异常/未用导入/危险调用），返回问题与修复建议。",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "可选，审查单个文件相对路径，缺省审全项目"}},
                "required": [],
            },
        },
    },
    _code_review,
)

register(
    "vuln_scan",
    {
        "type": "function",
        "function": {
            "name": "vuln_scan",
            "description": "扫描代码与第三方依赖的安全漏洞（SQL 注入/硬编码密钥/危险调用）。",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "可选，扫描单个文件相对路径"}},
                "required": [],
            },
        },
    },
    _vuln_scan,
)

register(
    "audit_read",
    {
        "type": "function",
        "function": {
            "name": "audit_read",
            "description": "读取操作审计记录（AI 对项目做过的所有生成/修改/查询）。",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
                "required": [],
            },
        },
    },
    _audit_read,
)

# ============================ 快照/回滚 / diff ============================

register(
    "snapshot_list",
    {
        "type": "function",
        "function": {
            "name": "snapshot_list",
            "description": "列出项目已有的文件快照（写入前自动备份的上一版本）。",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "可选，只看某文件的快照"}},
                "required": [],
            },
        },
    },
    _snapshot_list,
)

register(
    "snapshot_rollback",
    {
        "type": "function",
        "function": {
            "name": "snapshot_rollback",
            "description": "按快照名把文件回滚到上一版本（写坏可无损还原）。快照名见 snapshot_list。",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "快照文件名，如 snap_169XXXXXXXXX.json"}},
                "required": ["name"],
            },
        },
    },
    _snapshot_rollback,
)

register(
    "diff_file",
    {
        "type": "function",
        "function": {
            "name": "diff_file",
            "description": "查看某文件相对上次快照的改动差异（类 unified diff）。",
            "parameters": {
                "type": "object",
                "properties": {"rel_path": {"type": "string", "description": "项目内相对路径"}},
                "required": ["rel_path"],
            },
        },
    },
    _diff_file,
)

# ============================ 工程全流程生成器 ============================

register(
    "gen_tests",
    {
        "type": "function",
        "function": {
            "name": "gen_tests",
            "description": "为指定 .py 模块自动生成 pytest 单测骨架（正常/边界/异常用例）并写入项目。",
            "parameters": {
                "type": "object",
                "properties": {
                    "module": {"type": "string", "description": "模块相对路径，如 app/core.py"},
                    "out": {"type": "string", "description": "可选输出文件名"},
                },
                "required": ["module"],
            },
        },
    },
    _gen_tests,
)

register(
    "gen_docker",
    {
        "type": "function",
        "function": {
            "name": "gen_docker",
            "description": "为项目生成 Dockerfile + .dockerignore（部署配置）。",
            "parameters": {"type": "object", "properties": {}},
            "required": [],
        },
    },
    _gen_docker,
)

register(
    "gen_doc",
    {
        "type": "function",
        "function": {
            "name": "gen_doc",
            "description": "基于语义图谱生成项目接口/结构概览文档 README.auto.md。",
            "parameters": {"type": "object", "properties": {}},
            "required": [],
        },
    },
    _gen_doc,
)


# ============================ skill 生态工具 ============================
def _skill_dir_root() -> str:
    return get_project_root()


def _skill_list(catalog: bool = False, category: str | None = None) -> str:
    from core.skill.registry import list_skills, list_catalog
    if catalog:
        rows = [f"[catalog][{x['category'] or '未分类'}] {x['name']} v{x['version']} — {x['desc']}"
                for x in list_catalog(category)]
        return "\n".join(rows) if rows else "内置目录为空"
    rows = [f"[installed][{x['category'] or '未分类'}]{'(禁用)' if not x['enabled'] else ''} "
            f"{x['name']} v{x['version']} — {x['desc']}"
            for x in list_skills(_skill_dir_root(), category)]
    return "\n".join(rows) if rows else "尚未安装任何技能（可用 skill_install 从内置目录或 source 安装）"


def _skill_install(name: str, source: str | None = None,
                   category: str | None = None, perms: dict | None = None) -> str:
    from core.skill.registry import install_skill
    return install_skill(name, source, _skill_dir_root(), category=category, perms=perms)


def _skill_uninstall(name: str) -> str:
    from core.skill.registry import uninstall_skill
    return uninstall_skill(name, _skill_dir_root())


def _skill_set(name: str, enabled: bool | None = None,
               permissions: dict | None = None) -> str:
    """启用/禁用技能，或按其名单独授权能力。"""
    from core.skill import registry as sk
    if enabled is not None:
        return sk.set_enabled(name, enabled, _skill_dir_root())
    if permissions is not None:
        return sk.set_permissions(name, permissions, _skill_dir_root())
    return sk.get_skill(name, _skill_dir_root()).get("error") or "未指定操作（enabled 或 permissions 至少一项）"


def _skill_run(name: str, args: str = "", pipeline: str | None = None) -> str:
    """在沙箱独立进程中运行单个技能（entry.py），或运行一条流水线。"""
    from core.skill import registry as sk
    from core.skill.runner import run_skill
    if pipeline:
        return sk.run_pipeline(pipeline, _skill_dir_root(), {"text": args} if args else {})
    return run_skill(name, args, root=_skill_dir_root())


def _skill_scaffold(name: str, category: str = "文档生成", desc: str = "") -> str:
    from core.skill.registry import scaffold_skill
    return scaffold_skill(name, category, desc, _skill_dir_root())


def _skill_upgrade(name: str, source: str | None = None) -> str:
    from core.skill.registry import upgrade_skill
    return upgrade_skill(name, source, _skill_dir_root())


def _skill_rollback(name: str) -> str:
    from core.skill.registry import rollback_skill
    return rollback_skill(name, _skill_dir_root())


def _skill_pipeline(action: str = "view", name: str = "", skills: str = "") -> str:
    from core.skill import registry as sk
    root = _skill_dir_root()
    if action == "save":
        seq = [s.strip() for s in skills.split(",") if s.strip()]
        return sk.save_pipeline(name, seq, root)
    if action == "del":
        return sk.delete_pipeline(name, root)
    pipes = sk.list_pipelines(root)
    if not pipes:
        return "尚无流水线（用 skill_pipeline save name='<名>' skills='A,B' 创建）"
    return "\n".join(f"[pipeline] {p['name']}: {' → '.join(p['skills'])}" for p in pipes)


register(
    "skill_list",
    {
        "type": "function",
        "function": {
            "name": "skill_list",
            "description": "枚举已安装的技能包；catalog=True 查看内置可安装目录；可按八分类（代码重构/测试生成/前端页面生成/数据库操作/容器构建/文档生成/模型微调/安全扫描）筛选。",
            "parameters": {"type": "object",
                           "properties": {"catalog": {"type": "boolean", "description": "是否列出内置可安装目录"},
                                          "category": {"type": "string", "description": "可选八分类筛选"}},
                           "required": []},
        },
    },
    _skill_list,
)

register(
    "skill_install",
    {
        "type": "function",
        "function": {
            "name": "skill_install",
            "description": "安装技能包。source 可为 URL/本地文件/技能包目录；省略时从内置目录安装。可指定 category(八分类) 与 perms(能力授权，默认全关)。",
            "parameters": {"type": "object",
                           "properties": {
                               "name": {"type": "string", "description": "技能包名称"},
                               "source": {"type": "string", "description": "可选的 SKILL.md/技能包来源 URL 或本地路径"},
                               "category": {"type": "string", "description": "可选八分类，如 安全扫描"},
                               "perms": {"type": "object", "description": "可选授权矩阵，如 {\"file_read\": true}"},
                           },
                           "required": ["name"]},
        },
    },
    _skill_install,
)

register(
    "skill_uninstall",
    {
        "type": "function",
        "function": {
            "name": "skill_uninstall",
            "description": "卸载已安装的技能包。",
            "parameters": {"type": "object",
                           "properties": {"name": {"type": "string"}},
                           "required": ["name"]},
        },
    },
    _skill_uninstall,
)

register(
    "skill_set",
    {
        "type": "function",
        "function": {
            "name": "skill_set",
            "description": "启用/禁用技能（enabled），或按技能单独授权能力（permissions：file_read/file_write/shell/mcp/train/network，默认全关）。",
            "parameters": {"type": "object",
                           "properties": {
                               "name": {"type": "string"},
                               "enabled": {"type": "boolean", "description": "true 启用 / false 禁用"},
                               "permissions": {"type": "object", "description": "能力授权，如 {\"shell\": true}"},
                           },
                           "required": ["name"]},
        },
    },
    _skill_set,
)

register(
    "skill_run",
    {
        "type": "function",
        "function": {
            "name": "skill_run",
            "description": "在沙箱独立进程中运行技能的 entry.py（默认断网、限时），或运行一条流水线（pipeline）。",
            "parameters": {"type": "object",
                           "properties": {
                               "name": {"type": "string", "description": "要运行的技能名"},
                               "args": {"type": "string", "description": "传给技能 main 的入参文本"},
                               "pipeline": {"type": "string", "description": "若指定则改为运行该流水线"},
                           },
                           "required": ["name"]},
        },
    },
    _skill_run,
)

register(
    "skill_scaffold",
    {
        "type": "function",
        "function": {
            "name": "skill_scaffold",
            "description": f"创建自定义技能脚手架（skill.json+SKILL.md+entry.py），分类限（{'、'.join(__import__('core.skill.registry', fromlist=['CATEGORIES']).CATEGORIES)}）。",
            "parameters": {"type": "object",
                           "properties": {
                               "name": {"type": "string"},
                               "category": {"type": "string", "description": "八分类之一"},
                               "desc": {"type": "string", "description": "玩法描述"},
                           },
                           "required": ["name"]},
        },
    },
    _skill_scaffold,
)

register(
    "skill_upgrade",
    {
        "type": "function",
        "function": {
            "name": "skill_upgrade",
            "description": "升级技能（旧版本自动入 history 快照）；source 省略时按内置目录升级。",
            "parameters": {"type": "object",
                           "properties": {"name": {"type": "string"}, "source": {"type": "string", "description": "可选升级来源"}},
                           "required": ["name"]},
        },
    },
    _skill_upgrade,
)

register(
    "skill_rollback",
    {
        "type": "function",
        "function": {
            "name": "skill_rollback",
            "description": "将技能回滚到上一个历史版本（避免升级后功能崩溃）。",
            "parameters": {"type": "object",
                           "properties": {"name": {"type": "string"}},
                           "required": ["name"]},
        },
    },
    _skill_rollback,
)

register(
    "skill_pipeline",
    {
        "type": "function",
        "function": {
            "name": "skill_pipeline",
            "description": "管理技能组合流水线：save 保存（skills 用逗号分隔顺序执行）、del 删除、view 查看列表。",
            "parameters": {"type": "object",
                           "properties": {
                               "action": {"type": "string", "description": "view / save / del"},
                               "name": {"type": "string", "description": "流水线名"},
                               "skills": {"type": "string", "description": "save 时逗号分隔的技能名，如 'scan,fix,test,deploy'"},
                           },
                           "required": ["action"]},
        },
    },
    _skill_pipeline,
)

register(
    "skill_disable_toggle",
    {
        "type": "function",
        "function": {
            "name": "skill_disable_toggle",
            "description": "快速禁用/启用技能（方便开关第三方能力）。",
            "parameters": {"type": "object",
                           "properties": {"name": {"type": "string"}, "enabled": {"type": "boolean"}},
                           "required": ["name", "enabled"]},
        },
    },
    lambda name, enabled: _skill_set(name, enabled=enabled),
)

# ============================ 后台并行子智能体编排 ============================
# 主 agent 把独立、互不依赖的子任务下发给后台线程的独立子 Agent 并行执行，
# 主循环保持同步，仅用 sub_list / sub_result 轮询收集。子 agent 与主上下文隔离。
# 延迟 import subagent 避免循环依赖（subagent 依赖 agent，agent 顶部不 import subagent）。


def _sub_spawn(prompt: str, role: str = "coder", model: str | None = None) -> str:
    """把独立子任务下发给后台并行子 agent，返回 task_id；可再用 sub_result 取回。"""
    from . import subagent
    return subagent.spawn_task(prompt, role=role, model=model)


def _sub_list() -> str:
    from . import subagent
    return subagent.sub_list()


def _sub_result(task_id: str, keep: bool = False) -> str:
    from . import subagent
    return subagent.sub_result(task_id, keep=keep)


register(
    "sub_spawn",
    {
        "type": "function",
        "function": {
            "name": "sub_spawn",
            "description": "把独立子任务下发给后台并行子 agent，然后可用 sub_result 取回；"
                           "适合可并行、互不依赖的片段。role 可选 planner/checker/coder/research。",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "给子 agent 的独立任务指令"},
                    "role": {"type": "string", "default": "coder",
                             "description": "子智能体角色：planner/checker/coder/research"},
                    "model": {"type": "string", "description": "可选，指定子任务模型，缺省用默认"},
                },
                "required": ["prompt"],
            },
        },
    },
    _sub_spawn,
)

register(
    "sub_list",
    {
        "type": "function",
        "function": {
            "name": "sub_list",
            "description": "列出所有后台子任务的 id/状态/角色/任务摘要/时间，用于查看进度。",
            "parameters": {"type": "object", "properties": {}},
            "required": [],
        },
    },
    _sub_list,
)

register(
    "sub_result",
    {
        "type": "function",
        "function": {
            "name": "sub_result",
            "description": "取回某个子任务（sub_spawn 返回的 task_id）的结果；已完成则一次性取走（"
                           "原任务从清单删除，除非 keep=true）。未完成时返回仍在运行提示。",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "sub_spawn 返回的子任务 id"},
                    "keep": {"type": "boolean", "default": False,
                             "description": "True 则取走后保留该任务，便于重试/复查"},
                },
                "required": ["task_id"],
            },
        },
    },
    _sub_result,
)


def _route_chain(task: str, kind: str | None = None) -> str:
    """把任务交予多智能体链（planner→coder→checker）协作，返回汇总结果。

    由模型在执行中自行判断复杂度后主动调用；子 agent 已被禁用本工具，防递归。
    """
    from . import router  # 延迟导入，避免循环导入
    if not task or not str(task).strip():
        return ("❌ route_chain 缺少必要参数 task。请把要拆给多智能体协作的完整任务"
                "（含目标与验收标准）写进 task 后重试用。")
    kind = kind or router.classify(task)
    return router.run_chain(str(task), kind)


register(
    "route_chain",
    {
        "type": "function",
        "function": {
            "name": "route_chain",
            "description": "当你自主判断当前任务较复杂（多文件、需先规划再实现再审查、或规模较大）时，"
                           "主动调用本工具把任务拆成多智能体链（planner→coder→checker）协作完成，"
                           "并返回协作汇总结果。串行派发多个子智能体并阻塞等待各自结果，"
                           "因此仅在确实需要时调用；简单/单一小任务不要用，直接自行完成。",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string",
                             "description": "要交予多智能体协作的完整任务描述（含目标与验收标准）"},
                    "kind": {"type": "string", "default": "code",
                             "description": "链类型：code/research/review；缺省按 task 自动识别"},
                },
                "required": ["task"],
            },
        },
    },
    _route_chain,
)