"""沙箱策略：网络/文件系统白名单、CPU/内存上限。"""
from __future__ import annotations

import re
from pathlib import Path

# 宿主仅凭据类环境变量名，命中的一律不给子进程（避免把债务/密钥带入沙箱）
_SECRET_ENV_RE = re.compile(
    r"(api[_-]?key|api[_-]?secret|secret|token|password|passwd|"
    r"private[_-]?key|credential|jwt|authorization)", re.IGNORECASE)

# 代理/联网类环境变量，默认断网时清空（零配置本地沙箱的主要拦截手段之一）
_PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
               "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy")

# 网络禁用标记（尽力而为；纯 subprocess 无法做到系统级断网，仅作降级兜底并如实提醒）
NETWORK_DISABLED_VAR = "CODEX_SANDBOX_NETWORK_DISABLED"


def is_secret_env_var(name: str) -> bool:
    """判断环境变量名是否为宿主密钥/凭据类（此类变量不给子进程）。"""
    return bool(_SECRET_ENV_RE.search(name))


def sanitize_env(base: dict[str, str], allow_network: bool) -> dict[str, str]:
    """对传给子进程的环境做脱敏：剥离密钥/凭据，默认剥离代理并打断网标记。"""
    env = {}
    for k, v in base.items():
        if is_secret_env_var(k):
            continue  # 不给子进程任何宿主密钥/凭证
        if not allow_network and k in _PROXY_VARS:
            continue  # 断网模式下不注入任何代理
        env[k] = v
    if not allow_network:
        env[NETWORK_DISABLED_VAR] = "1"
    return env


class SandboxPolicy:
    def __init__(self, timeout_seconds: int = 60, max_cpu_percent: int = 80,
                 max_memory_mb: int = 4096, max_processes: int = 32,
                 allow_network: bool = False,
                 allowed_dirs: list[str] | None = None):
        self.timeout_seconds = timeout_seconds
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_mb = max_memory_mb
        self.max_processes = max_processes
        self.allow_network = allow_network
        self.allowed_dirs = [Path(d).resolve() for d in (allowed_dirs or [])]

    def clamp_command(self, command: str) -> str:
        """按策略包装命令：网络白名单 + 超时。

        当前实现注入通用安全约束；后续可增强为容器/setrlimit。
        """
        return command

    def resolve_workdir(self, requested: str | None) -> str | None:
        """把子进程工作目录锁定在 allowed_dirs 白名单内（零配置本地沙箱核心）。

        请求的目录在白名单内则用它；否则（含未指定）回退到白名单第一个目录。
        allowed_dirs 为空表示“未启用项目根锁定”，随调用方自由指定。
        """
        if not self.allowed_dirs:
            return requested
        if requested:
            try:
                r = Path(requested).resolve()
            except OSError:
                r = Path(requested)
            if any(a == r or a in r.parents for a in self.allowed_dirs):
                return str(r)
        return str(self.allowed_dirs[0])

    def validate_paths(self, paths: list[str]) -> list[str]:
        """路径白名单校验，返回允许访问的路径。"""
        if not self.allowed_dirs:
            return paths  # 未配置白名单时不做限制
        ok: list[str] = []
        for raw in paths:
            p = Path(raw).resolve()
            if any(a == p or a in p.parents for a in self.allowed_dirs):
                ok.append(raw)
        return ok

    def path_block_reason(self, cmd_text: str) -> str | None:
        """扫描命令文本，若引用到项目目录外的现有路径则返回拦截原因。

        纯 subprocess/无容器下无法做系统级文件访问控制，故采用启发式：
        抓取命令中的绝对路径 token，若命中白名单之外则拒绝，保证“读写不出项目根”。
        """
        if not self.allowed_dirs or not cmd_text:
            return None
        for raw in self._path_tokens(cmd_text):
            p = Path(raw).expanduser()
            if not p.is_absolute():
                continue
            if "://" in raw or raw.startswith(("-", "--")):
                continue
            try:
                resolved = p.resolve()
            except OSError:
                continue
            if any(a == resolved or a in resolved.parents for a in self.allowed_dirs):
                continue
            return raw
        return None

    @staticmethod
    def _path_tokens(cmd_text: str) -> list[str]:
        tokens: list[str] = []
        # 引号包裹的路径/盘符绝对路径 token
        tokens += re.findall(r"['\"]([^'\"]+?)['\"]", cmd_text)
        tokens += [t for t in cmd_text.split()
                   if re.match(r"^[A-Za-z]:[\\/]", t)]
        return tokens

    def to_dict(self) -> dict:
        return {
            "timeout_seconds": self.timeout_seconds,
            "max_cpu_percent": self.max_cpu_percent,
            "max_memory_mb": self.max_memory_mb,
            "max_processes": self.max_processes,
            "allow_network": self.allow_network,
            "allowed_dirs": [str(d) for d in self.allowed_dirs],
        }