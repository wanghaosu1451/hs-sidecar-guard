"""生命周期钩子：PreToolUse / PostToolUse / Stop（生产安全护栏）。

仅借鉴 Claude Code 的设计思路（确定性护栏 + 生命周期事件），完全用本项目自有
架构实现，不引入任何 Anthropic SDK / 后门。

钩子来源（读取「项目根」下的文件）：
  (a) 结构化 JSON 规则文件 `.hs_hooks.json` —— 默认启用、最安全（不执行任意代码）；
  (b) 可选本地脚本 `hooks.py`（导出 pre_tool_use / post_tool_use / stop 函数）——
      默认关闭，需在 JSON 里显式设置 `script_enabled: true` 才启用；用隔离子进程
      执行、带超时、失败静默，避免恶意/出错脚本影响主流程。

设计原则：钩子自身任何异常都不影响主流程（try/except 静默降级），
即「钩子宁可失效，也不炸主流程」。

`.hs_hooks.json` 规则示例：
{
  "enabled": true,
  "pre_tool_use": [
    {"name": "run_shell", "action": "block", "message": "此工具已被全局禁用"},
    {"name": "run_*", "arguments_contain": "rm -rf", "action": "block",
     "message": "检测到危险删除参数，已拦截"},
    {"name": "project_write", "action": "inject", "inject": "写文件前请先确认父目录已存在"}
  ],
  "hooks_module": {"path": "hooks.py", "timeout": 5},
  "script_enabled": false
}
"""
from __future__ import annotations

import fnmatch
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# 项目根下的钩子规则文件名（结构化 JSON，模式 a）
HOOKS_FILENAME = ".hs_hooks.json"
# 可选脚本文件名（模式 b，默认关闭）
HOOKS_MODULE_FILENAME = "hooks.py"
# 脚本执行的默认超时（秒），防止钩子脚本卡死等待
_DEFAULT_TIMEOUT = 5


class PreHookResult:
    """PreToolUse 的判定结果。

    - blocked=True 表示应阻断本次工具调用（不执行）。
    - inject 为实时 prompt 修正提示列表（非阻断，随工具结果回传给模型）。
    """

    __slots__ = ("blocked", "message", "inject")

    def __init__(self, blocked: bool = False, message: str | None = None,
                 inject: list[str] | None = None) -> None:
        self.blocked = blocked
        self.message = message
        self.inject = inject or []

    def __repr__(self) -> str:  # pragma: no cover - 仅调试
        return f"<PreHookResult blocked={self.blocked} message={self.message!r} inject={self.inject}>"


def _resolve_root(root: Any = None) -> str:
    """把传入 root 或当前项目写入目录收敛为绝对路径；空/`.`时回退到 cwd。"""
    if not root or root == ".":
        root = None
    base = root if root else None
    if base is None:
        # 惰性取工具注册表里的项目根，避免模块加载时产生循环 import
        from . import tools as _tr
        base = _tr.get_project_root()
    if not base or base == ".":
        base = "."
    return str(Path(os.fspath(base)).expanduser().resolve())


def _items(value: Any) -> list[dict]:
    """把 JSON 里可能不是列表的 pre_tool_use 字段规整为列表。"""
    if isinstance(value, list):
        return [r for r in value if isinstance(r, dict)]
    if isinstance(value, dict):
        return [value]
    return []


class HookManager:
    """钩子管理器：解析并持有项目根下的钩子规则，提供三个事件点。

    每次工具调用、会话结束时都会走到这里；规则与脚本执行全部 try/except 兜住，
    任何异常都静默降级（视为无钩子），绝不抛给上层。

    额外通道（可选）：
      .sidecar —— 指向独立 Sidecar 进程的 SidecarClient。
                  接入后每次工具调用会自动走 HTTP RPC 给 Sidecar 做漂移/安全校验。
                  Sidecar 挂了自动降级到规则级，绝不阻塞主 Agent。
    """

    def __init__(self, root: Any) -> None:
        self.root = _resolve_root(root)
        self.enabled = True
        self.rules: list[dict] = []
        self.script_enabled = False
        self.script_path: str | None = None
        self.module_timeout = _DEFAULT_TIMEOUT
        self.sidecar = None  # 运行时外部注入: SidecarClient(...)
        self.load()

    # ---------------- 加载 / 解析 ----------------
    def load(self) -> None:
        """从 root/.hs_hooks.json 加载规则。破损/缺失文件一律还原为默认（无钩子）。"""
        self.enabled = True
        self.rules = []
        self.script_enabled = False
        self.script_path = None
        self.module_timeout = _DEFAULT_TIMEOUT
        p = Path(self.root) / HOOKS_FILENAME
        if not p.is_file():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8") or "{}")
        except Exception:  # noqa: BLE001 - 破损 JSON 静默忽略
            return
        if not isinstance(data, dict):
            return
        self.enabled = bool(data.get("enabled", True))
        self.rules = _items(data.get("pre_tool_use"))
        mod = data.get("hooks_module")
        if isinstance(mod, dict):
            self.script_path = str(mod.get("path") or HOOKS_MODULE_FILENAME)
            try:
                self.module_timeout = float(mod.get("timeout") or _DEFAULT_TIMEOUT)
            except (TypeError, ValueError):
                self.module_timeout = _DEFAULT_TIMEOUT
        self.script_enabled = bool(data.get("script_enabled", False))

    # ---------------- 事件点 ----------------
    def pre_tool_use(self, name: str, arguments: Any) -> PreHookResult:
        """工具执行前触发。返回阻断/注入判定；钩子异常一律静默放行。"""
        result = PreHookResult()
        if not self.enabled:
            return result
        try:
            for rule in self.rules:
                if not self._rule_matches(rule, name, arguments):
                    continue
                action = str(rule.get("action", "inject"))
                if action == "block":
                    result.blocked = True
                    result.message = str(rule.get("message")
                                         or f"钩子已阻断工具 {name}")
                    return result
                # inject：非阻断，收集修正提示，执行后随结果回传给模型
                inj = rule.get("inject")
                if inj:
                    result.inject.append(str(inj))
        except Exception:  # noqa: BLE001 - 规则解析出错不阻塞主流程
            return PreHookResult()
        # ===== Sidecar 独立进程通道（可选）=====
        if self.sidecar is not None:
            try:
                sc = self.sidecar.pre_tool_use(name, arguments)
                if sc.get("block"):
                    result.blocked = True
                    reason = str(sc.get("msg") or "Sidecar 拦截")
                    src = sc.get("source", "sidecar")
                    result.message = f"[{src}] {reason}"
                    return result
            except Exception:  # noqa: BLE001 - Sidecar 挂了静默降级
                pass
        # 可选脚本（默认关闭）：其返回里如需阻断则合并进来
        if self.script_enabled and self.script_path:
            try:
                out = self._run_script("pre_tool_use", name, arguments)
                if isinstance(out, dict) and out.get("block"):
                    result.blocked = True
                    result.message = str(
                        out.get("message")
                        or result.message
                        or f"钩子已阻断工具 {name}")
            except Exception:  # noqa: BLE001
                pass
        return result

    def post_tool_use(self, name: str, arguments: Any, result: str) -> None:
        """工具执行后触发（仅回调，不改变返回值）。异常静默。"""
        if not self.enabled:
            return
        try:
            if self.sidecar is not None:
                self.sidecar.post_tool_use(name, arguments)
        except Exception:  # noqa: BLE001
            pass
        try:
            if self.script_enabled and self.script_path:
                self._run_script("post_tool_use", name, arguments, result)
        except Exception:  # noqa: BLE001
            pass

    def stop(self) -> None:
        """一次任务/会话结束时触发。异常静默。"""
        if not self.enabled:
            return
        try:
            if self.script_enabled and self.script_path:
                self._run_script("stop")
        except Exception:  # noqa: BLE001
            pass

    # ---------------- 规则匹配 ----------------
    def _rule_matches(self, rule: dict, name: str, arguments: Any) -> bool:
        """单条规则是否命中：名称匹配（支持 fnmatch 通配）+ 参数关键字命中。"""
        try:
            if "name" in rule:
                n = rule["name"]
                if not (n == name or fnmatch.fnmatch(name, str(n))):
                    return False
            if "arguments_contain" in rule:
                needle = str(rule["arguments_contain"])
                hay = arguments if isinstance(arguments, str) else json.dumps(
                    arguments, ensure_ascii=False)
                if needle not in str(hay):
                    return False
        except Exception:  # noqa: BLE001
            return False
        return True

    # ---------------- 可选脚本隔离执行 ----------------
    def _run_script(self, event: str, *args: Any):
        """用隔离子进程执行 hooks.py 里名为 event 的函数，超时/失败返回 None（静默）。"""
        script = Path(self.root) / self.script_path
        if not script.is_file():
            self.script_enabled = False  # 脚本缺失：关掉，避免每次重复探测
            return None
        # 极简 runner：加载脚本模块 -> 调用 event 函数 -> 输出 JSON。全程隔离、有限参数。
        runner = (
            "import sys, json, importlib.util\n"
            "p=sys.argv[1]; ev=sys.argv[2]\n"
            "try:\n"
            "    payload=json.loads(sys.stdin.read()) if not sys.stdin.isatty() else None\n"
            "except Exception:\n    payload=None\n"
            "spec=importlib.util.spec_from_file_location('_hs_hooks_src', p)\n"
            "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            "fn=getattr(m, ev, None)\n"
            "if fn is None:\n    print('null'); sys.exit(0)\n"
            "try:\n"
            "    out=fn(*payload) if isinstance(payload, list) else fn()\n"
            "except Exception:\n    print('null'); sys.exit(1)\n"
            "try:\n    print(json.dumps(out, ensure_ascii=False))\n"
            "except Exception:\n    print('null')\n"
        )
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            r = subprocess.run(
                [sys.executable, "-c", runner, str(script), event],
                input=json.dumps(list(args), ensure_ascii=False),
                capture_output=True, text=True, timeout=self.module_timeout,
                env=env, cwd=self.root)
        except subprocess.TimeoutExpired:
            return None
        except Exception:  # noqa: BLE001
            return None
        try:
            return json.loads(r.stdout or "null")
        except Exception:  # noqa: BLE001
            return None


# ---------------- 进程内管理器缓存（root 变化自动重载） ----------------
_lazy: dict[str, Any] = {"mgr": None, "root": None}


def get_manager(root: Any = None) -> HookManager:
    """返回缓存的管理器；当项目根变化时自动重建，保证读到最新规则。"""
    key = _resolve_root(root)
    mgr, cached_root = _lazy["mgr"], _lazy["root"]
    if mgr is None or cached_root != key:
        mgr = HookManager(key)
        _lazy["mgr"] = mgr
        _lazy["root"] = key
    return mgr


def pre_tool_use(name: str, arguments: Any) -> PreHookResult:
    """模块级便捷入口：工具执行前触发（供 tools.invoke 调用）。"""
    return get_manager().pre_tool_use(name, arguments)


def post_tool_use(name: str, arguments: Any, result: str) -> None:
    """模块级便捷入口：工具执行后触发。"""
    get_manager().post_tool_use(name, arguments, result)


def stop() -> None:
    """模块级便捷入口：一次任务/会话结束时触发。"""
    get_manager().stop()


def reload(root: Any = None) -> HookManager:
    """强制重载（丢弃缓存管理器），返回新管理器；供 /init 或测试使用。"""
    _lazy["mgr"] = None
    _lazy["root"] = None
    return get_manager(root)


def rules_path(root: Any = None) -> Path:
    """返回项目根下钩子规则文件的路径（用于 /init 探测，无论是否存在）。"""
    return Path(_resolve_root(root)) / HOOKS_FILENAME


def rules_loaded(root: Any = None) -> bool:
    """项目根下是否存在可用的钩子规则文件。"""
    return rules_path(root).is_file()


def script_path(root: Any = None) -> Path:
    """返回可选钩子脚本的路径（用于 /init 探测）。"""
    return Path(_resolve_root(root)) / HOOKS_MODULE_FILENAME