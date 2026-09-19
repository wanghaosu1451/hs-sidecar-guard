"""纯终端字符 UI：告警、暂停、回溯、因果树渲染。

不引入任何 GUI 依赖，用 Rich 做终端着色/进度条/树状展示。
没有 Rich 的 fallback 也能用（降级为纯文本）。
"""
from __future__ import annotations

from typing import Any


def _has_rich() -> bool:
    try:
        import rich  # noqa: F401
        return True
    except ImportError:
        return False


class TerminalUI:
    """Sidecar 在终端输出的统一入口。全部走这里便于后续接入 hooks / Trace 日志。"""

    def __init__(self, force_plain: bool = False) -> None:
        self._rich = (not force_plain) and _has_rich()

    # ---------- 通用 ----------
    def print(self, msg: str, level: str = "info") -> None:
        prefix = {"info": "[SIDE]", "warn": "[WARN]",
                  "block": "[BLOCK]", "ok": "[OK]"}.get(level, "[SIDE]")
        if self._rich:
            from rich.console import Console
            c = Console()
            color = {"info": "cyan", "warn": "yellow",
                     "block": "red", "ok": "green"}.get(level, "white")
            c.print(f"[{color}]{prefix}[/{color}] {msg}")
        else:
            print(f"{prefix} {msg}")

    # ---------- Goal Drift ----------
    def anchor_frozen(self, anchor_summary: dict) -> None:
        self.print("任务锚点已固化（不会被上下文压缩抹掉）", "ok")
        for key, items in anchor_summary.items():
            if items:
                self.print(f"  {key}:")
                for it in items:
                    self.print(f"    - {it}", "info")

    def drift_alert(self, score: float, level: str,
                     violations: list[str], rationale: str) -> None:
        self.print(f"偏离度 {score:.2f} | 级别 {level.upper()}", "warn" if level == "warn" else "block")
        for v in violations:
            self.print(f"  ✗ {v}", "block" if level == "block" else "warn")
        if rationale:
            self.print(f"  {rationale}", "info")
        if self._rich:
            from rich.console import Console
            c = Console()
            c.print("[yellow]━━ 任务已暂停，请确认是否继续 ━━[/yellow]")

    # ---------- Shell 防火墙 ----------
    def fw_blocked(self, tool_name: str, reason: str) -> None:
        self.print(f"Shell 防火墙拦截 {tool_name}: {reason}", "block")

    def fw_confirm(self, tool_name: str, hint: str) -> None:
        self.print(f"中危操作 {tool_name} 待确认: {hint}", "warn")

    def fw_allowed(self, tool_name: str) -> None:
        self.print(f"放行 {tool_name}", "ok")

    # ---------- 因果依赖 ----------
    def render_dep_tree(self, root: str, deps: dict[str, list[str]]) -> str:
        """把依赖 dict 渲染成字符树，返回字符串（终端直接 print）。"""
        lines = [root]
        visited = set()

        def walk(node: str, depth: int) -> None:
            if node in visited:
                lines.append("  " * depth + "└─ (循环)")
                return
            visited.add(node)
            children = deps.get(node, [])
            for c in children:
                lines.append("  " * depth + "├─ " + c)
                walk(c, depth + 1)

        walk(root, 1)
        tree = "\n".join(lines)
        if self._rich:
            from rich.console import Console
            c = Console()
            c.print(tree, style="dim")
        else:
            print(tree)
        return tree