"""Sidecar 主进程：与主 Agent 通过 hooks 交互的独立校验层。

不抢主流程的控制权——主 Agent 在 pre_tool_use 时把事件丢给 Sidecar，
Sidecar 返回 "block / confirm / allow"，主 Agent 决定是否继续执行。
"""
from __future__ import annotations

from typing import Any

from .anchor import TaskAnchor
from .drift import DriftDetector, DriftReport
from .ui import TerminalUI
from ..sidecar.firewall.gate import FirewallGate


class Sidecar:
    """Sidecar 侧校验进程的单例门面（每个项目根一个实例）。"""

    def __init__(self, project_root: str | Path,
                 llm_client=None,
                 force_plain: bool = False) -> None:
        self.project_root = str(project_root)
        self.llm_client = llm_client
        self.ui = TerminalUI(force_plain=force_plain)
        self.anchor = TaskAnchor.load(self.project_root)
        self.detector: DriftDetector | None = None
        self.firewall: FirewallGate | None = None
        if self.anchor is not None:
            self.detector = DriftDetector(self.anchor, llm_client=llm_client)
        try:
            from ..sidecar.firewall.gate import FirewallGate
            self.firewall = FirewallGate(project_root=self.project_root,
                                         llm_client=llm_client)
        except Exception:
            self.firewall = None

    # -------- 初始化锚点 --------
    def bootstrap(self, original_instruction: str,
                  requirements: list[str] | None = None,
                  constraints: list[str] | None = None,
                  forbidden: list[str] | None = None) -> TaskAnchor:
        self.anchor = TaskAnchor.freeze(
            self.project_root, original_instruction,
            requirements, constraints, forbidden)
        self.detector = DriftDetector(self.anchor, llm_client=self.llm_client)
        self.ui.anchor_frozen(self.anchor.summary())
        return self.anchor

    # -------- 工具调用拦截（对接 hooks.py） --------
    def pre_tool_use(self, tool_name: str, arguments: Any) -> dict:
        """主 Agent 在 pre_tool_use 时调用。返回 {"block": bool, "msg": str, "pause": bool}。"""
        out = {"block": False, "msg": "", "pause": False}

        # 1) Shell 防火墙（最前置，毫秒级）
        if self.firewall is not None:
            fw = self.firewall.judge(tool_name, arguments)
            if fw.blocked:
                self.ui.fw_blocked(tool_name, fw.reason)
                out.update(block=True, msg=fw.reason)
                return out
            if fw.confirm_needed:
                self.ui.fw_confirm(tool_name, fw.reason)

        # 2) Goal Drift 检测
        if self.detector is not None:
            report: DriftReport = self.detector.check(tool_name, arguments)
            if report.level == "block":
                self.ui.drift_alert(report.score, report.level,
                                    report.violations, report.rationale)
                out.update(block=True, msg="目标漂移严重：" +
                           "; ".join(report.violations) or report.rationale,
                           pause=True)
                return out
            if report.level == "warn":
                self.ui.drift_alert(report.score, report.level,
                                    report.violations, report.rationale)
                out.update(pause=True)

        return out

    def post_tool_use(self, tool_name: str, arguments: Any,
                      result: str) -> None:
        """工具执行后：当前只做日志打点，后续可扩因果一致性校验。"""
        if self.detector is not None:
            self.detector.check(tool_name, arguments)  # 记录历史但不阻断