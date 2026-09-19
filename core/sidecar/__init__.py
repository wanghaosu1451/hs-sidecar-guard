"""Sidecar 纯终端校验层（Goal Drift + 防火墙 + 因果依赖）。

与主 Agent 解耦：主 Agent 通过 hooks.py 的 pre/post 拦截点把工具调用事件丢过来，
Sidecar 独立做校验、输出、阻断——不引入任何 GUI 依赖，纯 CLI/Rich 字符渲染。
"""
from .anchor import TaskAnchor
from .drift import DriftDetector, DriftReport
from .sidecar import Sidecar
from .ui import TerminalUI

__all__ = [
    "TaskAnchor",
    "DriftDetector",
    "DriftReport",
    "Sidecar",
    "TerminalUI",
]