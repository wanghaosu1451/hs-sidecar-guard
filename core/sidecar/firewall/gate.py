"""Shell 防火墙：三层放行网关（allow → forbid → confirm → 可选语义层）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .rules import RuleSet, load_rules


@dataclass
class FirewallDecision:
    blocked: bool = False
    confirm_needed: bool = False
    reason: str = ""


SHELL_TOOLS = {"run_shell", "run_bash", "run_powershell", "run_cmd",
               "shell", "bash", "powershell", "cmd"}


def _text_of(args: Any) -> str:
    import json
    if isinstance(args, str):
        return args
    try:
        return json.dumps(args, ensure_ascii=False)
    except Exception:
        return str(args)


class FirewallGate:
    def __init__(self, project_root: str | Path, llm_client=None) -> None:
        self.project_root = str(project_root)
        self.rules: RuleSet = load_rules(self.project_root)
        self.llm_client = llm_client  # 可选，用于模糊 case 的语义判定

    def judge(self, tool_name: str, arguments: Any) -> FirewallDecision:
        """工具调用拦截判定。非 Shell 工具默认放行。"""
        if tool_name not in SHELL_TOOLS:
            return FirewallDecision()

        hay = _text_of(arguments)
        low = hay.lower()

        # Layer 0: allow 白名单优先
        if self.rules.is_allow(low):
            return FirewallDecision()

        # Layer 1: forbid 黑名单（毫秒级）
        if self.rules.is_forbid(low):
            return FirewallDecision(blocked=True,
                                     reason="命中禁止规则: 高危命令模式")

        # Layer 2: confirm 中危
        if self.rules.is_confirm(low):
            # 如果挂了 1.5B，额外问一次"这是在删临时文件还是生产数据？"
            if self.llm_client is not None:
                try:
                    out = self.llm_client.generate(
                        f"判断这条 Shell 命令的意图：{hay[:300]}\n"
                        "只回答 DELETE_DATA / CLEAN_TMP / BUILD / OTHER",
                        max_new_tokens=40)
                    out = out.strip().upper()
                    if "DELETE" in out and "TMP" not in out:
                        return FirewallDecision(
                            blocked=True,
                            reason=f"语义判定为高危删除: {out}")
                    if "CLEAN_TMP" in out or "BUILD" in out:
                        return FirewallDecision()  # 放行
                except Exception:
                    pass  # LLM 挂了就退回 confirm
            return FirewallDecision(confirm_needed=True,
                                     reason="命中中危规则，需人工确认")

        # Layer 3: allow
        return FirewallDecision()