"""自然语言安全规则解析 + YAML 模板。

用户写的规则是自然语言，我们离线编译成正则/关键词列表——避免让 1.5B 在线编译。
规则文件示例（项目根 `.hs_safety.yaml`）：

    shell:
      forbid:
        - 禁止递归删除项目目录      # → 编译成 rm -rf . / rm -rf 项目根 的正则
        - 禁止执行远程下载脚本     # → 编译成 curl/wget | sh 正则
      confirm:
        - 禁止修改系统密钥配置
      allow:
        - 允许删除 /tmp 临时文件
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_RULES = {
    # 这些规则内置，用户可在 .hs_safety.yaml 覆写或追加
    "forbid_regex": [
        r"rm\s+(-[a-zA-Z]*r[a-zA-Z]*|\-[a-zA-Z]*rf[a-zA-Z]*)\s+(\.\s|--preserve-root|/etc|/usr|/var)",
        r"curl.*\|\s*(ba|z)?sh",
        r"wget.*\|\s*(ba|z)?sh",
        r"git\s+push\s+.*--force(?!.*--lease)",
        r">\s*[/]?etc[/](passwd|shadow)",
        r"chmod\s+.*4\d{3}",
    ],
    "confirm_regex": [
        r"rm\s+(-[a-zA-Z]*r[a-zA-Z]*|\-[a-zA-Z]*rf[a-zA-Z]*)",
        r"mv\s+.*\.\.",
        r"dd\s+if=",
        r"systemctl\s+(stop|disable)",
    ],
    "allow_regex": [
        r"rm\s+(-f[a-zA-Z]*)*\s+/tmp/",
        r"pip\s+install",
    ],
}


@dataclass
class RuleSet:
    """编译后的规则集。"""

    forbid_regex: list[str] = field(default_factory=list)
    confirm_regex: list[str] = field(default_factory=list)
    allow_regex: list[str] = field(default_factory=list)

    # 把用户自然语言规则转成正则/关键词的编译映射（离线做）
    _NL_TRIGGERS: dict[str, list[str]] = field(default_factory=lambda: {
        "递归删除项目目录": [r"rm\s+.*-r", r"rm\s+.*\.\s"],
        "远程下载脚本": [r"curl.*\|", r"wget.*\|"],
        "系统密钥": [r"/etc/(passwd|shadow)", r"chmod.*4\d"],
        "强制推送": [r"git push.*--force(?!.*--lease)"],
    })

    def is_forbid(self, text: str) -> bool:
        return any(re.search(p, text, re.IGNORECASE)
                   for p in self.forbid_regex)

    def is_confirm(self, text: str) -> bool:
        return any(re.search(p, text, re.IGNORECASE)
                   for p in self.confirm_regex)

    def is_allow(self, text: str) -> bool:
        return any(re.search(p, text, re.IGNORECASE)
                   for p in self.allow_regex)


def load_rules(project_root: str | Path) -> RuleSet:
    """加载并合并项目自定义规则；未找到或格式破损就用默认。"""
    rs = RuleSet(
        forbid_regex=list(DEFAULT_RULES["forbid_regex"]),
        confirm_regex=list(DEFAULT_RULES["confirm_regex"]),
        allow_regex=list(DEFAULT_RULES["allow_regex"]),
    )
    p = Path(project_root) / ".hs_safety.yaml"
    if not p.is_file():
        return rs
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception:
        # yaml 未装或文件破损，静默退回默认
        return rs
    if not isinstance(data, dict):
        return rs
    shell = data.get("shell", {}) if isinstance(data, dict) else {}
    # 用户自然语言规则 → 查 NL_TRIGGERS 做离线编译
    for level, key in [("forbid", "forbid_regex"),
                       ("confirm", "confirm_regex"),
                       ("allow", "allow_regex")]:
        for rule in shell.get(level, []) or []:
            needle = str(rule).strip()
            matched = False
            for trigger, pats in rs._NL_TRIGGERS.items():
                if trigger in needle or needle in trigger:
                    for pat in pats:
                        if pat not in getattr(rs, key):
                            getattr(rs, key).append(pat)
                    matched = True
                    break
            # 如果自然语言规则没匹配到预编译模板，就直接当正则用
            if not matched and needle.startswith("regex:"):
                pat = needle.split(":", 1)[1].strip()
                if pat not in getattr(rs, key):
                    getattr(rs, key).append(pat)
    return rs