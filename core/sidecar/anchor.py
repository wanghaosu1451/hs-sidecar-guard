"""任务锚点固化与持久化（不可篡改）。

首次会话启动时从用户初始指令里提取：
  - requirements  硬性需求
  - constraints   约束（必须/禁止）
  - context       背景/目标
写进独立 JSON 文件，不和对话历史绑在一起——上下文压缩、模型失忆都不会抹掉锚点。
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


ANCHOR_FILENAME = ".hs_anchor.json"


@dataclass
class TaskAnchor:
    """一次会话的不可篡改锚点。"""

    project_root: str
    original_instruction: str
    requirements: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    frozen_at: float = 0.0
    instruction_hash: str = ""  # 用来判定"这次 anchor 是从哪条原始指令来的"

    # ---------------- 持久化 ----------------
    @classmethod
    def load(cls, project_root: str | Path) -> "TaskAnchor | None":
        p = Path(project_root) / ANCHOR_FILENAME
        if not p.is_file():
            return None
        try:
            return cls(**json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            return None

    def save(self) -> Path:
        self.frozen_at = time.time()
        self.instruction_hash = hashlib.sha256(
            self.original_instruction.encode("utf-8")).hexdigest()[:16]
        p = Path(self.project_root) / ANCHOR_FILENAME
        p.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return p

    # ---------------- 固化 ----------------
    @classmethod
    def freeze(cls, project_root: str | Path,
               original_instruction: str,
               requirements: list[str] | None = None,
               constraints: list[str] | None = None,
               forbidden: list[str] | None = None) -> "TaskAnchor":
        """从原始指令固化锚点；若已存在则返回已有（防篡改）。"""
        existing = cls.load(project_root)
        if existing is not None:
            # 只在 hash 匹配（同一条原始指令）时才覆盖，否则视为新会话
            if existing.instruction_hash == hashlib.sha256(
                    original_instruction.encode("utf-8")).hexdigest()[:16]:
                return existing
        anchor = cls(
            project_root=str(Path(project_root).resolve()),
            original_instruction=original_instruction,
            requirements=list(requirements or []),
            constraints=list(constraints or []),
            forbidden=list(forbidden or []),
        )
        anchor.save()
        return anchor

    # ---------------- 查询 ----------------
    def has(self) -> bool:
        return bool(self.requirements or self.constraints or self.forbidden)

    def summary(self) -> dict:
        return {
            "requirements": self.requirements,
            "constraints": self.constraints,
            "forbidden": self.forbidden,
        }