"""任务锚点分层持久化：三层。

  Layer 1: .hs_anchor.json     不可修改的锚点（hash 校验，漂移必须先改原始指令）
  Layer 2: .hs_progress.json   会话推进记录（哪些锚点已满足、遇到过哪些漂移）
  Layer 3: .hs_trace.jsonl     完整 Trace（每次工具调用 → Sidecar 判定 → 锚点状态变化）

  Sidecar 重启后锚点不丢、会话暂停再回来也知道做到哪一步。
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


PROGRESS_FILENAME = ".hs_progress.json"


@dataclass
class ProgressTracker:
    """Layer 2：会话进展追踪（每次 bootstrapped 时创建一次，会话全周期写）。"""

    project_root: str
    anchor_hash: str                        # 关联哪条 anchor
    fulfilled: list[str] = field(default_factory=list)   # 已满足的需求/约束原文
    drift_events: list[dict] = field(default_factory=list)   # 遇到过的漂移事件
    started_at: float = 0.0
    updated_at: float = 0.0

    # -------- 持久化 --------
    @classmethod
    def load(cls, project_root: str | Path) -> "ProgressTracker | None":
        p = Path(project_root) / PROGRESS_FILENAME
        if not p.is_file():
            return None
        try:
            return cls(**json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            return None

    def save(self) -> Path:
        self.updated_at = time.time()
        p = Path(self.project_root) / PROGRESS_FILENAME
        p.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8")
        return p

    # -------- 构造 --------
    @classmethod
    def create(cls, project_root: str | Path, anchor_hash: str) -> "ProgressTracker":
        p = cls(project_root=str(Path(project_root).resolve()),
                anchor_hash=anchor_hash, started_at=time.time())
        p.save()
        return p

    # -------- 更新 --------
    def mark_fulfilled(self, item: str) -> None:
        if item not in self.fulfilled:
            self.fulfilled.append(item)
        self.save()

    def record_drift(self, score: float, violations: list[str],
                     tool_name: str = "", args_preview: str = "") -> None:
        self.drift_events.append({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "score": score,
            "violations": violations,
            "tool": tool_name,
            "args": args_preview[:200],
        })
        self.save()

    def summary(self) -> dict:
        return {
            "fulfilled": self.fulfilled,
            "total_drift_events": len(self.drift_events),
            "started_at": self.started_at,
            "updated_at": self.updated_at,
        }


@dataclass
class TraceEvent:
    """Layer 3：每次工具调用的完整 Trace 事件。"""
    ts: str
    event: str                 # bootstrap / pre_tool_use / drift_alert / firewall_block / merged / discarded
    tool: str = ""
    args: str = ""
    details: dict = field(default_factory=dict)
    score: float | None = None


TRACE_FILENAME = ".hs_trace.jsonl"


class TraceLogger:
    """Layer 3：完整事件日志（JSONL 追加写，可查询可回放）。"""

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.path = self.project_root / TRACE_FILENAME
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 创建时写一个 session 头
        if not self.path.is_file():
            self.append("session_start",
                        details={"ts": time.strftime("%Y-%m-%d %H:%M:%S")})

    def append(self, event: str, tool: str = "",
               args: str = "", details: dict | None = None,
               score: float | None = None) -> None:
        try:
            rec = TraceEvent(
                ts=time.strftime("%Y-%m-%d %H:%M:%S"),
                event=event, tool=tool, args=args,
                details=details or {}, score=score)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
        except Exception:
            pass  # Trace 写失败绝不炸主流程

    def tail(self, n: int = 20) -> list[dict]:
        if not self.path.is_file():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        out: list[dict] = []
        for l in lines[-n:]:
            try: out.append(json.loads(l))
            except: pass
        return out

    def filter_by(self, event: str | None = None,
                   tool: str | None = None) -> list[dict]:
        out: list[dict] = []
        if not self.path.is_file():
            return out
        for l in self.path.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(l)
            except Exception:
                continue
            if event and e.get("event") != event:
                continue
            if tool and e.get("tool") != tool:
                continue
            out.append(e)
        return out
