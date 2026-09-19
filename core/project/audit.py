"""操作审计：记录所有 AI 生成/修改/查询行为到 <root>/.hs/audit.jsonl。

满足金融/政企“谁改了什么、什么时候、改了哪些文件”的审计与等保要求。
"""
from __future__ import annotations

import json
import time
from pathlib import Path


def record(root: str | Path, action: str, detail: str = "", path: str | None = None) -> None:
    f = Path(root).resolve() / ".hs" / "audit.jsonl"
    f.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "detail": (detail or "")[:300],
        "path": path,
    }
    try:
        with f.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        pass


def read(root: str | Path, limit: int = 50) -> str:
    f = Path(root).resolve() / ".hs" / "audit.jsonl"
    if not f.is_file():
        return "(暂无操作审计记录)"
    lines = f.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    lines = lines[-limit:]
    out = []
    for ln in lines:
        try:
            e = json.loads(ln)
            out.append(f"[{e.get('ts')}] {e.get('action')} {e.get('path') or ''} {e.get('detail') or ''}")
        except Exception:  # noqa: BLE001
            pass
    return "操作审计（最近 %d 条）:\n" % len(out) + "\n".join(out) if out else "(暂无操作审计记录)"


def stats(root: str | Path) -> dict:
    """聚合审计记录：按日/周按月归、按动作类型统计、高危操作筛选，供可视化报表。"""
    f = Path(root).resolve() / ".hs" / "audit.jsonl"
    records: list[dict] = []
    if f.is_file():
        for ln in f.read_text(encoding="utf-8", errors="ignore").strip().splitlines():
            try:
                records.append(json.loads(ln))
            except Exception:  # noqa: BLE001
                continue

    def _week_key(date: str) -> int:
        try:
            return time.strptime(date, "%Y-%m-%d").tm_wday  # 0=周一
        except Exception:  # noqa: BLE001
            return 0

    by_day: dict[str, int] = {}
    by_action: dict[str, int] = {}
    by_week: dict[str, int] = {}
    for r in records:
        ts = r.get("ts") or ""
        by_day[ts[:10]] = by_day.get(ts[:10], 0) + 1
        wk = _week_key(ts[:10])
        week_label = f"{ts[:10]} (周{wk})"
        by_week[week_label] = by_week.get(week_label, 0) + 1
        act = (r.get("action") or "unknown").replace("tool_", "")
        by_action[act] = by_action.get(act, 0) + 1
    # 高危动作：删除/写文件/训练任务/技能调用等
    _HIGH_ACTIONS = ("delete", "write", "run", "train")
    high = {k: v for k, v in by_action.items()
            if any(h in k for h in _HIGH_ACTIONS)}
    return {
        "total": len(records),
        "by_day": _desc(by_day),
        "by_week": _desc(by_week),
        "by_action": _desc(by_action),
        "high_risk": _desc(high),
    }


def _desc(d: dict) -> dict:
    """按值降序排序，便于报表/前端展示。"""
    return dict(sorted(d.items(), key=lambda kv: kv[1], reverse=True))