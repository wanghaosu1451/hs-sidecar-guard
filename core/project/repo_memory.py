"""持久化项目记忆：跨会话保留架构决策、编码规范、历史修改记录。

不依赖上下文窗口——即使早期对话被摘要压缩，这里的决策仍可在任意会话被读取，
多任务共享同一份项目认知，避免多会话输出冲突。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

_DEFAULT_NOTES: list[dict] = []


def _file(root: str | Path) -> Path:
    return Path(root).resolve() / ".hs" / "project_memory.json"


def _load(root: str | Path) -> list[dict]:
    f = _file(root)
    if not f.is_file():
        return []
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []


def remember(root: str | Path, topic: str, body: str, kind: str = "note") -> str:
    """记录一条项目记忆（架构决策/规范/历史），跨会话保留。"""
    notes = _load(root)
    notes.append({
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "topic": (topic or "").strip(),
        "body": (body or "").strip(),
        "kind": kind or "note",
    })
    _file(root).parent.mkdir(parents=True, exist_ok=True)
    _file(root).write_text(json.dumps(notes, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    return f"已存入项目记忆[{kind}]: {topic or body[:40]}"


def _tokens(text: str) -> set[str]:
    """把文本切成小写词条，用于关键词重叠打分。"""
    return {t for t in re.split(r"[^0-9a-zA-Z\u4e00-\u9fff]+", (text or "").lower()) if t}


def _note_score(query_tokens: set[str], note: dict) -> tuple[int, int]:
    """按关键词子串命中打分：(score, recency_idx)。topic 命中权重×2，body ×1。

    用子串包含而非整词相等——兼容无空格分词的中文（如"数据库"命中"数据库迁移方案"）。
    """
    topic_txt = (note.get("topic") or "").lower()
    body_txt = (note.get("body") or "").lower()
    score = 0
    for q in query_tokens:
        if q and q in topic_txt:
            score += 2
        elif q and q in body_txt:
            score += 1
    return score, 0


def read(root: str | Path, topic: str = "", limit: int = 10) -> str:
    """读取项目记忆；给 topic 时按关键词重叠打分取最相关 limit 条，否则取最近 limit 条。"""
    notes = _load(root)
    if not notes:
        return "(项目暂无持久记忆，可先让智能体调用 remember 记录架构决策/规范)"
    if topic:
        query = _tokens(topic)
        # 按分数降序、同分取索引更新（enumerate 索引近似时间序）取最相关 limit 条
        ordered = sorted(enumerate(notes),
                         key=lambda it: (_note_score(query, it[1]), it[0]),
                         reverse=True)
        top = [n for _, n in ordered[:limit]]
        lines = [f"- [{n.get('ts')}][{n.get('kind','note')}] "
                 f"{n.get('topic') or ''}｜{n.get('body')}" for n in top]
    else:
        recent = notes[-limit:]
        lines = [f"- [{n.get('ts')}][{n.get('kind','note')}] "
                 f"{n.get('topic') or ''}｜{n.get('body')}" for n in reversed(recent)]
    return "项目持久记忆（跨会话共享）:\n" + "\n".join(lines)


def is_empty(root: str | Path) -> bool:
    return not _load(root)