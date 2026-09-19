"""文件快照 / 回滚：每次覆盖写文件前自动备份上一版本，支持实时 diff 与一键回滚。

“可逆操作直接做，不可逆操作可回滚”的落点——AI 写坏文件可无损还原，
避免“写错就丢了原版”。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

PREFIX = "snap_"


def _dir(root: str | Path) -> Path:
    return Path(root).resolve() / ".hs" / "snapshots"


def backup(root: str | Path, rel_path: str, content: str) -> str | None:
    """记录 rel_path 在写新内容前的上一版本。content 为将要写入的新内容。
    返回快照名；若这是首次写入（无旧版）则返回 None 不建快照。"""
    root = Path(root).resolve()
    target = (root / rel_path).resolve()
    if not target.is_file():
        return None  # 首次创建，无需快照
    old = target.read_text(encoding="utf-8", errors="ignore")
    if old == content:
        return None  # 内容未变，不建快照
    d = _dir(root)
    d.mkdir(parents=True, exist_ok=True)
    name = f"{PREFIX}{int(time.time() * 1000)}.json"
    (d / name).write_text(json.dumps({
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "path": rel_path, "content": old,
    }, ensure_ascii=False), encoding="utf-8")
    return name


def list_snapshots(root: str | Path, rel_path: str = "") -> str:
    d = _dir(root)
    if not d.is_dir():
        return "(暂无快照)"
    snaps = []
    for f in sorted(d.glob(PREFIX + "*.json"), key=lambda p: p.stat().st_mtime):
        try:
            e = json.loads(f.read_text(encoding="utf-8"))
            p = e.get("path", "")
            if rel_path and p != rel_path:
                continue
            snaps.append(f"[{e.get('ts')}] {p}  <-  {f.name}")
        except Exception:  # noqa: BLE001
            continue
    return "\n".join(snaps) if snaps else "(该文件暂无快照)"


def rollback(root: str | Path, snapshot_name: str) -> str:
    """按快照名把文件还原到上一版本。"""
    root = Path(root).resolve()
    d = _dir(root)
    f = (d / snapshot_name)
    if not f.is_file():
        return f"错误：快照不存在 {snapshot_name}（可用 snapshot_list 查看）"
    try:
        e = json.loads(f.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return f"错误：快照损坏 - {exc}"
    target = (root / e["path"]).resolve()
    if not target.is_relative_to(root):
        return "错误：快照路径越界，拒绝回滚"
    target.write_text(e["content"], encoding="utf-8")
    f.unlink(missing_ok=True)  # 回滚后作废该快照
    return f"✅ 已回滚 {e['path']}（快照 {snapshot_name} 已作废）"


def diff(root: str | Path, rel_path: str) -> str:
    """对比当前文件与最近一次快照，输出类 unified diff 的差异。"""
    root = Path(root).resolve()
    target = (root / rel_path).resolve()
    if not target.is_file():
        return f"错误：文件不存在 {rel_path}"
    old = None
    # 取最近一个属于该文件的快照
    for p in reversed(sorted(_dir(root).glob(PREFIX + "*.json"),
                             key=lambda p: p.stat().st_mtime)):
        try:
            e = json.loads(p.read_text(encoding="utf-8"))
            if e.get("path") == rel_path:
                old = e["content"]
                break
        except Exception:  # noqa: BLE001
            continue
    if old is None:
        return "(该文件暂无历史快照，无法 diff)"
    new = target.read_text(encoding="utf-8", errors="ignore")
    if old == new:
        return f"（{rel_path} 与上版无差异）"
    old_l, new_l = old.splitlines(), new.splitlines()
    out = [f"--- {rel_path} (上次快照)", f"+++ {rel_path} (当前)"]
    o, n = 0, 0
    while o < len(old_l) or n < len(new_l):
        if o < len(old_l) and n < len(new_l) and old_l[o] == new_l[n]:
            o, n = o + 1, n + 1
        elif n < len(new_l) and (o >= len(old_l) or new_l[n] != old_l[o]):
            out.append("  + " + new_l[n]); n += 1
        else:
            out.append("  - " + old_l[o]); o += 1
    return "\n".join(out[:200]) + ("\n…(截断)" if (len(out) > 200) else "")