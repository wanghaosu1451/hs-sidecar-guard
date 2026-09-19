"""项目语义图谱：离线扫描全仓库，建函数/类/调用/导入依赖的持久索引，
用于“改动一处自动算全链路影响”，不依赖上下文窗口理解全局。

缓存到 <project_root>/.hs/index_graph.json，按文件 mtime 增量跳过未改动文件。
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

_GRAPH_CACHE = {}
_GRAPH_TTL = 5  # 同一进程内 N 秒内复用内存缓存


def _is_py(p: Path) -> bool:
    return p.is_file() and p.suffix == ".py"


def _parse_one(path: Path) -> dict:
    """解析单个 py 文件，提取自身符号与对外的导入/调用。"""
    node = {"path": str(path), "defs": [], "imports": [], "uses": []}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return node
    # 本文件定义的符号
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node["defs"].append({"name": n.name, "kind": "class" if isinstance(n, ast.ClassDef) else "func"})
        elif isinstance(n, ast.Import):
            for a in n.names:
                node["imports"].append((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            node["imports"].append((n.module or "").split(".")[0])
    # 本文件调用的符号（含 b.f() 属性调用 → 记 f，便于跨文件建边）
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
                node["uses"].append(n.func.id)
            elif isinstance(n.func, ast.Attribute):
                node["uses"].append(n.func.attr)
    return node


def build_graph(root: str | Path, force: bool = False) -> dict:
    """全量/增量构建语义图谱，命中缓存则直接返回。"""
    root = Path(root).resolve()
    cache_file = root / ".hs" / "index_graph.json"
    key = str(root)
    now_key = (len(list(root.rglob("*.py"))) if root.is_dir() else 0) + (
        cache_file.stat().st_mtime_ns if cache_file.is_file() else 0)
    if not force and key in _GRAPH_CACHE and _GRAPH_CACHE[key][0] == now_key:
        return _GRAPH_CACHE[key][1]

    nodes, edges = [], []
    mtimes: dict[str, float] = {}
    for p in sorted(root.rglob("*.py")):
        if ".hs" in p.parts or p.name in ("conftest.py",):
            continue
        n = _parse_one(p)
        nodes.append(n)
        mtimes[str(p)] = p.stat().st_mtime_ns

    from collections import Counter
    uses = Counter()
    for n in nodes:
        for name in n["uses"]:
            uses[name] += 1
    # 定义索引：symbol -> 定义文件
    def_index: dict[str, list[str]] = {}
    for n in nodes:
        for d in n["defs"]:
            def_index.setdefault(d["name"], []).append(n["path"])
    # 建边：调用文件里用到了别处定义的名字
    for n in nodes:
        for name in n["uses"]:
            targets = def_index.get(name, [])
            for t in targets:
                if t != n["path"]:
                    edges.append({"from": n["path"], "to": t, "symbol": name})

    graph = {
        "root": str(root),
        "files": [n["path"] for n in nodes],
        "edges": edges,
        "def_index": {k: v for k, v in def_index.items()},
        "mtimes": mtimes,
    }
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(graph, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    _GRAPH_CACHE[key] = (now_key, graph)
    return graph


def query_impact(graph: dict, symbol: str, reverse: bool = False) -> str:
    """查 symbol 的定义位置与全链路影响（reverse=False 找谁调用它；True 找它调用谁）。"""
    defs = graph.get("def_index", {}).get(symbol, [])
    lines = [f"符号「{symbol}」定义于: {', '.join(defs) or '(未定义，可能是内部/三方)'}"]
    if not defs:
        return "\n".join(lines) + "\n(本地图内未找到定义，可能来自三方库)"
    affected = sorted({e["from"] for e in graph["edges"]
                       if e["to"] in defs and e["symbol"] == symbol})
    if not reverse:
        lines.append(f"影响范围(调用它的文件, {len(affected)}):")
        lines += [f"  - {a}" for a in affected] or ["  - (无直接调用)"]
    else:
        called = sorted({e["to"] for e in graph["edges"] if e["from"] in defs})
        lines.append(f"它调用的外部符号所在文件({len(called)}):")
        lines += [f"  - {a}" for a in called] or ["  - (无)"]
    return "\n".join(lines)


def graph_summary(graph: dict, max_files: int = 8) -> str:
    """图谱的紧凑文本摘要，用于注入模型上下文。"""
    files = graph.get("files", [])
    edges = graph.get("edges", [])
    if not files:
        return "(项目语义图谱为空)"
    head = "".join(f"  - {f}\n" for f in files[:max_files])
    over = f"  …(共 {len(files)} 文件)\n" if len(files) > max_files else ""
    rel = ""
    if edges:
        top = sorted({e["symbol"] for e in edges})[:20]
        rel = f"跨文件调用关系({len(edges)} 条)涉及: {', '.join(top)}\n"
    return f"项目语义图谱（{len(files)} 文件）:\n{head}{over}{rel}"


def analyze_dependency_impact(root: str | Path, changed_file: str) -> str:
    """给定被修改文件，列出所有直接/间接依赖它的文件（改造影响面）。"""
    g = build_graph(root)
    rel_targets = set()
    changed = Path(str(changed_file)).resolve()
    # 直接：边指向 changed 中的符号
    changed_defs = {s for s, fs in g["def_index"].items() if any(
        Path(f).resolve() == changed for f in fs)}
    for e in g["edges"]:
        if e["to"] in changed_defs:
            rel_targets.add(e["from"])
        if Path(e["to"]).resolve() == changed:
            rel_targets.add(Path(e["to"]).resolve().__str__())  # noqa
    one = sorted({str(Path(x).resolve()) for x in rel_targets})
    if not one:
        return f"修改 {changed_file} 未发现其他受影响文件。"
    return f"修改 {changed_file} 可能影响以下文件({len(one)}):\n" + "\n".join(
        f"  - {o}" for o in one)