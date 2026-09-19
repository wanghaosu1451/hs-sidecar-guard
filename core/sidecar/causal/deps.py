"""语法级多文件依赖图：import/require/from、同符号名引用（变量/函数/类）。

不做真正的语义理解，靠模式匹配抓**显性**跨文件依赖——够拦截"改 A 文件会崩 B 文件"的初级错误。
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


# 按扩展名匹配 import 语句的正则
_IMPORT_PATTERNS = {
    ".py": [
        re.compile(r"^\s*import\s+([\w\.]+)"),
        re.compile(r"^\s*from\s+([\w\.]+)\s+import\s+([\w\*, ]+)"),
    ],
    ".js": [
        re.compile(r'(?:import|require)\s*\(?\s*["\']([\w\.\/-]+)["\']'),
    ],
    ".ts": [
        re.compile(r'(?:import|require)\s*\(?\s*["\']([\w\.\/-]+)["\']'),
    ],
}


@dataclass
class CausalGraph:
    """项目文件间的依赖图。"""

    root: str
    files: set[str] = field(default_factory=set)          # 所有文件绝对路径
    imports: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    reverse: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def affects(self, target_file: str) -> set[str]:
        """改 target_file 后可能受影响的文件（反向依赖）。"""
        return self.reverse.get(target_file, set())

    def who_affects_me(self, target_file: str) -> set[str]:
        """target_file 依赖了谁（正向依赖）。"""
        return self.imports.get(target_file, set())

    def side_effects_of_write(self, target_file: str,
                               changed_symbols: list[str] | None = None) -> dict:
        """高阶 API：给 agent 输出"改 A 文件 → 会影响 B/C"，用于修改前预判。"""
        affected = self.affects(target_file)
        if changed_symbols:
            # 语义级细化（待 1.5B：这里按符号名在 affected 文件里 grep 做近似）
            sym = set(changed_symbols)
            refined = set()
            for f in affected:
                try:
                    content = Path(f).read_text(encoding="utf-8",
                                                errors="ignore")
                except Exception:
                    continue
                if any(s in content for s in sym):
                    refined.add(f)
            if refined:
                affected = refined
        return {"changed": target_file, "affects": sorted(affected)}


def build_graph(project_root: str | Path,
                 include_ext: tuple[str, ...] = (".py", ".js", ".ts")) -> CausalGraph:
    """扫描项目根，构建 CausalGraph。"""
    root = Path(project_root).resolve()
    g = CausalGraph(root=str(root))

    # 收集文件
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in include_ext:
            continue
        # 跳过隐藏目录 / venv / node_modules / __pycache__
        rel = p.relative_to(root).as_posix()
        skip_dirs = (".git", "node_modules", "__pycache__", ".venv", "venv", "env",
                     ".hs_snapshots", ".hs_checkpoints")
        if any(part in skip_dirs for part in p.parts):
            continue
        g.files.add(str(p))

    # 解析 imports
    files_by_stem: dict[str, str] = {}  # stem → 绝对路径（同名冲突时后者覆盖）
    for f in g.files:
        files_by_stem[Path(f).stem] = f

    for f in g.files:
        ext = Path(f).suffix.lower()
        pats = _IMPORT_PATTERNS.get(ext, [])
        try:
            content = Path(f).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pat in pats:
            for m in pat.finditer(content):
                mod = m.group(1).split(".")[0]
                target = files_by_stem.get(mod)
                if target and target != f:
                    g.imports[f].add(target)
                    g.reverse[target].add(f)

    return g