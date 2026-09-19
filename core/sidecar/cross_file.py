"""Cross-file 语义依赖检测：规则级 trigger_files 精确匹配 + embedding 语义验证。

三层 pipeline：
1. **文件存在性** —— trigger_files 提到的关联文件在项目里是否真实存在
2. **语义关联** —— embed changed file 与 trigger file 的函数/代码片段，计算 cosine
3. **模式描述** —— cross_file_patterns.json 的 description 给出具体修复建议

设计约束：
- 每次 project_write 工具调用时触发，O(k) 开销（k=关联文件数，通常 <10）
- 用已有的 gte-small-zh embedder，不新增任何依赖
- 纯函数式，不修改任何文件，sidecar 只读检查

作者：hs-sidecar-guard sidecar pipeline
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .knowledge import load_knowledge, query_cross_file_knowledge
from .embedding import get_embedder


# —— 代码片段提取器（从单个文件抽函数/类签名 + 常量赋值）——

_FUNC_CLASS_RE = re.compile(
    r'^(?:class|def)\s+([A-Za-z_][\w]*)',
    re.MULTILINE | re.DOTALL,
)
_ASSIGN_RE = re.compile(
    r'^([A-Z][A-Z0-9_]*)\s*=\s*(.+)$',
    re.MULTILINE,
)


def _extract_signatures(source: str, max_chunks: int = 10) -> list[str]:
    """从源码文件里提取函数/类签名 + 顶层常量赋值作为语义 chunk。"""
    chunks = []
    for m in _FUNC_CLASS_RE.finditer(source):
        # 取签名 + 后续 2 行
        start = m.start()
        end = source.find("\n", start + 1)
        if end == -1:
            end = len(source)
        header = source[start:end].strip()
        # 取简短描述（如果有 docstring 的第一行）
        doc_match = re.search(
            r'"""(.*?)(?:\n.*?)?"""|\'\'\'(.*?)(?:\n.*?)?\'\'',
            source[end:end + 200], re.DOTALL)
        if doc_match:
            desc = (doc_match.group(1) or doc_match.group(2) or "").strip()
            if desc:
                header = f"{header}  # {desc[:60]}"
        chunks.append(header)

    for m in _ASSIGN_RE.finditer(source):
        name = m.group(1)
        val = m.group(2).strip()[:60]
        if val and not val.startswith("#"):
            chunks.append(f"const {name} = {val}")

    if not chunks:
        # 回退：取前 3 行非空非注释内容
        for line in source.splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                chunks.append(s[:80])
                if len(chunks) >= max_chunks:
                    break

    return chunks[:max_chunks]


def _read_file_utf8(p: Path) -> str | None:
    try:
        return p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


# —— 主检测函数 ——

def detect_cross_file_dependency(
    changed_path: str | Path,
    project_root: str | Path = ".",
    top_k_patterns: int = 2,
    semantic_cosine_threshold: float = 0.65,
) -> list[dict]:
    """检测单个文件修改是否需要同步修改其他文件。

    返回：[{"pattern": str, "trigger_file": str, "cosine": float,
            "semantic_hit": bool, "description": str}, ...]
    """
    changed_path = Path(changed_path)
    project_root = Path(project_root)
    changed_name = changed_path.name.lower()

    # 1) 规则级：cross_file_patterns.json 里的 trigger_files 精确匹配
    kb = load_knowledge("cross_file_patterns")
    if not kb:
        return []

    embedder = get_embedder()
    results = []

    for pattern in kb:
        triggers = [t.lower() for t in pattern.get("trigger_files", [])]
        # 该 pattern 是否与当前修改的文件有关？
        matched = False
        matched_trigger = None
        for t in triggers:
            if t == changed_name or changed_name.endswith("/" + t):
                matched = True
                matched_trigger = t
                break
            # 也可能改的是 trigger 文件里提到的其他文件
            # 但 trigger 文件本身还存在吗？
            candidate = (project_root / t).resolve()
            if candidate.exists() and candidate.is_file():
                # 跳过——changed 文件不是 trigger 本身，而是 trigger 关联的文件
                # 这种情况我们要提醒用户"改了 X 记得同步改 trigger"
                pass

        if not matched:
            # 换方向：changed 文件是关联文件，但 trigger 文件还存在吗？
            triggers_exist = [
                t for t in triggers
                if (project_root / t).exists()
            ]
            if not triggers_exist:
                continue
            # 检查 changed_name 是否在 triggers 里 OR 是 trigger 关联的文件
            # 如果 trigger_files 里有多个文件（如 config.py + server.py）
            # 且我改了其中一个，就要提醒另一个
            if changed_name not in triggers:
                continue
            matched_trigger = changed_name

        # 找到了相关 pattern，现在做语义验证
        desc = pattern.get("description", "")
        semantic_hit = False
        cosine = 0.0

        # embed changed file
        full_changed = project_root / changed_path
        changed_source = _read_file_utf8(full_changed)
        if changed_source:
            changed_chunks = _extract_signatures(changed_source)
            changed_vec = embedder.encode(
                [" ".join(changed_chunks) if changed_chunks else changed_source[:500]]
            )[0]

            # embed trigger_files 里的其他文件
            for t in triggers:
                if t == matched_trigger:
                    continue  # 跳过自己
                t_path = (project_root / t).resolve()
                t_source = _read_file_utf8(t_path)
                if not t_source:
                    continue
                t_chunks = _extract_signatures(t_source)
                t_vec = embedder.encode(
                    [" ".join(t_chunks) if t_chunks else t_source[:500]]
                )[0]
                cosine = float((changed_vec * t_vec).sum())
                if cosine >= semantic_cosine_threshold:
                    semantic_hit = True
                    break

        results.append({
            "pattern": pattern.get("pattern", ""),
            "trigger_file": t if semantic_hit else matched_trigger,
            "cosine": round(cosine, 3),
            "semantic_hit": semantic_hit,
            "description": desc,
        })

    # 按 cosine 排序，取 top_k
    results.sort(key=lambda r: r["cosine"], reverse=True)
    return results[:top_k_patterns]
