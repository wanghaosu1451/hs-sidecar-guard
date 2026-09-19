"""工作区管理：根目录、文件扫描、打开/保存。"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

IGNORE_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules",
               ".idea", ".vscode", "dist", "build", ".trae"}
TEXT_EXT = {".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".md", ".txt",
            ".html", ".css", ".html", ".yaml", ".yml", ".toml", ".ini",
            ".xml", ".csv", ".rs", ".go", ".java", ".c", ".cpp", ".h",
            ".sh", ".bat", ".ps1", ".sql", ".vue", ".svelte"}


class Workspace:
    def __init__(self, root: str | Path | None = None):
        self.root: Path | None = Path(root) if root else None

    def open(self, path: str | Path) -> None:
        self.root = Path(path).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(str(self.root))

    def iter_files(self) -> Iterator[Path]:
        """遍历工作区文本文件（跳过忽略目录）。"""
        if self.root is None:
            return
        for p in self.root.rglob("*"):
            if p.is_dir():
                continue
            if any(part in IGNORE_DIRS for part in p.parts):
                continue
            if p.suffix in TEXT_EXT:
                yield p

    def rel(self, path: str | Path) -> str:
        if self.root is None:
            return str(path)
        try:
            return str(Path(path).resolve().relative_to(self.root))
        except ValueError:
            return str(path)

    def resolve(self, rel: str) -> Path:
        return (self.root / rel).resolve() if self.root else Path(rel).resolve()

    def read(self, path: str | Path, errors: str = "replace") -> str:
        p = Path(path)
        if not p.exists():
            return ""
        return p.read_text(encoding="utf-8", errors=errors)

    def write(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


# ============================ 全仓上下文索引（喂给 AI，避免只看片段） ============================
_SYMBOL_EXTS = {".py": "python", ".js": "js", ".ts": "ts", ".tsx": "tsx"}


def build_repo_context(ws: Workspace, max_files: int = 80,
                       max_depth: int = 4) -> str:
    """生成工作区代码索引：目录树 + 各文件顶层符号。供注入系统提示。"""
    if ws.root is None or not ws.root.is_dir():
        return ""
    lines: list[str] = []
    root = ws.root

    # 1) 目录树（裁剪深度/忽略目录）
    def walk(d: Path, prefix: str, depth: int) -> None:
        try:
            entries = [p for p in d.iterdir()
                       if p.name not in IGNORE_DIRS and not p.name.startswith(".")]
        except OSError:
            return
        entries.sort(key=lambda p: (not p.is_dir(), p.name.lower()))
        for i, p in enumerate(entries):
            last = i == len(entries) - 1
            branch = "└─ " if last else "├─ "
            lines.append(prefix + branch + p.name + ("/" if p.is_dir() else ""))
            if p.is_dir() and depth < max_depth:
                walk(p, prefix + ("   " if last else "│  "), depth + 1)

    lines.append("# 目录结构")
    try:
        walk(root, "", 0)
    except Exception:  # noqa: BLE001
        pass

    # 2) 符号索引（只对源码文件采样，控制体积）
    symbols = []
    count = 0
    for p in ws.iter_files():
        if count >= max_files:
            break
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = str(p.relative_to(root))
        sig = _symbols_for(src, p.suffix)
        if sig:
            symbols.append(f"## {rel}\n" + "\n".join(f"- {s}" for s in sig))
            count += 1

    if symbols:
        lines.append("\n# 顶层符号（函数/类）")
        lines.extend(symbols)
    return "\n".join(lines)


def _symbols_for(src: str, suffix: str) -> list[str]:
    sig: list[str] = []
    try:
        import ast
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = ", ".join(a.arg for a in node.args.args)
                sig.append(f"def {node.name}({args})")
            elif isinstance(node, ast.ClassDef):
                bases = ", ".join(ast.unparse(b) for b in node.bases)
                sig.append(f"class {node.name}({bases})" if bases else f"class {node.name}")
    except (SyntaxError, ValueError):
        # 非 Python 或语法不完整：退化为只列关键行关键词
        for ln in src.splitlines():
            s = ln.strip()
            if s.startswith(("def ", "class ", "function ", "export ")):
                sig.append(s.split("{")[0].rstrip(" (:"))
            elif s.startswith(("async def ",)):
                sig.append(s)
    return sig[:80]