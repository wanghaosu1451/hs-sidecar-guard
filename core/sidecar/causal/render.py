"""终端字符树渲染（Rich 可选，没有就降级纯文本）。"""
from __future__ import annotations


def _has_rich() -> bool:
    try:
        import rich  # noqa: F401
        return True
    except ImportError:
        return False


def render_tree(root: str, deps: dict[str, list[str]],
                 color_branches: bool = True) -> str:
    """渲染"文件 → 它依赖的文件"为字符树，返回字符串。"""
    lines = [root]
    visited = set()

    def walk(node: str, depth: int) -> None:
        if node in visited:
            lines.append("  " * depth + "└─ (循环)")
            return
        visited.add(node)
        children = deps.get(node, [])
        for c in children:
            lines.append("  " * depth + "├─ " + c)
            walk(c, depth + 1)

    walk(root, 1)
    tree = "\n".join(lines)
    if color_branches and _has_rich():
        try:
            from rich.console import Console
            c = Console()
            c.print(tree, style="dim")
        except Exception:
            print(tree)
    else:
        print(tree)
    return tree


def render_affects_tree(target: str, affected: list[str]) -> str:
    """给 Agent 输出"改 target 会影响哪些文件"的精简树。"""
    deps = {target: affected}
    return render_tree(target, deps)