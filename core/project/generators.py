"""工程全流程生成器：自动产出单元测试骨架、部署配置、接口/结构文档。

生成的产物直接写入项目目录，与“编码 - 测试 - 文档 - 部署”闭环对齐。
"""
from __future__ import annotations

import ast
from pathlib import Path


def _load_module(root: str | Path, module: str) -> Path:
    p = Path(root).resolve() / str(module).lstrip("/")
    if not p.is_file():
        raise FileNotFoundError(f"模块不存在: {module}")
    return p


def gen_tests(root: str | Path, module: str, out: str = "") -> str:
    """为单个 .py 模块生成 pytest 单元测试骨架（含边界/异常用例）。"""
    try:
        p = _load_module(root, module)
    except FileNotFoundError as e:
        return f"错误：{e}"
    try:
        tree = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError as e:
        return f"错误：{module} 无法解析 - {e}"
    funcs = [n.name for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]
    if not funcs:
        return f"错误：{module} 中没有可测的公开函数"
    mod_name = p.stem
    out_name = out or f"test_{mod_name}.py"
    lines = [f"# 由 HS 自动生成：{mod_name} 单元测试骨架（覆盖正常/边界/异常）",
             f"import pytest", f"import {mod_name}", ""]
    for f in funcs:
        lines += [
            f"def test_{mod_name}_{f}_normal():",
            f"    # TODO: 按 {mod_name}.{f} 的输入替换样例",
            f"    # ret = {mod_name}.{f}('样例')",
            f"    # assert ret is not None",
            f"    pass", "",
            f"def test_{mod_name}_{f}_boundary():",
            f"    # TODO: 覆盖空值/极值/None 等边界",
            f"    pass", "",
            f"def test_{mod_name}_{f}_error():",
            f"    # TODO: 覆盖非法入参期望抛出的异常",
            f"    with pytest.raises(Exception):",
            f"        pass", "",
        ]
    target = Path(root).resolve() / out_name
    target.write_text("\n".join(lines), encoding="utf-8")
    return f"✅ 已生成 {target}（覆盖 {len(funcs)} 个函数）"


def gen_docker(root: str | Path) -> str:
    """根据项目特征生成 Dockerfile 与 .dockerignore。"""
    root = Path(root).resolve()
    has_py = any(root.rglob("*.py"))
    if not has_py:
        return "错误：未检测到 Python 工程，暂不支持自动生成部署配置"
    req = root / "requirements.txt"
    install = "RUN pip install --no-cache-dir -r requirements.txt" if req.is_file() else ""
    docker = f"""# 由 HS 自动生成
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
{install}
COPY . .
EXPOSE 8000
CMD ["python", "app.py"]
"""
    (root / "Dockerfile").write_text(docker, encoding="utf-8")
    (root / ".dockerignore").write_text(".hs\n__pycache__\n*.pyc\n.venv\n", encoding="utf-8")
    return f"✅ 已生成 Dockerfile + .dockerignore（推荐: docker build -t hs-app .）"


def gen_doc(root: str | Path) -> str:
    """基于语义图谱生成接口/结构概览文档 README.auto.md。"""
    from core.project.graph import build_graph
    root = Path(root).resolve()
    g = build_graph(root)
    files = g.get("files", [])
    lines = ["# 项目接口 / 结构概览（HS 自动生成，跟随代码更新）", ""]
    if not files:
        return "错误：项目内没有 Python 文件，无法生成结构文档"
    lines += [f"- 文件数: {len(files)}；跨文件调用关系: {len(g.get('edges', []))} 条", ""]
    lines.append("## 公开符号（函数/类）")
    defs = g.get("def_index", {})
    for name, locs in list(defs.items())[:120]:
        lines.append(f"- `{name}` 定义于 {', '.join(locs)}")
    lines += ["", "## 说明", "- 本文件由 `gen_doc` 工具生成；重新生成可覆盖以保持同步。"]
    (root / "README.auto.md").write_text("\n".join(lines), encoding="utf-8")
    return f"✅ 已生成 README.auto.md（{len(defs)} 个符号）"