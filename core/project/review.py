"""静态代码审查 + 漏洞扫描（离线，零出域）。

- code_review：命名/异常/未用导入/裸 except/print 等风格与隐患，输出可修复建议。
- vuln_scan：SQL 注入、eval/exec、密钥硬编码等隐性安全问题，并扫描第三方依赖。
"""
from __future__ import annotations

import ast
import re
from pathlib import Path


def _py_files(root: str | Path, single: str | None = None):
    root = Path(root)
    if single:
        p = root / str(single).lstrip("/")
        return [p] if p.is_file() else []
    return [p for p in root.rglob("*.py") if ".hs" not in p.parts]


def code_review(root: str | Path, path: str | None = None) -> str:
    """对项目（default）或单个 py 文件做静态审查，返回问题清单。"""
    root = Path(root)
    files = _py_files(root, path)
    if not files:
        return f"(未找到可审查的 Python 文件: {path or root})"
    issues: list[str] = []
    for p in files:
        rel = p.relative_to(root)
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError as e:
            issues.append(f"[{rel}] 语法错误: {e}")
            continue
        imported = set()
        used = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imported.update(a.asname or a.name.split(".")[0] for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                imported.add((n.module or "").split(".")[0])
            elif isinstance(n, ast.Name):
                used.add(n.id)
            elif isinstance(n, ast.FunctionDef):
                if not re.match(r"^[a-z][a-z0-9_]*$", n.name):
                    issues.append(f"[{rel}] 函数命名应 snake_case: {n.name}")
        unused = sorted(imported - used)
        if unused:
            issues.append(f"[{rel}] 未使用的导入: {', '.join(unused)}")
        for n in ast.walk(tree):
            if isinstance(n, ast.ExceptHandler) and not (n.type or n.name):
                issues.append(f"[{rel}] 裸 except 吞掉所有异常（应捕获具体类型）")
            if isinstance(n, ast.Call) and isinstance(n.func,
                    ast.Name) and n.func.id in ("eval", "exec", "compile"):
                issues.append(f"[{rel}] 使用 {n.func.id}() 可能造成代码注入，需校验输入")
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if len(line) > 120:
                issues.append(f"[{rel}:{i}] 行过长({len(line)}>120)，建议拆分")
    if not issues:
        return f"✅ 审查通过，未发现问题: {path or root}"
    return f"发现 {len(issues)} 处待处理:\n" + "\n".join(f"  {x}" for x in issues)


# 硬编码密钥 / 危险调用识别
_SECRET_LIT = re.compile(r"(password|passwd|secret|token|api[_-]?key)\s*=\s*['\"][^'\"]+['\"]",
                         re.IGNORECASE)
_SQL_CONCAT = re.compile(
    r"execute\s*\([^)]*f['\"]|cursor\.executemany?\s*\(\s*f['\"]|"
    r"format\s*\([^)]*\)\s*[;,]?\s*#\s*sql|['\"].*SELECT.*\{\}|INSERT\s+.*\{\}",
    re.IGNORECASE)


def vuln_scan(root: str | Path, path: str | None = None) -> str:
    """扫描代码与依赖的安全问题，输出发现项与修复建议。"""
    root = Path(root)
    files = _py_files(root, path)
    findings: list[str] = []
    for p in files:
        rel = p.relative_to(root)
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if _SECRET_LIT.search(line):
                findings.append(f"[{rel}:{i}] 疑似密钥/口令硬编码 → 建议改用环境变量")
            if _SQL_CONCAT.search(line) or ("SELECT " in line and "{" in line and "execute" in text.lower()):
                findings.append(f"[{rel}:{i}] 疑似 SQL 拼接注入 → 建议用参数化查询")
            if re.search(r"\b(os\.system|subprocess\.call|shell=True)\b", line):
                findings.append(f"[{rel}:{i}] shell 调用需校验输入，避免命令注入")
    # 第三方依赖：解析 requirements 中固死版本与已知危险动作
    req = root / "requirements.txt"
    deps = []
    if req.is_file():
        for line in req.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip() and not line.startswith("#") and re.match(r"^[A-Za-z0-9_.\-]+==", line.strip()):
                deps.append(line.strip())
    if deps:
        findings.append("检测到已锁版依赖（共 %d 个）→ 建议用 pip-audit 核查已知漏洞: %s"
                        % (len(deps), ", ".join(d.split("==")[0] for d in deps[:8])))
    if not findings:
        return f"✅ 未发现明显安全问题: {path or root}"
    return f"发现 {len(findings)} 项安全隐患:\n" + "\n".join(f"  {x}" for x in findings)