"""定时安全巡检：后台周期扫描项目的密钥/高危命令/危险文件读写/漏洞依赖，生成告警。

离线、零出域；结果以结构化 JSON 落盘（.hs/scan_last.json），高危发现追加进
.hs/scan_alerts.jsonl，供可视化报表与 CLI 使用。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

SCAN_CATEGORIES = ["secrets", "shell", "filesystem", "deps"]
_SEVERITY = {"secrets": "high", "shell": "medium", "filesystem": "high", "deps": "low"}

_SKIP_DIRS = {".hs", ".git", ".pytest_cache", "__pycache__", "node_modules",
              ".venv", "venv"}

# 高危 shell 命令（尽量精确，避免误报普通用法）
_HIGH_SHELL = [
    (r"\brm\s+-rf\s+[/~]", "rm -rf 根/家目录，可能毁机"),
    (r">\s*/dev/sda", "写裸盘设备"),
    (r"\bdd\s+if=.*\sof=/dev/sd", "dd 写盘"),
    (r"\bmkfs(\.\w+)?\s+", "格式化磁盘"),
    (r":\(\)\s*\{\s*:\|:&\s*\};", "fork 炸弹"),
    (r"chmod\s+777\s+", "放开 777 权限"),
    (r"\b(shutdown|reboot)\b", "关机/重启"),
    (r"(curl|wget).*\|.*\b(sh|bash)\b", "管道直灌 shell 执行远程脚本"),
    (r"\bDROP\s+DATABASE\b|\bTRUNCATE\b", "删除/截断数据库"),
    (r"\bgit\s+reset\s+--hard\b", "git 硬重置丢失提交"),
]
# 容器/数据库等危险文件操作
_FS_DANGEROUS = [
    (r"\bos\.remove\(|os\.unlink\(|os\.rmdir\(|shutil\.rmtree\(|shutil\.rmtree\b",
     "直接删除文件/目录"),
    (r"\.unlink\(|\.rmdir\(|\.unlink\(missing_ok", "Path 删除操作"),
    (r"\.write_bytes?.*(\.env|credentials|id_rsa|\.pem)", "向敏感凭据文件写入"),
]
# 硬编码密钥
_SECRET_LIT = re.compile(
    r"(password|passwd|secret|token|api[_-]?key|access[_-]?key)\s*[:=]\s*['\"][^'\"\s]{4,}['\"]",
    re.IGNORECASE)
_KEY_TOK = re.compile(r"(sk-|eyJ)[A-Za-z0-9_.\-]{16,}")

_TEXT_EXT = {".py", ".js", ".ts", ".tsx", ".jsx", ".vue", ".java", ".go", ".rs",
             ".c", ".cpp", ".h", ".sh", ".bat", ".ps1", ".sql", ".yaml", ".yml",
             ".json", ".toml", ".ini", ".cfg", ".conf", ".env.example"}


def _iter_text_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and ".hs" not in p.parts and not any(s in p.parts for s in _SKIP_DIRS):
            if p.suffix in _TEXT_EXT or p.name.startswith(".env"):
                yield p


def _file_kind(p: Path) -> str:
    if p.suffix in (".py", ".js", ".ts", ".jsx", ".tsx", ".vue", ".java", ".go", ".rs", ".c", ".cpp"):
        return "code"
    if p.suffix in (".sh", ".bat", ".ps1"):
        return "script"
    return "config"


def scan_project(root: str | Path, path: str | None = None) -> dict[str, Any]:
    """扫描单个文件(path)或整个项目(root)，返回结构化巡检结果。"""
    root = Path(root).resolve()
    findings: list[dict] = []
    files = []
    if path:
        _p = root / str(path).lstrip("/\\")
        files = [_p] if _p.is_file() else []
    else:
        files = list(_iter_text_files(root))
    for p in files:
        rel = p.relative_to(root).as_posix()
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            continue
        if "\x00" in text:  # 二进制跳过
            continue
        kind = _file_kind(p)
        lines = text.splitlines()
        for i, line in enumerate(lines, 1):
            if _SECRET_LIT.search(line) or bool(_KEY_TOK.search(line)):
                findings.append({"category": "secrets", "severity": "high",
                                 "file": rel, "line": i, "message": "疑似硬编码密钥/口令"})
            if kind in ("code", "script"):
                for rx, msg in _HIGH_SHELL:
                    if re.search(rx, line):
                        findings.append({"category": "shell", "severity": "medium",
                                         "file": rel, "line": i, "message": "高危 shell 命令：" + msg})
            for rx, msg in _FS_DANGEROUS:
                if re.search(rx, line):
                    findings.append({"category": "filesystem", "severity": "high",
                                     "file": rel, "line": i, "message": "危险文件操作：" + msg})
    # 依赖（去重、按严重度计数）
    deps = _scan_deps(root)
    findings += deps
    counts = {c: 0 for c in SCAN_CATEGORIES}
    for f in findings:
        counts[f["category"]] = counts.get(f["category"], 0) + 1
    result = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": len(files),
        "counts": counts,
        "total": len(findings),
        "findings": findings[:200],
    }
    return result


def _scan_deps(root: Path) -> list[dict]:
    out: list[dict] = []
    req = root / "requirements.txt"
    if req.is_file():
        for line in req.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "==" not in s:
                continue
            # 空版本（== ""）或通配版本属危险依赖配置
            if re.match(r"^[A-Za-z0-9_.\-]+==[\"']?$", s) or "*" in s or re.search(r"==\s*0\.0\.0", s):
                out.append({"category": "deps", "severity": "low", "file": "requirements.txt",
                            "line": 0, "message": f"危险/未锁依赖：{s}"})
    pkg = root / "package.json"
    if pkg.is_file():
        import json as _json
        try:
            d = _json.loads(pkg.read_text(encoding="utf-8", errors="ignore"))
            for section in ("dependencies", "devDependencies"):
                for name, ver in (d.get(section) or {}).items():
                    if "*" in str(ver) or str(ver) == "":
                        out.append({"category": "deps", "severity": "low", "file": "package.json",
                                    "line": 0, "message": f"未锁依赖：{name}@{ver}"})
        except Exception:  # noqa: BLE001
            pass
    return out


def _hs_dir(root: Path) -> Path:
    return root / ".hs"


def persist(result: dict, root: str | Path) -> Path:
    """落盘最新巡检结果 + 追加高危告警日志，返回告警条数。"""
    root = Path(root).resolve()
    _hs_dir(root).mkdir(parents=True, exist_ok=True)
    last = _hs_dir(root) / "scan_last.json"
    last.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    alert_file = _hs_dir(root) / "scan_alerts.jsonl"
    new_alerts = [f for f in result["findings"] if f.get("severity") == "high"]
    with alert_file.open("a", encoding="utf-8") as fh:
        for a in new_alerts:
            fh.write(json.dumps({"ts": result["ts"], **a}, ensure_ascii=False) + "\n")
    return last


def scan_and_persist(root: str | Path) -> dict:
    """执行巡检并落盘；返回结果（供调度与接口复用）。"""
    result = scan_project(root)
    persist(result, root)
    return result


def summary(root: str | Path) -> str:
    """人类可读的巡检摘要（CLI / 审计报表用）。"""
    result = scan_and_persist(root)
    c = result["counts"]
    line = (f"巡检时间 {result['ts']} · 扫描 {result['files']} 个文件 · "
            f"共 {result['total']} 项：密钥{c['secrets']} / Shell{c['shell']} / "
            f"文件操作{c['filesystem']} / 依赖{c['deps']}")
    if not result["findings"]:
        return line + "\n✅ 未发现明显安全问题"
    rows = [f"  [{f['severity']}] {f['file']}:{f['line']} {f['message']}"
            for f in result["findings"][:40]]
    return line + "\n发现:\n" + "\n".join(rows)