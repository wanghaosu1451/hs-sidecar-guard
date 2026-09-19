"""GitHub 开源 MCP 插件安装器。

把用户指定的 GitHub 仓库（https://github.com/o/r、裸 o/r、或 git@…）clone 到本机，
按 MCP 服务器登记（deny-first，approved 默认 False），由 operator 批准后方可被模型调用。
只做登记不做启用 → 安全时序：“先登记 intent，后按需注入”。
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from . import registry

# 插件落盘目录：hs-sidecar-guard/mcp_plugins/<repo>（随项目存在，不污染系统路径）。
_plugins_root = Path(__file__).resolve().parent.parent.parent / "mcp_plugins"


def plugins_root() -> Path:
    return _plugins_root


def _parse_ref(ref: str) -> tuple[str, str]:
    """解析 GitHub 引用，返回 (可 clone 的 git URL, 仓库名)。"""
    ref = (ref or "").strip()
    if not ref:
        raise ValueError("缺少 GitHub 仓库地址，例如：owner/repo 或 https://github.com/owner/repo")
    ref = ref.rstrip("/")
    url = ref
    if "://" not in url and not url.startswith("git@"):
        parts = ref.split("/")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"无法解析 GitHub 仓库地址：{ref!r}（应为 owner/repo）")
        url = f"https://github.com/{ref}"

    # 从 URL 中提取 owner/repo（git@host:o/r、https://github.com/o/r、含子树路径均可）。
    if url.startswith("git@"):
        _path = url.split(":", 1)[1]
        segs = _path.split("/")
    else:
        eh = url.split("://", 1)[1] if "://" in url else url
        host, _, _path = eh.partition("/")
        segs = _path.split("/")
    # 去空与 .git；取前两段作为 owner, repo
    segs = [s for s in segs if s and s != ".git"]
    if len(segs) < 2:
        raise ValueError(f"无法从地址解析 owner/repo：{ref!r}")
    repo = segs[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    return url, repo


def _infer_command(repo_dir: Path) -> tuple[str, list[str]]:
    """启发式推断启动命令，返回 (command, args)；无法推断返回空串（需 /mcp add 手动配）。"""
    pj = repo_dir / "package.json"
    if pj.is_file():
        try:
            data = json.loads(pj.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            data = {}
        bin_ = data.get("bin") or {}
        if isinstance(bin_, dict):
            candidates = list(bin_.values())
        else:
            candidates = [bin_] if bin_ else []
        for c in candidates:
            if isinstance(c, str) and (repo_dir / c).is_file():
                return "node", [c]
        main = data.get("main")
        if isinstance(main, str) and main and (repo_dir / main).is_file():
            return "node", [main]
        return "node", [candidates[0]] if candidates else []
    if any((repo_dir / f).is_file()
           for f in ("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt")):
        for cand in ("main.py", "server.py", "__main__.py", "src/main.py", "src/server.py"):
            if (repo_dir / cand).is_file():
                return "python", [cand]
        return "python", []
    if (repo_dir / "Cargo.toml").is_file():
        return "cargo", ["run"]
    return "", []


def install_from_github(ref: str) -> str:
    """把 GitHub 仓库 clone 并按 MCP 服务器登记。返回友好提示（需 /mcp approve 启用）。"""
    url, repo = _parse_ref(ref)
    target = _plugins_root / repo
    if target.exists():
        raise ValueError(f"插件目录已存在：{target}，请先删除或改地址重新安装。")

    _plugins_root.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["git", "clone", "--depth", "1", url, str(target)],
        capture_output=True, text=True, timeout=180,
    )
    if proc.returncode != 0:
        raise ValueError(f"git clone 失败：{(proc.stderr or proc.stdout).strip()[:400]}")

    command, args = _infer_command(target)
    if not command:
        raise ValueError(
            f"未能自动推断 {repo} 的启动命令（未识别到 Node/Python/Cargo 工程），"
            f"请用 /mcp add <name> <command> [args...] 手动登记。")

    srv = registry.add_server(repo, command, args, desc=f"GitHub 插件：{url}")
    return (f"✔ 已从 GitHub 安装并登记 MCP 插件：{repo}（{url}）\n"
            f"  本地路径：{target}\n"
            f"  推断启动命令：{command} {' '.join(args)}\n"
            f"  [deny-first] 该插件默认【未启用】，请用 /mcp approve {repo} 批准后模型才能调用。"
            f"（批准时若未声明工具，将自动反填白名单）")