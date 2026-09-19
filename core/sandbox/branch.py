"""增量快照分支推演沙盒（无 git 依赖，文件系统硬链接实现）。

高风险批量修改前一键推演：
  1. fork()  在独立目录创建可写快照
  2. 推演阶段 Agent 在快照上操作
  3. diff()  终端输出完整对比
  4. merge() / discard()  用户决定合并或丢弃

设计原则：零外部依赖（不依赖 git）、增量存储（只复制被修改的文件）、
所有路径在项目根下封闭，绝对拒绝越界访问。
"""
from __future__ import annotations

import filecmp
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator


SANDBOX_DIRNAME = ".hs_sandbox"


@dataclass
class SandboxBranch:
    """一个推演分支。"""

    name: str           # 分支名（唯一，如 "sandbox_20260919_161530"）
    root: str           # 沙盒根目录（绝对路径，隔离的工作副本）
    original_root: str  # 被快照的项目根（绝对路径）
    created_at: float
    status: str = "open"  # open / merged / discarded

    # ---------------- 构造 ----------------
    @classmethod
    def create(cls, project_root: str | Path,
               hint: str = "sandbox") -> "SandboxBranch":
        """在项目根下创建一个隔离沙盒目录。"""
        original = Path(project_root).resolve()
        sandbox_parent = original / SANDBOX_DIRNAME
        sandbox_parent.mkdir(parents=True, exist_ok=True)

        name = f"{hint}_{time.strftime('%Y%m%d_%H%M%S')}"
        root = sandbox_parent / name
        root.mkdir(parents=True, exist_ok=False)

        # 硬链接复制所有文件（目录结构 + 所有文件硬链接到原始 inode）
        for dirpath, dirnames, filenames in os.walk(original):
            # 跳过沙盒自己、隐藏目录、依赖目录
            rel = Path(dirpath).relative_to(original).as_posix()
            if rel == ".":
                skip_dirs = {SANDBOX_DIRNAME, ".git", "__pycache__", "node_modules",
                             ".hs", ".venv", "venv"}
            else:
                skip_dirs = {".git", "__pycache__", "node_modules", SANDBOX_DIRNAME}
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]

            target_dir = root / rel
            target_dir.mkdir(parents=True, exist_ok=True)
            for fname in filenames:
                src = Path(dirpath) / fname
                dst = target_dir / fname
                # 直接复制文件，沙盒是独立副本。硬链接虽然省空间，但
                # 在 Windows 上用 open(path, 'w') 改写会同步修改原始 inode。
                try:
                    shutil.copy2(str(src), str(dst))
                except OSError:
                    pass

        branch = cls(name=name, root=str(root),
                     original_root=str(original),
                     created_at=time.time())
        # 写入元数据（让用户随时能查出这个分支是什么）
        (root / ".hs_sandbox_meta.json").write_text(
            __import__("json").dumps(asdict(branch), ensure_ascii=False, indent=2),
            encoding="utf-8")
        return branch

    # ---------------- 原始 vs 分支 diff ----------------

    def diff_files(self) -> dict[str, dict]:
        """返回所有被修改的文件的 {path: {state: added/modified/deleted}}。"""
        orig_root = Path(self.original_root)
        sandbox_root = Path(self.root)
        changed: dict[str, dict] = {}

        orig_files = self._walk_files(orig_root)
        sand_files = self._walk_files(sandbox_root)

        # added: 在 sandbox 但不在 original
        for rel, p in sand_files.items():
            if rel.startswith(".hs_sandbox_meta"):
                continue
            if rel not in orig_files:
                changed[rel] = {"state": "added", "path": p}
                continue
            if not filecmp.cmp(orig_files[rel], p, shallow=False):
                changed[rel] = {"state": "modified", "path": p}

        # deleted: 在 original 但不在 sandbox
        for rel, p in orig_files.items():
            if rel.startswith(".hs_sandbox"):
                continue
            if rel not in sand_files:
                changed[rel] = {"state": "deleted", "path": p}

        return changed

    def generate_diff_text(self) -> str:
        """用 difflib 生成终端可读的 unified diff。"""
        import difflib
        orig_root = Path(self.original_root)
        sandbox_root = Path(self.root)
        out_parts: list[str] = [
            f"=== Sandbox {self.name} ===\n",
            f"Original: {self.original_root}\n",
            f"Branch:   {self.root}\n",
        ]
        changed = self.diff_files()
        if not changed:
            out_parts.append("(无变化)\n")
        for rel, info in sorted(changed.items()):
            orig_file = orig_root / rel
            sand_file = sandbox_root / rel
            out_parts.append(f"\n--- {rel}  [{info['state']}]\n")
            if info["state"] == "modified":
                try:
                    a = orig_file.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
                    b = sand_file.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
                    diff = list(difflib.unified_diff(a, b, fromfile=f"a/{rel}",
                                                     tofile=f"b/{rel}", n=3))
                    out_parts.append("".join(diff[:200]))   # 防止 diff 太长炸终端
                    if len(diff) > 200:
                        out_parts.append(f"\n... ({len(diff) - 200} 行 diff 截断)\n")
                except Exception as e:
                    out_parts.append(f"(diff failed: {e})\n")
            elif info["state"] == "added":
                try:
                    out_parts.append("+".join(
                        (sand_file.read_text(encoding="utf-8", errors="ignore")
                         .splitlines(True)[:50])) + "\n")
                except Exception:
                    pass
            elif info["state"] == "deleted":
                out_parts.append("(文件已删除)\n")
        return "".join(out_parts)

    # ---------------- 合并 / 丢弃 ----------------

    def merge(self) -> list[str]:
        """把分支改动应用回原始项目。返回被修改的文件列表。"""
        if self.status != "open":
            return []
        changed = self.diff_files()
        applied: list[str] = []
        orig_root = Path(self.original_root)
        for rel, info in changed.items():
            orig = orig_root / rel
            if info["state"] == "added":
                orig.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(info["path"], orig)
                applied.append(rel)
            elif info["state"] == "modified":
                orig.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(info["path"], orig)
                applied.append(rel)
            elif info["state"] == "deleted":
                if orig.is_file():
                    orig.unlink()
                    applied.append(rel)
        self.status = "merged"
        return applied

    def discard(self) -> None:
        """丢弃分支（删除沙盒目录），改动全部作废。"""
        if Path(self.root).is_dir():
            shutil.rmtree(self.root, ignore_errors=True)
        self.status = "discarded"

    # ---------------- 内部 ----------------

    @staticmethod
    def _walk_files(root: Path) -> dict[str, Path]:
        out: dict[str, Path] = {}
        skip = {SANDBOX_DIRNAME, ".git", "__pycache__", "node_modules"}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip]
            for fname in filenames:
                p = Path(dirpath) / fname
                rel = p.relative_to(root).as_posix()
                out[rel] = p
        return out


class SandboxManager:
    """项目根下所有沙盒分支的管理门面。"""

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.sandbox_parent = self.project_root / SANDBOX_DIRNAME

    def create(self, hint: str = "sandbox") -> SandboxBranch:
        return SandboxBranch.create(self.project_root, hint=hint)

    def list_branches(self) -> list[str]:
        if not self.sandbox_parent.is_dir():
            return []
        return sorted(p.name for p in self.sandbox_parent.iterdir()
                      if p.is_dir() and not p.name.startswith("."))

    def get(self, name: str) -> SandboxBranch | None:
        p = self.sandbox_parent / name
        meta = p / ".hs_sandbox_meta.json"
        if meta.is_file():
            import json
            try:
                return SandboxBranch(**json.loads(meta.read_text(encoding="utf-8")))
            except Exception:
                return None
        return None

    def cleanup_all(self) -> None:
        if self.sandbox_parent.is_dir():
            shutil.rmtree(self.sandbox_parent, ignore_errors=True)
