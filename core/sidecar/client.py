"""主 Agent 端的 Sidecar 独立进程 Client。

职责：
  1. 用 subprocess.Popen 拉起 Sidecar server.py（独立 OS 进程）
  2. 通过 HTTP 调用 /bootstrap /pre_tool_use /post_tool_use
  3. 主 Agent 退出时优雅 stop 子进程

物理隔离保证：
  - Sidecar 在独立 Python 解释器里跑，和主 Agent 进程完全分离
  - Sidecar 进程只加载 Qwen2.5-1.5B + Sidecar 专用 LoRA，和主 Agent 的
    远程 LLM（Claude Code / cloud）完全隔离，GPU 显存各占各的
  - Sidecar 不持有主 Agent 对话历史，只拿「锚点 + 当前工具调用」做校验
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _port_free(port: int, host: str = "127.0.0.1") -> bool:
    s = socket.socket()
    try:
        s.connect((host, port))
        return False
    except (ConnectionRefusedError, OSError):
        return True
    finally:
        s.close()


class SidecarClient:
    """主 Agent 端的 Sidecar 进程管理 + RPC 封装。"""

    def __init__(self, project_root: str | Path,
                 port: int = 8765,
                 model_path: str | None = None,
                 adapter_path: str | None = None,
                 host: str = "127.0.0.1",
                 python_exe: str | None = None) -> None:
        self.project_root = str(Path(project_root).resolve())
        self.port = port
        self.host = host
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.python_exe = python_exe or sys.executable
        self._proc: subprocess.Popen | None = None
        self._base = f"http://{host}:{port}"

    # ---------- 进程生命周期 ----------

    def start(self, timeout: float = 30.0) -> bool:
        """拉起 Sidecar server 子进程并等待就绪。"""
        if self._proc is not None:
            return True
        if not _port_free(self.port, self.host):
            print(f"[Sidecar] 端口 {self.port} 已被占用，假定已有 Sidecar 实例在跑")
            return self._wait_ready(timeout)

        # 找项目根（有 core/__init__.py 的目录）
        import importlib.util
        spec = importlib.util.find_spec("core.sidecar.server")
        if spec and spec.origin:
            # .../hs-sidecar-guard/core/sidecar/server.py → project_root=.../hs-sidecar-guard
            project_root = str(Path(spec.origin).resolve().parent.parent.parent)
        else:
            project_root = self.project_root

        cmd = [
            self.python_exe, "-m", "core.sidecar.server",
            "--port", str(self.port),
            "--project-root", self.project_root,
        ]
        if self.model_path:
            cmd += ["--model-path", self.model_path]
        if self.adapter_path:
            cmd += ["--adapter-path", self.adapter_path]

        # Windows 下沉寂进程（不弹黑窗）
        creationflags = 0
        if sys.platform.startswith("win"):
            creationflags = subprocess.CREATE_NO_WINDOW

        self._proc = subprocess.Popen(
            cmd,
            cwd=project_root,          # ← 关键：在项目根启动，让 -m core.sidecar.server 能 import
            creationflags=creationflags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        return self._wait_ready(timeout)

    def stop(self, timeout: float = 5.0) -> None:
        """优雅终止 Sidecar 子进程。"""
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        except Exception:
            pass
        self._proc = None

    def __enter__(self) -> "SidecarClient":
        self.start()
        return self

    def __exit__(self, *a) -> None:
        self.stop()

    # ---------- RPC ----------

    def _post(self, path: str, payload: dict) -> dict:
        import urllib.request
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=3) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            # Sidecar 没启动 / 崩了 → 安全降级：放行
            return {"block": False, "msg": f"sidecar error: {e}", "pause": False}

    def health(self) -> dict:
        return self._post("/health", {})

    def bootstrap(self, original_instruction: str,
                  requirements: list[str] | None = None,
                  constraints: list[str] | None = None,
                  forbidden: list[str] | None = None) -> dict:
        return self._post("/bootstrap", {
            "original_instruction": original_instruction,
            "requirements": requirements or [],
            "constraints": constraints or [],
            "forbidden": forbidden or [],
        })

    def pre_tool_use(self, tool_name: str, arguments: Any) -> dict:
        return self._post("/pre_tool_use", {
            "tool_name": tool_name,
            "arguments": arguments if isinstance(arguments, str)
                        else json.dumps(arguments, ensure_ascii=False),
        })

    def post_tool_use(self, tool_name: str, arguments: Any) -> dict:
        return self._post("/post_tool_use", {
            "tool_name": tool_name,
            "arguments": arguments if isinstance(arguments, str)
                        else json.dumps(arguments, ensure_ascii=False),
        })

    # ---------- 内部 ----------

    def _wait_ready(self, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                self.health()
                return True
            except Exception:
                time.sleep(0.5)
        return False
