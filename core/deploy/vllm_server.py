"""VLLM / Ollama 后端启动封装：启停本地推理服务，注入 OpenAI 兼容端点。"""
from __future__ import annotations

import shutil
import subprocess
import time
import urllib.request
from pathlib import Path


class BackendLauncher:
    def __init__(self, backend: str = "ollama", host: str = "127.0.0.1",
                 port: int = 11434):
        self.backend = backend
        self.host = host
        self.port = port
        self._proc: subprocess.Popen | None = None

    @property
    def api_base(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def health_url(self) -> str:
        if self.backend == "ollama":
            return f"http://{self.host}:{self.port}/api/tags"
        return f"http://{self.host}:{self.port}/v1/models"

    def binary(self) -> str:
        return "ollama" if self.backend == "ollama" else "vllm"

    def is_installed(self) -> bool:
        return shutil.which(self.binary()) is not None

    def start(self) -> dict:
        if not self.is_installed():
            return {"ok": False,
                    "error": f"未检测到 {self.binary()}，请先安装后再部署。"}
        if self.is_healthy():
            return {"ok": True, "already": True, "api_base": self.api_base}
        try:
            if self.backend == "ollama":
                cmd = ["ollama", "serve"]
            else:
                cmd = ["vllm", "serve",
                       "--host", self.host, "--port", str(self.port)]
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"启动失败: {e}"}
        # 等待健康检查
        deadline = time.time() + 30
        while time.time() < deadline:
            if self.is_healthy():
                return {"ok": True, "api_base": self.api_base}
            time.sleep(1)
        return {"ok": False, "error": "启动超时，未通过健康检查",
                "api_base": self.api_base}

    def stop(self) -> dict:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
            return {"ok": True, "stopped": True}
        return {"ok": True, "stopped": False}

    def is_healthy(self) -> bool:
        try:
            with urllib.request.urlopen(self.health_url, timeout=2) as r:
                return r.status == 200
        except Exception:  # noqa: BLE001
            return False

    def status(self) -> dict:
        return {
            "backend": self.backend,
            "api_base": self.api_base,
            "healthy": self.is_healthy(),
            "installed": self.is_installed(),
            "running": bool(self._proc and self._proc.poll() is None),
        }