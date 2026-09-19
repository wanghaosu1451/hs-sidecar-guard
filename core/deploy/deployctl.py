"""部署管理：编排启动/停止/健康检查，并把本地后端注册到 registry。"""
from __future__ import annotations

from ..llm.keys import load as load_cfg
from . import registry
from .vllm_server import BackendLauncher


class DeployController:
    def __init__(self, backend: str | None = None):
        cfg = load_cfg().get("deploy", {})
        self.backend = backend or cfg.get("backend", "ollama")
        host = cfg.get("host", "127.0.0.1")
        self.port = int(cfg.get("port", 11434))
        self.launcher = BackendLauncher(self.backend, host, self.port)
        self._default_model = "llama3"

    def start(self, model: str | None = None) -> dict:
        r = self.launcher.start()
        if r.get("ok"):
            used_model = model or self._default_model
            registry.register_local_backend(self.backend,
                                            self.launcher.api_base, used_model)
        return {**r, "backend": self.backend}

    def stop(self) -> dict:
        return self.launcher.stop()

    def status(self) -> dict:
        return self.launcher.status()

    def deploy_trained_model(self, model_path: str,
                             model_name: str | None = None) -> dict:
        """把训练产物注册（供 UI 选用），并登记为可接入端点的模型。"""
        registry.register_trained_model(model_path, model_name)
        name = model_name or f"trained-{model_path.split('/')[-1]}"
        registry.register_local_backend(self.backend, self.launcher.api_base, name)
        return {"ok": True, "model": name, "api_base": self.launcher.api_base}