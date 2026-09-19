"""训练/导出产物注册表：便于在 UI 中选用私有化端点或已训练模型。"""
from __future__ import annotations

import json
from pathlib import Path

REGISTRY_FILE = Path(__file__).resolve().parent.parent.parent / "config" / "registry.json"


def _load() -> dict:
    if REGISTRY_FILE.exists():
        try:
            return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"local_backends": [], "trained_models": []}


def _save(data: dict) -> None:
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                             encoding="utf-8")


def register_local_backend(backend: str, api_base: str, model: str) -> None:
    data = _load()
    entry = {"backend": backend, "api_base": api_base, "model": model,
             "provider_model": f"{backend}/{model}"}
    data["local_backends"] = [e for e in data["local_backends"]
                              if not (e["api_base"] == api_base and e["model"] == model)]
    data["local_backends"].append(entry)
    _save(data)


def register_trained_model(path: str, name: str | None = None) -> None:
    data = _load()
    entry = {"path": path, "name": name or f"trained@{Path(path).name}"}
    data["trained_models"] = [e for e in data["trained_models"]
                              if e["path"] != path]
    data["trained_models"].append(entry)
    _save(data)


def list_local_backends() -> list[dict]:
    return _load()["local_backends"]


def list_trained_models() -> list[dict]:
    return _load()["trained_models"]