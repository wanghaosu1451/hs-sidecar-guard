"""本地后端接入：统一 OpenAI 兼容层。

把 Ollama / vLLM / LM Studio / 私有化端点抽象成 gateway 可用的参数字典。
LiteLLM 多数本地服务天然映射，本层只负责配置组装并暴露统一结构。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class LocalEndpoint:
    """一个本地/私有化部署端点的统一描述。"""
    name: str                 # 显示名
    backend: str              # ollama / vllm / lmstudio / custom
    api_base: str             # OpenAI 兼容 base_url，如 http://127.0.0.1:11434/v1
    model: str                # 端点上的模型名
    api_key: str = "local"    # 本地服务多不校验 key，占位

    def to_litellm(self) -> dict[str, Any]:
        """转成 gateway.completion 可消费的 provider/model + kwargs。"""
        prefix = {
            "ollama": "ollama",
            "vllm": "vllm",
            "lmstudio": "lmstudio",
            "custom": "custom",
        }.get(self.backend, "ollama")
        return {
            "provider_model": f"{prefix}/{self.model}",
            "base_url": self.api_base,
            "api_key": self.api_key,
        }


KNOWN_LOCAL_DEFAULTS = [
    # 注意：litellm 的 ollama provider 走原生 /api/chat，base_url 不能带 /v1（带 /v1 会 404）
    LocalEndpoint("Ollama (默认)", "ollama",
                  "http://127.0.0.1:11434", "llama3"),
    LocalEndpoint("Ollama Qwen2.5", "ollama",
                  "http://127.0.0.1:11434", "qwen2.5"),
    # vLLM / LM Studio 走 OpenAI 兼容端点，必须带 /v1
    LocalEndpoint("vLLM", "vllm",
                  "http://127.0.0.1:8000/v1", "Qwen/Qwen2.5-7B-Instruct"),
    LocalEndpoint("LM Studio", "lmstudio",
                  "http://127.0.0.1:1234/v1", "local-model"),
    LocalEndpoint("自定义私有化端点", "custom",
                  "http://127.0.0.1:8080/v1", "my-model"),
]


def build_local(backend: str, api_base: str, model: str) -> LocalEndpoint:
    return LocalEndpoint(name=f"{backend}:{model}", backend=backend,
                         api_base=api_base, model=model)


def list_local_defaults() -> list[dict[str, Any]]:
    return [asdict(e) for e in KNOWN_LOCAL_DEFAULTS]