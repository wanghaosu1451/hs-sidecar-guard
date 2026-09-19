"""配置读写与安全存放 API key。

使用配置文件 config/settings.json；key 仅存本地，不做加密（首版）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
CONFIG_FILE = CONFIG_DIR / "settings.json"

_DEFAULTS: dict[str, Any] = {
    "llm": {
        "provider": "openai/gpt-4o-mini",
        "api_key": "",
        "base_url": "",
        "temperature": 0.7,
        "max_tokens": 2048,
        "stream": True,
        "retry": True,  # 模型调用自动重试开关（默认开）
        "show_thinking": True,  # 云端推理模型思考过程是否折叠展示（/thinking on|off 切换）
    },
    "budget": {
        "max_steps": 300,    # 任务最大执行轮次（缺省 300 步）
        "max_tokens": 0,     # token 预算；0 表示本地不下发预算（不限额）
    },
    "sandbox": {
        "enabled": True,
        "timeout_seconds": 60,
        "max_cpu_percent": 80,
        "max_memory_mb": 4096,
        "allow_network": False,
        "allowed_dirs": [],
    },
    "training": {
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "output_dir": "./artifacts",
        "quantization": "4bit",
        "lora_r": 16,
        "learning_rate": 2e-4,
        "epochs": 1,
        "batch_size": 4,
        "use_gpu": True,
    },
    "deploy": {
        "backend": "ollama",
        "host": "127.0.0.1",
        "port": 11434,
        # litellm ollama provider 走 /api/chat，不能带 /v1
        "api_base": "http://127.0.0.1:11434",
    },
}

_lock: dict[str, Any] = {}

# 环境变量覆盖表：(section, key) -> 环境变量名。让整包可移植——每台机器在项目根
# 放一个 .env（或直接设环境变量）即可改模型地址/Key/联网开关，无需改 settings.json。
_ENV_OVERRIDES: dict[tuple[str, str], str] = {
    ("llm", "provider"): "HS_LLM_PROVIDER",
    ("llm", "api_key"): "HS_LLM_API_KEY",
    ("llm", "base_url"): "HS_LLM_BASE_URL",
    ("llm", "temperature"): "HS_LLM_TEMPERATURE",
    ("llm", "max_tokens"): "HS_LLM_MAX_TOKENS",
    ("sandbox", "enabled"): "HS_SANDBOX",
    ("sandbox", "allow_network"): "HS_NETWORK",
}


def _load_env() -> None:
    """把项目根 .env 读入 os.environ。已存在的环境变量优先，不覆盖。"""
    env_file = CONFIG_DIR.parent / ".env"
    if not env_file.is_file():
        return
    try:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = (k or "").strip()
            v = (v or "").strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except OSError:
        pass


def _apply_env(merged: dict[str, Any]) -> dict[str, Any]:
    """按 _ENV_OVERRIDES 用环境变量覆盖已合并的配置（优先级：env > settings > 默认值）。"""
    for (section, key), var in _ENV_OVERRIDES.items():
        if var not in os.environ:
            continue
        raw = os.environ[var].strip()
        if not raw:
            continue
        cur = merged.get(section, {}).get(key)
        sec = merged.setdefault(section, {})
        if isinstance(cur, bool):
            sec[key] = raw.lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int):
            try:
                sec[key] = int(raw)
            except ValueError:
                sec[key] = raw
        else:
            sec[key] = raw
    return merged


def _load() -> dict[str, Any]:
    _load_env()
    if not CONFIG_FILE.exists():
        save(_DEFAULTS)
        return _apply_env(json.loads(json.dumps(_DEFAULTS)))
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        data = {}
    # 合并默认值，缺字段时补默认
    merged = json.loads(json.dumps(_DEFAULTS))
    merged.update(data)
    for k, v in _DEFAULTS.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k].update(data.get(k, {}))
    return _apply_env(merged)


def load() -> dict[str, Any]:
    # 不去 _lock 缓存，直接每次读文件，确保 settings.json 修改后立即生效（JSON 很小）
    data = _load()
    _lock["data"] = data  # 仅用于 save 保持一致性，读时始终取最新
    return data


def get(section: str | None = None, key: str | None = None) -> Any:
    data = load()
    if section:
        val = data.get(section, _DEFAULTS.get(section, {}))
        if key:
            return val.get(key, _DEFAULTS.get(section, {}).get(key))
        return val
    return data


def save(data: dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    _lock["data"] = data


def update(section: str, key: str, value: Any) -> None:
    data = load()
    data.setdefault(section, {})
    data[section][key] = value
    save(data)


def get_api_key(provider_model: str) -> str:
    """按 provider 前缀尝试取环境变量中的 key。

    优先级：settings 中的 api_key > 环境变量（OPENAI_API_KEY 等大写）。
    """
    saved = get("llm", "api_key")
    if saved:
        return saved
    provider = (provider_model or "").split("/")[0].upper()
    env_map = {
        "OPENAI": "OPENAI_API_KEY",
        "ANTHROPIC": "ANTHROPIC_API_KEY",
        "AZURE": "AZURE_API_KEY",
        "GEMINI": "GEMINI_API_KEY",
        "DEEPSEEK": "DEEPSEEK_API_KEY",
        "QWEN": "DASHSCOPE_API_KEY",
        "ZHIPU": "ZHIPU_API_KEY",
        "BAIDU": "BAIDU_API_KEY",
        "GROQ": "GROQ_API_KEY",
        "TOGETHER": "TOGETHERAI_API_KEY",
        "MISTRAL": "MISTRAL_API_KEY",
        "REPLICATE": "REPLICATE_API_TOKEN",
        "OPENROUTER": "OPENROUTER_API_KEY",
    }
    return os.environ.get(env_map.get(provider, ""), "")