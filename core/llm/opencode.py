"""OpenCode(SST) 配置兼容：读取 opencode.json，翻译成本地 settings llm 字段。

opencode.json 的核心是 provider 键驱动：每个 provider 带 baseURL / apiKey /
npm / 嵌套 models；顶层 model 用 "pid/mid" 选中「默认模型」。
这里把「被选中的那个 provider + model」翻译成 hs-sidecar-guard 的 llm 配置
(provider=pid/mid, base_url, api_key, model_kwargs=options)，便于复用同一份
远端开放 API 配置，不必在两边各配一遍。

仅支持纯 JSON 的 opencode.json；.ts/.mjs 的动态配置不在本解析范围。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _find_config(cwd: str | None = None) -> Path | None:
    """在 cwd 及用户目录下定位 opencode.json（OpenCode 的惯例查找路径）。"""
    candidates = []
    root = Path(cwd or os.getcwd())
    candidates.append(root / "opencode.json")
    # 逐级向上找，直到文件系统根
    for p in root.resolve().parents:
        candidates.append(p / "opencode.json")
    # 用户主目录兜底
    home = Path.home() / "opencode.json"
    if home not in candidates:
        candidates.append(home)
    for c in candidates:
        if c.is_file():
            return c
    return None


def _resolve_key(value: str) -> str:
    """opencode 里 apiKey 可写作 "{env:VAR}" 占位或字面量。这里解析环境变量。"""
    value = (value or "").strip()
    if not value:
        return ""
    if value.startswith("{env:") and value.endswith("}"):
        var = value[5:-1]
        return os.environ.get(var, "")
    return value


def _get_first(cfg: dict, *names: str) -> Any:
    for n in names:
        if n in cfg:
            return cfg[n]
    return None


def parse(path: str | Path | None = None,
          cwd: str | None = None) -> dict[str, Any] | None:
    """解析 opencode.json，返回 {provider, base_url, api_key, model_kwargs, meta}。

    找不到文件 / 无默认 model / 对应 provider 缺失时返回 None。
    """
    cfg_path = Path(path) if path else _find_config(cwd)
    if not cfg_path or not cfg_path.is_file():
        return None
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None

    model_ref = data.get("model")
    providers = data.get("provider")
    if not model_ref or not isinstance(providers, dict) or "/" not in str(model_ref):
        return None

    pid, mid = str(model_ref).split("/", 1)
    prov = providers.get(pid)
    if not isinstance(prov, dict):
        return None

    base_url = _get_first(prov, "baseURL", "baseUrl", "base_url", "baseURLV") or ""
    api_key = _resolve_key(str(_get_first(prov, "apiKey", "api_key") or ""))
    npm = (prov.get("npm") or "") if isinstance(prov.get("npm"), str) else ""
    opts = {}
    models = prov.get("models")
    if isinstance(models, dict) and isinstance(models.get(mid), dict):
        mopts = models[mid].get("options") or {}
        if isinstance(mopts, dict):
            opts = dict(mopts)  # 直接作为 model_kwargs（如 {"thinking": false}）

    return {
        "provider": model_ref,                 # "pid/mid"，与 litellm provider/model 一致
        "base_url": str(base_url).rstrip("/") or "",
        "api_key": api_key,
        "model_kwargs": opts,
        "meta": {"npm": npm, "pid": pid, "mid": mid, "path": str(cfg_path)},
    }


def import_to_settings(path: str | Path | None = None,
                       cwd: str | None = None) -> dict[str, Any]:
    """把 opencode.json 的默认模型导入本地 llm 配置并落盘。

    返回 {"imported": bool, "applied": {...}}。
    """
    from . import keys as _k
    parsed = parse(path, cwd)
    if not parsed:
        return {"imported": False, "applied": {}, "reason": "未找到可用的 opencode.json / 默认模型"}
    applied: dict[str, Any] = {
        "provider": parsed["provider"],
        "base_url": parsed["base_url"],
        "api_key": parsed["api_key"],
    }
    _k.update("llm", "provider", applied["provider"])
    _k.update("llm", "base_url", applied["base_url"])
    _k.update("llm", "api_key", applied["api_key"])
    if parsed["model_kwargs"]:
        _k.update("llm", "model_kwargs", parsed["model_kwargs"])
        applied["model_kwargs"] = parsed["model_kwargs"]
    return {"imported": True, "applied": applied, "meta": parsed["meta"]}