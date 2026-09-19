"""LiteLLM 统一网关（支柱A）。

所有对话/沙箱/部署产物统一从这里发起调用。支持同步、流式、多轮消息，
并按 provider 前缀自动组装 key 与 base_url。tools 支持 function calling。
"""
from __future__ import annotations

import os
import time
from types import SimpleNamespace
from typing import Any, Callable, Iterator

# 断网/首启优化：用本地模型 cost map，避免每次 import litellm 都去 GitHub 拉价格表（网络不通会重试3次、慢15s+）
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("LITELLM_DROP_PARAMS", "true")

from . import keys

try:  # LiteLLM 可选依赖，缺失时给出明确错误
    import litellm
    litellm.request_timeout = 600  # 本地推理可能 115s+，放大 httpx 超时避免误判卡死/connection timeout
except Exception as _e:  # pragma: no cover
    litellm = None
    _LITELLM_IMPORT_ERROR = _e
else:
    _LITELLM_IMPORT_ERROR = None


class LLMError(Exception):
    pass


def ensure_ready() -> None:
    if litellm is None:
        raise LLMError(
            f"未安装 litellm，请先执行: pip install litellm（原因: {_LITELLM_IMPORT_ERROR}）")


# ============================ 重试状态机（仿 response_retry） ============================
# 仅对“网络/超时/5xx”等瞬断重试（指数退避），业务 4xx（鉴权/模型不存在）不重试；
# 重试上限用尽时把错误作为面向用户的提示返回，不让上层误判为工具失败而无限反思。
_RETRY_MARKERS = (
    "timeout", "timed out", "connection", "network", "econnreset",
    "temporarily unavailable", "overloaded", "internal server",
    "server error", "bad gateway", "service unavailable", "gateway timeout",
    "token repeat limit reached", "repeat limit", "aborted",
)


def _retry_cfg() -> bool:
    try:
        return bool(keys.load().get("llm", {}).get("retry", True))
    except Exception:  # noqa: BLE001
        return True


def _is_retryable(exc: BaseException) -> bool:
    """判断异常是否为可重试的瞬断。业务 4xx（鉴权/模型不存在）不重试。"""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "code", None)
    try:
        status_int = int(status)
    except (TypeError, ValueError):
        status_int = None
    if status_int is not None:
        if status_int == 429 or 500 <= status_int < 600:
            return True
        if 400 <= status_int < 500:
            return False  # 鉴权/模型不存在等业务性错误，重试无意义
    blob = f"{type(exc).__name__} {exc}".lower()
    return any(m in blob for m in _RETRY_MARKERS)


def _backoff(attempt: int, base: float = 1.0) -> float:
    """指数退避：attempt=0 -> 1s，attempt=1 -> 2s。"""
    return base * (2 ** attempt)


def _split_provider(provider_model: str) -> tuple[str, str]:
    parts = (provider_model or "").split("/", 1)
    if len(parts) != 2:
        return "openai", provider_model or "gpt-4o-mini"
    return parts[0], parts[1]


def _build_kwargs(provider_model: str, **overrides: Any) -> dict[str, Any]:
    provider, model = _split_provider(provider_model)
    kwargs: dict[str, Any] = {"model": provider_model}
    # 本地后端把 base_url 映射到 api_base
    if provider in ("ollama", "vllm", "lmstudio", "custom"):
        base = overrides.get("base_url") or overrides.get("api_base")
        if base:
            kwargs["api_base"] = base
        # 优先用 overrides 里传的 api_key（如带鉴权的 Ollama 网关），没有才用 "local"
        key = overrides.get("api_key")
        if key:
            kwargs["api_key"] = key
        else:
            kwargs.setdefault("api_key", "local")
    else:
        key = overrides.get("api_key") or keys.get_api_key(provider_model)
        if key:
            kwargs["api_key"] = key
        base = overrides.get("base_url") or overrides.get("api_base")
        if base:
            kwargs["api_base"] = base
    # 传递温度/最大token（仅当显式给出）
    for k, v in overrides.items():
        if k in ("temperature", "max_tokens", "stream", "tools", "messages"):
            kwargs[k] = v
    # 本地 Ollama 网关：抬升 repeat_penalty 抑制重复（避免触发 "token repeat limit reached"，
    # 长代码/长输出被端上打断），并按需放宽输出上限以复用长生成。
    if any(s in str(kwargs.get("api_base", "")).lower() for s in ("ollama", "1143")):
        kwargs.setdefault("repeat_penalty", 1.3)
        if "num_predict" not in kwargs and "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 4096
    return kwargs


def chat(messages: list[dict], provider_model: str | None = None,
         temperature: float | None = None, max_tokens: int | None = None,
         tools: list[dict] | None = None, base_url: str | None = None,
         api_key: str | None = None, **kv) -> dict[str, Any]:
    """同步对话。messages: [{"role","content"}，可选 tool_calls/tool_call_id]。"""
    ensure_ready()
    pm = provider_model or keys.get("llm", "provider")
    cfg = keys.load().get("llm", {})
    temp = temperature if temperature is not None else cfg.get("temperature", 0.7)
    mtoks = max_tokens if max_tokens is not None else cfg.get("max_tokens", 2048)
    kwargs = _build_kwargs(pm, temperature=temp, max_tokens=mtoks,
                           messages=messages, tools=tools,
                           base_url=base_url or cfg.get("base_url") or "",
                           api_key=api_key or cfg.get("api_key") or "")
    # 透传 settings.llm.model_kwargs（如 {"options": {"thinking": false}} 控制 thinking）
    extra = cfg.get("model_kwargs") or {}
    if extra:
        kwargs.update(extra)
    retry_max = 2 if _retry_cfg() else 0
    attempt = 0
    while True:
        try:
            resp = litellm.completion(**kwargs)
            return _normalize(resp)
        except Exception as e:  # noqa: BLE001
            attempt += 1
            if attempt <= retry_max and _is_retryable(e):
                time.sleep(_backoff(attempt - 1))
                continue
            # 上限用尽或业务性错误：把错误作为面向用户的提示返回，而非抛给上层
            # 误判为工具失败而无限反思。附 error 标记便于上层区分“真实内容”与“报错”。
            return {"content": f"调用模型失败（已按配重试 {attempt - 1} 次）: {e}",
                    "role": "assistant", "tool_calls": None,
                    "finish_reason": None, "usage": None, "error": str(e)}


def chat_stream(messages: list[dict], provider_model: str | None = None,
                temperature: float | None = None, max_tokens: int | None = None,
                tools: list[dict] | None = None, base_url: str | None = None,
                api_key: str | None = None, **kv) -> Iterator[str]:
    """流式对话，逐段产出增量文本。"""
    ensure_ready()
    pm = provider_model or keys.get("llm", "provider")
    cfg = keys.load().get("llm", {})
    temp = temperature if temperature is not None else cfg.get("temperature", 0.7)
    mtoks = max_tokens if max_tokens is not None else cfg.get("max_tokens", 2048)
    kwargs = _build_kwargs(pm, temperature=temp, max_tokens=mtoks, stream=True,
                           messages=messages, tools=tools,
                           base_url=base_url or cfg.get("base_url") or "",
                           api_key=api_key or cfg.get("api_key") or "")
    # 透传 settings.llm.model_kwargs（与 chat 一致）
    extra = cfg.get("model_kwargs") or {}
    if extra:
        kwargs.update(extra)
    retry_max = 2 if _retry_cfg() else 0
    attempt = 0
    while True:
        sent_any = False
        try:
            stream = litellm.completion(**kwargs)
            for chunk in stream:
                if not chunk or not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None) or ""
                if text:
                    sent_any = True
                    yield text
            return
        except Exception as e:  # noqa: BLE001
            attempt += 1
            if not sent_any and attempt <= retry_max and _is_retryable(e):
                time.sleep(_backoff(attempt - 1))
                continue
            # 未产出任何增量时整体失败：把错误作为面向用户的提示吐出，不再抛给上层
            # 误判为工具失败。已产出增量后再失败则就地补充一句报错。
            yield f"\n[调用模型失败（已按配重试 {attempt - 1} 次）] {e}"
            return


def chat_structured_stream(messages: list[dict], provider_model: str | None = None,
                           temperature: float | None = None, max_tokens: int | None = None,
                           tools: list[dict] | None = None, base_url: str | None = None,
                           api_key: str | None = None,
                           on_delta: Callable[[str], None] | None = None,
                           on_reasoning: Callable[[str], None] | None = None,
                           **kv) -> dict[str, Any]:
    """流式调用并聚合出结构化结果（含 tool_calls），用于既有工具循环。

    - 流式逐字回调 on_delta(增量文本)，立即获得"在生成"的体感；
    - 支持云端推理模型（如 DeepSeek-R1 等）：增量地回调 on_reasoning(chunk)，
      并把聚合后的推理内容放进返回 dict 的 "reasoning" 字段（无则 None）；
    - 流结束后返回与 `chat()` 同构的 dict（content / tool_calls / finish_reason / usage）
      + 追加 "reasoning"，工具循环无需改动即可直接消费 tool_calls。
    """
    ensure_ready()
    pm = provider_model or keys.get("llm", "provider")
    cfg = keys.load().get("llm", {})
    temp = temperature if temperature is not None else cfg.get("temperature", 0.7)
    mtoks = max_tokens if max_tokens is not None else cfg.get("max_tokens", 2048)
    kwargs = _build_kwargs(pm, temperature=temp, max_tokens=mtoks, stream=True,
                           messages=messages, tools=tools,
                           base_url=base_url or cfg.get("base_url") or "",
                           api_key=api_key or cfg.get("api_key") or "")
    extra = cfg.get("model_kwargs") or {}
    if extra:
        kwargs.update(extra)
    retry_max = 2 if _retry_cfg() else 0
    attempt = 0
    while True:
        sent_any = False
        try:
            stream = litellm.completion(**kwargs)
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_acc: dict[int, dict[str, str]] = {}
            finish_reason = None
            usage = None
            for chunk in stream:
                if not chunk:
                    continue
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                ch = chunk.choices[0]
                if getattr(ch, "finish_reason", None):
                    finish_reason = ch.finish_reason
                delta = ch.delta
                txt = getattr(delta, "content", None)
                if txt:
                    sent_any = True
                    content_parts.append(txt)
                    if on_delta:
                        on_delta(txt)
                # 云端推理模型（DeepSeek-R1 等）把思考内容放在 reasoning_content 里：
                # 聚合到 reasoning_parts，并实时回调给调用方（用于可折叠展示）。
                rz = getattr(delta, "reasoning_content", None)
                if rz is None:
                    rz = getattr(delta, "reasoning", None)
                if rz:
                    reasoning_parts.append(rz)
                    if on_reasoning:
                        on_reasoning(rz)
                tcs = getattr(delta, "tool_calls", None)
                if tcs:
                    for tc in tcs:
                        idx = getattr(tc, "index", 0) or 0
                        slot = tool_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                        tid = getattr(tc, "id", None)
                        if tid:
                            slot["id"] = tid
                        fn = getattr(tc, "function", None)
                        if fn:
                            if getattr(fn, "name", None):
                                slot["name"] = fn.name
                            if getattr(fn, "arguments", None):
                                slot["arguments"] += fn.arguments
            # 组装 tool_calls（参数此时已由 LiteLLM 拼好，通常为 JSON 文本）
            tool_calls = []
            for idx in sorted(tool_acc):
                s = tool_acc[idx]
                if s["name"]:
                    tool_calls.append(SimpleNamespace(
                        type="function",
                        id=s["id"] or f"call_stream_{idx}",
                        function=SimpleNamespace(name=s["name"],
                                                 arguments=s["arguments"] or "{}"),
                    ))
            content = "".join(content_parts) if content_parts else None
            reasoning = "".join(reasoning_parts) if reasoning_parts else None
            return {"content": content, "role": "assistant",
                    "tool_calls": tool_calls or None,
                    "finish_reason": finish_reason, "usage": usage,
                    "reasoning": reasoning}
        except Exception as e:  # noqa: BLE001
            attempt += 1
            if not sent_any and attempt <= retry_max and _is_retryable(e):
                time.sleep(_backoff(attempt - 1))
                continue
            return {"content": f"调用模型失败（已按配重试 {attempt - 1} 次）: {e}",
                    "role": "assistant", "tool_calls": None,
                    "finish_reason": None, "usage": None, "error": str(e)}


def _normalize(resp: Any) -> dict[str, Any]:
    choice = resp.choices[0]
    message = choice.message
    return {
        "content": getattr(message, "content", None),
        "role": getattr(message, "role", "assistant"),
        "tool_calls": getattr(message, "tool_calls", None),
        "finish_reason": getattr(choice, "finish_reason", None),
        "usage": getattr(resp, "usage", None),
    }


def list_models(provider_model: str) -> list[str]:
    """尝试列出某 provider 的模型（本地 ollama/vllm 常用）。"""
    ensure_ready()
    try:
        models = litellm.get_model_list(provider_model)
    except Exception:  # noqa: BLE001
        return []
    if isinstance(models, dict):
        models = models.get("data", [])
        return [getattr(m, "id", None) or m for m in models]
    return models or []