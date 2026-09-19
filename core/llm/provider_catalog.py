"""80+ 模型供应商目录。

以 LiteLLM 支持的 `provider/model` 标识作为统一键。列表覆盖主流云厂商
与本地后端；UI 据此渲染下拉，用户选择后由 gateway.py 统一调用。
"""
from __future__ import annotations

# 格式：{"provider/model": "显示名"}
# provider 是 LiteLLM 支持的 provider 前缀；本地后端使用 ollama/vllm/lmstudio/bedrock。
PROVIDERS: dict[str, str] = {
    # ---- OpenAI 系 ----
    "openai/gpt-4o": "OpenAI GPT-4o",
    "openai/gpt-4o-mini": "OpenAI GPT-4o-mini",
    "openai/gpt-4-turbo": "OpenAI GPT-4 Turbo",
    "openai/o1": "OpenAI o1",
    "openai/o3-mini": "OpenAI o3-mini",
    "openai/gpt-3.5-turbo": "OpenAI GPT-3.5 Turbo",
    # ---- Azure OpenAI ----
    "azure/gpt-4o": "Azure OpenAI GPT-4o",
    "azure/gpt-4": "Azure OpenAI GPT-4",
    # ---- Anthropic / Claude ----
    "anthropic/claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
    "anthropic/claude-3-5-haiku": "Claude 3.5 Haiku",
    "anthropic/claude-3-opus": "Claude 3 Opus",
    "anthropic/claude-sonnet-4": "Claude Sonnet 4",
    # ---- Google / Gemini ----
    "gemini/gemini-1.5-pro": "Gemini 1.5 Pro",
    "gemini/gemini-1.5-flash": "Gemini 1.5 Flash",
    "gemini/gemini-2.0-flash": "Gemini 2.0 Flash",
    "gemini/gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini/gemini-2.5-flash": "Gemini 2.5 Flash",
    # ---- 国内大厂 ----
    "deepseek/deepseek-chat": "DeepSeek Chat",
    "deepseek/deepseek-reasoner": "DeepSeek R1",
    "qwen/qwen-turbo": "通义千问 Turbo",
    "qwen/qwen-plus": "通义千问 Plus",
    "qwen/qwen-max": "通义千问 Max",
    "qwen/qwen3-235b": "通义千问 Qwen3 235B",
    "qwen/qwen2.5-72b-instruct": "通义千问 2.5 72B",
    "zhipu/glm-4": "智谱 GLM-4",
    "zhipu/glm-4-plus": "智谱 GLM-4-Plus",
    "zhipu/glm-4-flash": "智谱 GLM-4-Flash",
    "baidu/ernie-bot-4": "文心一言 4",
    "baidu/ernie-3.5-8k": "文心一言 3.5",
    "mistral/mistral-large": "百度云 Mistral Large",
    # ---- 国际主流与开源托管 ----
    "groq/llama-3.3-70b-versatile": "Groq Llama3 70B",
    "groq/qwen-2.5-32b": "Groq Qwen 32B",
    "together_ai/meta-llama-3.1-405b": "Together Llama3.1 405B",
    "replicate/llama-3.3-70b-instruct": "Replicate Llama3.3 70B",
    "fireworks_ai/accounts/fireworks/models/llama-v3p1-405b-instruct": "Fireworks Llama3.1 405B",
    "codestral/codestral-latest": "Mistral Codestral",
    "mistral/open-mistral-nemo": "Mistral Large Nemo",
    "mistral/mistral-medium": "Mistral Medium",
    "mistral/pixtral-large-latest": "Mistral Pixtral Large",
    "meta-llama/llama-3.1-8b-instruct": "Meta Llama 3.1 8B",
    # ---- Hugging Face / 通用 ----
    "huggingface/meta-llama/llama-3.1-70b": "HF Llama 3.1 70B",
    "text-generation-inference/Qwen/Qwen2.5-7B-Instruct": "TGI Qwen2.5 7B",
    "vertex_ai/gemini-1.5-pro": "Vertex Gemini 1.5 Pro",
    "bedrock/anthropic.claude-3-5-sonnet": "Bedrock Claude 3.5",
    "cohere/command": "Cohere Command",
    "cohere/command-r-plus": "Cohere Command R+",
    "jina/jina-embeddings-v3": "Jina Embeddings v3",
    "ai21/j2-ultra": "AI21 J2 Ultra",
    "nvidia_ai_endpoints/meta/llama3-70b-instruct": "NVIDIA Llama3 70B",
    "perplexity/sonar": "Perplexity Sonar",
    "sambanova/meta-llama-3.1-70b-instruct": "SambaNova Llama3 70B",
    "databricks/databricks-meta-llama-3-1-70b-instruct": "Databricks Llama3 70B",
    "watsonx/meta-llama-llama-3-2-3b-instruct": "Watsonx Llama3 2B",
    "groq/mixtral-8x7b": "Groq Mixtral 8x7B",
    "openrouter/openai/gpt-4o": "OpenRouter GPT-4o",
    "openrouter/anthropic/claude-3.5-sonnet": "OpenRouter Claude 3.5",
    "openrouter/mistralai/mixtral-8x22b": "OpenRouter Mixtral 8x22B",
    "openrouter/cohere/command-r": "OpenRouter Command R",
    "openrouter/meta-llama/llama-3.3-70b": "OpenRouter Llama3.3 70B",
    "ollama/llama3": "Ollama Llama3 (本地)",
    "ollama/qwen2.5": "Ollama Qwen2.5 (本地)",
    "ollama/mistral": "Ollama Mistral (本地)",
    "vllm/Qwen/Qwen2.5-7B-Instruct": "vLLM Qwen2.5 7B (本地)",
    "lmstudio/Qwen/Qwen2.5-7B-Instruct": "LM Studio Qwen2.5 7B (本地)",
    "custom/openai-compatible": "自定义 OpenAI 兼容端点 (私有化)",
    # ---- 更多云端供应商（补充至 80+）----
    "moonshot/moonshot-v1-8k": "月之暗面 Kimi 8k",
    "moonshot/moonshot-v1-32k": "月之暗面 Kimi 32k",
    "moonshot/moonshot-v1-128k": "月之暗面 Kimi 128k",
    "minimax/abab6.5-chat": "MiniMax 6.5",
    "stepfun/step-1-128k": "阶跃星辰 Step-1 128k",
    "doubao/doubao-pro-32k": "字节豆包 Pro 32k",
    "baichuan/baichuan2-53b": "百川 Baichuan2 53B",
    "hunyuan/hunyuan-turbo": "腾讯混元 Turbo",
    "ali/qwen-max": "阿里云(ali) Qwen-Max",
    "mistral/open-mixtral-8x22b": "Mistral Mixtral 8x22B",
    "mistral/codestral-latest": "Mistral Codestral-latest",
    "groq/llama-3.1-8b-instant": "Groq Llama3.1 8B",
    "groq/llava-v1.5-7b": "Groq LLaVA 7B",
    "together_ai/upstage/SOLAR-10.7B-Instruct": "Together Solar 10.7B",
    "together_ai/mistralai/Mistral-7B-Instruct-v0.3": "Together Mistral 7B",
    "replicate/meta/llama-3-70b-instruct": "Replicate Llama3 70B",
    "replicate/mistralai/mistral-7b-instruct": "Replicate Mistral 7B",
    "fireworks_ai/meta-llama-3-70b-instruct": "Fireworks Llama3 70B",
    "perplexity/sonar-pro": "Perplexity Sonar Pro",
    "cohere/command-r": "Cohere Command R",
    "cohere/embed-multilingual-v3.0": "Cohere Embed v3.0",
    "nvidia_ai_endpoints/meta/llama-3.3-70b-instruct": "NVIDIA Llama3.3 70B",
    "anyscale/meta-llama/Llama-3.2-3B": "Anyscale Llama3.2 3B",
    "openrouter/deepseek/deepseek-chat": "OpenRouter DeepSeek",
    "openrouter/qwen/qwen2.5-72b-instruct": "OpenRouter Qwen2.5 72B",
    "openrouter/google/gemini-flash-1.5": "OpenRouter Gemini Flash",
    "anthropic/claude-3-haiku": "Claude 3 Haiku",
    "gemini/gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite",
    "qwen/qwen-turbo-latest": "通义千问 Turbo(最新)",
    "zhipu/glm-4-air": "智谱 GLM-4-Air",
    "deepseek/deepseek-chat-v3": "DeepSeek V3",
    "openai/gpt-4.1": "OpenAI GPT-4.1",
    "openai/gpt-4.1-mini": "OpenAI GPT-4.1-mini",
    "openai/gpt-4.1-nano": "OpenAI GPT-4.1-nano",
}

# 别名分组，供 UI 分类展示
GROUPS: dict[str, list[str]] = {
    "OpenAI / Azure": [k for k in PROVIDERS if k.startswith(("openai/", "azure/"))],
    "Anthropic / Claude": [k for k in PROVIDERS if k.startswith("anthropic/")],
    "Google / Gemini": [k for k in PROVIDERS if k.startswith("gemini/")],
    "国内厂商": [k for k in PROVIDERS if k.startswith(
        ("deepseek/", "qwen/", "zhipu/", "baidu/", "moonshot/", "minimax/",
         "stepfun/", "doubao/", "baichuan/", "hunyuan/", "ali/"))],
    "开源托管平台": [k for k in PROVIDERS if k.startswith(
        ("groq/", "together_ai/", "replicate/", "fireworks_ai/",
         "mistral/", "codestral/", "openrouter/", "nvidia_", "perplexity/",
         "sambanova/", "databricks/", "watsonx/", "text-generation-inference/",
         "huggingface/", "anyscale/"))],
    "AWS / 云厂商": [k for k in PROVIDERS if k.startswith(
        ("bedrock/", "vertex_ai/", "cohere/", "ai21/", "jina/", "meta-llama/"))],
    "本地 / 私有化": [k for k in PROVIDERS if k.startswith(
        ("ollama/", "vllm/", "lmstudio/", "custom/"))],
    "其他": [k for k in PROVIDERS],
}


def count() -> int:
    return len(PROVIDERS)