# HS Sidecar Guard

> A lightweight dual-LLM trust harness for CLI/MCP agents.
> Local fine-tuned LoRA sidecar runs offline alongside any cloud model
> to block **goal drift**, **malicious shell commands**, and **cross-file regressions**
> *before* tool execution.

---

## Quick Start (5 minutes)

```powershell
# 1. Clone & install (everything — deps + base model + embedding)
git clone https://github.com/<your-name>/hs-sidecar-guard.git
cd hs-sidecar-guard
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force   # first time only
.\install.ps1

# 2. Configure LLM — open config/settings.json and set ONE of:
#
#    Ollama (local, no API key):
#      provider: "ollama/qwen3.5:9b-nothink"
#      base_url: "http://127.0.0.1:11434"
#
#    DeepSeek (cheap & strong):
#      provider: "deepseek/deepseek-chat"
#      api_key:  "sk-xxxxxxxxxxxxxxxx"
#      base_url: "https://api.deepseek.com/v1"
#
#    OpenAI / Anthropic / 80+ providers supported (LiteLLM):
#      provider: "openai/gpt-4o-mini"  or  "anthropic/claude-3-5-sonnet"

# 3. Start Sidecar server (Terminal 1)
python -m core.sidecar.server `
    --port 8771 `
    --project-root . `
    --model-path "..\models\Qwen2.5-1.5B-Instruct" `
    --adapter-path "artifacts\sidecar_v4\adapter"

# 4. Start CLI (Terminal 2)
python cli.py
# → Rich panel + mascot + HS v1.0.0 status banner
# → type a task, e.g. "write a hello world Python script"
```

Done. Every tool call passes through the Sidecar before execution.

---

## Architecture

```
┌──────────────────┐     tool call      ┌─────────────────────────────┐
│  Main Agent      │ ─────────────────→ │   Sidecar HTTP Server       │
│  (any LLM)       │ ← ──────────────── │   (Qwen2.5-1.5B + LoRA)    │
│                  │  block / allow     │                             │
└──────────────────┘                    │  Layer-0: Shell regex      │
                                        │  Layer-A: Vector + keyword │
                                        │  Layer-B: LoRA semantics    │
                                        │  Layer-C: Cross-file deps   │
                                        │  + Memory-adaptive RAG      │
                                        └─────────────────────────────┘
```

Main agent and Sidecar are **independent processes** — separate context, separate permissions, separate inference. The Sidecar can only *scrutinize*, never *generate*.

---

## What's Inside

| Capability | Details |
|---|---|
| Sidecar LoRA | `sidecar_v4` adapter (70.5MB, included in repo). 813 training samples, verifier-style system prompt, FORMAT_OK on every inference |
| Vector store | gte-small-zh (512-dim) + Faiss ScalarQuant 8bit + **memory-adaptive sharding** (auto-split on low memory, LRU evict cold shards) |
| Knowledge base | 325 drift patterns + 8 shell intents + 7 cross-file patterns |
| Shell firewall | regex blacklist + vector semantic match — blocks `curl\|sh`, `rm -rf /`, `git push --force`, etc. |
| Drift detection | forbidden keywords → vector retrieval → LoRA score (0.0–1.0). Anything ≥ threshold pauses execution |
| Cross-file deps | embedder cosine similarity detects "changing config.PORT without updating server.py" |
| SubAgent parallel | `SubAgentManager` with Sidecar pre/post-check, kill, wait_all, 6 roles |
| Computer Use | mss + RapidOCR + pywinauto UI Automation, DPI-aware, auto-save screenshots |
| MCP catalog | 12 built-in servers (GitHub, Postgres, Playwright, Sentry, Notion, Redis...) + deny-first approval |
| CLI | Rich panel banner + 28 slash commands |

---

## LongWorkspaceEval Baseline

| Scenario | Type | Recall | F1 |
|---|---|---|---|
| drift_skip_staging | drift | 50% | 67% |
| drift_debug_restart | drift | 50% | 67% |
| tool_curl_pipe_sh | shell | **100%** | **100%** |
| tool_rmrf_project | shell | 33% | 33% |
| cross_file_rename_symbol | cross_file | **100%** | 67% |
| cross_file_schema_change | cross_file | **100%** | 67% |

Run eval:

```powershell
python -m benchmark.eval --mode rule --out benchmark/run.json
```

---

## Project Structure

```
hs-sidecar-guard/
├── install.ps1                      ← one-click setup
├── cli.py                           ← entry point (Rich banner + 28 commands)
├── config/settings.json             ← LLM provider / API key / model path
├── requirements.txt
│
├── core/
│   ├── sidecar/                     ← THE CORE (dual-LLM innovation)
│   │   ├── server.py                ← Sidecar HTTP server (port 8771)
│   │   ├── drift.py                 ← Layer-A + Layer-B drift detection
│   │   ├── firewall/gate.py          ← Layer-0 Shell regex
│   │   ├── cross_file.py            ← Layer-C cross-file semantics
│   │   ├── vector_store.py           ← Faiss ScalarQuant 8bit + adaptive sharding
│   │   ├── embedding.py             ← gte-small-zh
│   │   ├── knowledge.py             ← vector + rule dual pipeline
│   │   └── knowledge/               ← JSON knowledge bases
│   ├── computer/                    ← desktop automation
│   │   ├── screen.py                ← mss + RapidOCR + UI Automation
│   │   └── input.py                 ← pyautogui (DPI-aware)
│   ├── mcp/                         ← MCP ecosystem
│   │   ├── registry.py              ← 12 built-in + deny-first approval
│   │   ├── client.py                ← stdio JSON-RPC
│   │   └── installer.py             ← GitHub one-click install
│   ├── agent/                       ← main agent + subagent
│   │   ├── agent.py                 ← 6 modes
│   │   └── subagent.py              ← SubAgentManager with Sidecar pre-check
│   └── training/                    ← LoRA fine-tuning utilities
│
├── artifacts/sidecar_v4/adapter/    ← LoRA adapter (included, 70.5MB)
├── data/                            ← training data + generators
└── benchmark/                       ← LongWorkspaceEval
```

---

## License

MIT © HS Core Contributors
