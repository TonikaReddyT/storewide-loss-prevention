---
description: "Use when adding, modifying, or debugging LLM (Large Language Model) tools in the MCP server, integrating text generation into the POI pipeline, working with OpenAI-compatible endpoints (Ollama, vLLM, LM Studio), configuring LLM_BASE_URL / LLM_MODEL / LLM_API_KEY, or implementing llm_chat / llm_complete / llm_list_models tools."
---

# LLM Integration — POI MCP Server

## Overview

LLM tools live in `person-of-interest/mcp_server/tools/llm_tools.py` and are registered via:

```python
def register(mcp: FastMCP, cfg: MCPConfig) -> None:
    ...
```

Called from `mcp_server/server.py` at startup. Follow this same `register(mcp, cfg)` signature for all new LLM tool modules.

---

## Configuration (`MCPConfig`)

| Env Var | `MCPConfig` Field | Default | Description |
|---|---|---|---|
| `LLM_BASE_URL` | `cfg.llm_base_url` | `http://localhost:11434/v1` | OpenAI-compatible endpoint |
| `LLM_API_KEY` | `cfg.llm_api_key` | `ollama` | API key (use `ollama` for local) |
| `LLM_MODEL` | `cfg.llm_model` | `""` | Default model; empty = caller must specify |
| `MCP_LLM_TIMEOUT` | `cfg.llm_timeout` | `60` | Request timeout in seconds |
| `MCP_ALLOW_EXTERNAL_AI` | `cfg.allow_external_ai` | `False` | Must be `true` for non-local endpoints |

---

## Client Instantiation Pattern

Always use the `_client(cfg)` helper — never instantiate `OpenAI` inline:

```python
from openai import OpenAI

def _client(cfg: MCPConfig):
    """Return a configured OpenAI client, raising on missing dependency."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")
    return OpenAI(
        base_url=cfg.llm_base_url,
        api_key=cfg.llm_api_key,
        timeout=cfg.llm_timeout,
    )
```

---

## External AI Guard

Gate every LLM call behind `_guard_external(cfg)` — return the error dict early if set:

```python
def _guard_external(cfg: MCPConfig) -> Optional[dict]:
    if not _is_local(cfg.llm_base_url) and not cfg.allow_external_ai:
        return {
            "error": (
                f"LLM endpoint '{cfg.llm_base_url}' is not local. "
                "Set MCP_ALLOW_EXTERNAL_AI=true to enable calls to remote AI endpoints."
            )
        }
    return None

# Usage in a tool:
@mcp.tool()
def llm_chat(prompt: str, model: str = "") -> dict:
    err = _guard_external(cfg)
    if err:
        return err
    ...
```

Local hosts: `localhost`, `127.0.0.1`, `::1`, `0.0.0.0`.

---

## Model Resolution

Always use the `_resolve_model` pattern — prefer the per-call `model` parameter, fall back to `cfg.llm_model`:

```python
def _resolve_model(cfg: MCPConfig, model: str) -> str:
    return model or cfg.llm_model or ""

# In tool body:
resolved_model = _resolve_model(cfg, model)
if not resolved_model:
    return {
        "error": "No model specified. Set LLM_MODEL or pass model= parameter. "
                 "Use llm_list_models() to see available models."
    }
```

---

## Data Privacy Before Sending

When sending POI events or surveillance data to the LLM, **strip** embedding vectors and images first — never send biometric data to a remote LLM:

```python
safe_events = []
for ev in events[:50]:   # also cap at 50 events
    safe_ev = {k: v for k, v in ev.items()
               if k not in ("embedding", "image", "image_b64", "embedding_vector")}
    safe_events.append(safe_ev)
```

---

## Standard Tool Return Shape

Tools should return dicts, not raise exceptions. Errors go in the `"error"` key:

```python
try:
    client = _client(cfg)
    resp = client.chat.completions.create(
        model=resolved_model,
        messages=messages,
        max_tokens=1024,
        temperature=0.3,
    )
    return {
        "result": resp.choices[0].message.content,
        "model": resp.model,
    }
except Exception as exc:
    log.exception("LLM call failed")
    return {"error": str(exc)}
```

---

## Existing LLM Tools (do not duplicate)

| Tool | Description |
|---|---|
| `llm_list_models` | Lists models from `LLM_BASE_URL/models` |
| `llm_chat` | Single-turn chat completion (messages list) |
| `llm_complete` | Raw text completion (legacy prompt style) |
| `llm_summarize_events` | Summarises POI detection events with a retail loss-prevention system prompt; strips embeddings/images before sending |

---

## Loss-Prevention System Prompt Pattern

When building LLM prompts about POI events, use this domain-aware framing:

```python
prompt = (
    "You are an analyst for a retail loss-prevention system. "
    "Summarize the following POI (Person of Interest) detection events. "
    "Highlight: detection frequency, time patterns, zones involved, and anything unusual.\n\n"
    f"Events ({len(safe_events)}):\n{json.dumps(safe_events, indent=2, default=str)}"
)
```

---

## Supported Backends

| Backend | `LLM_BASE_URL` | `LLM_API_KEY` |
|---|---|---|
| Ollama (local) | `http://localhost:11434/v1` | `ollama` |
| vLLM | `http://localhost:8080/v1` | `token-abc` |
| LM Studio | `http://localhost:1234/v1` | `lm-studio` |
| OpenAI | `https://api.openai.com/v1` | your key + `MCP_ALLOW_EXTERNAL_AI=true` |
