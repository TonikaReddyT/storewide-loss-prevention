---
description: "Use when adding, modifying, or debugging VLM (Vision Language Model) tools in the MCP server, analyzing surveillance or face images with multimodal models, working with base64 image inputs, configuring VLM_BASE_URL / VLM_MODEL, implementing vlm_describe_image / vlm_analyze_scene / vlm_list_models tools, or integrating vision models like LLaVA, Qwen-VL, GPT-4o, or MiniCPM-V."
---

# VLM Integration — POI MCP Server

## Overview

VLM tools live in `person-of-interest/mcp_server/tools/vlm_tools.py` and are registered via:

```python
def register(mcp: FastMCP, cfg: MCPConfig) -> None:
    ...
```

Called from `mcp_server/server.py` at startup. All VLM tools use the same `register(mcp, cfg)` signature.

---

## Configuration (`MCPConfig`)

| Env Var | `MCPConfig` Field | Default | Description |
|---|---|---|---|
| `VLM_BASE_URL` | `cfg.vlm_base_url` | `http://localhost:11434/v1` | OpenAI-compatible vision endpoint |
| `VLM_API_KEY` | `cfg.vlm_api_key` | `ollama` | API key |
| `VLM_MODEL` | `cfg.vlm_model` | `""` | Default vision model; empty = caller must specify |
| `MCP_VLM_TIMEOUT` | `cfg.vlm_timeout` | `120` | Timeout in seconds (longer than LLM — vision is slower) |
| `MCP_ALLOW_EXTERNAL_AI` | `cfg.allow_external_ai` | `False` | Must be `true` for non-local endpoints |

---

## Client Instantiation

Use `_client(cfg)` — same OpenAI client, pointed at the VLM endpoint:

```python
def _client(cfg: MCPConfig):
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")
    return OpenAI(
        base_url=cfg.vlm_base_url,
        api_key=cfg.vlm_api_key,
        timeout=cfg.vlm_timeout,
    )
```

---

## Guards — Always Apply Both

Every VLM tool that accepts an image must apply both guards before processing:

```python
@mcp.tool()
def vlm_describe_image(image_b64: str, prompt: str = "Describe this image.", model: str = "") -> dict:
    err = _guard_external(cfg) or _guard_image_size(image_b64)
    if err:
        return err
    ...
```

### External AI Guard

```python
def _guard_external(cfg: MCPConfig) -> Optional[dict]:
    if not _is_local(cfg.vlm_base_url) and not cfg.allow_external_ai:
        return {
            "error": (
                f"VLM endpoint '{cfg.vlm_base_url}' is not local. "
                "Set MCP_ALLOW_EXTERNAL_AI=true to enable calls to remote AI endpoints."
            )
        }
    return None
```

### Image Size Guard

Maximum accepted: **4 MB base64-encoded** (~3 MB raw image).

```python
_MAX_IMAGE_B64_BYTES = 4 * 1024 * 1024  # 4 MB encoded

def _guard_image_size(image_b64: str) -> Optional[dict]:
    if len(image_b64) > _MAX_IMAGE_B64_BYTES:
        size_mb = len(image_b64) / 1_048_576
        return {"error": f"Image too large ({size_mb:.1f} MB encoded). Maximum is 4 MB."}
    return None
```

---

## Image Input — OpenAI Vision Format

Build image content blocks using the `_image_url` helper — detect MIME type from magic bytes:

```python
def _image_url(image_b64: str) -> dict:
    """Build an OpenAI image_url content block from raw base64 bytes."""
    try:
        import base64
        raw = base64.b64decode(image_b64[:16])
        if raw[:2] == b"\xff\xd8":
            mime = "image/jpeg"
        elif raw[:8] == b"\x89PNG\r\n\x1a\n":
            mime = "image/png"
        else:
            mime = "image/jpeg"  # safe fallback
    except Exception:
        mime = "image/jpeg"
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}}
```

Message structure for multimodal requests:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            _image_url(image_b64),
        ],
    }
]
```

---

## Model Resolution

Same pattern as LLM tools — per-call override, then config default:

```python
def _resolve_model(cfg: MCPConfig, model: str) -> str:
    return model or cfg.vlm_model or ""

resolved_model = _resolve_model(cfg, model)
if not resolved_model:
    return {
        "error": "No model specified. Set VLM_MODEL or pass model= parameter. "
                 "Use vlm_list_models() to see available models."
    }
```

---

## Token Limits

Cap `max_tokens` to safe bounds before calling the model:

```python
max_tokens = max(1, min(max_tokens, 2048))
```

Recommended defaults:
- `vlm_describe_image`: `max_tokens=512`
- `vlm_analyze_scene`: `max_tokens=768`

---

## Existing VLM Tools (do not duplicate)

| Tool | Input | Output | Use Case |
|---|---|---|---|
| `vlm_list_models` | — | list of `{id, created, owned_by}` | Discover available vision models |
| `vlm_describe_image` | `image_b64`, `prompt`, `model`, `max_tokens` | `{description, model, finish_reason}` | General image description |
| `vlm_analyze_scene` | `image_b64`, `context`, `model`, `max_tokens` | `{analysis, model, finish_reason}` | Surveillance scene analysis for loss-prevention |

---

## Loss-Prevention Scene Analysis Prompt Pattern

When analyzing surveillance frames for suspicious activity:

```python
system_context = (
    "You are a retail loss-prevention analyst reviewing a surveillance camera frame. "
    "Identify: number of people visible, their approximate locations, any items being handled, "
    "and any behaviour that may warrant attention (concealment, loitering, unusual movement)."
)
if context:
    system_context += f"\n\nAdditional context: {context}"
```

---

## Privacy Rules for VLM

- Never send face crop images containing identified individuals to a remote VLM without `MCP_ALLOW_EXTERNAL_AI=true`.
- The `image_b64` parameter must never be logged — even at DEBUG level.
- If generating thumbnails for display, use `person-of-interest/backend/api/thumbnail_routes.py` paths, not raw base64 in API responses.

---

## Supported Vision Backends

| Backend | `VLM_BASE_URL` | Recommended Models |
|---|---|---|
| Ollama (local) | `http://localhost:11434/v1` | `llava`, `llava-phi3`, `moondream` |
| vLLM | `http://localhost:8080/v1` | `Qwen/Qwen2-VL-7B-Instruct` |
| LM Studio | `http://localhost:1234/v1` | LLaVA variants |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o` + `MCP_ALLOW_EXTERNAL_AI=true` |
