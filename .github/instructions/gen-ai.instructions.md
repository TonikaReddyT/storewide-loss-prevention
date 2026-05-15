---
description: "Use when developing GenAI features, adding new MCP tools to the POI MCP server, implementing OpenVINO inference tools, building AI-powered analysis features, working with FastMCP tool registration, configuring MCP transport modes (stdio vs streamable-http), or integrating generative AI into the retail loss-prevention pipeline."
---

# Generative AI Development — POI MCP Server

## MCP Server Overview

The POI MCP server (`person-of-interest/mcp_server/`) exposes AI capabilities to Claude Desktop and other MCP clients via structured tools. Entry point: `mcp_server/server.py`.

Transport modes:
- **`stdio`** — for Claude Desktop integration (local, no network port needed)
- **`streamable-http`** — containerised, port `9000`, used in Docker Compose

---

## Tool Registration Pattern

Every tool module must implement a `register(mcp, cfg)` function and call it from `server.py`:

```python
# mcp_server/tools/my_tools.py
from mcp.server.fastmcp import FastMCP
from mcp_server.config import MCPConfig

def register(mcp: FastMCP, cfg: MCPConfig) -> None:
    """Register my tools on the MCP server."""

    @mcp.tool()
    def my_tool_name(param: str) -> dict:
        """One-line summary of what this tool does.

        Args:
            param: Description of the parameter.

        Returns:
            Dict with result or error key.
        """
        ...
```

Register in `server.py`:

```python
from mcp_server.tools import my_tools
my_tools.register(mcp, cfg)
```

---

## Tool Naming Conventions

Use the domain-prefix naming convention — all tools in this codebase follow this pattern:

| Prefix | Domain | Examples |
|---|---|---|
| `llm_` | LLM text generation | `llm_chat`, `llm_list_models`, `llm_summarize_events` |
| `vlm_` | Vision language models | `vlm_describe_image`, `vlm_analyze_scene`, `vlm_list_models` |
| `openvino_` | OpenVINO inference | `openvino_generate_face_embedding`, `openvino_benchmark_inference` |
| `docker_` | Container management | `docker_list_containers`, `docker_get_logs` |
| `poi_` | POI backend data | `poi_list`, `poi_get_events`, `poi_search` |
| `redis_` | Redis introspection | `redis_get_stats`, `redis_list_keys` |

---

## Write Operation Guard

All tools that mutate state (enroll POI, delete, clear index, etc.) must check `MCP_ALLOW_MUTATIONS`:

```python
def _guard_mutations(cfg: MCPConfig) -> Optional[dict]:
    if not cfg.allow_mutations:
        return {
            "error": "Mutation operations are disabled. "
                     "Set MCP_ALLOW_MUTATIONS=true to enable write operations."
        }
    return None

# Usage:
@mcp.tool()
def poi_delete(poi_id: str) -> dict:
    err = _guard_mutations(cfg)
    if err:
        return err
    ...
```

`MCPConfig` field: `cfg.allow_mutations` — defaults to `False`.

---

## External AI Guard

Both LLM and VLM tools check `cfg.allow_external_ai` before sending data to non-local endpoints. See `llm.instructions.md` and `vlm.instructions.md` for the exact implementation.

Local hosts definition: `("localhost", "127.0.0.1", "::1", "0.0.0.0")`.

---

## Tool Return Contract

All tools return `dict` — never raise exceptions to the MCP client:

```python
# Success
return {"result": ..., "model": ..., "count": ...}

# Error
return {"error": "Human-readable error message"}

# Partial success
return {"results": [...], "warnings": ["..."], "errors": []}
```

- Return dicts for single items, lists of dicts for collections.
- Include `"model"` in AI tool responses so the caller knows which model was used.
- Include `"count"` for listing tools.

---

## OpenVINO Tool Patterns

OpenVINO inference tools wrap the same models used by the backend:

- Detection: `face-detection-retail-0004`
- Re-ID: `face-reidentification-retail-0095` (outputs 256-d float32)
- Models base path: `cfg.model_base` (default `/models/intel`)
- Device: `cfg.inference_device` (default `CPU`)

```python
@mcp.tool()
def openvino_generate_face_embedding(image_b64: str) -> dict:
    """Generate a 256-d face embedding from a base64 image using OpenVINO.

    Returns:
        Dict with embedding (list of 256 floats) and inference_time_ms.
    """
    err = _guard_image_size(image_b64)
    if err:
        return err
    ...
```

Always validate image size (4 MB max) before passing to OpenVINO — same `_guard_image_size` helper as VLM tools.

---

## MCPConfig — Key Fields Reference

```python
# person-of-interest/mcp_server/config.py
cfg.poi_backend_url       # "http://localhost:8000" — POI REST API
cfg.redis_host            # "localhost"
cfg.redis_port            # 6379
cfg.llm_base_url          # "http://localhost:11434/v1"
cfg.llm_model             # "" (must be specified per-call if empty)
cfg.vlm_base_url          # "http://localhost:11434/v1"
cfg.vlm_model             # ""
cfg.allow_external_ai     # False — gate for remote AI endpoints
cfg.allow_mutations       # False — gate for write operations
cfg.model_base            # "/models/intel"
cfg.inference_device      # "CPU"
cfg.filesystem_root       # "/workspace/person-of-interest"
cfg.docker_base_url       # "" (uses default Docker socket)
```

---

## Privacy & Security Rules

1. **Biometric data never leaves the local network** without explicit `MCP_ALLOW_EXTERNAL_AI=true`.
2. Strip `embedding`, `embedding_vector`, `image`, `image_b64` fields from any data before sending to external LLMs.
3. Tools that expose raw Redis keys or FAISS internals should require read permissions — document clearly in the tool's docstring.
4. The Docker socket mount is read-only (`:ro`) — `docker_tools` must never start/stop/modify containers, only inspect and read logs.
5. All filesystem tools must be sandboxed to `cfg.filesystem_root` — reject paths that escape via `..` traversal.

---

## Testing MCP Tools

Test tools as plain Python functions — the `@mcp.tool()` decorator is transparent:

```python
# In mcp_server/tests/test_llm_tools.py
from unittest.mock import patch, MagicMock
from mcp_server.config import MCPConfig
from mcp_server.tools import llm_tools
from mcp.server.fastmcp import FastMCP

def test_llm_chat_blocks_external_without_flag():
    cfg = MCPConfig(llm_base_url="https://api.openai.com/v1", allow_external_ai=False)
    mcp = FastMCP("test")
    llm_tools.register(mcp, cfg)
    result = mcp.call_tool("llm_chat", {"prompt": "hello"})
    assert "error" in result
    assert "MCP_ALLOW_EXTERNAL_AI" in result["error"]
```

---

## Adding a New Tool Module — Checklist

- [ ] Create `mcp_server/tools/<domain>_tools.py`
- [ ] Implement `register(mcp: FastMCP, cfg: MCPConfig) -> None`
- [ ] Name tools with the correct domain prefix (`llm_`, `vlm_`, `openvino_`, etc.)
- [ ] Apply `_guard_external` for AI endpoint calls
- [ ] Apply `_guard_mutations` for any write operations
- [ ] Apply `_guard_image_size` for any image inputs
- [ ] Return `{"error": "..."}` on failure — never raise
- [ ] Add module to `server.py` registration block
- [ ] Add any new env vars to `MCPConfig` in `mcp_server/config.py`
- [ ] Add tests under `mcp_server/tests/`
