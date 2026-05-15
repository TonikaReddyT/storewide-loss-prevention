"""LLM MCP tools.

Provides tools for interacting with Large Language Models via an
OpenAI-compatible API endpoint (OpenAI, Ollama, vLLM, LM Studio, etc.).

Privacy note: These tools send prompt data to the configured LLM endpoint.
When LLM_BASE_URL points to a non-local host, calls are gated behind
MCP_ALLOW_EXTERNAL_AI=true to prevent accidental data exfiltration.

Configuration environment variables:
    LLM_BASE_URL          API base URL  (default: http://localhost:11434/v1)
    LLM_API_KEY           API key       (default: ollama)
    LLM_MODEL             Default model (default: empty — use llm_list_models)
    MCP_LLM_TIMEOUT       Request timeout in seconds (default: 60)
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

from mcp_server.config import MCPConfig

log = logging.getLogger("poi.mcp.llm")

_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "0.0.0.0")


def _is_local(base_url: str) -> bool:
    """Return True if *base_url* resolves to a local/loopback address."""
    try:
        from urllib.parse import urlparse
        host = urlparse(base_url).hostname or ""
        return host in _LOCAL_HOSTS
    except Exception:
        return False


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


def _guard_external(cfg: MCPConfig) -> Optional[dict]:
    """Return an error dict if external AI calls are not allowed, else None."""
    if not _is_local(cfg.llm_base_url) and not cfg.allow_external_ai:
        return {
            "error": (
                f"LLM endpoint '{cfg.llm_base_url}' is not local. "
                "Set MCP_ALLOW_EXTERNAL_AI=true to enable calls to remote AI endpoints."
            )
        }
    return None


def _resolve_model(cfg: MCPConfig, model: str) -> str:
    return model or cfg.llm_model or ""


def register(mcp: FastMCP, cfg: MCPConfig) -> None:
    """Register LLM tools on the MCP server."""

    @mcp.tool()
    def llm_list_models() -> list[dict]:
        """List available models from the configured LLM endpoint.

        Uses the OpenAI-compatible /models endpoint.

        Returns:
            List of dicts with id, created, and owned_by for each model.
        """
        err = _guard_external(cfg)
        if err:
            return [err]
        try:
            client = _client(cfg)
            models = client.models.list()
            return [
                {
                    "id": m.id,
                    "created": getattr(m, "created", None),
                    "owned_by": getattr(m, "owned_by", None),
                }
                for m in models.data
            ]
        except Exception as exc:
            return [{"error": str(exc)}]

    @mcp.tool()
    def llm_chat_complete(
        messages: list[dict],
        model: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> dict:
        """Send a chat message sequence to the LLM and return the response.

        Each message dict must have 'role' ('system'|'user'|'assistant')
        and 'content' (string).

        Privacy note: Message content is sent to the configured LLM endpoint.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            model: Model name override. Uses LLM_MODEL default when empty.
            max_tokens: Maximum tokens in the response (1–4096, default 1024).
            temperature: Sampling temperature 0.0–2.0 (default 0.7).

        Returns:
            Dict with model, content, finish_reason, prompt_tokens,
            completion_tokens, and total_tokens.
        """
        err = _guard_external(cfg)
        if err:
            return err
        resolved_model = _resolve_model(cfg, model)
        if not resolved_model:
            return {
                "error": "No model specified. Set LLM_MODEL or pass model= parameter. "
                "Use llm_list_models() to see available models."
            }
        max_tokens = max(1, min(max_tokens, 4096))
        temperature = max(0.0, min(temperature, 2.0))
        try:
            client = _client(cfg)
            resp = client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            choice = resp.choices[0]
            usage = resp.usage or {}
            return {
                "model": resp.model,
                "content": choice.message.content,
                "finish_reason": choice.finish_reason,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def llm_summarize_events(events: list[dict], model: str = "") -> dict:
        """Summarize a list of POI detection events using the LLM.

        Generates a concise natural-language summary of detection activity,
        highlighting frequency, timing, and any patterns worth investigating.

        Privacy note: Event metadata (timestamps, zone names, detection counts)
        is sent to the configured LLM endpoint. Do NOT include raw embeddings
        or image data in the events list.

        Args:
            events: List of POI event dicts (e.g. from mqtt or FAISS tools).
                    Truncated to 50 events maximum.
            model: Model name override. Uses LLM_MODEL default when empty.

        Returns:
            Dict with summary (string), event_count, and model used.
        """
        err = _guard_external(cfg)
        if err:
            return err
        resolved_model = _resolve_model(cfg, model)
        if not resolved_model:
            return {
                "error": "No model specified. Set LLM_MODEL or pass model= parameter. "
                "Use llm_list_models() to see available models."
            }
        # Truncate and strip any embedding vectors before sending
        safe_events = []
        for ev in events[:50]:
            safe_ev = {k: v for k, v in ev.items() if k not in ("embedding", "image", "image_b64")}
            safe_events.append(safe_ev)

        prompt = (
            "You are an analyst for a retail loss-prevention system. "
            "Summarize the following POI (Person of Interest) detection events. "
            "Highlight: detection frequency, time patterns, zones involved, and anything unusual.\n\n"
            f"Events ({len(safe_events)}):\n{json.dumps(safe_events, indent=2, default=str)}"
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            client = _client(cfg)
            resp = client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                max_tokens=1024,
                temperature=0.3,
            )
            return {
                "summary": resp.choices[0].message.content,
                "event_count": len(safe_events),
                "model": resp.model,
            }
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def llm_analyze_alert(alert: dict, model: str = "") -> dict:
        """Analyze a POI alert and suggest investigative actions.

        Uses the LLM to interpret an alert, assess risk level, and recommend
        next steps for loss-prevention operators.

        Privacy note: Alert metadata is sent to the configured LLM endpoint.
        Do NOT include raw embeddings or image data in the alert dict.

        Args:
            alert: Alert dict (e.g. from FAISS/MQTT tools). Embedding and
                   image fields are automatically stripped before sending.
            model: Model name override. Uses LLM_MODEL default when empty.

        Returns:
            Dict with summary, risk_level, recommended_actions (list),
            confidence, and model.
        """
        err = _guard_external(cfg)
        if err:
            return err
        resolved_model = _resolve_model(cfg, model)
        if not resolved_model:
            return {
                "error": "No model specified. Set LLM_MODEL or pass model= parameter. "
                "Use llm_list_models() to see available models."
            }
        safe_alert = {k: v for k, v in alert.items() if k not in ("embedding", "image", "image_b64")}
        prompt = (
            "You are a loss-prevention analyst. Analyze the following POI alert "
            "and respond in JSON with these exact fields: "
            '"summary" (string), "risk_level" ("low"|"medium"|"high"|"critical"), '
            '"recommended_actions" (list of strings), "confidence" ("low"|"medium"|"high"), '
            '"uncertainties" (list of strings).\n\n'
            f"Alert data:\n{json.dumps(safe_alert, indent=2, default=str)}"
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            client = _client(cfg)
            resp = client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                max_tokens=512,
                temperature=0.2,
            )
            content = resp.choices[0].message.content or ""
            # Attempt to parse structured JSON from the response
            try:
                # Strip markdown code fences if present
                stripped = content.strip()
                if stripped.startswith("```"):
                    stripped = "\n".join(stripped.split("\n")[1:])
                if stripped.endswith("```"):
                    stripped = stripped[: stripped.rfind("```")]
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = {"summary": content}
            parsed["model"] = resp.model
            return parsed
        except Exception as exc:
            return {"error": str(exc)}

    log.info(
        "LLM tools registered (base_url=%s, model=%s, timeout=%ds, external_ai=%s)",
        cfg.llm_base_url,
        cfg.llm_model or "<unset>",
        cfg.llm_timeout,
        cfg.allow_external_ai,
    )
