"""VLM MCP tools.

Provides tools for interacting with Vision Language Models (VLMs) via an
OpenAI-compatible API endpoint that supports image inputs (e.g. LLaVA via
Ollama, GPT-4o, Qwen-VL, MiniCPM-V, etc.).

Privacy note: These tools send image data and prompts to the configured VLM
endpoint. When VLM_BASE_URL points to a non-local host, calls are gated
behind MCP_ALLOW_EXTERNAL_AI=true to prevent accidental data exfiltration.

Configuration environment variables:
    VLM_BASE_URL          API base URL  (default: http://localhost:11434/v1)
    VLM_API_KEY           API key       (default: ollama)
    VLM_MODEL             Default model (default: empty — use vlm_list_models)
    MCP_VLM_TIMEOUT       Request timeout in seconds (default: 120)
"""

from __future__ import annotations

import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

from mcp_server.config import MCPConfig

log = logging.getLogger("poi.mcp.vlm")

_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "0.0.0.0")

# Maximum base64 image size accepted (4 MB encoded ≈ ~3 MB raw image)
_MAX_IMAGE_B64_BYTES = 4 * 1024 * 1024


def _is_local(base_url: str) -> bool:
    try:
        from urllib.parse import urlparse
        host = urlparse(base_url).hostname or ""
        return host in _LOCAL_HOSTS
    except Exception:
        return False


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


def _guard_external(cfg: MCPConfig) -> Optional[dict]:
    if not _is_local(cfg.vlm_base_url) and not cfg.allow_external_ai:
        return {
            "error": (
                f"VLM endpoint '{cfg.vlm_base_url}' is not local. "
                "Set MCP_ALLOW_EXTERNAL_AI=true to enable calls to remote AI endpoints."
            )
        }
    return None


def _guard_image_size(image_b64: str) -> Optional[dict]:
    if len(image_b64) > _MAX_IMAGE_B64_BYTES:
        size_mb = len(image_b64) / 1_048_576
        return {"error": f"Image too large ({size_mb:.1f} MB encoded). Maximum is 4 MB."}
    return None


def _resolve_model(cfg: MCPConfig, model: str) -> str:
    return model or cfg.vlm_model or ""


def _image_url(image_b64: str) -> dict:
    """Build an OpenAI image_url content block from raw base64 bytes."""
    # Detect JPEG vs PNG from magic bytes
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


def register(mcp: FastMCP, cfg: MCPConfig) -> None:
    """Register VLM (Vision Language Model) tools on the MCP server."""

    @mcp.tool()
    def vlm_list_models() -> list[dict]:
        """List available models from the configured VLM endpoint.

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
    def vlm_describe_image(
        image_b64: str,
        prompt: str = "Describe this image in detail.",
        model: str = "",
        max_tokens: int = 512,
    ) -> dict:
        """Describe the contents of an image using a Vision Language Model.

        Privacy note: The image and prompt are sent to the configured VLM
        endpoint. Do not pass images containing sensitive PII unless the
        endpoint is local or MCP_ALLOW_EXTERNAL_AI is explicitly set.

        Args:
            image_b64: Base64-encoded image bytes (JPEG or PNG, max 4 MB).
            prompt: Instruction for the VLM (default: describe the image).
            model: Model name override. Uses VLM_MODEL default when empty.
            max_tokens: Maximum tokens in the response (1–2048, default 512).

        Returns:
            Dict with description (string), model, and finish_reason.
        """
        err = _guard_external(cfg) or _guard_image_size(image_b64)
        if err:
            return err
        resolved_model = _resolve_model(cfg, model)
        if not resolved_model:
            return {
                "error": "No model specified. Set VLM_MODEL or pass model= parameter. "
                "Use vlm_list_models() to see available models."
            }
        max_tokens = max(1, min(max_tokens, 2048))
        try:
            client = _client(cfg)
            resp = client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            _image_url(image_b64),
                        ],
                    }
                ],
                max_tokens=max_tokens,
            )
            return {
                "description": resp.choices[0].message.content,
                "model": resp.model,
                "finish_reason": resp.choices[0].finish_reason,
            }
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def vlm_analyze_scene(
        image_b64: str,
        context: str = "",
        model: str = "",
        max_tokens: int = 768,
    ) -> dict:
        """Analyze a surveillance scene image for unusual or suspicious activity.

        Uses a VLM to describe the scene, identify the number of people,
        notable behaviours, and any activity that may warrant attention in
        a retail loss-prevention context.

        Privacy note: Image data is sent to the configured VLM endpoint.

        Args:
            image_b64: Base64-encoded scene image (JPEG or PNG, max 4 MB).
            context: Optional additional context (e.g. zone name, time of day).
            model: Model name override. Uses VLM_MODEL default when empty.
            max_tokens: Maximum tokens in the response (default 768).

        Returns:
            Dict with scene_description, people_count_estimate,
            notable_activity (list), risk_indicators (list), model.
        """
        err = _guard_external(cfg) or _guard_image_size(image_b64)
        if err:
            return err
        resolved_model = _resolve_model(cfg, model)
        if not resolved_model:
            return {
                "error": "No model specified. Set VLM_MODEL or pass model= parameter. "
                "Use vlm_list_models() to see available models."
            }
        context_clause = f"\nAdditional context: {context}" if context else ""
        prompt = (
            "You are assisting a retail loss-prevention team. Analyze this scene image and respond "
            "in JSON with these exact fields: "
            '"scene_description" (string), '
            '"people_count_estimate" (integer or null), '
            '"notable_activity" (list of strings), '
            '"risk_indicators" (list of strings — only genuine concerns, empty list if none).'
            + context_clause
        )
        max_tokens = max(1, min(max_tokens, 2048))
        try:
            import json
            client = _client(cfg)
            resp = client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            _image_url(image_b64),
                        ],
                    }
                ],
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            try:
                stripped = content.strip()
                if stripped.startswith("```"):
                    stripped = "\n".join(stripped.split("\n")[1:])
                if stripped.endswith("```"):
                    stripped = stripped[: stripped.rfind("```")]
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = {"scene_description": content}
            parsed["model"] = resp.model
            return parsed
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def vlm_describe_face_images(
        image_b64_1: str,
        image_b64_2: str,
        model: str = "",
        max_tokens: int = 512,
    ) -> dict:
        """Provide a descriptive visual comparison of two face-crop images.

        IMPORTANT: This tool provides a natural-language description only.
        It is NOT a biometric identification system and must NOT be used
        to make identity decisions. For identity matching, use the
        openvino_generate_face_embedding tool and cosine similarity.

        Privacy note: Both images are sent to the configured VLM endpoint.

        Args:
            image_b64_1: Base64-encoded first face image (JPEG or PNG, max 4 MB).
            image_b64_2: Base64-encoded second face image (JPEG or PNG, max 4 MB).
            model: Model name override. Uses VLM_MODEL default when empty.
            max_tokens: Maximum tokens in the response (default 512).

        Returns:
            Dict with description_image1, description_image2,
            visual_comparison (descriptive only), disclaimer, and model.
        """
        for img, label in ((image_b64_1, "image_b64_1"), (image_b64_2, "image_b64_2")):
            err = _guard_external(cfg) or _guard_image_size(img)
            if err:
                return {**err, "field": label}
        resolved_model = _resolve_model(cfg, model)
        if not resolved_model:
            return {
                "error": "No model specified. Set VLM_MODEL or pass model= parameter. "
                "Use vlm_list_models() to see available models."
            }
        prompt = (
            "Describe the appearance of the person in each image separately (hair colour, "
            "approximate age range, clothing, notable features). Then provide a brief descriptive "
            "comparison of visible attributes. Do NOT make any claims about identity or whether "
            "the images show the same person — that determination requires biometric analysis."
        )
        max_tokens = max(1, min(max_tokens, 2048))
        try:
            import json
            client = _client(cfg)
            resp = client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            _image_url(image_b64_1),
                            _image_url(image_b64_2),
                        ],
                    }
                ],
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            return {
                "visual_comparison": content,
                "disclaimer": (
                    "This is a descriptive comparison only. "
                    "It is NOT a biometric identity match. "
                    "Use openvino_generate_face_embedding for identity decisions."
                ),
                "model": resp.model,
            }
        except Exception as exc:
            return {"error": str(exc)}

    log.info(
        "VLM tools registered (base_url=%s, model=%s, timeout=%ds, external_ai=%s)",
        cfg.vlm_base_url,
        cfg.vlm_model or "<unset>",
        cfg.vlm_timeout,
        cfg.allow_external_ai,
    )
