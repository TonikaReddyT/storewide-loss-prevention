"""MCP Server configuration — loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class MCPConfig:
    """All configuration for the POI MCP server.

    Mirrors the POI backend env vars where applicable and adds MCP-specific ones.
    """

    # ── GitHub ────────────────────────────────────────────
    github_token: str = ""
    github_org: str = ""

    # ── Jira ─────────────────────────────────────────────
    jira_url: str = ""
    jira_username: str = ""
    jira_api_token: str = ""
    jira_project_key: str = ""

    # ── Filesystem sandbox ────────────────────────────────
    filesystem_root: str = "/workspace/person-of-interest"

    # ── Redis ─────────────────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # ── POI Backend REST API ──────────────────────────────
    poi_backend_url: str = "http://localhost:8000"

    # ── MQTT ─────────────────────────────────────────────
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_ca_cert: str = ""
    mqtt_scene_uid: str = ""
    mqtt_event_buffer_size: int = 100

    # ── OpenVINO ─────────────────────────────────────────
    model_base: str = "/models/intel"
    det_model: str = ""
    lm_model: str = ""
    reid_model: str = ""
    inference_device: str = "CPU"

    # ── Docker ────────────────────────────────────────────
    docker_base_url: str = ""  # empty = use default socket

    # ── LLM ───────────────────────────────────────────────
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = ""  # empty = must be passed per-call or use llm_list_models
    llm_timeout: int = 60  # seconds

    # ── VLM ───────────────────────────────────────────────
    vlm_base_url: str = "http://localhost:11434/v1"
    vlm_api_key: str = "ollama"
    vlm_model: str = ""  # empty = must be passed per-call or use vlm_list_models
    vlm_timeout: int = 120  # seconds (vision inference is slower)

    # ── Security ─────────────────────────────────────────
    # When False (default), tools that mutate state (write files, create
    # issues, publish MQTT, start/stop containers, etc.) will refuse to run.
    allow_mutations: bool = False
    # When False (default), LLM/VLM calls to non-local endpoints are blocked
    # to prevent accidental exfiltration of surveillance or biometric data.
    allow_external_ai: bool = False

    # ── Deep Learning ─────────────────────────────────────
    # When False (default), DL model path tools are restricted to MODEL_BASE.
    dl_allow_all_paths: bool = False

    # ── Python execution sandbox ──────────────────────────
    python_exec_timeout: int = 30  # seconds

    # ── MCP transport ─────────────────────────────────────
    mcp_transport: str = "stdio"  # "stdio" or "streamable-http"
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 9000

    @classmethod
    def from_env(cls) -> MCPConfig:
        """Build config from environment variables."""
        model_base = os.getenv("MODEL_BASE", "/models/intel")
        scene_uid = os.getenv("SCENE_UID", "")
        return cls(
            github_token=os.getenv("GITHUB_TOKEN", ""),
            github_org=os.getenv("GITHUB_ORG", ""),
            jira_url=os.getenv("JIRA_URL", ""),
            jira_username=os.getenv("JIRA_USERNAME", ""),
            jira_api_token=os.getenv("JIRA_API_TOKEN", ""),
            jira_project_key=os.getenv("JIRA_PROJECT_KEY", ""),
            filesystem_root=os.getenv("MCP_FILESYSTEM_ROOT", "/workspace/person-of-interest"),
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_db=int(os.getenv("REDIS_DB", "0")),
            poi_backend_url=os.getenv("POI_BACKEND_URL", "http://localhost:8000"),
            mqtt_host=os.getenv("MQTT_HOST", "localhost"),
            mqtt_port=int(os.getenv("MQTT_PORT", "1883")),
            mqtt_ca_cert=os.getenv("MQTT_CA_CERT", ""),
            mqtt_scene_uid=scene_uid,
            mqtt_event_buffer_size=int(os.getenv("MQTT_EVENT_BUFFER_SIZE", "100")),
            model_base=model_base,
            det_model=os.getenv(
                "DET_MODEL",
                f"{model_base}/face-detection-retail-0004/FP32/face-detection-retail-0004.xml",
            ),
            lm_model=os.getenv(
                "LM_MODEL",
                f"{model_base}/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml",
            ),
            reid_model=os.getenv(
                "REID_MODEL",
                f"{model_base}/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml",
            ),
            inference_device=os.getenv("INFERENCE_DEVICE", "CPU"),
            docker_base_url=os.getenv("DOCKER_BASE_URL", ""),
            llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
            llm_api_key=os.getenv("LLM_API_KEY", "ollama"),
            llm_model=os.getenv("LLM_MODEL", ""),
            llm_timeout=int(os.getenv("MCP_LLM_TIMEOUT", "60")),
            vlm_base_url=os.getenv("VLM_BASE_URL", "http://localhost:11434/v1"),
            vlm_api_key=os.getenv("VLM_API_KEY", "ollama"),
            vlm_model=os.getenv("VLM_MODEL", ""),
            vlm_timeout=int(os.getenv("MCP_VLM_TIMEOUT", "120")),
            allow_mutations=os.getenv("MCP_ALLOW_MUTATIONS", "false").lower() == "true",
            allow_external_ai=os.getenv("MCP_ALLOW_EXTERNAL_AI", "false").lower() == "true",
            dl_allow_all_paths=os.getenv("MCP_DL_ALLOW_ALL_PATHS", "false").lower() == "true",
            python_exec_timeout=int(os.getenv("MCP_PYTHON_EXEC_TIMEOUT", "30")),
            mcp_transport=os.getenv("MCP_TRANSPORT", "stdio"),
            mcp_host=os.getenv("MCP_HOST", "0.0.0.0"),
            mcp_port=int(os.getenv("MCP_PORT", "9000")),
        )
