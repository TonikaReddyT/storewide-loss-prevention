"""POI MCP Server — entry point.

Wires together all tool categories and runs the MCP server.

Transport is controlled by the MCP_TRANSPORT env var:
  - "stdio"            (default) — for Claude Desktop integration
  - "streamable-http"            — for containerized HTTP/SSE deployment

All nine tool categories are registered on a single FastMCP instance.
Mutating tools are gated behind MCP_ALLOW_MUTATIONS=true.

Usage:
    python -m mcp_server.server
    MCP_TRANSPORT=streamable-http MCP_PORT=9000 python -m mcp_server.server
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from mcp_server.config import MCPConfig
from mcp_server.tools import (
    deep_learning_tools,
    docker_tools,
    faiss_tools,
    filesystem_tools,
    github_tools,
    jira_tools,
    llm_tools,
    mqtt_tools,
    openvino_tools,
    python_tools,
    redis_tools,
    vlm_tools,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("poi.mcp.server")

cfg = MCPConfig.from_env()


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Server lifecycle — start background services on startup, stop on shutdown."""
    log.info("=== POI MCP Server Starting (transport=%s, mutations=%s) ===", cfg.mcp_transport, cfg.allow_mutations)
    mqtt_tools.start_subscriber(cfg)
    try:
        yield
    finally:
        log.info("=== POI MCP Server Shutting Down ===")
        mqtt_tools.stop_subscriber()


mcp = FastMCP(
    "poi-mcp-server",
    instructions=(
        "MCP server for the POI (Person of Interest) retail loss-prevention re-identification system. "
        "Integrated with SceneScape via MQTT and REST. "
        "Tool categories: GitHub, Jira, Filesystem, Redis, OpenVINO, Python, MQTT Events, "
        "FAISS/POI management, Docker, LLM (text generation), VLM (vision-language), "
        "and Deep Learning utilities. "
        "Read operations are always available. Mutating operations require MCP_ALLOW_MUTATIONS=true. "
        "LLM/VLM calls to non-local endpoints require MCP_ALLOW_EXTERNAL_AI=true."
    ),
    host=cfg.mcp_host,
    port=cfg.mcp_port,
    lifespan=lifespan,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
)

# ── Register all tool categories ─────────────────────────────────────────────
github_tools.register(mcp, cfg)
jira_tools.register(mcp, cfg)
filesystem_tools.register(mcp, cfg)
redis_tools.register(mcp, cfg)
openvino_tools.register(mcp, cfg)
python_tools.register(mcp, cfg)
mqtt_tools.register(mcp, cfg)
faiss_tools.register(mcp, cfg)
docker_tools.register(mcp, cfg)
llm_tools.register(mcp, cfg)
vlm_tools.register(mcp, cfg)
deep_learning_tools.register(mcp, cfg)


def main() -> None:
    transport = cfg.mcp_transport
    if transport == "streamable-http":
        log.info("Starting HTTP/SSE server on %s:%d", cfg.mcp_host, cfg.mcp_port)
        mcp.run(transport="streamable-http")
    else:
        log.info("Starting stdio MCP server")
        mcp.run()


if __name__ == "__main__":
    main()
