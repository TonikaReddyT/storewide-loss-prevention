FROM python:3.11-slim

LABEL org.opencontainers.image.title="POI MCP Server"
LABEL org.opencontainers.image.description="MCP server exposing GitHub, Jira, Filesystem, Redis, OpenVINO, Python, MQTT, FAISS, and Docker tools for the POI system"

# System dependencies for OpenCV and paho-mqtt
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY mcp_server/requirements.txt /app/mcp_server/requirements.txt
RUN pip install --no-cache-dir -r /app/mcp_server/requirements.txt

# Copy the entire person-of-interest project so that 'backend' package is importable
COPY backend/ /app/backend/
COPY mcp_server/ /app/mcp_server/

# PYTHONPATH must include /app so both 'backend' and 'mcp_server' are importable
ENV PYTHONPATH=/app

# Default transport: streamable-http for containerized deployment
ENV MCP_TRANSPORT=streamable-http
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=9000
ENV LOG_LEVEL=INFO

EXPOSE 9000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -sf -o /dev/null -w "%{http_code}" http://localhost:9000/mcp | grep -qE "^(200|406)$" || exit 1

CMD ["python", "-m", "mcp_server.server"]
