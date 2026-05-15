---
description: "Use when writing, editing, or reviewing Dockerfiles or docker-compose files in the POI project. Covers service naming conventions, port assignments, named volume usage, external network wiring, multi-stage Dockerfile patterns, read-only socket mounts, and environment variable conventions."
applyTo: ["**/Dockerfile", "**/docker-compose*.yml", "**/docker-compose*.yaml"]
---

# Docker Conventions — POI Project

## Service Naming

All POI services use the `poi-` prefix:

| Service Name | Image | Port | Role |
|---|---|---|---|
| `poi-backend` | Built from `backend/Dockerfile` | `8000` | FastAPI backend |
| `poi-ui` | Built from `ui/Dockerfile` | `3000→80` | React UI (nginx) |
| `poi-redis` | `redis:8.6.2-alpine` | `6379` | Metadata + event store |
| `poi-alert-service` | `intel/alert-service:0.0.1` | `8001` | Alert fan-out |
| `poi-mcp-server` | Built from `mcp_server/Dockerfile` | `9000` | MCP server (LLM/VLM/OpenVINO tools) |

---

## Port Conventions

| Service | Host Port | Container Port |
|---|---|---|
| poi-backend | `8000` | `8000` |
| poi-ui | `3000` | `80` |
| poi-redis | `6379` | `6379` |
| poi-alert-service | `8001` | `8001` |
| poi-mcp-server | `9000` | `9000` |

Do not reassign these ports without updating all dependent services and the `MCPConfig.poi_backend_url`.

---

## Named Volumes

| Volume | Mounted In | Access | Contents |
|---|---|---|---|
| `faiss-data` | `poi-backend` (rw), `poi-mcp-server` (ro) | see note | FAISS index files (`poi.index`, `id_map.json`) |
| `upload-data` | `poi-backend` (rw), `poi-mcp-server` (ro) | see note | Uploaded reference images + captured face thumbnails |

- The MCP server must **always** mount these volumes read-only (`:ro`) — it never writes to the index.
- Mount paths: `/data/faiss/` and `/data/uploads/`.

```yaml
volumes:
  - faiss-data:/data/faiss
  - upload-data:/data/uploads:ro   # :ro in mcp-server
```

---

## External Network

The backend connects to SceneScape's MQTT broker via the `storewide-lp` external network:

```yaml
networks:
  storewide-lp:
    external: true
  poi-internal:
    driver: bridge
```

- `poi-backend` must be on both `storewide-lp` and `poi-internal`.
- `poi-redis`, `poi-ui`, `poi-mcp-server`, `poi-alert-service` only need `poi-internal`.
- Never attach the Redis container to `storewide-lp`.

---

## Docker Socket Mount (MCP Server Only)

The MCP server mounts the Docker socket for `docker_tools` container management — always read-only:

```yaml
volumes:
  - /var/run/docker.sock:/var/run/docker.sock:ro
```

---

## Dockerfile Patterns

### Multi-stage Python service

```dockerfile
# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim
WORKDIR /app

# System deps for OpenCV (required for image processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY . .

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### React UI (nginx)

```dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY package-lock.json package.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
```

---

## Environment Variable Conventions in Compose

Use environment variables for all runtime configuration — never hardcode URLs or credentials:

```yaml
environment:
  - REDIS_HOST=poi-redis
  - REDIS_PORT=6379
  - MQTT_HOST=${MQTT_HOST}
  - MQTT_CA_CERT=/run/secrets/ca.crt
  - SIMILARITY_THRESHOLD=0.6
  - FAISS_INDEX_PATH=/data/faiss/poi.index
```

Secrets (TLS certs, auth tokens) must use Docker secrets or bind-mounted files — never plain env vars.

---

## Health Checks

Add health checks to all long-running services:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/status"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 20s
```

For Redis:

```yaml
healthcheck:
  test: ["CMD", "redis-cli", "ping"]
  interval: 10s
  timeout: 5s
  retries: 5
```

---

## Dependency Order

```yaml
depends_on:
  poi-redis:
    condition: service_healthy
```

Always wait for Redis to be healthy before starting `poi-backend` or `poi-mcp-server`.

---

## Image Pinning

Pin all base images to specific versions — never use `latest`:

```yaml
image: redis:8.6.2-alpine      # good
image: redis:latest             # bad — non-reproducible
```
