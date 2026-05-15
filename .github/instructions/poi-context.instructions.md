---
description: "Use when working on the Person of Interest (POI) re-identification system: adding features, debugging, understanding the pipeline, writing tests, or explaining architecture. Covers the POI application domain, service boundaries, real-time MQTT pipeline flow, domain entities, Redis key schema, FAISS index usage, and key file locations."
---

# POI Re-identification System — Application Context

## Purpose

Real-time retail loss-prevention system that answers:
- **Live**: "Is this enrolled suspect visible on any camera right now?" → triggers an alert.
- **Historical**: "Where did this person appear across all cameras in a given time window?" → returns a visit timeline.

Domain: biometric person re-identification in a multi-camera retail environment, integrated with Intel SceneScape spatial computing.

---

## Architecture

Follows **Clean Architecture**: `API → Service → Domain ← Infrastructure`

```
React UI (Vite/TS)         :3000
    ↕ WebSocket / REST
poi-backend (FastAPI)       :8000
    ↕ Redis (metadata/cache) :6379
    ↕ FAISS (vector index)   on-disk (/data/faiss/)
    ↕ MQTT (paho)
poi-alert-service            :8001   alert fan-out (log + WebSocket + MQTT)
poi-mcp-server (FastMCP)    :9000   Claude Desktop / LLM+VLM+OpenVINO tools
SceneScape / DLStreamer               upstream inference + MQTT publisher
```

---

## Domain Entities (`backend/domain/entities/`)

| Entity | Key Fields |
|---|---|
| `POI` | `poi_id`, `severity` (low/medium/high), `status` (active/inactive), `reference_images: list[ReferenceImage]`, `embedding_ids: list[str]` |
| `PersonEvent` / `MovementEvent` | `object_id`, `timestamp`, `camera_id`, `region_id`, `scene_id`, `embedding_vector: list[float]` (256-d), `poi_id` (populated on match), `dwell` |
| `MatchResult` | `poi_id`, `similarity_score` (0–1 cosine), `faiss_distance` |
| `AlertPayload` | `alert_id`, `poi_id`, `severity`, `camera_id`, `confidence`, `match.bbox`, `match.thumbnail_path`, `poi_metadata` |
| `ReferenceImage` | `embedding_id`, `vector_dim=256`, stored image path |

---

## Redis Key Schema

| Key Pattern | Contents | TTL |
|---|---|---|
| `poi:{poi_id}` | POI metadata JSON | permanent |
| `event:{object_id}:{timestamp}` | MovementEvent JSON | 7 days |
| `alert:{alert_id}` | AlertPayload JSON | 7 days |
| `embedding_map:{embedding_id}` | poi_id mapping | permanent |
| Cache-aside per object_id | FAISS result | 5 min |

---

## Real-Time Pipeline Flow

1. **DLStreamer** detects persons per camera → publishes to `scenescape/data/camera/{id}` with face sub-object embeddings (base64 IEEE-754 floats, 256-d, `face-reidentification-retail-0095` model).
2. **`mqtt_consumer.py`** decodes embedding → `MatchingService` checks Redis cache first (5 min TTL), then FAISS cosine search (threshold `0.60`, `IndexFlatIP` on L2-normed vectors).
3. On match: `EventBus.publish("match_found")` → `AlertService._on_match_found` → `AlertServiceStrategy.send()` → `poi-alert-service` fan-out (log + WebSocket to UI).
4. All movement events (matched or not) stored in Redis `event:{object_id}:{timestamp}` (7-day TTL).
5. Region entry/exit from `scenescape/regulated/scene/+` tracked by `scenescape_consumer.py` with dwell time computation.

MQTT topics:
- `scenescape/data/camera/+` — per-camera person detection events
- `scenescape/regulated/scene/+` — region entry/exit events
- `scenescape/external/+/person` — external person events

---

## Key Technologies

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn (async Python) |
| Vector search | FAISS `IndexFlatIP` (cosine on L2-normed 256-d vectors) |
| Metadata & event store | Redis 8.x |
| Computer vision / inference | OpenVINO — `face-detection-retail-0004`, `face-reidentification-retail-0095` |
| MQTT messaging | paho-mqtt |
| Frontend | React + TypeScript, Vite, TailwindCSS |
| MCP server | `mcp.server.fastmcp.FastMCP` |
| LLM/VLM integration | OpenAI-compatible API (Ollama / vLLM / LM Studio, default `localhost:11434/v1`) |
| Containerisation | Docker Compose, multi-service |

---

## Key File Locations

| File | Purpose |
|---|---|
| `person-of-interest/backend/main.py` | App entry — wires all layers at startup |
| `person-of-interest/backend/core/config.py` | `Config` singleton, all env vars |
| `person-of-interest/backend/domain/entities/` | Domain entity definitions |
| `person-of-interest/backend/service/matching_service.py` | FAISS cosine matching + cache-aside |
| `person-of-interest/backend/service/alert_service.py` | Alert dedup + EventBus wiring |
| `person-of-interest/backend/consumers/mqtt_consumer.py` | MQTT ingestion + embedding decode |
| `person-of-interest/backend/consumers/scenescape_consumer.py` | Region/dwell tracking |
| `person-of-interest/backend/infrastructure/faiss/repository.py` | FAISS CRUD + persistence |
| `person-of-interest/backend/infrastructure/redis/repository.py` | Redis repositories |
| `person-of-interest/backend/api/poi_routes.py` | POI CRUD endpoints |
| `person-of-interest/backend/api/search_routes.py` | Historical search endpoint |
| `person-of-interest/mcp_server/server.py` | MCP server entry point |
| `person-of-interest/mcp_server/config.py` | `MCPConfig` dataclass |
| `person-of-interest/mcp_server/tools/llm_tools.py` | LLM MCP tools |
| `person-of-interest/mcp_server/tools/vlm_tools.py` | VLM MCP tools |
| `person-of-interest/docker-compose.yml` | All services |
| `person-of-interest/configs/pipeline-config.json` | DLStreamer pipeline config |

---

## API Routes (`/api/v1`)

| Method | Path | Description |
|---|---|---|
| `POST` | `/poi` | Enroll new POI (1–5 images) |
| `GET` | `/poi` | List all POIs |
| `GET` | `/poi/{poi_id}` | Get single POI |
| `DELETE` | `/poi/{poi_id}` | Delete POI + FAISS vectors |
| `POST` | `/search` | Historical search by image + time range |
| `GET` | `/cameras` | List cameras (proxies SceneScape) |
| `GET` | `/alerts` | Last 50 POI match alerts |
| `GET` | `/status` | Health + FAISS vector count + MQTT state |
| `GET` | `/thumbnails/...` | Serve captured thumbnails |

---

## Security & Privacy Rules

- Face embeddings and biometric data must never be logged at DEBUG level in full.
- MCP tools sending data to non-local AI endpoints must be gated behind `MCP_ALLOW_EXTERNAL_AI=true`.
- MCP write operations must check `MCP_ALLOW_MUTATIONS=true`.
- Alert dedup: 60s TTL per `object_id` in backend; 60s SHA-1 hash window on `(poi_id, camera_id)` in alert-service.
- FAISS index and upload volumes are read-only in the MCP server container.
