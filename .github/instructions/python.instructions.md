---
description: "Use when writing, editing, or reviewing Python code in this project. Covers async FastAPI patterns, Clean Architecture layer boundaries, Redis repository conventions, FAISS usage, paho-mqtt patterns, OpenVINO embedding pipeline, configuration via dataclass singletons, and error handling idioms used throughout the POI backend."
applyTo: "**/*.py"
---

# Python Coding Standards — POI Project

## Architecture Layers (Clean Architecture)

Respect the following strict layer boundary — never import upward:

```
API (api/)
  ↓ calls
Service (service/)
  ↓ calls
Domain (domain/)        ← pure Python, no infrastructure imports
  ↑ implemented by
Infrastructure (infrastructure/)
```

- `api/` routes must only call `service/` classes — never repositories directly.
- `service/` classes receive repository interfaces via constructor injection (DI at `main.py`).
- `domain/entities/` contains plain dataclasses/Pydantic models — zero external dependencies.
- `infrastructure/` contains Redis, FAISS, MQTT, and SceneScape adapters.

---

## FastAPI Patterns

```python
# Prefer async route handlers
@router.get("/items/{id}")
async def get_item(id: str) -> ItemResponse:
    ...

# Dependency injection via FastAPI Depends
@router.post("/poi")
async def create_poi(
    payload: POICreateRequest,
    service: POIService = Depends(get_poi_service),
) -> POIResponse:
    ...
```

- Return Pydantic response models, not raw dicts.
- Raise `HTTPException` with appropriate status codes; never let domain exceptions bubble to the client unhandled.
- Register routers in `main.py` under the `/api/v1` prefix.

---

## Configuration

Use `get_config()` (singleton) from `backend.core.config`:

```python
from backend.core.config import get_config

cfg = get_config()
threshold = cfg.similarity_threshold   # 0.6
```

- All env vars are centralised in the `Config` dataclass in `backend/core/config.py`.
- MCP server uses `MCPConfig` from `mcp_server/config.py` — same pattern.
- Never `os.environ.get()` inline outside config files.

---

## Redis Repository Conventions

Key naming — always use these exact patterns:

| Pattern | Example | TTL |
|---|---|---|
| `poi:{poi_id}` | `poi:abc-123` | permanent |
| `event:{object_id}:{timestamp}` | `event:obj-7:1714500000.0` | 7 days (`appearance_ttl_days`) |
| `alert:{alert_id}` | `alert:xyz-456` | 7 days |
| `embedding_map:{embedding_id}` | `embedding_map:emb-0` | permanent |

```python
# Store with TTL
await redis.setex(f"event:{obj_id}:{ts}", ttl_seconds, json.dumps(payload))

# Fetch and deserialise
raw = await redis.get(f"poi:{poi_id}")
if raw:
    poi = POI(**json.loads(raw))
```

---

## FAISS Usage

```python
# Always L2-normalise before adding/searching
import faiss, numpy as np

vec = np.array(embedding, dtype=np.float32)
faiss.normalize_L2(vec.reshape(1, -1))

# Index type: IndexFlatIP (inner product = cosine after normalisation)
# Dimension: 256 (face-reidentification-retail-0095 output)
# Threshold: cfg.similarity_threshold (default 0.6)
distances, indices = index.search(vec.reshape(1, -1), cfg.search_top_k)
```

- Persist index to disk at `cfg.faiss_index_path` after every write.
- Use `cfg.faiss_id_map_path` (JSON) to map FAISS integer indices → `embedding_id` strings.
- FAISS operations are not thread-safe — protect with a threading lock.

---

## paho-mqtt Patterns

```python
import paho.mqtt.client as mqtt

# Always set clean_session=False for durable subscriptions
client = mqtt.Client(client_id="poi-backend", clean_session=False)

# TLS when ca_cert is set
if cfg.mqtt_ca_cert:
    client.tls_set(ca_certs=cfg.mqtt_ca_cert)

# Reconnect on disconnect
def on_disconnect(client, userdata, rc):
    if rc != 0:
        client.reconnect()
```

MQTT topics consumed by the POI backend:
- `scenescape/data/camera/+` — person detection events with face embeddings
- `scenescape/regulated/scene/+` — region entry/exit events
- `scenescape/external/+/person` — external person events

Embeddings arrive as base64-encoded IEEE-754 float32 arrays (256 floats):

```python
import base64, struct, numpy as np

raw = base64.b64decode(embedding_b64)
vector = list(struct.unpack(f"{len(raw)//4}f", raw))
```

---

## OpenVINO Embedding Pipeline

Models live at `cfg.model_base` (default `/models/intel`):
- Detection: `face-detection-retail-0004` (`cfg.det_model`)
- Landmark: `cfg.lm_model`
- Re-ID: `face-reidentification-retail-0095` (`cfg.reid_model`), outputs 256-d float32 vectors

```python
from openvino.runtime import Core

core = Core()
det_model = core.read_model(cfg.det_model)
compiled = core.compile_model(det_model, cfg.inference_device)
```

- Prefer `cfg.inference_device = "CPU"` unless GPU is confirmed available.
- Wrap inference in try/except — OpenVINO raises on shape mismatches; log and skip the frame, never crash.

---

## Logging

```python
import logging
log = logging.getLogger("poi.<module_name>")
# e.g. "poi.service.matching", "poi.consumer.mqtt", "poi.mcp.llm"
```

- Never log full embedding vectors or raw face images — biometric PII.
- Log at INFO for service-level events, DEBUG for frame-level detail.
- Use structured messages: `log.info("Match found: poi=%s score=%.3f camera=%s", poi_id, score, cam_id)`

---

## Error Handling

- Domain/service errors: raise custom exceptions from `backend/domain/` — never `Exception` bare.
- Infrastructure errors: catch, log, and either re-raise or return a safe default — never silently swallow.
- In async route handlers, always `await` coroutines; use `asyncio.gather` for parallel I/O.
- OpenVINO inference and FAISS calls are CPU-bound — run in `asyncio.get_event_loop().run_in_executor(None, ...)` if called from async context.
