"""MQTT Event MCP tools.

Maintains a lifecycle-managed MQTT subscriber that buffers incoming
SceneScape scene events. Exposes tools to:
  - Inspect recent buffered events
  - Publish messages to MQTT topics
  - Simulate SceneScape object events for testing

The subscriber is started/stopped via start_subscriber() / stop_subscriber()
called from the server lifespan, not on module import.

Raw re-identification embeddings are intentionally excluded from the event
buffer to protect biometric data. Only metadata (object IDs, timestamps,
camera IDs, confidence scores) is retained.
"""

from __future__ import annotations

import json
import logging
import ssl
import threading
import time
from collections import deque
from typing import Optional

from mcp.server.fastmcp import FastMCP

from mcp_server.config import MCPConfig

log = logging.getLogger("poi.mcp.mqtt")

# Module-level subscriber state — managed exclusively by start/stop_subscriber()
_subscriber: Optional["_MQTTSubscriber"] = None
_subscriber_lock = threading.Lock()


# ── Lifecycle functions called from server.py ──────────────────────────────


def start_subscriber(cfg: MCPConfig) -> None:
    """Start the background MQTT subscriber. Safe to call multiple times."""
    global _subscriber
    with _subscriber_lock:
        if _subscriber is not None:
            return
        if not cfg.mqtt_host:
            log.warning("MQTT_HOST not configured — subscriber not started")
            return
        _subscriber = _MQTTSubscriber(cfg)
        _subscriber.start()
        log.info("MQTT subscriber started (broker=%s:%d)", cfg.mqtt_host, cfg.mqtt_port)


def stop_subscriber() -> None:
    """Stop the background MQTT subscriber. Safe to call when not running."""
    global _subscriber
    with _subscriber_lock:
        if _subscriber is not None:
            _subscriber.stop()
            _subscriber = None
            log.info("MQTT subscriber stopped")


# ── Tool registration ──────────────────────────────────────────────────────


def register(mcp: FastMCP, cfg: MCPConfig) -> None:
    """Register MQTT tools on the MCP server."""

    @mcp.tool()
    def mqtt_get_recent_events(limit: int = 20) -> list[dict]:
        """Get recently received MQTT SceneScape events from the in-memory buffer.

        Raw re-identification embeddings are excluded. Only metadata is returned.

        Args:
            limit: Number of events to return from newest to oldest (default 20, max 100).

        Returns:
            List of event dicts with topic, scene_id, timestamp, object_count,
            person_count, and per-object metadata (id, confidence, camera_id).
        """
        with _subscriber_lock:
            sub = _subscriber
        if sub is None:
            return [{"error": "MQTT subscriber is not running. MQTT_HOST may not be configured."}]
        events = sub.get_recent(min(limit, 100))
        return events

    @mcp.tool()
    def mqtt_get_subscriber_status() -> dict:
        """Get the current status of the MQTT subscriber.

        Returns:
            Dict with connected, broker, topic, buffered_events, and last_event_at.
        """
        with _subscriber_lock:
            sub = _subscriber
        if sub is None:
            return {
                "connected": False,
                "broker": f"{cfg.mqtt_host}:{cfg.mqtt_port}",
                "topic": "",
                "buffered_events": 0,
                "last_event_at": None,
                "note": "Subscriber not started — MQTT_HOST may not be configured",
            }
        return sub.status()

    @mcp.tool()
    def mqtt_publish(topic: str, payload: str, qos: int = 0, retain: bool = False) -> dict:
        """Publish a message to an MQTT topic.

        Requires MCP_ALLOW_MUTATIONS=true. Opens a short-lived connection.

        Args:
            topic: MQTT topic to publish to.
            payload: Message payload string (JSON or plain text).
            qos: Quality of Service level — 0, 1, or 2. Default: 0.
            retain: Whether the broker should retain this message. Default: False.

        Returns:
            Confirmation dict with topic and message_id, or an error.
        """
        if not cfg.allow_mutations:
            return {"error": "Mutations are disabled. Set MCP_ALLOW_MUTATIONS=true to enable."}
        try:
            import paho.mqtt.publish as publish

            kwargs: dict = {
                "hostname": cfg.mqtt_host,
                "port": cfg.mqtt_port,
                "qos": qos,
                "retain": retain,
            }
            if cfg.mqtt_ca_cert:
                kwargs["tls"] = {
                    "ca_certs": cfg.mqtt_ca_cert,
                    "cert_reqs": ssl.CERT_REQUIRED,
                    "tls_version": ssl.PROTOCOL_TLS_CLIENT,
                }
            publish.single(topic, payload=payload, **kwargs)
            return {"status": "published", "topic": topic, "payload_length": len(payload)}
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def mqtt_simulate_scenescape_event(
        scene_id: str = "",
        object_id: str = "obj-sim-001",
        confidence: float = 0.92,
        camera_id: str = "cam-01",
    ) -> dict:
        """Publish a simulated SceneScape person detection event to MQTT.

        Useful for testing the POI pipeline end-to-end. The event uses a
        zero-vector embedding (no actual biometric data).
        Requires MCP_ALLOW_MUTATIONS=true.

        Args:
            scene_id: Scene UID. Defaults to SCENE_UID env var.
            object_id: Simulated object/person identifier.
            confidence: Detection confidence score (0.0 – 1.0).
            camera_id: Camera identifier included in visibility list.

        Returns:
            Confirmation with the topic and payload summary, or an error.
        """
        if not cfg.allow_mutations:
            return {"error": "Mutations are disabled. Set MCP_ALLOW_MUTATIONS=true to enable."}

        sid = scene_id or cfg.mqtt_scene_uid or "test-scene-001"
        topic = f"scenescape/event/{sid}/objects"
        payload = {
            "id": sid,
            "name": f"scene-{sid[:8]}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "objects": [
                {
                    "id": object_id,
                    "category": "person",
                    "type": "person",
                    "confidence": confidence,
                    "visibility": [camera_id],
                    "center_of_mass": [0.5, 0.5, 0.0],
                    "metadata": {
                        "reid": {
                            # Zero vector — no biometric data
                            "embedding_vector": [[0.0] * 256]
                        }
                    },
                }
            ],
        }
        return mqtt_publish(topic=topic, payload=json.dumps(payload))

    log.info("MQTT tools registered (broker=%s:%d)", cfg.mqtt_host, cfg.mqtt_port)


# ── Subscriber implementation ──────────────────────────────────────────────


class _MQTTSubscriber:
    """Background paho-mqtt subscriber with a bounded event buffer."""

    def __init__(self, cfg: MCPConfig) -> None:
        self._cfg = cfg
        self._buffer: deque[dict] = deque(maxlen=cfg.mqtt_event_buffer_size)
        self._buffer_lock = threading.Lock()
        self._last_event_at: Optional[float] = None
        self._client = None
        self._running = False
        self._connected = False

        topic_default = f"scenescape/event/{cfg.mqtt_scene_uid}/objects" if cfg.mqtt_scene_uid else "scenescape/event/+/objects"
        self._topic = topic_default

    def start(self) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            log.error("paho-mqtt not installed — MQTT subscriber disabled")
            return

        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.on_disconnect = self._on_disconnect
        self._client.reconnect_delay_set(min_delay=2, max_delay=60)

        if self._cfg.mqtt_ca_cert:
            self._client.tls_set(
                ca_certs=self._cfg.mqtt_ca_cert,
                cert_reqs=ssl.CERT_REQUIRED,
                tls_version=ssl.PROTOCOL_TLS_CLIENT,
            )
            self._client.tls_insecure_set(True)

        self._running = True
        self._client.connect_async(self._cfg.mqtt_host, self._cfg.mqtt_port)
        self._client.loop_start()

    def stop(self) -> None:
        self._running = False
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()

    def get_recent(self, limit: int) -> list[dict]:
        with self._buffer_lock:
            items = list(self._buffer)
        return list(reversed(items))[:limit]

    def status(self) -> dict:
        with self._buffer_lock:
            count = len(self._buffer)
        return {
            "connected": self._connected,
            "broker": f"{self._cfg.mqtt_host}:{self._cfg.mqtt_port}",
            "topic": self._topic,
            "buffered_events": count,
            "last_event_at": self._last_event_at,
        }

    def _on_connect(self, client, userdata, flags, rc, properties=None) -> None:
        if rc == 0:
            self._connected = True
            client.subscribe(self._topic)
            log.info("MQTT subscriber connected, subscribed to %s", self._topic)
        else:
            log.error("MQTT subscriber connection failed: rc=%s", rc)

    def _on_disconnect(self, client, userdata, flags, rc, properties=None) -> None:
        self._connected = False
        if self._running:
            log.warning("MQTT subscriber disconnected (rc=%s), reconnecting...", rc)

    def _on_message(self, client, userdata, msg) -> None:
        if not self._running:
            return
        try:
            payload = json.loads(msg.payload)
        except (json.JSONDecodeError, TypeError):
            return

        # Extract metadata only — strip raw embeddings to protect biometric data
        objects = payload.get("objects", []) + payload.get("entered", [])
        person_objects = []
        for obj in objects:
            if obj.get("category") == "person" or obj.get("type") == "person":
                person_objects.append(
                    {
                        "id": obj.get("id"),
                        "confidence": obj.get("confidence"),
                        "camera_id": (obj.get("visibility") or [None])[0],
                        "center_of_mass": obj.get("center_of_mass"),
                        "has_embedding": bool(
                            obj.get("metadata", {}).get("reid", {}).get("embedding_vector")
                        ),
                    }
                )

        event_summary = {
            "topic": msg.topic,
            "scene_id": payload.get("id", ""),
            "scene_name": payload.get("name", ""),
            "timestamp": payload.get("timestamp", ""),
            "object_count": len(objects),
            "person_count": len(person_objects),
            "persons": person_objects,
            "received_at": time.time(),
        }

        with self._buffer_lock:
            self._buffer.append(event_summary)
            self._last_event_at = time.time()
