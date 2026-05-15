"""Redis MCP tools.

Provides tools to inspect and interact with the POI system's Redis store:
POIs, movement events, alerts, cache entries, and raw key access.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

from mcp_server.config import MCPConfig

log = logging.getLogger("poi.mcp.redis")


def register(mcp: FastMCP, cfg: MCPConfig) -> None:
    """Register Redis tools on the MCP server."""

    def _client():
        try:
            import redis
        except ImportError:
            raise RuntimeError("redis-py not installed. Run: pip install redis")
        return redis.Redis(
            host=cfg.redis_host,
            port=cfg.redis_port,
            db=cfg.redis_db,
            decode_responses=True,
        )

    @mcp.tool()
    def redis_list_pois(limit: int = 50) -> list[dict]:
        """List all enrolled Persons of Interest (POIs) from Redis.

        Args:
            limit: Maximum number of POIs to return (default 50).

        Returns:
            List of POI dicts with poi_id, severity, status, notes,
            enrolled_by, created_at, and reference_image count.
        """
        r = _client()
        try:
            poi_ids = list(r.smembers("poi:index"))
            pois = []
            for pid in sorted(poi_ids, reverse=True)[: min(limit, 200)]:
                raw = r.get(f"poi:{pid}")
                if raw:
                    data = json.loads(raw)
                    pois.append(
                        {
                            "poi_id": data.get("poi_id"),
                            "severity": data.get("severity"),
                            "status": data.get("status"),
                            "notes": data.get("notes", ""),
                            "enrolled_by": data.get("enrolled_by", ""),
                            "created_at": data.get("timestamp", ""),
                            "reference_image_count": len(data.get("reference_images", [])),
                        }
                    )
            return pois
        except Exception as exc:
            return [{"error": str(exc)}]

    @mcp.tool()
    def redis_get_poi(poi_id: str) -> dict:
        """Get full details of a specific POI from Redis.

        Args:
            poi_id: The POI identifier, e.g. 'poi-a1b2c3d4'.

        Returns:
            Full POI dict including reference images and embedding IDs,
            or an error dict if not found.
        """
        r = _client()
        try:
            raw = r.get(f"poi:{poi_id}")
            if raw is None:
                return {"error": f"POI '{poi_id}' not found"}
            return json.loads(raw)
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def redis_get_recent_alerts(limit: int = 20) -> list[dict]:
        """Get the most recent POI match alerts from Redis.

        Args:
            limit: Number of alerts to retrieve (default 20, max 100).

        Returns:
            List of alert dicts with poi_id, object_id, timestamp, camera_id,
            region_name, similarity_score, and severity.
        """
        r = _client()
        try:
            raw_list = r.lrange("alerts:recent", 0, min(limit, 100) - 1)
            return [json.loads(item) for item in raw_list]
        except Exception as exc:
            return [{"error": str(exc)}]

    @mcp.tool()
    def redis_get_events_for_poi(
        poi_id: str,
        start_time: str = "",
        end_time: str = "",
    ) -> list[dict]:
        """Get historical movement events for a specific POI.

        Args:
            poi_id: The POI identifier, e.g. 'poi-a1b2c3d4'.
            start_time: ISO 8601 start filter, e.g. '2024-01-01T00:00:00Z'.
            end_time: ISO 8601 end filter, e.g. '2024-12-31T23:59:59Z'.

        Returns:
            List of event dicts sorted by timestamp, containing timestamp,
            camera_id, region, and object_id.
        """
        r = _client()
        try:
            keys = r.smembers(f"events:poi:{poi_id}")
            events = []
            for key in keys:
                raw = r.get(key)
                if raw:
                    evt = json.loads(raw)
                    ts = evt.get("timestamp", "")
                    if start_time and ts < start_time:
                        continue
                    if end_time and ts > end_time:
                        continue
                    events.append(evt)
            events.sort(key=lambda e: e.get("timestamp", ""))
            return events
        except Exception as exc:
            return [{"error": str(exc)}]

    @mcp.tool()
    def redis_get_key(key: str) -> dict:
        """Get the value of a specific Redis key (string keys only).

        Args:
            key: Redis key to retrieve.

        Returns:
            Dict with 'key', 'value' (parsed JSON if possible), 'type', and 'ttl'.
        """
        r = _client()
        try:
            key_type = r.type(key)
            ttl = r.ttl(key)
            if key_type == "string":
                raw = r.get(key)
                try:
                    value = json.loads(raw) if raw else None
                except (json.JSONDecodeError, TypeError):
                    value = raw
                return {"key": key, "value": value, "type": key_type, "ttl": ttl}
            elif key_type == "list":
                value = r.lrange(key, 0, 9)
                return {"key": key, "value": value, "type": key_type, "ttl": ttl, "note": "showing first 10"}
            elif key_type == "set":
                value = list(r.smembers(key))
                return {"key": key, "value": value, "type": key_type, "ttl": ttl}
            elif key_type == "hash":
                value = r.hgetall(key)
                return {"key": key, "value": value, "type": key_type, "ttl": ttl}
            elif key_type == "none":
                return {"error": f"Key '{key}' does not exist"}
            return {"key": key, "type": key_type, "ttl": ttl, "note": "unsupported type for value display"}
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def redis_set_key(key: str, value: str, ttl: int = 0) -> dict:
        """Set a Redis key to a string value.

        Requires MCP_ALLOW_MUTATIONS=true.

        Args:
            key: Redis key to set.
            value: String value to store.
            ttl: Time-to-live in seconds (0 = no expiry).

        Returns:
            Confirmation dict or error.
        """
        if not cfg.allow_mutations:
            return {"error": "Mutations are disabled. Set MCP_ALLOW_MUTATIONS=true to enable."}
        r = _client()
        try:
            if ttl > 0:
                r.setex(key, ttl, value)
            else:
                r.set(key, value)
            return {"status": "set", "key": key, "ttl": ttl}
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def redis_delete_key(key: str) -> dict:
        """Delete a Redis key.

        Requires MCP_ALLOW_MUTATIONS=true.

        Args:
            key: Redis key to delete.

        Returns:
            Confirmation dict with deleted count.
        """
        if not cfg.allow_mutations:
            return {"error": "Mutations are disabled. Set MCP_ALLOW_MUTATIONS=true to enable."}
        r = _client()
        try:
            deleted = r.delete(key)
            return {"deleted": deleted, "key": key}
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def redis_search_keys(pattern: str, limit: int = 50) -> list[str]:
        """Scan Redis keys matching a pattern (safe, cursor-based scan).

        Args:
            pattern: Redis key pattern with wildcards, e.g. 'poi:*' or 'event:obj-*'.
            limit: Maximum number of keys to return (default 50, max 200).

        Returns:
            List of matching key names.
        """
        r = _client()
        try:
            keys = []
            cursor = 0
            max_keys = min(limit, 200)
            while True:
                cursor, batch = r.scan(cursor=cursor, match=pattern, count=100)
                keys.extend(batch)
                if cursor == 0 or len(keys) >= max_keys:
                    break
            return keys[:max_keys]
        except Exception as exc:
            return [str(exc)]

    @mcp.tool()
    def redis_get_stats() -> dict:
        """Get Redis server statistics and memory usage.

        Returns:
            Dict with redis_version, connected_clients, used_memory_human,
            total_commands_processed, poi_count, and alert_count.
        """
        r = _client()
        try:
            info = r.info()
            poi_count = r.scard("poi:index")
            alert_count = r.llen("alerts:recent")
            return {
                "redis_version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory_human": info.get("used_memory_human"),
                "total_commands_processed": info.get("total_commands_processed"),
                "uptime_in_seconds": info.get("uptime_in_seconds"),
                "poi_count": poi_count,
                "alert_count": alert_count,
            }
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def redis_flush_alerts() -> dict:
        """Clear all recent alerts from Redis.

        Requires MCP_ALLOW_MUTATIONS=true.

        Returns:
            Confirmation with the number of alerts cleared.
        """
        if not cfg.allow_mutations:
            return {"error": "Mutations are disabled. Set MCP_ALLOW_MUTATIONS=true to enable."}
        r = _client()
        try:
            count = r.llen("alerts:recent")
            r.delete("alerts:recent")
            return {"status": "cleared", "alerts_removed": count}
        except Exception as exc:
            return {"error": str(exc)}

    log.info("Redis tools registered (host=%s:%d, mutations=%s)", cfg.redis_host, cfg.redis_port, cfg.allow_mutations)
