"""Custom FAISS MCP tools.

Provides read-only FAISS index operations by calling the POI backend REST API.
Using the backend as the single source of truth avoids cross-process memory
inconsistency (the FAISS index lives in the backend's process memory and is
only persisted to disk on shutdown).

Available operations:
  - faiss_get_stats   — index vector count and dimension via /api/v1/status
  - faiss_search_poi  — upload an image and search for matching POIs
  - faiss_list_poi_vectors — mapping of POI IDs to their FAISS vector count

All write operations (add/remove vectors) are intentionally absent here;
use the POI management tools (poi_create, poi_delete) which go through the
backend's coordinated service layer.
"""

from __future__ import annotations

import base64
import logging
from typing import Optional

import requests

from mcp.server.fastmcp import FastMCP

from mcp_server.config import MCPConfig

log = logging.getLogger("poi.mcp.faiss")


def register(mcp: FastMCP, cfg: MCPConfig) -> None:
    """Register FAISS tools on the MCP server."""

    def _backend(path: str, method: str = "GET", **kwargs) -> dict:
        """Thin wrapper for backend REST API calls."""
        url = f"{cfg.poi_backend_url.rstrip('/')}{path}"
        try:
            resp = requests.request(method, url, timeout=30, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def faiss_get_stats() -> dict:
        """Get FAISS index statistics from the POI backend.

        Returns:
            Dict with faiss_vectors (total indexed vectors), status, and
            mqtt_connected flag.
        """
        return _backend("/api/v1/status")

    @mcp.tool()
    def faiss_search_by_image(image_b64: str, start_time: str = "", end_time: str = "") -> dict:
        """Search the FAISS index by uploading a face image.

        Submits the image to the backend's search endpoint which:
        1. Generates a 256-d face embedding using OpenVINO
        2. Searches the FAISS index for the nearest POI
        3. Returns historical movement events for the matched POI

        Args:
            image_b64: Base64-encoded image bytes (JPEG or PNG).
            start_time: ISO 8601 start of time range filter, e.g. '2024-01-01T00:00:00Z'.
            end_time: ISO 8601 end of time range filter.

        Returns:
            Search results with poi_id, visits list, total_visits, and search_stats
            (vectors_searched, query_latency_ms). Or error dict on failure.
        """
        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception:
            return {"error": "Invalid base64 image data"}

        url = f"{cfg.poi_backend_url.rstrip('/')}/api/v1/search"
        try:
            resp = requests.post(
                url,
                files={"image": ("query.jpg", image_bytes, "image/jpeg")},
                data={"start_time": start_time, "end_time": end_time},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def faiss_list_pois() -> list[dict]:
        """List all enrolled POIs with their FAISS vector counts.

        Fetches POI list from the backend and annotates each POI with the
        number of reference images (proxy for indexed vectors).

        Returns:
            List of POI dicts with poi_id, severity, status, and reference_image_count.
        """
        result = _backend("/api/v1/poi")
        if isinstance(result, dict) and "error" in result:
            return [result]
        if isinstance(result, list):
            return [
                {
                    "poi_id": p.get("poi_id"),
                    "severity": p.get("severity"),
                    "status": p.get("status"),
                    "reference_image_count": len(p.get("reference_images", [])),
                    "created_at": p.get("timestamp", ""),
                }
                for p in result
            ]
        return [{"error": "Unexpected response from backend", "raw": str(result)[:200]}]

    @mcp.tool()
    def poi_get(poi_id: str) -> dict:
        """Get full details of a specific POI from the backend.

        Args:
            poi_id: The POI identifier, e.g. 'poi-a1b2c3d4'.

        Returns:
            Full POI record including reference_images, severity, status, and notes.
        """
        return _backend(f"/api/v1/poi/{poi_id}")

    @mcp.tool()
    def poi_delete(poi_id: str) -> dict:
        """Delete a POI and remove its vectors from the FAISS index.

        Requires MCP_ALLOW_MUTATIONS=true. Calls the backend DELETE endpoint
        which coordinates FAISS removal and Redis cleanup atomically.

        Args:
            poi_id: The POI identifier, e.g. 'poi-a1b2c3d4'.

        Returns:
            Confirmation dict or error.
        """
        if not cfg.allow_mutations:
            return {"error": "Mutations are disabled. Set MCP_ALLOW_MUTATIONS=true to enable."}
        return _backend(f"/api/v1/poi/{poi_id}", method="DELETE")

    @mcp.tool()
    def poi_create_from_image(
        image_b64: str,
        severity: str = "medium",
        description: str = "",
        image_filename: str = "reference.jpg",
    ) -> dict:
        """Enroll a new POI by uploading a reference face image.

        Requires MCP_ALLOW_MUTATIONS=true. The backend generates a 256-d
        face embedding and adds it to the FAISS index alongside the POI
        metadata in Redis.

        Args:
            image_b64: Base64-encoded image bytes (JPEG or PNG).
            severity: Risk severity — 'low', 'medium', or 'high'.
            description: Optional notes about the POI.
            image_filename: Filename hint for the image (affects MIME type detection).

        Returns:
            Created POI dict with poi_id, severity, and embedding status.
        """
        if not cfg.allow_mutations:
            return {"error": "Mutations are disabled. Set MCP_ALLOW_MUTATIONS=true to enable."}
        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception:
            return {"error": "Invalid base64 image data"}

        mime = "image/png" if image_filename.lower().endswith(".png") else "image/jpeg"
        url = f"{cfg.poi_backend_url.rstrip('/')}/api/v1/poi"
        try:
            resp = requests.post(
                url,
                files={"images": (image_filename, image_bytes, mime)},
                data={"severity": severity, "description": description},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def faiss_get_recent_alerts(limit: int = 20) -> list[dict]:
        """Get the most recent POI match alerts from the backend.

        Args:
            limit: Number of alerts to return (default 20, max 100).

        Returns:
            List of alert dicts with poi_id, object_id, camera_id,
            region_name, similarity_score, severity, and timestamp.
        """
        result = _backend(f"/api/v1/alerts")
        if isinstance(result, dict) and "error" in result:
            return [result]
        if isinstance(result, list):
            return result[:min(limit, 100)]
        return [{"error": "Unexpected response", "raw": str(result)[:200]}]

    @mcp.tool()
    def faiss_list_cameras() -> dict:
        """List cameras registered in SceneScape (proxied through the backend).

        Returns:
            Dict with cameras list and count.
        """
        return _backend("/api/v1/cameras")

    log.info("FAISS/POI tools registered (backend=%s, mutations=%s)", cfg.poi_backend_url, cfg.allow_mutations)
