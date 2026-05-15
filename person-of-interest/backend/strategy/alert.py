"""Alert delivery strategies — Strategy pattern.

POI backend uses a single strategy: forward every alert to the intel/alert-service
via HTTP POST. All delivery concerns (logging, WebSocket broadcast, MQTT publish)
are handled inside the alert-service, not here.
"""

from __future__ import annotations

import logging

import requests

from backend.domain.entities.match_result import AlertPayload
from backend.domain.interfaces.alert import AlertStrategy

log = logging.getLogger("poi.strategy.alert")


class AlertServiceStrategy(AlertStrategy):
    """POST alerts to intel/alert-service REST API.

    Raises on any HTTP or network failure so the caller can decide whether
    to mark the alert as sent.
    """

    def __init__(self, alert_service_url: str) -> None:
        self._url = alert_service_url.rstrip("/")

    def send(self, alert: AlertPayload) -> None:
        payload = {
            "alert_type": "POI_MATCH",
            "timestamp": alert.timestamp,
            "metadata": {
                "alert_id": alert.alert_id,
                "poi_id": alert.poi_id,
                "severity": alert.severity,
                "camera_id": alert.match.get("camera_id", ""),
                "similarity_score": alert.match.get("similarity_score", 0),
                "confidence": alert.match.get("confidence", 0),
                "bbox": alert.match.get("bbox", [0, 0, 0, 0]),
                "frame_number": alert.match.get("frame_number", 0),
                "thumbnail_path": alert.match.get("thumbnail_path", ""),
                "notes": alert.poi_metadata.get("notes", ""),
                "enrollment_date": alert.poi_metadata.get("enrollment_date", ""),
                "total_previous_matches": alert.poi_metadata.get("total_previous_matches", 0),
            },
        }
        resp = requests.post(
            f"{self._url}/api/v1/alerts",
            json=payload,
            timeout=5,
            proxies={"http": None, "https": None},  # bypass system proxy for internal calls
        )
        resp.raise_for_status()
        log.info("Alert forwarded to alert-service: %s", alert.alert_id)

    def name(self) -> str:
        return "alert_service"
