"""SceneScape REST API adapter."""

from __future__ import annotations

import logging
from typing import Optional

import requests
import urllib3

from backend.core.config import get_config

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

log = logging.getLogger("poi.scenescape")


class ScenescapeAPIAdapter:
    """Adapter Pattern — wraps the SceneScape REST API."""

    def __init__(self) -> None:
        self._cfg = get_config()
        self._base_url = self._cfg.scenescape_api_url.rstrip("/")
        self._token = self._cfg.scenescape_api_token
        self._session = requests.Session()
        self._session.verify = False
        if self._token:
            self._session.headers["Authorization"] = f"Token {self._token}"

    def list_cameras(self) -> list[dict]:
        if not self._base_url:
            log.warning("SceneScape API URL not configured")
            return []
        try:
            resp = self._session.get(f"{self._base_url}/api/v1/cameras", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return data
            return data.get("results", data.get("cameras", []))
        except requests.RequestException:
            log.exception("Failed to fetch cameras from SceneScape")
            return []

    def get_camera(self, camera_id: str) -> Optional[dict]:
        if not self._base_url:
            return None
        try:
            resp = self._session.get(f"{self._base_url}/api/v1/cameras/{camera_id}", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            log.exception("Failed to fetch camera %s", camera_id)
            return None
