"""Camera API routes — proxy to SceneScape API."""

from __future__ import annotations

import logging

from fastapi import APIRouter

log = logging.getLogger("poi.api.camera")

router = APIRouter(prefix="/cameras", tags=["Cameras"])

_scenescape_adapter = None


def init(scenescape_adapter) -> None:
    global _scenescape_adapter
    _scenescape_adapter = scenescape_adapter


@router.get("")
async def list_cameras():
    """List all cameras from SceneScape."""
    cameras = _scenescape_adapter.list_cameras()
    return {"cameras": cameras, "count": len(cameras)}


@router.get("/{camera_id}")
async def get_camera(camera_id: str):
    """Get a single camera from SceneScape."""
    camera = _scenescape_adapter.get_camera(camera_id)
    if camera is None:
        return {"error": f"Camera {camera_id} not found"}
    return camera
