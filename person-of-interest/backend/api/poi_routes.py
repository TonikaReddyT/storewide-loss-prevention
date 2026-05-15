"""POI API routes — thin controllers."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

log = logging.getLogger("poi.api.poi")

router = APIRouter(prefix="/poi", tags=["POI"])

# Service injected at startup
_poi_service = None


def init(poi_service) -> None:
    global _poi_service
    _poi_service = poi_service


@router.post("", status_code=201)
async def create_poi(
    images: list[UploadFile] = File(...),
    severity: str = Form("medium"),
    description: str = Form(""),
):
    """Create a new POI from one or more uploaded images."""
    if not images:
        raise HTTPException(400, "At least one image is required")
    if len(images) > 5:
        raise HTTPException(400, "Maximum 5 images allowed")

    image_bytes = []
    for img in images:
        data = await img.read()
        if not data:
            continue
        image_bytes.append(data)

    if not image_bytes:
        raise HTTPException(400, "No valid images uploaded")

    result = await _poi_service.create_poi(image_bytes, severity, description)
    if "error" in result:
        raise HTTPException(422, result["error"])
    return result


@router.get("")
async def list_pois():
    """List all enrolled POIs in descending order by date."""
    return _poi_service.list_pois()


@router.get("/{poi_id}")
async def get_poi(poi_id: str):
    """Get a single POI by ID."""
    result = _poi_service.get_poi(poi_id)
    if result is None:
        raise HTTPException(404, f"POI {poi_id} not found")
    return result


@router.delete("/{poi_id}")
async def delete_poi(poi_id: str):
    """Delete a POI by ID."""
    deleted = _poi_service.delete_poi(poi_id)
    if not deleted:
        raise HTTPException(404, f"POI {poi_id} not found")
    return {"status": "deleted", "poi_id": poi_id}
