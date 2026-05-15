"""Event domain entity for tracking person movements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PersonEvent:
    object_id: str
    timestamp: str
    camera_id: str
    region_id: str
    region_name: str
    scene_id: str
    confidence: float
    embedding_vector: list[float]
    poi_id: Optional[str] = None
    dwell: Optional[float] = None
    first_seen: Optional[str] = None
    center_of_mass: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "timestamp": self.timestamp,
            "camera_id": self.camera_id,
            "region_id": self.region_id,
            "region_name": self.region_name,
            "scene_id": self.scene_id,
            "confidence": self.confidence,
            "poi_id": self.poi_id,
            "dwell": self.dwell,
            "first_seen": self.first_seen,
        }


@dataclass
class MovementEvent:
    object_id: str
    timestamp: str
    camera_id: str
    region: str
    poi_id: Optional[str] = None
    embedding_reference: Optional[str] = None
    thumbnail_path: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "object_id": self.object_id,
            "timestamp": self.timestamp,
            "camera_id": self.camera_id,
            "region": self.region,
            "poi_id": self.poi_id,
            "embedding_reference": self.embedding_reference,
        }
        if self.thumbnail_path:
            d["thumbnail_path"] = self.thumbnail_path
        return d
