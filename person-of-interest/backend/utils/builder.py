"""POI Builder — Builder pattern for constructing POI from multiple images."""

from __future__ import annotations

import uuid
from typing import Optional

import numpy as np

from backend.domain.entities.poi import POI, POIStatus, ReferenceImage, Severity


class POIBuilder:
    """Builder Pattern — Constructs a POI entity step by step from multiple images."""

    def __init__(self) -> None:
        self._poi_id: Optional[str] = None
        self._severity: Severity = Severity.MEDIUM
        self._notes: str = ""
        self._enrolled_by: str = "system"
        self._reference_images: list[ReferenceImage] = []
        self._embedding_ids: list[str] = []
        self._status: POIStatus = POIStatus.ACTIVE

    def with_id(self, poi_id: str) -> POIBuilder:
        self._poi_id = poi_id
        return self

    def with_severity(self, severity: str) -> POIBuilder:
        self._severity = Severity(severity)
        return self

    def with_notes(self, notes: str) -> POIBuilder:
        self._notes = notes
        return self

    def with_enrolled_by(self, enrolled_by: str) -> POIBuilder:
        self._enrolled_by = enrolled_by
        return self

    def add_image(self, embedding_id: str, image_path: str, vector_dim: int = 256) -> POIBuilder:
        self._reference_images.append(
            ReferenceImage(
                source="uploaded_image",
                embedding_id=embedding_id,
                vector_dim=vector_dim,
                image_path=image_path,
            )
        )
        self._embedding_ids.append(embedding_id)
        return self

    def with_status(self, status: str) -> POIBuilder:
        self._status = POIStatus(status)
        return self

    def build(self) -> POI:
        if self._poi_id is None:
            self._poi_id = POI.generate_id()
        return POI(
            poi_id=self._poi_id,
            severity=self._severity,
            notes=self._notes,
            reference_images=self._reference_images,
            status=self._status,
            enrolled_by=self._enrolled_by,
            embedding_ids=self._embedding_ids,
        )
