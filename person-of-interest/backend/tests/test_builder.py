"""Tests for POI Builder pattern."""

from __future__ import annotations

from backend.domain.entities.poi import POIStatus, Severity
from backend.utils.builder import POIBuilder


class TestPOIBuilder:
    def test_build_defaults(self):
        poi = POIBuilder().with_id("poi-b001").build()
        assert poi.poi_id == "poi-b001"
        assert poi.severity == Severity.MEDIUM
        assert poi.notes == ""
        assert poi.status == POIStatus.ACTIVE
        assert poi.reference_images == []

    def test_build_full(self):
        poi = (
            POIBuilder()
            .with_id("poi-b002")
            .with_severity("high")
            .with_notes("armed suspect")
            .with_enrolled_by("admin")
            .add_image("emb-01", "/uploads/poi-b002/ref_0.jpg")
            .add_image("emb-02", "/uploads/poi-b002/ref_1.jpg")
            .build()
        )
        assert poi.poi_id == "poi-b002"
        assert poi.severity == Severity.HIGH
        assert poi.notes == "armed suspect"
        assert poi.enrolled_by == "admin"
        assert len(poi.reference_images) == 2
        assert poi.embedding_ids == ["emb-01", "emb-02"]

    def test_build_auto_id(self):
        poi = POIBuilder().build()
        assert poi.poi_id.startswith("poi-")

    def test_fluent_chain(self):
        builder = POIBuilder()
        result = builder.with_id("x").with_severity("low").with_notes("n")
        assert result is builder  # Same object — fluent interface

    def test_with_status(self):
        poi = POIBuilder().with_id("poi-s1").with_status("inactive").build()
        assert poi.status == POIStatus.INACTIVE

    def test_image_vector_dim(self):
        poi = (
            POIBuilder()
            .with_id("poi-d1")
            .add_image("emb-01", "/path.jpg", vector_dim=512)
            .build()
        )
        assert poi.reference_images[0].vector_dim == 512
