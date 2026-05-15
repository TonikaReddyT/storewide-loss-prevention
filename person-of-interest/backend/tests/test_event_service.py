"""Tests for EventService."""

from __future__ import annotations

from unittest.mock import MagicMock

from backend.service.event_service import EventService


class TestEventService:
    def _make_service(self):
        repo = MagicMock()
        # Default: no thumbnails or dwells in Redis
        repo.get_thumbnail.return_value = None
        repo.get_region_dwells_for_object.return_value = []
        return EventService(repo), repo

    def test_store_movement(self):
        service, repo = self._make_service()
        service.store_movement(
            object_id="obj-1",
            timestamp="2025-01-15T12:00:00Z",
            camera_id="cam-01",
            region="aisle1",
            poi_id="poi-x",
        )
        repo.store_event.assert_called_once()
        evt = repo.store_event.call_args[0][0]
        assert evt.object_id == "obj-1"
        assert evt.camera_id == "cam-01"
        assert evt.poi_id == "poi-x"

    def test_store_movement_with_thumbnail(self):
        service, repo = self._make_service()
        service.store_movement(
            object_id="obj-1",
            timestamp="2025-01-15T12:00:00Z",
            camera_id="cam-01",
            region="aisle1",
            poi_id="poi-x",
            thumbnail_path="/api/v1/thumbnail/obj-1",
        )
        evt = repo.store_event.call_args[0][0]
        assert evt.thumbnail_path == "/api/v1/thumbnail/obj-1"
        d = evt.to_dict()
        assert d["thumbnail_path"] == "/api/v1/thumbnail/obj-1"

    def test_search_history_empty(self):
        service, repo = self._make_service()
        repo.get_events_for_poi.return_value = []

        result = service.search_history("poi-x")

        assert result["event_type"] == "poi_history_result"
        assert result["total_visits"] == 0
        assert result["visits"] == []

    def test_search_history_groups_by_date(self):
        service, repo = self._make_service()
        repo.get_events_for_poi.return_value = [
            {"timestamp": "2025-01-15T10:00:00Z", "camera_id": "cam-01", "region": "aisle1", "object_id": "obj-1"},
            {"timestamp": "2025-01-15T14:00:00Z", "camera_id": "cam-02", "region": "aisle2", "object_id": "obj-1"},
            {"timestamp": "2025-01-16T09:00:00Z", "camera_id": "cam-01", "region": "aisle1", "object_id": "obj-1"},
        ]

        result = service.search_history("poi-x")

        assert result["total_visits"] == 2
        assert result["visits"][0]["date"] == "2025-01-15"
        assert result["visits"][1]["date"] == "2025-01-16"
        # Day 1 had two cameras
        assert set(result["visits"][0]["cameras_visited"]) == {"cam-01", "cam-02"}
        # Region name should be present
        assert result["visits"][0]["region_name"] != ""

    def test_search_history_with_time_range(self):
        service, repo = self._make_service()
        repo.get_events_for_poi.return_value = []

        service.search_history("poi-x", start_time="2025-01-14", end_time="2025-01-16")

        repo.get_events_for_poi.assert_called_once_with("poi-x", "2025-01-14", "2025-01-16")

    def test_search_history_single_event_per_day(self):
        service, repo = self._make_service()
        repo.get_events_for_poi.return_value = [
            {"timestamp": "2025-01-15T10:00:00Z", "camera_id": "cam-01", "object_id": "obj-1"},
        ]

        result = service.search_history("poi-x")

        assert result["total_visits"] == 1
        visit = result["visits"][0]
        assert visit["entry_time"] == "2025-01-15T10:00:00Z"
        assert visit["exit_time"] is None  # Only one event

    def test_search_history_thumbnail_from_event(self):
        """Thumbnail persisted on event should appear in visit."""
        service, repo = self._make_service()
        repo.get_events_for_poi.return_value = [
            {
                "timestamp": "2025-01-15T10:00:00Z",
                "camera_id": "cam-01",
                "region": "aisle1",
                "object_id": "obj-1",
                "thumbnail_path": "/api/v1/thumbnail/obj-1",
            },
        ]

        result = service.search_history("poi-x")

        assert result["visits"][0]["thumbnail"] == "/api/v1/thumbnail/obj-1"

    def test_search_history_thumbnail_fallback_to_redis(self):
        """When event has no thumbnail_path, fall back to Redis thumbnail cache."""
        service, repo = self._make_service()
        repo.get_events_for_poi.return_value = [
            {
                "timestamp": "2025-01-15T10:00:00Z",
                "camera_id": "cam-01",
                "region": "aisle1",
                "object_id": "obj-1",
            },
        ]
        repo.get_thumbnail.return_value = b"base64data"

        result = service.search_history("poi-x")

        assert result["visits"][0]["thumbnail"] == "/api/v1/thumbnail/obj-1"
        repo.get_thumbnail.assert_called_with("obj-1")

    def test_search_history_no_thumbnail(self):
        """When no thumbnail is available, field should be empty string."""
        service, repo = self._make_service()
        repo.get_events_for_poi.return_value = [
            {
                "timestamp": "2025-01-15T10:00:00Z",
                "camera_id": "cam-01",
                "object_id": "obj-1",
            },
        ]

        result = service.search_history("poi-x")

        assert result["visits"][0]["thumbnail"] == ""

    def test_search_history_enriched_with_region_dwells(self):
        """Region dwell data should be merged into visit regions."""
        service, repo = self._make_service()
        repo.get_events_for_poi.return_value = [
            {
                "timestamp": "2025-01-15T10:00:00Z",
                "camera_id": "lp-camera1",
                "region": "lp-camera1",
                "object_id": "obj-1",
            },
        ]
        repo.get_region_dwells_for_object.return_value = [
            {
                "region_name": "checkout-zone",
                "entry_time": "2025-01-15T09:55:00Z",
                "exit_time": "2025-01-15T10:05:00Z",
                "dwell_sec": 600.0,
                "camera_id": "lp-camera1",
            },
        ]

        result = service.search_history("poi-x")

        visit = result["visits"][0]
        # Real region name should replace camera-ID-only entry
        assert "checkout-zone" in visit["regions"]
        assert visit["region_name"] == "checkout-zone"
        # Dwell details should be present
        assert len(visit["region_dwells"]) == 1
        assert visit["region_dwells"][0]["dwell_sec"] == 600.0

    def test_search_history_region_dwells_filtered_by_date(self):
        """Region dwell lookup should pass date filter."""
        service, repo = self._make_service()
        repo.get_events_for_poi.return_value = [
            {
                "timestamp": "2025-01-15T10:00:00Z",
                "camera_id": "cam-01",
                "region": "cam-01",
                "object_id": "obj-1",
            },
        ]
        repo.get_region_dwells_for_object.return_value = []

        service.search_history("poi-x")

        # Should call with date filter matching the visit date
        repo.get_region_dwells_for_object.assert_called_with("obj-1", date_filter="2025-01-15")

    def test_store_region_exit_uses_entry_region_name(self):
        """Region exit should use the human-readable name from entry, not raw ID."""
        service, repo = self._make_service()
        repo.get_region_presence.return_value = {
            "first_seen": "2025-01-15T09:00:00Z",
            "region_name": "Electronics Aisle",
            "camera_id": "cam-01",
        }

        service.store_region_exit("obj-1", "2025-01-15T09:30:00Z", "scene-1", "reg-42", "reg-42")

        call = repo.store_region_dwell.call_args
        # Should use "Electronics Aisle" from entry_data, not "reg-42"
        assert call[0][4] == "Electronics Aisle"
        # Should pass entry_time and camera_id
        assert call[1]["entry_time"] == "2025-01-15T09:00:00Z"
        assert call[1]["camera_id"] == "cam-01"
