"""Tests for AlertService — Observer-based alert dispatch."""

from __future__ import annotations

from unittest.mock import ANY, MagicMock

from backend.domain.entities.match_result import AlertPayload, MatchResult
from backend.domain.entities.poi import POI, POIStatus, ReferenceImage, Severity
from backend.observer.events import EventBus, MatchFoundEvent
from backend.service.alert_service import AlertService


def _make_alert_payload(alert_id="alert-001", poi_id="poi-a"):
    return AlertPayload(
        alert_id=alert_id,
        poi_id=poi_id,
        severity="high",
        timestamp="2025-01-15T12:00:00Z",
        match={"camera_id": "cam-01", "similarity_score": 0.9, "bbox": [0, 0, 0, 0]},
        poi_metadata={"notes": "test", "enrollment_date": "", "total_previous_matches": 0},
    )


class TestAlertService:
    def _make_service(self, is_sent=False):
        strategy = MagicMock()
        strategy.name.return_value = "mock"
        event_repo = MagicMock()
        event_repo.is_alert_sent.return_value = is_sent
        event_repo.get_events_for_poi.return_value = []
        poi_repo = MagicMock()
        poi_repo.get.return_value = POI(
            poi_id="poi-a",
            severity=Severity.HIGH,
            notes="test suspect",
            reference_images=[],
        )
        bus = EventBus()
        service = AlertService([strategy], event_repo, poi_repo, bus)
        return service, strategy, event_repo, poi_repo, bus

    def test_observer_dispatches_alert(self):
        service, strategy, event_repo, _, bus = self._make_service()

        alert = _make_alert_payload()
        event = MatchFoundEvent(alert=alert, object_id="obj-1", timestamp="t1")

        bus.publish("match_found", event)

        strategy.send.assert_called_once_with(alert)
        event_repo.store_alert.assert_called_once()
        event_repo.mark_alert_sent.assert_called_once_with("obj-1:poi-a", ttl=ANY)

    def test_idempotent_dedup(self):
        service, strategy, event_repo, _, bus = self._make_service(is_sent=True)

        alert = _make_alert_payload()
        event = MatchFoundEvent(alert=alert, object_id="obj-dup", timestamp="t1")

        bus.publish("match_found", event)

        strategy.send.assert_not_called()  # Dedup prevents dispatch
        event_repo.store_alert.assert_not_called()

    def test_multiple_strategies(self):
        s1 = MagicMock()
        s1.name.return_value = "s1"
        s2 = MagicMock()
        s2.name.return_value = "s2"
        event_repo = MagicMock()
        event_repo.is_alert_sent.return_value = False
        poi_repo = MagicMock()
        bus = EventBus()
        service = AlertService([s1, s2], event_repo, poi_repo, bus)

        alert = _make_alert_payload()
        event = MatchFoundEvent(alert=alert, object_id="obj-m", timestamp="t1")
        bus.publish("match_found", event)

        s1.send.assert_called_once()
        s2.send.assert_called_once()

    def test_create_alert_payload(self):
        service, _, event_repo, poi_repo, _ = self._make_service()
        match = MatchResult(poi_id="poi-a", similarity_score=0.88, faiss_distance=0.88)

        result = service.create_alert_payload(
            match=match,
            object_id="obj-x",
            timestamp="2025-01-15T12:00:00Z",
            camera_id="cam-01",
            region_name="aisle1",
            confidence=0.95,
            center_of_mass={"x": 320, "y": 240, "width": 80, "height": 160},
        )

        assert result.poi_id == "poi-a"
        assert result.severity == "high"
        assert result.match["camera_id"] == "cam-01"
        assert result.match["confidence"] == 0.95
        assert result.match["similarity_score"] == 0.88
        assert result.poi_metadata["name"] == "test suspect"
        assert result.poi_metadata["notes"] == ""

    def test_strategy_error_doesnt_halt_dispatch(self):
        """All strategies are attempted even if one fails, but alert is NOT
        persisted because all-must-succeed semantics require full delivery."""
        bad_strategy = MagicMock()
        bad_strategy.name.return_value = "bad"
        bad_strategy.send.side_effect = RuntimeError("network error")
        good_strategy = MagicMock()
        good_strategy.name.return_value = "good"

        event_repo = MagicMock()
        event_repo.is_alert_sent.return_value = False
        poi_repo = MagicMock()
        bus = EventBus()
        service = AlertService([bad_strategy, good_strategy], event_repo, poi_repo, bus)

        alert = _make_alert_payload()
        event = MatchFoundEvent(alert=alert, object_id="obj-e", timestamp="t1")
        bus.publish("match_found", event)

        bad_strategy.send.assert_called_once()
        good_strategy.send.assert_called_once()  # Not blocked by bad_strategy's error
        # Partial delivery → NOT persisted (all-must-succeed semantics)
        event_repo.store_alert.assert_not_called()
        event_repo.mark_alert_sent.assert_not_called()

    def test_failed_delivery_does_not_mark_sent(self):
        """If every strategy fails, the alert is NOT marked sent so it can retry."""
        bad_strategy = MagicMock()
        bad_strategy.name.return_value = "bad"
        bad_strategy.send.side_effect = RuntimeError("alert-service down")

        event_repo = MagicMock()
        event_repo.is_alert_sent.return_value = False
        poi_repo = MagicMock()
        bus = EventBus()
        AlertService([bad_strategy], event_repo, poi_repo, bus)

        alert = _make_alert_payload()
        event = MatchFoundEvent(alert=alert, object_id="obj-f", timestamp="t1")
        bus.publish("match_found", event)

        bad_strategy.send.assert_called_once()
        event_repo.store_alert.assert_not_called()
        event_repo.mark_alert_sent.assert_not_called()
