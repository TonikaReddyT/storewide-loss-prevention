"""Tests for Observer pattern (EventBus)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from backend.domain.entities.match_result import AlertPayload
from backend.observer.events import EventBus, MatchFoundEvent


class TestEventBus:
    def test_subscribe_and_publish(self):
        bus = EventBus()
        callback = MagicMock()
        bus.subscribe("test_event", callback)

        event = {"data": "hello"}
        bus.publish("test_event", event)

        callback.assert_called_once_with(event)

    def test_multiple_subscribers(self):
        bus = EventBus()
        cb1 = MagicMock()
        cb2 = MagicMock()
        bus.subscribe("evt", cb1)
        bus.subscribe("evt", cb2)

        bus.publish("evt", "payload")

        cb1.assert_called_once_with("payload")
        cb2.assert_called_once_with("payload")

    def test_no_subscribers(self):
        bus = EventBus()
        # Should not raise
        bus.publish("nonexistent", {})

    def test_subscriber_error_isolated(self):
        bus = EventBus()
        bad_cb = MagicMock(side_effect=RuntimeError("boom"))
        good_cb = MagicMock()
        bus.subscribe("evt", bad_cb)
        bus.subscribe("evt", good_cb)

        bus.publish("evt", "data")

        bad_cb.assert_called_once()
        good_cb.assert_called_once()  # not affected by bad_cb's error

    def test_publish_async(self):
        bus = EventBus()
        results = []

        async def async_cb(event):
            results.append(event)

        bus.subscribe("async_evt", async_cb)
        asyncio.get_event_loop().run_until_complete(
            bus.publish_async("async_evt", "async_data")
        )
        assert results == ["async_data"]

    def test_different_event_types(self):
        bus = EventBus()
        cb_a = MagicMock()
        cb_b = MagicMock()
        bus.subscribe("type_a", cb_a)
        bus.subscribe("type_b", cb_b)

        bus.publish("type_a", "only_a")

        cb_a.assert_called_once_with("only_a")
        cb_b.assert_not_called()


class TestMatchFoundEvent:
    def test_creation(self):
        alert = AlertPayload(
            alert_id="a1",
            poi_id="p1",
            severity="high",
            timestamp="2025-01-01T00:00:00Z",
            match={"camera_id": "c1"},
            poi_metadata={"notes": ""},
        )
        event = MatchFoundEvent(alert=alert, object_id="obj-1", timestamp="t1")
        assert event.object_id == "obj-1"
        assert event.alert.poi_id == "p1"
