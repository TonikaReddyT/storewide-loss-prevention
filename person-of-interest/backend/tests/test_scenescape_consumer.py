"""Tests for ScenescapeRegionConsumer — region entry/exit via diffing."""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from backend.consumers.scenescape_consumer import ScenescapeRegionConsumer


@pytest.fixture
def event_service():
    return MagicMock()


@pytest.fixture
def consumer(event_service):
    return ScenescapeRegionConsumer(event_service)


# Regulated scene topic format: scenescape/regulated/scene/{scene_id}
_TOPIC = "scenescape/regulated/scene/scene-001"


def _make_payload(timestamp, persons):
    """Build a regulated scene payload with persons list.

    Each person should be a dict with at least 'id' and 'regions' keys.
    """
    return {
        "timestamp": timestamp,
        "objects": persons,
    }


def _make_person(person_id, regions=None, category="person"):
    """Build a person object for the regulated scene payload."""
    p = {"id": person_id, "category": category}
    if regions is not None:
        p["regions"] = regions
    return p


def test_region_entry_detected(consumer, event_service):
    """First appearance of a region in a person's regions dict triggers entry."""
    person = _make_person("person-1", regions={"zone-a": {"name": "Zone A"}})
    payload = _make_payload("2025-01-15T12:00:00Z", [person])
    consumer.handle_event(_TOPIC, payload)
    event_service.store_region_entry.assert_called_once_with(
        "person-1", "2025-01-15T12:00:00Z", "scene-001", "zone-a", "Zone A", None,
        entry_frame_key=None,
    )
    event_service.store_region_exit.assert_not_called()


def test_region_exit_detected(consumer, event_service):
    """Region disappearing from a person's regions dict triggers exit."""
    # First message: person enters zone-a
    person1 = _make_person("person-1", regions={"zone-a": {"name": "Zone A"}})
    consumer.handle_event(_TOPIC, _make_payload("t1", [person1]))

    # Second message: person has no regions (exited zone-a)
    person2 = _make_person("person-1", regions={})
    consumer.handle_event(_TOPIC, _make_payload("t2", [person2]))

    event_service.store_region_exit.assert_called_once()
    call_args = event_service.store_region_exit.call_args[0]
    assert call_args[0] == "person-1"
    assert call_args[3] == "zone-a"


def test_non_person_ignored(consumer, event_service):
    """Non-person objects should not trigger region events."""
    obj = _make_person("cart-1", regions={"zone-a": {"name": "Zone A"}}, category="cart")
    payload = _make_payload("t1", [obj])
    consumer.handle_event(_TOPIC, payload)
    event_service.store_region_entry.assert_not_called()
    event_service.store_region_exit.assert_not_called()


def test_non_matching_topic_ignored(consumer, event_service):
    """Topics not matching the regulated pattern are silently ignored."""
    consumer.handle_event("scenescape/event/bfb9f86b/objects", {"timestamp": "t", "objects": []})
    event_service.store_region_entry.assert_not_called()
    event_service.store_region_exit.assert_not_called()


def test_service_exception_does_not_propagate(consumer, event_service):
    """Exceptions in event_service are caught and do not raise."""
    event_service.store_region_entry.side_effect = RuntimeError("Redis down")
    person = _make_person("person-2", regions={"zone-b": {"name": "Zone B"}})
    payload = _make_payload("t1", [person])
    # Should not raise
    consumer.handle_event(_TOPIC, payload)


def test_person_leaving_scene_triggers_exit(consumer, event_service):
    """Person present in message 1 but absent in message 2 triggers implicit exit."""
    person = _make_person("person-3", regions={"zone-a": {"name": "Zone A"}})
    consumer.handle_event(_TOPIC, _make_payload("t1", [person]))

    # Next message has no persons — person-3 left the scene
    consumer.handle_event(_TOPIC, _make_payload("t2", []))

    event_service.store_region_exit.assert_called_once()


def test_multiple_regions_tracked(consumer, event_service):
    """Entering multiple regions, then exiting one, only fires exit for the left region."""
    person = _make_person("p1", regions={
        "zone-a": {"name": "Zone A"},
        "zone-b": {"name": "Zone B"},
    })
    consumer.handle_event(_TOPIC, _make_payload("t1", [person]))
    assert event_service.store_region_entry.call_count == 2

    # Leave zone-a, stay in zone-b
    person2 = _make_person("p1", regions={"zone-b": {"name": "Zone B"}})
    consumer.handle_event(_TOPIC, _make_payload("t2", [person2]))
    event_service.store_region_exit.assert_called_once()
    assert event_service.store_region_exit.call_args[0][3] == "zone-a"
