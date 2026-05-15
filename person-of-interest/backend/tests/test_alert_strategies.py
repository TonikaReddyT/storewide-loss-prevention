"""Tests for AlertServiceStrategy."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from backend.domain.entities.match_result import AlertPayload
from backend.strategy.alert import AlertServiceStrategy


def _make_alert():
    return AlertPayload(
        alert_id="alert-001",
        poi_id="poi-a",
        severity="high",
        timestamp="2025-01-15T12:00:00Z",
        match={"camera_id": "cam-01", "similarity_score": 0.9, "confidence": 0.95,
               "bbox": [0, 0, 100, 100], "frame_number": 0, "thumbnail_path": ""},
        poi_metadata={"notes": "test", "enrollment_date": "", "total_previous_matches": 0},
    )


class TestAlertServiceStrategy:
    def test_name(self):
        assert AlertServiceStrategy("http://localhost:8001").name() == "alert_service"

    def test_send_posts_to_alert_service(self):
        strategy = AlertServiceStrategy("http://alert-svc:8001")
        alert = _make_alert()

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None

        with patch("backend.strategy.alert.requests.post", return_value=mock_resp) as mock_post:
            strategy.send(alert)

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[0][0] == "http://alert-svc:8001/api/v1/alerts"
        payload = call_kwargs[1]["json"]
        assert payload["alert_type"] == "POI_MATCH"
        assert payload["metadata"]["poi_id"] == "poi-a"
        assert payload["metadata"]["severity"] == "high"
        assert payload["metadata"]["camera_id"] == "cam-01"
        mock_resp.raise_for_status.assert_called_once()

    def test_send_raises_on_http_error(self):
        strategy = AlertServiceStrategy("http://alert-svc:8001")
        alert = _make_alert()

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("503")

        with patch("backend.strategy.alert.requests.post", return_value=mock_resp):
            with pytest.raises(requests.HTTPError):
                strategy.send(alert)

    def test_send_raises_on_connection_error(self):
        strategy = AlertServiceStrategy("http://alert-svc:8001")
        alert = _make_alert()

        with patch("backend.strategy.alert.requests.post",
                   side_effect=requests.ConnectionError("refused")):
            with pytest.raises(requests.ConnectionError):
                strategy.send(alert)

    def test_trailing_slash_stripped_from_url(self):
        strategy = AlertServiceStrategy("http://alert-svc:8001/")
        alert = _make_alert()

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None

        with patch("backend.strategy.alert.requests.post", return_value=mock_resp) as mock_post:
            strategy.send(alert)

        url = mock_post.call_args[0][0]
        assert url == "http://alert-svc:8001/api/v1/alerts"

