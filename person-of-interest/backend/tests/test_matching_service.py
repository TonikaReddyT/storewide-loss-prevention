"""Tests for MatchingService — Cache-Aside pattern."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from backend.domain.entities.match_result import MatchResult
from backend.service.matching_service import MatchingService


class TestMatchingService:
    def _make_service(self, strategy_result=None, cached_poi=None):
        strategy = MagicMock()
        strategy.match.return_value = strategy_result or []

        cache = MagicMock()
        cache.get_poi_for_object.return_value = cached_poi

        with patch("backend.service.matching_service.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.search_top_k = 10
            cfg.similarity_threshold = 0.6
            cfg.object_cache_ttl = 300
            cfg.benchmark_latency = False
            mock_cfg.return_value = cfg
            service = MatchingService(strategy, cache)

        return service, strategy, cache

    def test_cache_hit(self):
        service, strategy, cache = self._make_service(cached_poi="poi-cached")
        cache.get_similarity_for_object = MagicMock(return_value=0.92)
        embedding = np.random.randn(256).tolist()

        result = service.match_object("obj-1", embedding)

        assert result is not None
        assert result.poi_id == "poi-cached"
        assert result.similarity_score == 0.92
        strategy.match.assert_not_called()  # FAISS skipped

    def test_cache_miss_with_match(self):
        match = MatchResult(poi_id="poi-new", similarity_score=0.88, faiss_distance=0.88)
        service, strategy, cache = self._make_service(strategy_result=[match])
        embedding = np.random.randn(256).tolist()

        result = service.match_object("obj-2", embedding)

        assert result is not None
        assert result.poi_id == "poi-new"
        strategy.match.assert_called_once()
        cache.set_poi_for_object.assert_called_once_with("obj-2", "poi-new", ttl=300, similarity=0.88)

    def test_cache_miss_no_match(self):
        service, strategy, cache = self._make_service(strategy_result=[])
        embedding = np.random.randn(256).tolist()

        result = service.match_object("obj-3", embedding)

        assert result is None
        strategy.match.assert_called_once()
        cache.set_poi_for_object.assert_not_called()

    def test_nested_embedding_flattened(self):
        match = MatchResult(poi_id="poi-flat", similarity_score=0.9, faiss_distance=0.9)
        service, strategy, cache = self._make_service(strategy_result=[match])

        # 2D array [[...]]
        embedding = [np.random.randn(256).tolist()]

        result = service.match_object("obj-4", embedding)
        assert result is not None
        # The service should handle 2D input via np.array → ndim check

    def test_best_match_returned(self):
        m1 = MatchResult(poi_id="poi-a", similarity_score=0.7, faiss_distance=0.7)
        m2 = MatchResult(poi_id="poi-b", similarity_score=0.95, faiss_distance=0.95)
        service, strategy, cache = self._make_service(strategy_result=[m2, m1])
        embedding = np.random.randn(256).tolist()

        result = service.match_object("obj-5", embedding)

        # First match is taken (strategy already sorted)
        assert result.poi_id == "poi-b"
