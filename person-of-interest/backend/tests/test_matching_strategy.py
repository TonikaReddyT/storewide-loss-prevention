"""Tests for CosineSimilarityStrategy."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from backend.domain.entities.match_result import MatchResult
from backend.strategy.matching import CosineSimilarityStrategy


class TestCosineSimilarityStrategy:
    def _make_strategy(self, search_results, id_map=None):
        faiss_repo = MagicMock()
        faiss_repo.search.return_value = search_results
        if id_map is None:
            id_map = {}
        faiss_repo.get_poi_id_for_faiss_id.side_effect = lambda fid: id_map.get(fid)
        return CosineSimilarityStrategy(faiss_repo), faiss_repo

    def test_match_above_threshold(self):
        strategy, _ = self._make_strategy(
            search_results=[(0, 0.92), (1, 0.55)],
            id_map={0: "poi-a", 1: "poi-b"},
        )
        query = np.random.randn(256).astype(np.float32)
        results = strategy.match(query, top_k=5, threshold=0.6)

        assert len(results) == 1
        assert results[0].poi_id == "poi-a"
        assert results[0].similarity_score == 0.92

    def test_match_no_results_above_threshold(self):
        strategy, _ = self._make_strategy(
            search_results=[(0, 0.3), (1, 0.1)],
            id_map={0: "poi-a", 1: "poi-b"},
        )
        query = np.random.randn(256).astype(np.float32)
        results = strategy.match(query, top_k=5, threshold=0.6)
        assert results == []

    def test_match_empty_index(self):
        strategy, _ = self._make_strategy(search_results=[])
        query = np.random.randn(256).astype(np.float32)
        results = strategy.match(query, top_k=5, threshold=0.6)
        assert results == []

    def test_match_missing_poi_id_skipped(self):
        strategy, _ = self._make_strategy(
            search_results=[(0, 0.9), (1, 0.8)],
            id_map={0: "poi-a"},  # faiss_id=1 has no poi mapping
        )
        query = np.random.randn(256).astype(np.float32)
        results = strategy.match(query, top_k=5, threshold=0.6)
        assert len(results) == 1
        assert results[0].poi_id == "poi-a"

    def test_match_sorted_by_similarity(self):
        strategy, _ = self._make_strategy(
            search_results=[(0, 0.7), (1, 0.95), (2, 0.8)],
            id_map={0: "poi-a", 1: "poi-b", 2: "poi-c"},
        )
        query = np.random.randn(256).astype(np.float32)
        results = strategy.match(query, top_k=5, threshold=0.6)
        assert len(results) == 3
        assert results[0].poi_id == "poi-b"
        assert results[1].poi_id == "poi-c"
        assert results[2].poi_id == "poi-a"

    def test_custom_threshold(self):
        strategy, _ = self._make_strategy(
            search_results=[(0, 0.85)],
            id_map={0: "poi-a"},
        )
        query = np.random.randn(256).astype(np.float32)
        # Threshold higher than result
        results = strategy.match(query, top_k=5, threshold=0.9)
        assert results == []
