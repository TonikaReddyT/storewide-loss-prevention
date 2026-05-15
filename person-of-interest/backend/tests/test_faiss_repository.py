"""Tests for FAISS repository — uses real FAISS index in-memory."""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _reset_faiss_singleton():
    """Reset FAISSRepository singleton between tests."""
    from backend.infrastructure.faiss.repository import FAISSRepository

    FAISSRepository._instance = None
    yield
    FAISSRepository._instance = None


@pytest.fixture
def faiss_repo(tmp_path):
    """Create a FAISSRepository with unique temp paths per test."""
    with patch("backend.infrastructure.faiss.repository.get_config") as mock_cfg:
        cfg = type("C", (), {
            "faiss_dimension": 256,
            "faiss_index_path": str(tmp_path / "test.index"),
            "faiss_id_map_path": str(tmp_path / "test_id_map.json"),
        })()
        mock_cfg.return_value = cfg

        from backend.infrastructure.faiss.repository import FAISSRepository

        FAISSRepository._instance = None
        repo = FAISSRepository()
        return repo


class TestFAISSRepository:
    def test_add_and_search(self, faiss_repo):
        v1 = np.random.randn(256).astype(np.float32)
        v1 /= np.linalg.norm(v1)

        ids = faiss_repo.add("poi-1", [v1])
        assert len(ids) == 1
        assert faiss_repo.total_vectors() == 1

        results = faiss_repo.search(v1, top_k=1)
        assert len(results) == 1
        fid, score = results[0]
        assert fid == ids[0]
        assert score > 0.99  # Same vector → cosine ~1.0

    def test_add_multiple_vectors(self, faiss_repo):
        vectors = [np.random.randn(256).astype(np.float32) for _ in range(3)]
        ids = faiss_repo.add("poi-multi", vectors)
        assert len(ids) == 3
        assert faiss_repo.total_vectors() == 3

    def test_search_empty_index(self, faiss_repo):
        v = np.random.randn(256).astype(np.float32)
        results = faiss_repo.search(v, top_k=5)
        assert results == []

    def test_get_poi_id_for_faiss_id(self, faiss_repo):
        v = np.random.randn(256).astype(np.float32)
        ids = faiss_repo.add("poi-lookup", [v])

        assert faiss_repo.get_poi_id_for_faiss_id(ids[0]) == "poi-lookup"
        assert faiss_repo.get_poi_id_for_faiss_id(99999) is None

    def test_remove(self, faiss_repo):
        v = np.random.randn(256).astype(np.float32)
        faiss_repo.add("poi-rm", [v])
        assert faiss_repo.total_vectors() == 1

        faiss_repo.remove("poi-rm")
        assert faiss_repo.total_vectors() == 0
        assert faiss_repo.get_poi_id_for_faiss_id(0) is None

    def test_remove_nonexistent(self, faiss_repo):
        # Should not raise
        faiss_repo.remove("poi-nobody")

    def test_save_and_load(self, faiss_repo, tmp_path):
        v = np.random.randn(256).astype(np.float32)
        v /= np.linalg.norm(v)
        ids = faiss_repo.add("poi-persist", [v])
        faiss_repo.save_to_disk()

        # Reset singleton and reload
        from backend.infrastructure.faiss.repository import FAISSRepository
        FAISSRepository._instance = None

        with patch("backend.infrastructure.faiss.repository.get_config") as mock_cfg:
            cfg = type("C", (), {
                "faiss_dimension": 256,
                "faiss_index_path": str(tmp_path / "test.index"),
                "faiss_id_map_path": str(tmp_path / "test_id_map.json"),
            })()
            mock_cfg.return_value = cfg
            repo2 = FAISSRepository()

        assert repo2.total_vectors() == 1
        results = repo2.search(v, top_k=1)
        assert len(results) == 1
        assert results[0][1] > 0.99

    def test_cosine_similarity_ranking(self, faiss_repo):
        # Create a query vector
        query = np.random.randn(256).astype(np.float32)
        query /= np.linalg.norm(query)

        # Similar vector (small perturbation)
        similar = query + np.random.randn(256).astype(np.float32) * 0.1
        similar /= np.linalg.norm(similar)

        # Dissimilar vector
        dissimilar = np.random.randn(256).astype(np.float32)
        dissimilar /= np.linalg.norm(dissimilar)

        faiss_repo.add("poi-similar", [similar])
        faiss_repo.add("poi-dissimilar", [dissimilar])

        results = faiss_repo.search(query, top_k=2)
        assert len(results) == 2
        # Similar vector should rank first with higher score
        fid_first, score_first = results[0]
        assert faiss_repo.get_poi_id_for_faiss_id(fid_first) == "poi-similar"
        assert score_first > results[1][1]

    def test_multiple_pois_search(self, faiss_repo):
        for i in range(5):
            v = np.random.randn(256).astype(np.float32)
            faiss_repo.add(f"poi-{i}", [v])

        assert faiss_repo.total_vectors() == 5

        query = np.random.randn(256).astype(np.float32)
        results = faiss_repo.search(query, top_k=3)
        assert len(results) == 3

    def test_top_k_larger_than_index(self, faiss_repo):
        v = np.random.randn(256).astype(np.float32)
        faiss_repo.add("poi-only", [v])

        results = faiss_repo.search(v, top_k=10)
        assert len(results) == 1  # Only 1 vector in index
