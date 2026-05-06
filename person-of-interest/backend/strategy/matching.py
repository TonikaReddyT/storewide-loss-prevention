"""Matching strategies — Strategy pattern."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from backend.domain.entities.match_result import MatchResult
from backend.domain.interfaces.matcher import MatchingStrategy
from backend.infrastructure.faiss.repository import FAISSRepository

log = logging.getLogger("poi.strategy.matching")


class CosineSimilarityStrategy(MatchingStrategy):
    """FAISS Inner-Product search on L2-normalized vectors (≡ cosine similarity)."""

    def __init__(self, faiss_repo: FAISSRepository) -> None:
        self._faiss = faiss_repo

    def match(
        self, query_vector: np.ndarray, top_k: int = 5, threshold: float = 0.6
    ) -> list[MatchResult]:
        # Log query vector norm — should be ~1.0
        query_norm = float(np.linalg.norm(query_vector))
        if abs(query_norm - 1.0) > 0.01:
            log.warning("Query vector norm=%.6f (expected ~1.0) — possible normalisation issue", query_norm)

        results = self._faiss.search(query_vector, top_k)
        matches = []
        for rank, (faiss_id, distance) in enumerate(results):
            # Inner product of L2-normed vectors = cosine similarity ∈ [-1, 1]
            similarity = float(distance)
            poi_id = self._faiss.get_poi_id_for_faiss_id(faiss_id)
            is_match = similarity >= threshold
            log.info(
                "FAISS rank=%d: poi=%s similarity=%.4f threshold=%.2f %s",
                rank, poi_id or faiss_id, similarity, threshold,
                "✓ MATCH" if is_match else "✗ below",
            )
            if is_match and poi_id:
                matches.append(
                    MatchResult(
                        poi_id=poi_id,
                        similarity_score=similarity,
                        faiss_distance=distance,
                        embedding_id=str(faiss_id),
                    )
                )
        # Sort by similarity descending
        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        return matches
