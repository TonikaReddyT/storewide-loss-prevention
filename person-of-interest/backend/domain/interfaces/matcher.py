"""Matcher interface for the domain layer."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from backend.domain.entities.match_result import MatchResult


class MatchingStrategy(ABC):
    """Strategy interface for embedding matching."""

    @abstractmethod
    def match(
        self, query_vector: np.ndarray, top_k: int = 5, threshold: float = 0.6
    ) -> list[MatchResult]: ...
