"""Embedding domain entity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Embedding:
    embedding_id: str
    vector: np.ndarray  # 256-d float32
    poi_id: Optional[str] = None
    faiss_id: Optional[int] = None

    @property
    def dimension(self) -> int:
        return self.vector.shape[0]

    def normalized(self) -> np.ndarray:
        norm = np.linalg.norm(self.vector)
        if norm == 0:
            return self.vector
        return self.vector / norm
