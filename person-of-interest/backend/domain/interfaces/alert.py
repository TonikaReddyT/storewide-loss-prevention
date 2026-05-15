"""Alert interface for the domain layer."""

from __future__ import annotations

from abc import ABC, abstractmethod

from backend.domain.entities.match_result import AlertPayload


class AlertStrategy(ABC):
    """Strategy interface for alert delivery."""

    @abstractmethod
    def send(self, alert: AlertPayload) -> None: ...

    @abstractmethod
    def name(self) -> str: ...
