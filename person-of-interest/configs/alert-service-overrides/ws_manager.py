"""WebSocket connection manager — tracks active clients and broadcasts messages.

Extended with a history ring buffer: when a new client connects it immediately
receives the last HISTORY_SIZE alerts so page refresh doesn't lose alert history.
"""

from __future__ import annotations

import json
import logging
from collections import deque

from fastapi import WebSocket

logger = logging.getLogger(__name__)

# Number of recent alerts to replay to newly connected clients
HISTORY_SIZE = 100


class ConnectionManager:
    """Manages active WebSocket connections for alert broadcasting."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []
        self._history: deque[str] = deque(maxlen=HISTORY_SIZE)

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)
        logger.info("WebSocket client connected (%d total)", len(self._connections))

        # Replay history so the client sees past alerts immediately
        if self._history:
            logger.info("Replaying %d historical alerts to new client", len(self._history))
            for message in self._history:
                try:
                    await ws.send_text(message)
                except Exception:
                    logger.warning("Failed to send historical alert to new client")
                    break

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)
        logger.info("WebSocket client disconnected (%d remaining)", len(self._connections))

    async def broadcast(self, message: str) -> None:
        """Send a message to all connected clients and store in history ring buffer."""
        # Store in history before broadcast so new clients get it
        self._history.append(message)

        dead: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in self._connections:
                self._connections.remove(ws)
            logger.warning("Removed dead WebSocket connection (%d remaining)", len(self._connections))

    @property
    def active_count(self) -> int:
        return len(self._connections)

    @property
    def history_count(self) -> int:
        return len(self._history)

    def clear_history(self) -> int:
        """Clear the in-memory alert history ring buffer. Returns count cleared."""
        count = len(self._history)
        self._history.clear()
        logger.info("Alert history cleared (%d entries removed)", count)
        return count


# Singleton instance shared between the WS endpoint and the delivery handler
ws_manager = ConnectionManager()
