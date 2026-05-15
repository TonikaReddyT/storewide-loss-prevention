"""Alert ingestion endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from src.core.models import AlertEnvelope
from src.delivery.ws_manager import ws_manager

router = APIRouter(tags=["alerts"])


@router.post("/alerts")
async def ingest_alert(request: Request) -> dict:
    """Accept a flexible JSON alert payload, enqueue for async processing."""
    body: dict[str, Any] = await request.json()
    envelope = AlertEnvelope.from_raw(body)

    worker = request.app.state.worker
    await worker.enqueue(envelope)

    return {
        "status": "accepted",
        "alert_type": envelope.alert_type,
        "timestamp": envelope.timestamp,
    }


@router.delete("/alerts")
async def clear_alert_history() -> dict:
    """Clear the in-memory WebSocket history so reconnecting clients don't see old alerts."""
    cleared = ws_manager.clear_history()
    return {"status": "cleared", "history_cleared": cleared}
