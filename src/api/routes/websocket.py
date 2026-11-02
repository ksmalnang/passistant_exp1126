"""WebSocket endpoints."""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from src.api.models import ChatStreamEvent
from src.api.services import chat_service
from src.guardrails.input_guard import InputGuard

logger = logging.getLogger(__name__)
router = APIRouter()
_guard = InputGuard()


@router.websocket("/ws/{thread_id}")
async def websocket_endpoint(websocket: WebSocket, thread_id: str):
    """WebSocket endpoint for real-time chat."""
    await websocket.accept()

    try:
        while True:
            payload = await _receive_chat_payload(websocket)
            message = payload["message"]
            last_event_id = payload.get("last_event_id")

            if not websocket.app.state.rate_limiter.allow(
                getattr(getattr(websocket, "client", None), "host", None) or "unknown"
            ):
                await _send_socket_event(
                    websocket,
                    _build_socket_error_event(
                        thread_id=thread_id,
                        message="Rate limit exceeded. Please try again later.",
                    ),
                )
                continue

            guard_result = _guard.validate(message)
            if not guard_result.safe:
                await _send_socket_event(
                    websocket,
                    _build_socket_error_event(
                        thread_id=thread_id,
                        message=f"Message rejected: {guard_result.reason}",
                    ),
                )
                continue

            async for event in chat_service.stream_chat_message(
                message=guard_result.sanitized or message.strip(),
                thread_id=thread_id,
                last_event_id=last_event_id,
            ):
                await _send_socket_event(websocket, event)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: %s", thread_id)
    except Exception as exc:
        logger.error("WebSocket error: %s", exc, exc_info=True)
        if _socket_connected(websocket):
            # The transport may close between the state check and this cleanup.
            with suppress(WebSocketDisconnect, OSError, RuntimeError):
                await websocket.close(code=1011)


def _socket_connected(websocket: WebSocket) -> bool:
    return (
        websocket.client_state == WebSocketState.CONNECTED
        and websocket.application_state == WebSocketState.CONNECTED
    )


async def _send_socket_event(websocket: WebSocket, event: ChatStreamEvent) -> None:
    """Stop delivery when the peer closes, including a close racing with send."""
    if not _socket_connected(websocket):
        raise WebSocketDisconnect(code=1006)
    try:
        await websocket.send_json(event.model_dump(mode="json"))
    except OSError as exc:
        raise WebSocketDisconnect(code=1006) from exc
    except RuntimeError as exc:
        # Some ASGI transports report a closed socket as RuntimeError while
        # Starlette's local state still says CONNECTED. Preserve other errors.
        if not _socket_connected(websocket) or (
            "Unexpected ASGI message" in str(exc)
            and ("websocket.close" in str(exc) or "response already completed" in str(exc))
        ):
            raise WebSocketDisconnect(code=1006) from exc
        raise


async def _receive_chat_payload(websocket: WebSocket) -> dict[str, Any]:
    """Accept either JSON messages or raw text messages."""
    incoming = await websocket.receive()
    if incoming["type"] == "websocket.disconnect":
        raise WebSocketDisconnect(
            code=incoming.get("code", 1000), reason=incoming.get("reason", "")
        )

    text = incoming.get("text")
    if text is None:
        return {"message": ""}

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"message": text}

    if isinstance(payload, str):
        return {"message": payload}
    if isinstance(payload, dict):
        return {
            "message": str(payload.get("message", "")),
            "last_event_id": payload.get("last_event_id"),
        }
    return {"message": text}


def _build_socket_error_event(thread_id: str, message: str) -> ChatStreamEvent:
    """Build a typed websocket error event."""
    return ChatStreamEvent(
        event_id=f"rejected:{uuid.uuid4()}",
        event_type="run.failed",
        thread_id=thread_id,
        run_id="rejected",
        timestamp=datetime.now(UTC),
        sequence=0,
        data={
            "code": "REQUEST_REJECTED",
            "message": message,
            "retryable": False,
        },
    )
