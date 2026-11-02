"""Exercise WebSocket disconnects through the real ASGI app and Starlette state machine."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from unittest.mock import AsyncMock

import pytest
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from src.api.routes.websocket import _build_socket_error_event, _send_socket_event
from src.api.services import chat_service
from src.guardrails.rate_limit import InMemoryRateLimiter
from tests.doubles import StoppableChatAgent
from tests.factories import make_agent_state

LOGGER = "src.api.routes.websocket"
HOST = "203.0.113.10"
THREAD_ID = "websocket-test-thread"


def _closed_error(message_type):
    return RuntimeError(
        f"Unexpected ASGI message '{message_type}', after sending 'websocket.close' "
        "or response already completed."
    )


def _scope():
    return {
        "type": "websocket",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "scheme": "ws",
        "path": f"/ws/{THREAD_ID}",
        "root_path": "",
        "query_string": b"",
        "headers": [],
        "client": (HOST, 50000),
        "server": ("testserver", 80),
        "subprotocols": [],
    }


class _SocketTransport:
    """An ASGI transport that rejects sends after the peer closes, like Uvicorn."""

    def __init__(self, *messages, send_failure=None, fail_on_send=1, close_on_failure=True):
        self.incoming = asyncio.Queue()
        for message in ({"type": "websocket.connect"}, *messages):
            self.incoming.put_nowait(message)
        self.closed = False
        self.sent = []
        self.attempts = []
        self.send_failure = send_failure
        self.fail_on_send = fail_on_send
        self.close_on_failure = close_on_failure
        self.send_count = 0

    async def receive(self):
        message = await self.incoming.get()
        if message["type"] == "websocket.disconnect":
            self.closed = True
        return message

    async def send(self, message):
        self.attempts.append(message)
        if self.closed:
            raise _closed_error(message["type"])
        if message["type"] == "websocket.send":
            self.send_count += 1
            if self.send_failure is not None and self.send_count == self.fail_on_send:
                self.closed = self.close_on_failure
                raise self.send_failure
        self.sent.append(message)
        if message["type"] == "websocket.close":
            self.closed = True

    async def run(self, app):
        await asyncio.wait_for(app(_scope(), self.receive, self.send), timeout=5)
        # The resumable run may continue independently after socket delivery ends.
        run = chat_service._active_runs.get(THREAD_ID)
        if run is not None and run.task is not None:
            with suppress(asyncio.CancelledError):
                await asyncio.wait_for(run.task, timeout=5)

    def events(self):
        return [json.loads(m["text"]) for m in self.sent if m["type"] == "websocket.send"]


@pytest.fixture
def socket_backend(chat_backend):
    chat_backend.final_state_factory = lambda thread_id: make_agent_state(
        response="Syarat kelulusan adalah 144 SKS.", session_id=thread_id
    )
    return chat_backend


def _errors(caplog):
    return [r for r in caplog.records if r.name == LOGGER and r.levelno >= logging.ERROR]


async def test_http_stop_sends_cancelled_event_and_socket_accepts_next_chat(
    app, api_client, chat_backend, valid_academic_message, caplog
):
    agent = StoppableChatAgent(THREAD_ID)
    chat_backend.agent_factory = lambda thread_id: agent
    connection = _SocketTransport(
        {"type": "websocket.receive", "text": valid_academic_message},
        {"type": "websocket.receive", "text": valid_academic_message},
        {"type": "websocket.disconnect", "code": 1000},
    )
    pending = asyncio.create_task(connection.run(app))
    try:
        await asyncio.wait_for(agent.started.wait(), timeout=5)
        run = chat_service._active_runs[THREAD_ID]
        response = await api_client.post(f"/chat/{THREAD_ID}/cancel", params={"run_id": run.run_id})
        await asyncio.wait_for(pending, timeout=5)
    finally:
        agent.release.set()

    assert response.json()["status"] == "cancelled"
    events = connection.events()
    cancelled = [e for e in events if e["event_type"] == "run.cancelled"]
    assert len(cancelled) == 1
    assert events[-1]["event_type"] == "run.completed"
    assert events[-1]["run_id"] != cancelled[0]["run_id"]
    assert len(agent.stream_calls) == 2
    assert _errors(caplog) == []


async def test_disconnect_is_not_treated_as_an_empty_chat_message(app, socket_backend, caplog):
    connection = _SocketTransport({"type": "websocket.disconnect", "code": 1000})

    await connection.run(app)

    assert [m["type"] for m in connection.attempts] == ["websocket.accept"]
    assert socket_backend.agents == []
    assert _errors(caplog) == []


@pytest.mark.parametrize("encoding", ["plain", "json"])
async def test_text_chat_finishes_and_then_disconnects_cleanly(
    app, socket_backend, valid_academic_message, caplog, encoding
):
    text = valid_academic_message
    if encoding == "json":
        text = json.dumps({"message": text, "last_event_id": None})
    connection = _SocketTransport(
        {"type": "websocket.receive", "text": text},
        {"type": "websocket.disconnect", "code": 1000},
    )

    await connection.run(app)

    events = connection.events()
    assert events[0]["event_type"] == "run.started"
    assert events[-1]["event_type"] == "run.completed"
    assert len(socket_backend.last_agent.stream_calls) == 1
    assert not any(m["type"] == "websocket.close" for m in connection.attempts)
    assert _errors(caplog) == []


@pytest.mark.parametrize("failure", [OSError("peer closed"), _closed_error("websocket.send")])
@pytest.mark.parametrize("path", ["stream", "input_guard", "rate_limit"])
async def test_disconnect_during_send_stops_without_sending_close_again(
    app, socket_backend, valid_academic_message, prompt_injection_message, caplog, path, failure
):
    text = prompt_injection_message if path == "input_guard" else valid_academic_message
    if path == "rate_limit":
        app.state.rate_limiter = InMemoryRateLimiter(limit=1)
        assert app.state.rate_limiter.allow(HOST)
    connection = _SocketTransport({"type": "websocket.receive", "text": text}, send_failure=failure)

    await connection.run(app)

    assert [m["type"] for m in connection.attempts] == ["websocket.accept", "websocket.send"]
    assert _errors(caplog) == []


async def test_interrupted_socket_can_resume_the_same_run(
    app, socket_backend, valid_academic_message, caplog
):
    first = _SocketTransport(
        {"type": "websocket.receive", "text": valid_academic_message},
        send_failure=_closed_error("websocket.send"),
        fail_on_send=2,
    )
    await first.run(app)
    cursor = first.events()[-1]
    resumed = _SocketTransport(
        {
            "type": "websocket.receive",
            "text": json.dumps(
                {
                    "message": valid_academic_message,
                    "last_event_id": cursor["event_id"],
                }
            ),
        },
        {"type": "websocket.disconnect", "code": 1000},
    )

    await resumed.run(app)

    events = resumed.events()
    assert events[-1]["event_type"] == "run.completed"
    assert all(event["run_id"] == cursor["run_id"] for event in events)
    assert all(event["sequence"] > cursor["sequence"] for event in events)
    assert len(socket_backend.last_agent.stream_calls) == 1
    assert _errors(caplog) == []


@pytest.mark.parametrize("side", ["client_state", "application_state"])
async def test_known_closed_socket_is_never_sent_an_event(side):
    send = AsyncMock()
    socket = WebSocket(_scope(), AsyncMock(), send)
    socket.client_state = socket.application_state = WebSocketState.CONNECTED
    setattr(socket, side, WebSocketState.DISCONNECTED)

    with pytest.raises(WebSocketDisconnect):
        await _send_socket_event(socket, _build_socket_error_event(THREAD_ID, "test"))

    send.assert_not_awaited()


@pytest.mark.parametrize("closed_during_failure", [False, True])
async def test_unexpected_processing_error_closes_once_if_possible(
    app, socket_backend, valid_academic_message, monkeypatch, caplog, closed_during_failure
):
    connection = _SocketTransport({"type": "websocket.receive", "text": valid_academic_message})

    def fail_stream(**kwargs):
        connection.closed = closed_during_failure
        raise ValueError("backend unavailable")

    monkeypatch.setattr(chat_service, "stream_chat_message", fail_stream)
    await connection.run(app)

    assert connection.attempts[-1] == {"type": "websocket.close", "code": 1011, "reason": ""}
    assert len(_errors(caplog)) == 1
    assert "backend unavailable" in _errors(caplog)[0].getMessage()


async def test_unrelated_send_error_remains_visible(
    app, socket_backend, valid_academic_message, caplog
):
    connection = _SocketTransport(
        {"type": "websocket.receive", "text": valid_academic_message},
        send_failure=RuntimeError("unexpected send failure"),
        close_on_failure=False,
    )

    await connection.run(app)

    assert connection.sent[-1]["type"] == "websocket.close"
    assert connection.sent[-1]["code"] == 1011
    assert len(_errors(caplog)) == 1
    assert "unexpected send failure" in _errors(caplog)[0].getMessage()
