"""Lifespan wiring tests for the API app."""

import logging
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI

from src.api import lifespan
from src.clients import JevDecisionClient
from src.utils.vector_store import VectorStoreTools


async def test_api_lifespan_runs_graph_sync_until_shutdown(app, monkeypatch):
    events = []

    @asynccontextmanager
    async def sync_context():
        events.append("sync_started")
        try:
            yield
        finally:
            events.append("sync_stopped")

    async def close_clients():
        events.append("clients_closed")

    monkeypatch.setattr(VectorStoreTools, "warm_up", lambda self: None)
    monkeypatch.setattr("src.api.graph_sync_lifespan", sync_context)
    monkeypatch.setattr("src.api.aclose_all_clients", close_clients)

    async with lifespan(app):
        assert events == ["sync_started"]
    assert events == ["sync_started", "sync_stopped", "clients_closed"]


async def test_api_lifespan_01_shutdown_closes_decision_client(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """App shutdown on the event loop closes the decision client built during a request."""
    monkeypatch.setattr(VectorStoreTools, "warm_up", lambda self: None)
    client = JevDecisionClient(api_key="test-key")
    monkeypatch.setattr("src.clients.decision._decision_client", client)

    async with lifespan(app):
        pass

    assert client._client.is_closed


async def test_api_lifespan_02_cleanup_failure_is_structured(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failing closer reports its event, component, duration and traceback."""
    monkeypatch.setattr(VectorStoreTools, "warm_up", lambda self: None)

    class _BrokenClient:
        async def aclose(self) -> None:
            raise TimeoutError

    monkeypatch.setattr("src.clients.decision._decision_client", _BrokenClient())

    with caplog.at_level(logging.WARNING):
        async with lifespan(app):
            pass

    failures = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "shutdown_cleanup_failed"
    ]
    assert len(failures) == 1

    record = failures[0]
    assert record.component == "aclose_decision_client"
    assert record.phase == "shutdown"
    assert record.duration_ms >= 0
    assert record.exc_info is not None
    assert record.exc_info[0] is TimeoutError
