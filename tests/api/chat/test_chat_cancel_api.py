"""Stop requests cancel producers, finish subscribers, and preserve subsequent turns."""

import asyncio
import logging
from contextlib import suppress

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph

from src.agent import PASsistantAgent
from src.api.services import chat_service
from src.guardrails.rate_limit import InMemoryRateLimiter
from src.services.contracts import AgentStreamUpdate
from src.utils.state import AgentState
from tests.doubles import StoppableChatAgent
from tests.factories import make_chat_payload, make_upload
from tests.helpers import parse_sse_frames

THREAD = "stop-test-thread"
MESSAGE = "Apa syarat kelulusan?"


@pytest.fixture
def slow_agent(chat_backend):
    agent = StoppableChatAgent(THREAD)
    chat_backend.agent_factory = lambda thread_id: agent
    yield agent
    agent.release.set()


def _post(client, path="/chat/stream", *, files=None):
    return asyncio.create_task(
        client.post(
            path,
            data=make_chat_payload(MESSAGE, thread_id=THREAD),
            files=files,
        )
    )


async def _started(agent):
    await asyncio.wait_for(agent.started.wait(), timeout=5)
    return chat_service._active_runs[THREAD]


async def test_cancel_sse_stops_agent_and_emits_one_terminal_event(api_client, slow_agent, caplog):
    with caplog.at_level(logging.INFO):
        pending = _post(api_client)
        run = await _started(slow_agent)
        response = await api_client.post(f"/chat/{THREAD}/cancel", params={"run_id": run.run_id})
        stream = await asyncio.wait_for(pending, timeout=5)

    assert response.status_code == 200
    assert response.json() == {"thread_id": THREAD, "run_id": run.run_id, "status": "cancelled"}
    assert slow_agent.cleaned.is_set()
    assert slow_agent.cancel_count == 1
    assert run.task.cancelled()
    frames = parse_sse_frames(stream.text)
    assert frames[-1]["event"] == "run.cancelled"
    assert frames[-1]["data"]["response"] == "Jawaban awal."
    assert sum(frame["event"] == "run.cancelled" for frame in frames) == 1
    assert not any(frame["event"] in {"run.completed", "run.failed"} for frame in frames)
    assert [f["payload"]["sequence"] for f in frames] == list(range(1, len(frames) + 1))
    summaries = [r for r in caplog.records if getattr(r, "event", None) == "run_summary"]
    assert len(summaries) == 1
    assert summaries[0].outcome == "cancelled"


@pytest.mark.parametrize("with_files", [False, True])
async def test_cancel_plain_chat_returns_409_and_next_turn_succeeds(
    api_client, slow_agent, sample_pdf_bytes, with_files
):
    files = (
        [make_upload("context.pdf", sample_pdf_bytes, "application/pdf")] if with_files else None
    )
    pending = _post(api_client, "/chat", files=files)
    run = await _started(slow_agent)

    stopped = await api_client.post(f"/chat/{THREAD}/cancel")
    original = await asyncio.wait_for(pending, timeout=5)

    assert stopped.status_code == 200
    assert stopped.json()["status"] == "cancelled"
    assert original.status_code == 409
    assert "cancelled" in original.json()["detail"]
    assert slow_agent.cleaned.is_set()
    assert slow_agent.cancel_count == 1
    next_turn = await api_client.post("/chat", data=make_chat_payload(MESSAGE, thread_id=THREAD))
    assert next_turn.status_code == 200
    assert next_turn.json()["response"] == "Jawaban selesai."
    assert chat_service._active_runs[THREAD].run_id != run.run_id


async def test_repeated_concurrent_stop_requests_are_idempotent(api_client, slow_agent):
    pending = _post(api_client)
    run = await _started(slow_agent)
    responses = await asyncio.gather(
        *[
            api_client.post(f"/chat/{THREAD}/cancel", params={"run_id": run.run_id})
            for _ in range(3)
        ]
    )
    await asyncio.wait_for(pending, timeout=5)

    assert all(response.status_code == 200 for response in responses)
    assert all(response.json()["status"] == "cancelled" for response in responses)
    assert slow_agent.cancel_count == 1
    assert sum(event.event_type == "run.cancelled" for event in run.events) == 1


async def test_stale_run_id_cannot_stop_the_next_turn(api_client, slow_agent):
    pending = _post(api_client)
    first = await _started(slow_agent)
    await api_client.post(f"/chat/{THREAD}/cancel", params={"run_id": first.run_id})
    await asyncio.wait_for(pending, timeout=5)
    slow_agent.block_next = True
    slow_agent.started.clear()
    pending = _post(api_client)
    second = await _started(slow_agent)

    stale = await api_client.post(f"/chat/{THREAD}/cancel", params={"run_id": first.run_id})

    assert stale.status_code == 409
    assert not second.done.is_set()
    assert second.cancel_requested is False
    assert slow_agent.cancel_count == 1
    stopped = await api_client.post(f"/chat/{THREAD}/cancel", params={"run_id": second.run_id})
    await asyncio.wait_for(pending, timeout=5)
    assert stopped.json()["status"] == "cancelled"
    assert slow_agent.cancel_count == 2


async def test_unknown_thread_does_not_create_an_agent(api_client, chat_backend):
    response = await api_client.post("/chat/unknown-thread/cancel")
    assert response.status_code == 404
    assert chat_backend.agents == []


@pytest.mark.parametrize("failed", [False, True])
async def test_stop_finished_run_returns_its_actual_status(api_client, slow_agent, failed):
    slow_agent.release.set()
    if failed:
        slow_agent.final_state.error = "test failure"
    await api_client.post("/chat", data=make_chat_payload(MESSAGE, thread_id=THREAD))
    run = chat_service._active_runs[THREAD]
    events_before = list(run.events)

    response = await api_client.post(f"/chat/{THREAD}/cancel", params={"run_id": run.run_id})

    assert response.status_code == 200
    assert response.json()["status"] == ("failed" if failed else "completed")
    assert run.events == events_before
    assert slow_agent.cancel_count == 0


async def test_stop_is_available_when_chat_rate_limit_is_exhausted(app, api_client, slow_agent):
    app.state.rate_limiter = InMemoryRateLimiter(limit=1)
    pending = _post(api_client)
    await _started(slow_agent)

    response = await api_client.post(f"/chat/{THREAD}/cancel")
    await asyncio.wait_for(pending, timeout=5)

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"


async def test_cancel_before_producer_starts_still_finishes(api_client, slow_agent):
    run = await chat_service._start_or_resume_run(
        message=MESSAGE,
        files=None,
        thread_id=THREAD,
        last_event_id=None,
    )
    assert not run.started.is_set()

    response = await asyncio.wait_for(chat_service.cancel_run(THREAD, run.run_id), timeout=5)

    assert response.status == "cancelled"
    assert run.done.is_set()
    assert [event.event_type for event in run.events] == ["run.started", "run.cancelled"]
    assert slow_agent.stream_calls == []


async def test_cancel_finishes_all_subscribers_and_replay_does_not_restart(api_client, slow_agent):
    first = chat_service.stream_chat_message(MESSAGE, thread_id=THREAD)
    second = chat_service.stream_chat_message(MESSAGE, thread_id=THREAD)
    started = await anext(first)
    assert (await anext(second)).run_id == started.run_id
    await _started(slow_agent)

    await api_client.post(f"/chat/{THREAD}/cancel", params={"run_id": started.run_id})
    tails = await asyncio.gather(_collect(first), _collect(second))

    assert all(tail[-1].event_type == "run.cancelled" for tail in tails)
    replay = await api_client.post(
        "/chat/stream",
        data=make_chat_payload(MESSAGE, thread_id=THREAD),
        headers={"Last-Event-ID": started.event_id},
    )
    assert parse_sse_frames(replay.text)[-1]["event"] == "run.cancelled"
    assert len(slow_agent.stream_calls) == 1


async def _collect(events):
    return [event async for event in events]


async def test_stop_propagates_into_the_real_langgraph_workflow(api_client, chat_backend):
    started, cleaned, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def response_node(state):
        started.set()
        try:
            await release.wait()
        finally:
            cleaned.set()
        return {"draft_response": "Completed", "messages": [AIMessage(content="Completed")]}

    graph = StateGraph(AgentState)
    graph.add_node("generate_response", response_node)
    graph.set_entry_point("generate_response")
    graph.add_edge("generate_response", END)
    chat_backend.agent_factory = lambda thread_id: PASsistantAgent(
        session_id=thread_id,
        app_factory=lambda checkpointer: graph.compile(checkpointer=checkpointer),
    )
    pending = _post(api_client)
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        response = await api_client.post(f"/chat/{THREAD}/cancel")
        original = await asyncio.wait_for(pending, timeout=5)
        assert response.json()["status"] == "cancelled"
        assert cleaned.is_set()
        assert parse_sse_frames(original.text)[-1]["event"] == "run.cancelled"
    finally:
        release.set()


async def test_stop_only_targets_the_requested_thread(api_client, chat_backend):
    agents = {key: StoppableChatAgent(key) for key in (THREAD, "other-thread")}
    chat_backend.agent_factory = lambda thread_id: agents[thread_id]
    first = _post(api_client)
    second = asyncio.create_task(
        api_client.post(
            "/chat/stream",
            data=make_chat_payload(MESSAGE, thread_id="other-thread"),
        )
    )
    try:
        await asyncio.wait_for(asyncio.gather(*[a.started.wait() for a in agents.values()]), 5)
        other_run = chat_service._active_runs["other-thread"]
        mismatch = await api_client.post(
            f"/chat/{THREAD}/cancel", params={"run_id": other_run.run_id}
        )
        assert mismatch.status_code == 409
        stopped = await api_client.post(f"/chat/{THREAD}/cancel")
        assert stopped.json()["status"] == "cancelled"
        assert not other_run.done.is_set()
        assert agents["other-thread"].cancel_count == 0
    finally:
        for agent in agents.values():
            agent.release.set()
        await asyncio.wait_for(asyncio.gather(first, second), 5)


async def test_late_results_from_an_agent_that_suppresses_cancellation_are_discarded(
    api_client, slow_agent, monkeypatch
):
    async def ignore_cancellation(message, files=None):
        slow_agent.started.set()
        with suppress(asyncio.CancelledError):
            await slow_agent.release.wait()

        yield AgentStreamUpdate(kind="final", state=AgentState(draft_response="Late answer"))

    monkeypatch.setattr(slow_agent, "stream_chat", ignore_cancellation)
    pending = _post(api_client)
    await _started(slow_agent)

    stopped = await api_client.post(f"/chat/{THREAD}/cancel")
    stream = await asyncio.wait_for(pending, 5)

    assert stopped.json()["status"] == "cancelled"
    assert "Late answer" not in stream.text
    assert [f["event"] for f in parse_sse_frames(stream.text)] == ["run.started", "run.cancelled"]
