"""Checkpoint round trips preserve state models without permissive deserialization."""

import logging
import uuid

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

from src.agent import PASsistantAgent
from src.graphs.checkpointing import create_checkpointer
from src.graphs.workflow import compile_app
from src.utils.state import (
    AgentState,
    Citation,
    DocumentType,
    DocumentUpload,
    ProcessingStatus,
    StudentRecord,
)

SERDE_LOGGER = "langgraph.checkpoint.serde.jsonplus"


def _state() -> AgentState:
    document = DocumentUpload(
        document_id="doc-1",
        filename="handbook.pdf",
        file_path="data/raw/doc-1.pdf",
        mime_type="application/pdf",
        file_size=10,
        document_type=DocumentType.POLICY,
        processing_status=ProcessingStatus.COMPLETED,
    )
    return AgentState(
        messages=[HumanMessage(content="Syarat cuti akademik?")],
        citations=[Citation(id=1, document_id="doc-1", page=3, chunk_id="doc-1:chunk:0")],
        pending_documents=[document],
        processed_documents=[document],
        current_document=document,
        student_records={"test-student": StudentRecord(student_id="test-student")},
    )


def test_default_serializer_reads_existing_state_checkpoints_without_warnings(caplog):
    # LangGraph checkpoints hold channel values, not the enclosing Pydantic model.
    values = dict(_state())
    # The writer's default configuration matches checkpoints created before the fix.
    checkpoint = JsonPlusSerializer().dumps_typed(values)
    with caplog.at_level(logging.WARNING, logger=SERDE_LOGGER):
        restored = create_checkpointer().serde.loads_typed(checkpoint)

    assert restored == values
    assert isinstance(restored["messages"][0], HumanMessage)
    assert isinstance(restored["citations"][0], Citation)
    assert isinstance(restored["current_document"], DocumentUpload)
    assert restored["current_document"].document_type is DocumentType.POLICY
    assert restored["current_document"].processing_status is ProcessingStatus.COMPLETED
    assert isinstance(restored["student_records"]["test-student"], StudentRecord)
    assert not [record for record in caplog.records if record.name == SERDE_LOGGER]


@pytest.mark.parametrize("dedicated", [False, True], ids=["shared-app", "app-factory"])
def test_default_agent_checkpoints_keep_citations_across_turns_without_warnings(caplog, dedicated):
    agent = PASsistantAgent(app_factory=compile_app) if dedicated else PASsistantAgent()
    config = {"configurable": {"thread_id": f"checkpoint-test-{uuid.uuid4().hex}"}}
    state = _state()
    values = {
        "messages": state.messages,
        "citations": state.citations,
        "processed_documents": state.processed_documents,
        "current_document": state.current_document,
        "student_records": state.student_records,
    }

    with caplog.at_level(logging.WARNING, logger=SERDE_LOGGER):
        agent.app.update_state(config, values)
        agent.app.update_state(config, {"messages": [HumanMessage(content="Jelaskan lagi")]})
        restored = agent.app.get_state(config).values

    assert restored["citations"] == state.citations
    assert isinstance(restored["citations"][0], Citation)
    assert isinstance(restored["processed_documents"][0], DocumentUpload)
    assert isinstance(restored["current_document"], DocumentUpload)
    assert isinstance(restored["student_records"]["test-student"], StudentRecord)
    assert [message.content for message in restored["messages"]] == [
        "Syarat cuti akademik?",
        "Jelaskan lagi",
    ]
    assert not [record for record in caplog.records if record.name == SERDE_LOGGER]


class UnregisteredCheckpointModel(BaseModel):
    value: str


def test_default_serializer_does_not_allow_arbitrary_application_types(caplog):
    serde = create_checkpointer().serde
    checkpoint = serde.dumps_typed(UnregisteredCheckpointModel(value="test"))

    with caplog.at_level(logging.WARNING, logger=SERDE_LOGGER):
        restored = serde.loads_typed(checkpoint)

    assert restored == {"value": "test"}
    assert "Blocked deserialization" in caplog.text
