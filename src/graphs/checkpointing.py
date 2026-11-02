"""Checkpoint serialization for PASsistant's application state models."""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from src.utils.state import (
    AgentState,
    Citation,
    DocumentType,
    DocumentUpload,
    ProcessingStatus,
    StudentRecord,
)


def create_checkpointer() -> InMemorySaver:
    """Allow the state models explicitly, including when strict msgpack is enabled."""
    return InMemorySaver(
        serde=JsonPlusSerializer(
            allowed_msgpack_modules=(
                AgentState,
                Citation,
                DocumentType,
                DocumentUpload,
                ProcessingStatus,
                StudentRecord,
            )
        )
    )
