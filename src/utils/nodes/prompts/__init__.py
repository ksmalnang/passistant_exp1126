"""Prompt templates for node implementations."""

from src.utils.nodes.prompts.response import (
    GROUNDING_EDITOR_PROMPT,
    RESPONSE_SYSTEM_PROMPT,
)
from src.utils.nodes.prompts.retrieval import QUERY_REWRITE_PROMPT
from src.utils.nodes.prompts.router import ROUTER_INTENT_PROMPT
from src.utils.nodes.prompts.student_record import STUDENT_RECORD_PROMPT

__all__ = [
    "GROUNDING_EDITOR_PROMPT",
    "QUERY_REWRITE_PROMPT",
    "RESPONSE_SYSTEM_PROMPT",
    "ROUTER_INTENT_PROMPT",
    "STUDENT_RECORD_PROMPT",
]
