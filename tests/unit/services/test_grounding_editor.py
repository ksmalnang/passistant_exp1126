"""Unit tests for the grounding editor."""

from __future__ import annotations

from types import SimpleNamespace

from src.services.response_generation import GroundedAnswerEditor


class _LLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.content)


def test_editor_returns_the_llm_edit() -> None:
    llm = _LLM("Jawaban yang didukung [1].")
    editor = GroundedAnswerEditor(llm_provider=lambda: llm)

    assert editor.edit("Draft panjang.", ["excerpt"]) == "Jawaban yang didukung [1]."
    assert "Draft panjang." in llm.prompts[0]
    assert "excerpt" in llm.prompts[0]


def test_editor_returns_the_draft_when_the_edit_is_empty() -> None:
    editor = GroundedAnswerEditor(llm_provider=lambda: _LLM("  "))

    assert editor.edit("Draft panjang.", ["excerpt"]) == "Draft panjang."


def test_editor_returns_the_draft_when_the_model_fails() -> None:
    def failing_provider():
        raise RuntimeError("provider down")

    editor = GroundedAnswerEditor(llm_provider=failing_provider)

    assert editor.edit("Draft panjang.", ["excerpt"]) == "Draft panjang."


def test_editor_without_excerpts_skips_the_model() -> None:
    llm = _LLM("not used")
    editor = GroundedAnswerEditor(llm_provider=lambda: llm)

    assert editor.edit("Draft panjang.", []) == "Draft panjang."
    assert llm.prompts == []
