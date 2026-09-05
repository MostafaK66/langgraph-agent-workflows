"""Pure essay-writer graph node logic with injected model and search boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, Protocol, TypedDict

from langgraph_agents.config import EssayConfig, SearchConfig
from langgraph_agents.errors import ModelError, SearchError, StateError
from langgraph_agents.prompts import (
    PLAN_PROMPT,
    REFLECTION_PROMPT,
    RESEARCH_CRITIQUE_PROMPT,
    RESEARCH_PLAN_PROMPT,
    WRITER_PROMPT,
)


class EssayState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: list[str]
    revision_number: int
    max_revisions: int


class EssayModel(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return plain text from the configured model."""

    def queries(self, system_prompt: str, user_prompt: str) -> Sequence[str]:
        """Return proposed search queries."""


class SearchClient(Protocol):
    def search(self, query: str, *, max_results: int) -> Sequence[str]:
        """Return plain-text evidence snippets."""


class EssayWriter:
    def __init__(
        self,
        model: EssayModel,
        search: SearchClient,
        search_config: SearchConfig,
        essay_config: EssayConfig,
    ) -> None:
        self._model = model
        self._search = search
        self._search_config = search_config
        self._essay_config = essay_config

    def initial_state(self, task: str) -> EssayState:
        cleaned = task.strip()
        if not cleaned:
            raise StateError("essay topic cannot be empty")
        return {
            "task": cleaned,
            "plan": "",
            "draft": "",
            "critique": "",
            "content": [],
            "revision_number": 0,
            "max_revisions": self._essay_config.max_revisions,
        }

    def plan(self, state: EssayState) -> dict[str, str]:
        return {"plan": self._complete(PLAN_PROMPT, _required(state, "task"))}

    def research_plan(self, state: EssayState) -> dict[str, list[str]]:
        prompt = RESEARCH_PLAN_PROMPT.format(max_queries=self._search_config.max_queries)
        return self._research(state, prompt, _required(state, "task"))

    def generate(self, state: EssayState) -> dict[str, str | int]:
        content = "\n\n".join(state.get("content", []))
        system = WRITER_PROMPT.format(content=content)
        user = f"Topic: {_required(state, 'task')}\n\nPlan: {_required(state, 'plan')}"
        critique = state.get("critique", "").strip()
        if critique:
            user += f"\n\nCritique to address: {critique}"
        return {
            "draft": self._complete(system, user),
            "revision_number": state.get("revision_number", 0) + 1,
        }

    def reflect(self, state: EssayState) -> dict[str, str]:
        return {"critique": self._complete(REFLECTION_PROMPT, _required(state, "draft"))}

    def research_critique(self, state: EssayState) -> dict[str, list[str]]:
        prompt = RESEARCH_CRITIQUE_PROMPT.format(
            max_queries=self._search_config.max_queries
        )
        return self._research(state, prompt, _required(state, "critique"))

    @staticmethod
    def should_continue(state: EssayState) -> Literal["end", "reflect"]:
        maximum = state.get("max_revisions", 0)
        current = state.get("revision_number", 0)
        if maximum <= 0:
            raise StateError("max_revisions must be positive")
        return "end" if current >= maximum else "reflect"

    def _complete(self, system: str, user: str) -> str:
        try:
            response = self._model.complete(system, user).strip()
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(f"essay model failed: {exc}") from exc
        if not response:
            raise ModelError("essay model returned empty text")
        return response

    def _research(
        self, state: EssayState, system_prompt: str, user_prompt: str
    ) -> dict[str, list[str]]:
        try:
            proposed = self._model.queries(system_prompt, user_prompt)
        except Exception as exc:
            raise ModelError(f"query generation failed: {exc}") from exc
        queries = tuple(query.strip() for query in proposed if query.strip())
        if not queries:
            raise ModelError("query generation returned no usable queries")
        content = list(state.get("content", []))
        for query in queries[: self._search_config.max_queries]:
            try:
                snippets = self._search.search(
                    query, max_results=self._search_config.max_results
                )
            except SearchError:
                raise
            except Exception as exc:
                raise SearchError(f"search failed for {query!r}: {exc}") from exc
            content.extend(snippet.strip() for snippet in snippets if snippet.strip())
        return {"content": content}


def _required(state: Mapping[str, object], name: str) -> str:
    value = state.get(name, "")
    if not isinstance(value, str) or not value.strip():
        raise StateError(f"essay state field {name!r} cannot be empty")
    return value.strip()
