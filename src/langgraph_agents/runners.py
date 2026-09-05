"""Application-level graph execution and output validation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Protocol

from langgraph_agents.errors import GraphError
from langgraph_agents.essay import EssayState
from langgraph_agents.messages import Message, ToolCall
from langgraph_agents.research import ResearchState


class InvokableGraph(Protocol):
    def invoke(
        self,
        state: ResearchState | EssayState,
        config: Mapping[str, object],
    ) -> object:
        """Execute a compiled graph."""


class ResumableResearchGraph(Protocol):
    def invoke(
        self,
        state: ResearchState | None,
        config: Mapping[str, object],
    ) -> object:
        """Start or resume an interrupted research graph."""

    def update_state(
        self,
        config: Mapping[str, object],
        values: Mapping[str, object],
        *,
        as_node: str,
    ) -> object:
        """Replace pending graph state before resuming."""


def graph_config(thread_id: str) -> dict[str, object]:
    if not thread_id.strip():
        raise GraphError("thread_id cannot be empty")
    return {"configurable": {"thread_id": thread_id.strip()}}


def run_research(graph: InvokableGraph, question: str, thread_id: str) -> str:
    cleaned = question.strip()
    if not cleaned:
        raise GraphError("research question cannot be empty")
    state: ResearchState = {"messages": [Message(role="user", content=cleaned)]}
    result = graph.invoke(state, graph_config(thread_id))
    if not isinstance(result, Mapping):
        raise GraphError("research graph returned malformed state")
    messages = result.get("messages")
    if not isinstance(messages, list) or not messages:
        raise GraphError("research graph returned no messages")
    final = messages[-1]
    if not isinstance(final, Message) or final.role != "assistant":
        raise GraphError("research graph did not finish with an assistant answer")
    if not final.content.strip():
        raise GraphError("research graph returned an empty answer")
    return final.content.strip()


def run_research_with_review(
    graph: ResumableResearchGraph,
    question: str,
    thread_id: str,
    approve: Callable[[tuple[ToolCall, ...]], bool],
    *,
    max_review_rounds: int = 20,
) -> str:
    """Pause before each tool round and let a caller approve or skip it."""
    cleaned = question.strip()
    if not cleaned:
        raise GraphError("research question cannot be empty")
    if max_review_rounds <= 0:
        raise GraphError("max_review_rounds must be positive")
    config = graph_config(thread_id)
    initial: ResearchState = {"messages": [Message(role="user", content=cleaned)]}
    result = graph.invoke(initial, config)
    for _ in range(max_review_rounds):
        messages = _research_messages(result)
        pending = messages[-1]
        if pending.role != "assistant":
            raise GraphError("research graph paused without an assistant message")
        if not pending.tool_calls:
            if not pending.content.strip():
                raise GraphError("research graph returned an empty answer")
            return pending.content.strip()
        if not approve(pending.tool_calls):
            skipped = replace(
                pending,
                content="Tool execution skipped by the user.",
                tool_calls=(),
            )
            graph.update_state(
                config,
                {"messages": [skipped]},
                as_node="model",
            )
        result = graph.invoke(None, config)
    raise GraphError("research graph exceeded the tool-review safety limit")


def run_essay(
    graph: InvokableGraph,
    initial_state: EssayState,
    thread_id: str,
) -> str:
    result = graph.invoke(initial_state, graph_config(thread_id))
    if not isinstance(result, Mapping):
        raise GraphError("essay graph returned malformed state")
    draft = result.get("draft")
    if not isinstance(draft, str) or not draft.strip():
        raise GraphError("essay graph returned no final draft")
    return draft.strip()


def _research_messages(result: object) -> list[Message]:
    if not isinstance(result, Mapping):
        raise GraphError("research graph returned malformed state")
    messages = result.get("messages")
    if not isinstance(messages, list) or not messages:
        raise GraphError("research graph returned no messages")
    if not all(isinstance(message, Message) for message in messages):
        raise GraphError("research graph returned malformed messages")
    return messages
