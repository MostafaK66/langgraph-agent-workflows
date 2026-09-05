from __future__ import annotations

from collections.abc import Mapping

import pytest

from langgraph_agents.errors import GraphError
from langgraph_agents.essay import EssayState
from langgraph_agents.messages import Message, ToolCall
from langgraph_agents.research import ResearchState
from langgraph_agents.runners import (
    graph_config,
    run_essay,
    run_research,
    run_research_with_review,
)


class Graph:
    def __init__(self, result: object) -> None:
        self.result = result
        self.state: ResearchState | EssayState | None = None
        self.config: Mapping[str, object] = {}

    def invoke(
        self,
        state: ResearchState | EssayState,
        config: Mapping[str, object],
    ) -> object:
        self.state = state
        self.config = config
        return self.result


def essay_state() -> EssayState:
    return {
        "task": "topic",
        "plan": "",
        "draft": "",
        "critique": "",
        "content": [],
        "revision_number": 0,
        "max_revisions": 1,
    }


def test_graph_config() -> None:
    assert graph_config(" id ") == {"configurable": {"thread_id": "id"}}
    with pytest.raises(GraphError, match="thread_id"):
        graph_config(" ")


def test_run_research_validates_input_and_output() -> None:
    graph = Graph({"messages": [Message(role="assistant", content=" answer ")]})
    assert run_research(graph, " question ", "thread") == "answer"
    assert graph.state == {"messages": [Message(role="user", content="question")]}


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (None, "malformed"),
        ({}, "no messages"),
        ({"messages": []}, "no messages"),
        ({"messages": ["wrong"]}, "assistant answer"),
        ({"messages": [Message(role="user", content="x")]}, "assistant answer"),
        ({"messages": [Message(role="assistant", content=" ")]}, "empty answer"),
    ],
)
def test_run_research_rejects_bad_results(result: object, message: str) -> None:
    with pytest.raises(GraphError, match=message):
        run_research(Graph(result), "question", "thread")


def test_run_research_rejects_empty_question() -> None:
    with pytest.raises(GraphError, match="question"):
        run_research(Graph({}), " ", "thread")


def test_run_essay_validates_output() -> None:
    initial = essay_state()
    assert run_essay(Graph({"draft": " final "}), initial, "thread") == "final"
    for result in (None, {}, {"draft": ""}, {"draft": 4}):
        with pytest.raises(GraphError):
            run_essay(Graph(result), initial, "thread")


class ResumableGraph:
    def __init__(self, results: list[object]) -> None:
        self.results = iter(results)
        self.updates: list[tuple[Mapping[str, object], str]] = []

    def invoke(
        self,
        state: ResearchState | None,
        config: Mapping[str, object],
    ) -> object:
        del state, config
        return next(self.results)

    def update_state(
        self,
        config: Mapping[str, object],
        values: Mapping[str, object],
        *,
        as_node: str,
    ) -> object:
        del config
        self.updates.append((values, as_node))
        return object()


def pending_result() -> dict[str, list[Message]]:
    return {
        "messages": [
            Message(
                role="assistant",
                content="",
                identifier="pending",
                tool_calls=(ToolCall("call", "search", {"query": "topic"}),),
            )
        ]
    }


def test_review_approves_and_resumes_tool() -> None:
    graph = ResumableGraph(
        [pending_result(), {"messages": [Message(role="assistant", content="done")]}]
    )
    seen: list[tuple[ToolCall, ...]] = []
    assert (
        run_research_with_review(
            graph,
            "question",
            "thread",
            lambda calls: not seen.append(calls),
        )
        == "done"
    )
    assert seen[0][0].name == "search"
    assert not graph.updates


def test_review_skips_and_replaces_pending_message() -> None:
    graph = ResumableGraph(
        [pending_result(), {"messages": [Message(role="assistant", content="skipped")]}]
    )
    assert run_research_with_review(graph, "question", "thread", lambda calls: False) == (
        "skipped"
    )
    values, node = graph.updates[0]
    messages = values["messages"]
    assert isinstance(messages, list)
    assert not messages[0].tool_calls
    assert node == "model"


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (None, "malformed state"),
        ({}, "no messages"),
        ({"messages": ["bad"]}, "malformed messages"),
        ({"messages": [Message(role="tool", content="x")]}, "paused without"),
        ({"messages": [Message(role="assistant", content="")]}, "empty answer"),
    ],
)
def test_review_validates_graph_state(result: object, message: str) -> None:
    with pytest.raises(GraphError, match=message):
        run_research_with_review(
            ResumableGraph([result]), "question", "thread", lambda calls: True
        )


def test_review_validates_input_and_safety_limit() -> None:
    with pytest.raises(GraphError, match="question"):
        run_research_with_review(ResumableGraph([]), " ", "thread", lambda calls: True)
    with pytest.raises(GraphError, match="positive"):
        run_research_with_review(
            ResumableGraph([]),
            "question",
            "thread",
            lambda calls: True,
            max_review_rounds=0,
        )
    with pytest.raises(GraphError, match="safety limit"):
        run_research_with_review(
            ResumableGraph([pending_result(), pending_result()]),
            "question",
            "thread",
            lambda calls: True,
            max_review_rounds=1,
        )
