from __future__ import annotations

from types import SimpleNamespace

import pytest

from langgraph_agents.config import EssayConfig, SearchConfig
from langgraph_agents.errors import GraphError, IntegrationUnavailableError
from langgraph_agents.essay import EssayWriter
from langgraph_agents.graphs import (
    build_essay_graph,
    build_research_graph,
    sqlite_checkpointer,
)
from langgraph_agents.messages import Message
from langgraph_agents.research import ResearchAgent


class Model:
    def respond(self, messages: object) -> Message:
        del messages
        return Message(role="assistant", content="answer")


class EssayModel:
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        del system_prompt, user_prompt
        return "text"

    def queries(self, system_prompt: str, user_prompt: str) -> list[str]:
        del system_prompt, user_prompt
        return ["query"]


class Search:
    def search(self, query: str, *, max_results: int) -> list[str]:
        del query, max_results
        return ["result"]


class Builder:
    latest: Builder | None = None
    fail = False

    def __init__(self, state: object) -> None:
        del state
        self.nodes: list[str] = []
        self.edges: list[tuple[object, object]] = []
        self.conditionals: list[object] = []
        self.entry: str | None = None
        self.options: dict[str, object] = {}
        Builder.latest = self

    def add_node(self, name: str, operation: object) -> None:
        del operation
        self.nodes.append(name)

    def add_edge(self, start: object, end: object) -> None:
        self.edges.append((start, end))

    def add_conditional_edges(self, *args: object) -> None:
        self.conditionals.append(args)

    def set_entry_point(self, name: str) -> None:
        self.entry = name

    def compile(self, **options: object) -> object:
        if self.fail:
            raise ValueError("compile failed")
        self.options = options
        return "compiled"


class Saver:
    def __init__(self, database: object, *, serde: object) -> None:
        assert database is not None
        assert serde is not None


class Serializer:
    modules: object = None

    def __init__(self, *, allowed_msgpack_modules: object) -> None:
        Serializer.modules = allowed_msgpack_modules


def fake_import(name: str) -> object:
    if name == "langgraph.graph":
        return SimpleNamespace(StateGraph=Builder, END="end")
    if name == "langgraph.checkpoint.sqlite":
        return SimpleNamespace(SqliteSaver=Saver)
    if name == "langgraph.checkpoint.serde.jsonplus":
        return SimpleNamespace(JsonPlusSerializer=Serializer)
    raise ImportError(name)


def research_agent() -> ResearchAgent:
    return ResearchAgent(Model(), [])  # type: ignore[arg-type]


def essay_writer() -> EssayWriter:
    return EssayWriter(
        EssayModel(),
        Search(),
        SearchConfig(),
        EssayConfig(),
    )


def test_build_research_graph_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("langgraph_agents.graphs.import_module", fake_import)
    assert (
        build_research_graph(
            research_agent(), checkpointer="saver", interrupt_before_tools=True
        )
        == "compiled"
    )
    assert Builder.latest is not None
    assert Builder.latest.nodes == ["model", "tools"]
    assert Builder.latest.options == {
        "checkpointer": "saver",
        "interrupt_before": ["tools"],
    }
    assert build_research_graph(research_agent()) == "compiled"
    assert Builder.latest.options == {}


def test_build_essay_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("langgraph_agents.graphs.import_module", fake_import)
    assert build_essay_graph(essay_writer(), checkpointer="saver") == "compiled"
    assert Builder.latest is not None
    assert len(Builder.latest.nodes) == 5
    assert Builder.latest.entry == "planner"
    assert Builder.latest.options == {"checkpointer": "saver"}
    assert build_essay_graph(essay_writer()) == "compiled"
    assert Builder.latest.options == {}


def test_graph_build_errors_are_domain_specific(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("langgraph_agents.graphs.import_module", fake_import)
    Builder.fail = True
    with pytest.raises(GraphError, match="research"):
        build_research_graph(research_agent())
    with pytest.raises(GraphError, match="essay"):
        build_essay_graph(essay_writer())
    Builder.fail = False


def test_sqlite_checkpointer_lifetime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("langgraph_agents.graphs.import_module", fake_import)
    with sqlite_checkpointer(":memory:") as saver:
        assert isinstance(saver, Saver)
    assert Serializer.modules == [
        ("langgraph_agents.messages", "Message"),
        ("langgraph_agents.messages", "ToolCall"),
    ]


def test_checkpointer_normalizes_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class Broken:
        def __init__(self, database: object, *, serde: object) -> None:
            del database, serde
            raise ValueError("bad database")

    def broken_import(name: str) -> object:
        if name == "langgraph.checkpoint.sqlite":
            return SimpleNamespace(SqliteSaver=Broken)
        return SimpleNamespace(JsonPlusSerializer=Serializer)

    monkeypatch.setattr("langgraph_agents.graphs.import_module", broken_import)
    with (
        pytest.raises(GraphError, match="checkpoint"),
        sqlite_checkpointer(":memory:"),
    ):
        pass


def test_graph_dependency_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr("langgraph_agents.graphs.import_module", missing)
    with pytest.raises(IntegrationUnavailableError, match=r"\.\[agents\]"):
        build_research_graph(research_agent())
