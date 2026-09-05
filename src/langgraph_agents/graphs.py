"""Lazy LangGraph assembly and safe checkpointer lifetime management."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from importlib import import_module
from typing import Any

from langgraph_agents.errors import GraphError, IntegrationUnavailableError
from langgraph_agents.essay import EssayState, EssayWriter
from langgraph_agents.research import ResearchAgent, ResearchState


def _optional_module(name: str) -> Any:
    try:
        return import_module(name)
    except ImportError as exc:
        raise IntegrationUnavailableError(
            f"optional integration {name!r} is unavailable; install with "
            "`python -m pip install -e '.[agents]'`"
        ) from exc


def build_research_graph(
    agent: ResearchAgent,
    *,
    checkpointer: object | None = None,
    interrupt_before_tools: bool = False,
) -> Any:
    """Compile the research loop without importing LangGraph at module load time."""
    module = _optional_module("langgraph.graph")
    try:
        builder = module.StateGraph(ResearchState)
        builder.add_node("model", agent.call_model)
        builder.add_node("tools", agent.take_actions)
        builder.add_conditional_edges(
            "model",
            agent.has_tool_calls,
            {True: "tools", False: module.END},
        )
        builder.add_edge("tools", "model")
        builder.set_entry_point("model")
        options: dict[str, object] = {}
        if checkpointer is not None:
            options["checkpointer"] = checkpointer
        if interrupt_before_tools:
            options["interrupt_before"] = ["tools"]
        return builder.compile(**options)
    except (AttributeError, TypeError, ValueError) as exc:
        raise GraphError(f"could not build research graph: {exc}") from exc


def build_essay_graph(
    writer: EssayWriter,
    *,
    checkpointer: object | None = None,
) -> Any:
    """Compile the plan/research/revise graph."""
    module = _optional_module("langgraph.graph")
    try:
        builder = module.StateGraph(EssayState)
        builder.add_node("planner", writer.plan)
        builder.add_node("research_plan", writer.research_plan)
        builder.add_node("generate", writer.generate)
        builder.add_node("reflect", writer.reflect)
        builder.add_node("research_critique", writer.research_critique)
        builder.set_entry_point("planner")
        builder.add_edge("planner", "research_plan")
        builder.add_edge("research_plan", "generate")
        builder.add_conditional_edges(
            "generate",
            writer.should_continue,
            {"end": module.END, "reflect": "reflect"},
        )
        builder.add_edge("reflect", "research_critique")
        builder.add_edge("research_critique", "generate")
        options = {"checkpointer": checkpointer} if checkpointer is not None else {}
        return builder.compile(**options)
    except (AttributeError, TypeError, ValueError) as exc:
        raise GraphError(f"could not build essay graph: {exc}") from exc


@contextmanager
def sqlite_checkpointer(connection: str) -> Iterator[object]:
    """Keep the supported SqliteSaver context open for the graph lifetime."""
    sqlite_module = _optional_module("langgraph.checkpoint.sqlite")
    serde_module = _optional_module("langgraph.checkpoint.serde.jsonplus")
    try:
        serializer = serde_module.JsonPlusSerializer(
            allowed_msgpack_modules=[
                ("langgraph_agents.messages", "Message"),
                ("langgraph_agents.messages", "ToolCall"),
            ]
        )
        with closing(sqlite3.connect(connection, check_same_thread=False)) as database:
            saver = sqlite_module.SqliteSaver(database, serde=serializer)
            yield saver
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
        raise GraphError(f"could not open checkpoint database: {exc}") from exc
