"""Thin command-line composition root."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path

from langgraph_agents.config import AppConfig
from langgraph_agents.errors import AgentLabError
from langgraph_agents.essay import EssayWriter
from langgraph_agents.graphs import (
    build_essay_graph,
    build_research_graph,
    sqlite_checkpointer,
)
from langgraph_agents.integrations import (
    OpenAIEssayModel,
    OpenAIResearchModel,
    TavilySearchClient,
    TavilySearchTool,
)
from langgraph_agents.messages import ToolCall
from langgraph_agents.research import ResearchAgent
from langgraph_agents.runners import run_essay, run_research, run_research_with_review


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LangGraph research and essay agents")
    parser.add_argument("--config", type=Path, help="optional TOML configuration")
    commands = parser.add_subparsers(dest="command", required=True)
    research = commands.add_parser("research", help="answer a researched question")
    research.add_argument("question", nargs="+", help="question to research")
    research.add_argument(
        "--review-tools",
        action="store_true",
        help="approve or skip each proposed search before it runs",
    )
    essay = commands.add_parser("essay", help="write and revise a five-paragraph essay")
    essay.add_argument(
        "topic",
        nargs="*",
        help="essay topic; when omitted, read it from standard input",
    )
    return parser


def load_config(path: Path | None) -> AppConfig:
    return AppConfig.from_toml(path) if path is not None else AppConfig()


def load_local_environment() -> None:
    """Load `.env` only when the optional runtime dependency is installed."""
    try:
        module = import_module("dotenv")
    except ImportError:
        return
    module.load_dotenv()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    load_local_environment()
    try:
        config = load_config(arguments.config)
        config.require_credentials(os.environ)
        api_key = os.environ["TAVILY_API_KEY"]
        search = TavilySearchClient(api_key)
        with sqlite_checkpointer(config.runtime.checkpoint_database) as checkpointer:
            if arguments.command == "research":
                tool = TavilySearchTool(
                    search,
                    max_results=config.search.max_results,
                )
                model = OpenAIResearchModel(config.model, [tool.name])
                graph = build_research_graph(
                    ResearchAgent(model, [tool]),
                    checkpointer=checkpointer,
                    interrupt_before_tools=bool(arguments.review_tools),
                )
                question = " ".join(arguments.question)
                if arguments.review_tools:
                    answer = run_research_with_review(
                        graph,
                        question,
                        config.runtime.thread_id,
                        approve_tool_calls,
                    )
                else:
                    answer = run_research(
                        graph,
                        question,
                        config.runtime.thread_id,
                    )
                print(answer)
                return 0

            topic = " ".join(arguments.topic).strip()
            if not topic:
                topic = sys.stdin.read().strip()
            writer = EssayWriter(
                OpenAIEssayModel(config.model),
                search,
                config.search,
                config.essay,
            )
            graph = build_essay_graph(writer, checkpointer=checkpointer)
            print(run_essay(graph, writer.initial_state(topic), config.runtime.thread_id))
            return 0
    except AgentLabError as exc:
        parser.exit(2, f"error: {exc}\n")


def approve_tool_calls(calls: tuple[ToolCall, ...]) -> bool:
    """Present proposed tools without exposing secrets and request confirmation."""
    print("Proposed tool calls:")
    for call in calls:
        print(f"- {call.name}: {dict(call.arguments)}")
    response = input("Run these tools? [y/N] ").strip().lower()
    return response in {"y", "yes"}


if __name__ == "__main__":
    raise SystemExit(main())
