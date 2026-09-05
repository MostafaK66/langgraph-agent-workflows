from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from langgraph_agents import cli


class NoOp:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


class Tool:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    @property
    def name(self) -> str:
        return "web_search"


@contextmanager
def checkpointer(connection: str) -> Any:
    assert connection == ":memory:"
    yield "saver"


def patch_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily")
    monkeypatch.setattr(cli, "load_local_environment", lambda: None)
    monkeypatch.setattr(cli, "TavilySearchClient", NoOp)
    monkeypatch.setattr(cli, "TavilySearchTool", Tool)
    monkeypatch.setattr(cli, "OpenAIResearchModel", NoOp)
    monkeypatch.setattr(cli, "ResearchAgent", NoOp)
    monkeypatch.setattr(cli, "OpenAIEssayModel", NoOp)
    monkeypatch.setattr(cli, "EssayWriter", NoOp)
    monkeypatch.setattr(cli, "sqlite_checkpointer", checkpointer)


def test_research_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    patch_runtime(monkeypatch)
    monkeypatch.setattr(cli, "build_research_graph", lambda *args, **kwargs: "graph")

    def run(graph: object, question: str, thread_id: str) -> str:
        assert graph == "graph"
        assert question == "what is this?"
        assert thread_id == "default"
        return "answer"

    monkeypatch.setattr(cli, "run_research", run)
    assert cli.main(["research", "what", "is", "this?"]) == 0
    assert capsys.readouterr().out == "answer\n"


def test_research_review_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    patch_runtime(monkeypatch)
    monkeypatch.setattr(cli, "build_research_graph", lambda *args, **kwargs: "graph")
    monkeypatch.setattr(cli, "run_research_with_review", lambda *args: "reviewed")
    assert cli.main(["research", "--review-tools", "question"]) == 0
    assert capsys.readouterr().out == "reviewed\n"


def test_essay_command_with_argument(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    patch_runtime(monkeypatch)

    class Writer:
        def __init__(self, *args: object) -> None:
            del args

        def initial_state(self, topic: str) -> object:
            assert topic == "essay topic"
            return "state"

    monkeypatch.setattr(cli, "EssayWriter", Writer)
    monkeypatch.setattr(cli, "build_essay_graph", lambda *args, **kwargs: "graph")
    monkeypatch.setattr(cli, "run_essay", lambda *args: "final essay")
    assert cli.main(["essay", "essay", "topic"]) == 0
    assert capsys.readouterr().out == "final essay\n"


def test_essay_command_reads_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    patch_runtime(monkeypatch)
    topics: list[str] = []

    class Writer:
        def __init__(self, *args: object) -> None:
            del args

        def initial_state(self, topic: str) -> object:
            topics.append(topic)
            return "state"

    monkeypatch.setattr(
        cli, "sys", SimpleNamespace(stdin=SimpleNamespace(read=lambda: "topic\n"))
    )
    monkeypatch.setattr(cli, "EssayWriter", Writer)
    monkeypatch.setattr(cli, "build_essay_graph", lambda *args, **kwargs: "graph")
    monkeypatch.setattr(cli, "run_essay", lambda *args: "essay")
    assert cli.main(["essay"]) == 0
    assert topics == ["topic"]
    assert capsys.readouterr().out == "essay\n"


def test_load_config(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[essay]\nmax_revisions = 2\n", encoding="utf-8")
    assert cli.load_config(path).essay.max_revisions == 2
    assert cli.load_config(None).essay.max_revisions == 3


def test_load_environment_present_and_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    module = SimpleNamespace(load_dotenv=lambda: calls.append("loaded"))
    monkeypatch.setattr(cli, "import_module", lambda name: module)
    cli.load_local_environment()
    assert calls == ["loaded"]

    def missing(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr(cli, "import_module", missing)
    cli.load_local_environment()


def test_cli_reports_domain_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(cli, "load_local_environment", lambda: None)
    with pytest.raises(SystemExit) as raised:
        cli.main(["research", "question"])
    assert raised.value.code == 2


@pytest.mark.parametrize(("response", "expected"), [("yes", True), ("n", False)])
def test_tool_approval(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    response: str,
    expected: bool,
) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt: response)
    call = cli.ToolCall("id", "search", {"query": "topic"})
    assert cli.approve_tool_calls((call,)) is expected
    assert "search" in capsys.readouterr().out
