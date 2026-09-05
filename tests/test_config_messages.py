from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from langgraph_agents.config import (
    AppConfig,
    EssayConfig,
    ModelConfig,
    RuntimeConfig,
    SearchConfig,
)
from langgraph_agents.errors import ConfigurationError
from langgraph_agents.messages import Message, ToolCall, merge_messages


def test_defaults_are_immutable_and_current() -> None:
    config = AppConfig()
    assert config.model.name == "gpt-4o-mini"
    assert config.essay.max_revisions == 3
    with pytest.raises(FrozenInstanceError):
        config.model.name = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: ModelConfig(name=""), "name"),
        (lambda: ModelConfig(temperature=-0.1), "temperature"),
        (lambda: ModelConfig(temperature=2.1), "temperature"),
        (lambda: SearchConfig(max_results=0), "max_results"),
        (lambda: SearchConfig(max_queries=0), "max_queries"),
        (lambda: EssayConfig(max_revisions=0), "max_revisions"),
        (lambda: RuntimeConfig(thread_id=""), "thread_id"),
        (lambda: RuntimeConfig(checkpoint_database=""), "checkpoint_database"),
    ],
)
def test_config_validation(factory: object, message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        factory()  # type: ignore[operator]


def test_toml_load_and_checkpoint_resolution(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[model]
name = "custom"
[search]
max_results = 4
[essay]
max_revisions = 2
[runtime]
thread_id = "thread"
checkpoint_database = "state/checkpoints.sqlite"
""",
        encoding="utf-8",
    )
    config = AppConfig.from_toml(path)
    assert config.model.name == "custom"
    assert config.search.max_results == 4
    assert config.essay.max_revisions == 2
    assert config.runtime.thread_id == "thread"
    assert config.runtime.checkpoint_database == str(
        tmp_path / "state/checkpoints.sqlite"
    )


def test_memory_checkpoint_is_not_resolved(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[runtime]\ncheckpoint_database = ":memory:"\n', encoding="utf-8")
    assert AppConfig.from_toml(path).runtime.checkpoint_database == ":memory:"


@pytest.mark.parametrize(
    "content",
    [
        "bad = [",
        "[unknown]\nvalue = 1",
        "model = 2",
        "[model]\nunknown = true",
    ],
)
def test_toml_rejects_invalid_content(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigurationError):
        AppConfig.from_toml(path)


def test_toml_missing_file_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="could not load"):
        AppConfig.from_toml(tmp_path / "missing.toml")


def test_required_credentials() -> None:
    AppConfig().require_credentials(
        {"OPENAI_API_KEY": "openai", "TAVILY_API_KEY": "tavily"}
    )
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY, TAVILY_API_KEY"):
        AppConfig().require_credentials({})


def test_merge_messages_assigns_replaces_and_does_not_mutate() -> None:
    original = Message(role="user", content="old", identifier="same")
    left = [original]
    replacement = Message(role="user", content="new", identifier="same")
    appended = Message(role="assistant", content="answer")
    identifiers = iter(["generated"])
    result = merge_messages(
        left,
        [replacement, appended],
        identifier_factory=lambda: next(identifiers),
    )
    assert result == [
        replacement,
        Message(role="assistant", content="answer", identifier="generated"),
    ]
    assert left == [original]


def test_tool_call_is_immutable() -> None:
    call = ToolCall("id", "search", {"query": "python"})
    assert call.arguments["query"] == "python"
    with pytest.raises(FrozenInstanceError):
        call.name = "changed"  # type: ignore[misc]
