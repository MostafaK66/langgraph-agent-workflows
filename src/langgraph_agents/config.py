"""Immutable and validated application configuration."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langgraph_agents.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str = "gpt-4o-mini"
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ConfigurationError("model.name cannot be empty")
        if not 0 <= self.temperature <= 2:
            raise ConfigurationError("model.temperature must be between 0 and 2")


@dataclass(frozen=True, slots=True)
class SearchConfig:
    max_results: int = 2
    max_queries: int = 3

    def __post_init__(self) -> None:
        if self.max_results <= 0:
            raise ConfigurationError("search.max_results must be positive")
        if self.max_queries <= 0:
            raise ConfigurationError("search.max_queries must be positive")


@dataclass(frozen=True, slots=True)
class EssayConfig:
    max_revisions: int = 3

    def __post_init__(self) -> None:
        if self.max_revisions <= 0:
            raise ConfigurationError("essay.max_revisions must be positive")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    thread_id: str = "default"
    checkpoint_database: str = ":memory:"

    def __post_init__(self) -> None:
        if not self.thread_id.strip():
            raise ConfigurationError("runtime.thread_id cannot be empty")
        if not self.checkpoint_database.strip():
            raise ConfigurationError("runtime.checkpoint_database cannot be empty")


@dataclass(frozen=True, slots=True)
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    essay: EssayConfig = field(default_factory=EssayConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_toml(cls, path: Path) -> AppConfig:
        try:
            with path.open("rb") as stream:
                data = tomllib.load(stream)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigurationError(
                f"could not load configuration {path}: {exc}"
            ) from exc
        unknown = set(data) - {"model", "search", "essay", "runtime"}
        if unknown:
            raise ConfigurationError(f"unknown configuration sections: {sorted(unknown)}")
        try:
            runtime = _section(data, "runtime")
            checkpoint = runtime.get("checkpoint_database")
            if isinstance(checkpoint, str) and checkpoint != ":memory:":
                checkpoint_path = Path(checkpoint)
                if not checkpoint_path.is_absolute():
                    runtime["checkpoint_database"] = str(
                        path.resolve().parent / checkpoint_path
                    )
            return cls(
                model=ModelConfig(**_section(data, "model")),
                search=SearchConfig(**_section(data, "search")),
                essay=EssayConfig(**_section(data, "essay")),
                runtime=RuntimeConfig(**runtime),
            )
        except TypeError as exc:
            raise ConfigurationError(f"invalid configuration in {path}: {exc}") from exc

    def require_credentials(self, environment: Mapping[str, str]) -> None:
        missing = [
            name
            for name in ("OPENAI_API_KEY", "TAVILY_API_KEY")
            if not environment.get(name, "").strip()
        ]
        if missing:
            raise ConfigurationError(
                "missing required environment variable(s): " + ", ".join(missing)
            )


def _section(data: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ConfigurationError(f"configuration section [{name}] must be a table")
    return dict(value)
