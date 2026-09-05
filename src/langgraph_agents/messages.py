"""Immutable research-agent messages and deterministic state reduction."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, cast
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class ToolCall:
    identifier: str
    name: str
    arguments: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    identifier: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None


def merge_messages(
    left: Sequence[Message],
    right: Sequence[Message],
    *,
    identifier_factory: Callable[[], str] = lambda: str(uuid4()),
) -> list[Message]:
    """Append new messages and replace messages sharing an identifier."""
    merged = list(left)
    positions = {
        message.identifier: index
        for index, message in enumerate(merged)
        if message.identifier is not None
    }
    for message in right:
        identified = (
            message
            if message.identifier is not None
            else replace(message, identifier=identifier_factory())
        )
        identifier = cast(str, identified.identifier)
        position = positions.get(identifier)
        if position is None:
            positions[identifier] = len(merged)
            merged.append(identified)
        else:
            merged[position] = identified
    return merged
