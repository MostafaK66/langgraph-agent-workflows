"""Tool-using research-agent node logic."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Protocol, TypedDict

from langgraph_agents.errors import ModelError
from langgraph_agents.messages import Message, merge_messages
from langgraph_agents.prompts import RESEARCH_SYSTEM_PROMPT


class ResearchState(TypedDict):
    messages: Annotated[list[Message], merge_messages]


class ResearchModel(Protocol):
    def respond(self, messages: Sequence[Message]) -> Message:
        """Return the next assistant message."""


class Tool(Protocol):
    @property
    def name(self) -> str:
        """Stable tool name exposed to the model."""

    def invoke(self, arguments: Mapping[str, object]) -> str:
        """Execute a validated tool request."""


class ResearchAgent:
    """Nodes for a model → tool → model research loop."""

    def __init__(
        self,
        model: ResearchModel,
        tools: Sequence[Tool],
        *,
        system_prompt: str = RESEARCH_SYSTEM_PROMPT,
    ) -> None:
        if not system_prompt.strip():
            raise ValueError("system_prompt cannot be empty")
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique")
        self._model = model
        self._tools = {tool.name: tool for tool in tools}
        self._system_prompt = system_prompt.strip()

    def call_model(self, state: ResearchState) -> dict[str, list[Message]]:
        messages = state.get("messages", [])
        if not messages:
            raise ModelError("research state must contain at least one message")
        request = [Message(role="system", content=self._system_prompt), *messages]
        try:
            response = self._model.respond(request)
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(f"research model failed: {exc}") from exc
        if response.role != "assistant":
            raise ModelError("research model must return an assistant message")
        return {"messages": [response]}

    @staticmethod
    def has_tool_calls(state: ResearchState) -> bool:
        messages = state.get("messages", [])
        return bool(messages and messages[-1].tool_calls)

    def take_actions(self, state: ResearchState) -> dict[str, list[Message]]:
        messages = state.get("messages", [])
        if not messages:
            raise ModelError("research state must contain at least one message")
        results: list[Message] = []
        for call in messages[-1].tool_calls:
            tool = self._tools.get(call.name)
            if tool is None:
                content = (
                    f"Unknown tool {call.name!r}; choose one of {sorted(self._tools)}."
                )
            else:
                try:
                    content = tool.invoke(call.arguments)
                except Exception as exc:
                    content = f"Tool {call.name!r} failed: {exc}"
            results.append(
                Message(
                    role="tool",
                    content=content,
                    tool_call_id=call.identifier,
                    name=call.name,
                )
            )
        return {"messages": results}
