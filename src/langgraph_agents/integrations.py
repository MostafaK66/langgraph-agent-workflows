"""Lazy adapters for OpenAI via LangChain and Tavily search."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any

from langgraph_agents.config import ModelConfig
from langgraph_agents.errors import (
    IntegrationUnavailableError,
    ModelError,
    SearchError,
)
from langgraph_agents.messages import Message, ToolCall


def _optional_module(name: str) -> Any:
    try:
        return import_module(name)
    except ImportError as exc:
        raise IntegrationUnavailableError(
            f"optional integration {name!r} is unavailable; install with "
            "`python -m pip install -e '.[agents]'`"
        ) from exc


class OpenAIResearchModel:
    """LangChain ChatOpenAI adapter for domain research messages."""

    def __init__(self, config: ModelConfig, tool_names: Sequence[str]) -> None:
        openai_module = _optional_module("langchain_openai")
        self._messages = _optional_module("langchain_core.messages")
        model = openai_module.ChatOpenAI(
            model=config.name,
            temperature=config.temperature,
        )
        definitions = [_tool_definition(name) for name in tool_names]
        self._model = model.bind_tools(definitions)

    def respond(self, messages: Sequence[Message]) -> Message:
        external = [self._to_external(message) for message in messages]
        response = self._model.invoke(external)
        calls: list[ToolCall] = []
        for raw in getattr(response, "tool_calls", ()):
            if not isinstance(raw, Mapping):
                raise ModelError("model returned a malformed tool call")
            name = raw.get("name")
            identifier = raw.get("id")
            arguments = raw.get("args", {})
            if not isinstance(name, str) or not isinstance(identifier, str):
                raise ModelError("model tool call is missing a name or identifier")
            if not isinstance(arguments, Mapping):
                raise ModelError("model tool-call arguments must be an object")
            calls.append(ToolCall(identifier, name, dict(arguments)))
        content = getattr(response, "content", "")
        return Message(
            role="assistant",
            content=content if isinstance(content, str) else str(content),
            identifier=getattr(response, "id", None),
            tool_calls=tuple(calls),
        )

    def _to_external(self, message: Message) -> object:
        if message.role == "system":
            return self._messages.SystemMessage(content=message.content)
        if message.role == "user":
            return self._messages.HumanMessage(content=message.content)
        if message.role == "tool":
            if not message.tool_call_id:
                raise ModelError("tool messages require a tool_call_id")
            return self._messages.ToolMessage(
                content=message.content,
                tool_call_id=message.tool_call_id,
                name=message.name,
            )
        calls = [
            {
                "id": call.identifier,
                "name": call.name,
                "args": dict(call.arguments),
                "type": "tool_call",
            }
            for call in message.tool_calls
        ]
        return self._messages.AIMessage(content=message.content, tool_calls=calls)


class OpenAIEssayModel:
    """LangChain ChatOpenAI adapter for essay text and structured queries."""

    def __init__(self, config: ModelConfig) -> None:
        openai_module = _optional_module("langchain_openai")
        self._messages = _optional_module("langchain_core.messages")
        self._model = openai_module.ChatOpenAI(
            model=config.name,
            temperature=config.temperature,
        )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._model.invoke(
            [
                self._messages.SystemMessage(content=system_prompt),
                self._messages.HumanMessage(content=user_prompt),
            ]
        )
        content = getattr(response, "content", "")
        return content if isinstance(content, str) else str(content)

    def queries(self, system_prompt: str, user_prompt: str) -> Sequence[str]:
        schema = {
            "title": "SearchQueries",
            "description": "Focused web searches",
            "type": "object",
            "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
            "required": ["queries"],
        }
        structured = self._model.with_structured_output(schema)
        result = structured.invoke(
            [
                self._messages.SystemMessage(content=system_prompt),
                self._messages.HumanMessage(content=user_prompt),
            ]
        )
        if not isinstance(result, Mapping) or not isinstance(result.get("queries"), list):
            raise ModelError("model returned malformed search queries")
        queries = result["queries"]
        if not all(isinstance(query, str) for query in queries):
            raise ModelError("every generated search query must be text")
        return tuple(str(query) for query in queries)


class TavilySearchClient:
    """Validated Tavily response adapter."""

    def __init__(self, api_key: str) -> None:
        if not api_key.strip():
            raise SearchError("Tavily API key cannot be empty")
        module = _optional_module("tavily")
        self._client = module.TavilyClient(api_key=api_key)

    def search(self, query: str, *, max_results: int) -> Sequence[str]:
        if not query.strip():
            raise SearchError("search query cannot be empty")
        if max_results <= 0:
            raise SearchError("max_results must be positive")
        response = self._client.search(query=query, max_results=max_results)
        if not isinstance(response, Mapping) or not isinstance(
            response.get("results"), list
        ):
            raise SearchError("Tavily returned a malformed response")
        snippets: list[str] = []
        for result in response["results"]:
            if not isinstance(result, Mapping) or not isinstance(
                result.get("content"), str
            ):
                raise SearchError("Tavily result is missing text content")
            snippets.append(result["content"])
        return snippets


class TavilySearchTool:
    def __init__(self, client: TavilySearchClient, *, max_results: int) -> None:
        if max_results <= 0:
            raise ValueError("max_results must be positive")
        self._client = client
        self._max_results = max_results

    @property
    def name(self) -> str:
        return "web_search"

    def invoke(self, arguments: Mapping[str, object]) -> str:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise SearchError("web_search requires a non-empty string query")
        return "\n\n".join(
            self._client.search(query.strip(), max_results=self._max_results)
        )


def _tool_definition(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Search the web for current factual information.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
