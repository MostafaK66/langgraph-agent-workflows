from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from langgraph_agents.config import ModelConfig
from langgraph_agents.errors import (
    IntegrationUnavailableError,
    ModelError,
    SearchError,
)
from langgraph_agents.integrations import (
    OpenAIEssayModel,
    OpenAIResearchModel,
    TavilySearchClient,
    TavilySearchTool,
)
from langgraph_agents.messages import Message, ToolCall


class ExternalMessage:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class ChatModel:
    response: object = ExternalMessage(content="answer", id="message", tool_calls=[])
    structured_response: object = {"queries": ["one", "two"]}
    definitions: object = None
    invoked: object = None

    def __init__(self, **kwargs: Any) -> None:
        self.arguments = kwargs

    def bind_tools(self, definitions: object) -> ChatModel:
        ChatModel.definitions = definitions
        return self

    def invoke(self, messages: object) -> object:
        ChatModel.invoked = messages
        return self.response

    def with_structured_output(self, schema: object) -> ChatModel:
        assert schema
        self.response = self.structured_response
        return self


class TavilyClient:
    response: object = {"results": [{"content": "evidence"}]}
    arguments: dict[str, object] = {}

    def __init__(self, api_key: str) -> None:
        assert api_key == "credential"

    def search(self, **kwargs: object) -> object:
        TavilyClient.arguments = kwargs
        return self.response


def fake_import(name: str) -> object:
    if name == "langchain_openai":
        return SimpleNamespace(ChatOpenAI=ChatModel)
    if name == "langchain_core.messages":
        return SimpleNamespace(
            SystemMessage=ExternalMessage,
            HumanMessage=ExternalMessage,
            AIMessage=ExternalMessage,
            ToolMessage=ExternalMessage,
        )
    if name == "tavily":
        return SimpleNamespace(TavilyClient=TavilyClient)
    raise ImportError(name)


def test_research_model_maps_all_message_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("langgraph_agents.integrations.import_module", fake_import)
    ChatModel.response = ExternalMessage(
        content=["answer"],
        id="response",
        tool_calls=[{"id": "call", "name": "search", "args": {"query": "x"}}],
    )
    model = OpenAIResearchModel(ModelConfig(), ["search"])
    response = model.respond(
        [
            Message(role="system", content="system"),
            Message(role="user", content="user"),
            Message(
                role="assistant",
                content="assistant",
                tool_calls=(ToolCall("old", "search", {"query": "old"}),),
            ),
            Message(
                role="tool",
                content="result",
                tool_call_id="old",
                name="search",
            ),
        ]
    )
    assert response.content == "['answer']"
    assert response.tool_calls == (ToolCall("call", "search", {"query": "x"}),)
    assert ChatModel.definitions


def test_tool_message_requires_call_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("langgraph_agents.integrations.import_module", fake_import)
    model = OpenAIResearchModel(ModelConfig(), [])
    with pytest.raises(ModelError, match="tool_call_id"):
        model.respond([Message(role="tool", content="bad")])


@pytest.mark.parametrize(
    ("tool_calls", "message"),
    [
        (["bad"], "malformed"),
        ([{"id": "x", "args": {}}], "name or identifier"),
        ([{"id": "x", "name": "search", "args": []}], "must be an object"),
    ],
)
def test_research_model_validates_tool_calls(
    monkeypatch: pytest.MonkeyPatch, tool_calls: object, message: str
) -> None:
    monkeypatch.setattr("langgraph_agents.integrations.import_module", fake_import)
    ChatModel.response = ExternalMessage(content="", tool_calls=tool_calls)
    with pytest.raises(ModelError, match=message):
        OpenAIResearchModel(ModelConfig(), []).respond([])


def test_essay_model_text_and_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("langgraph_agents.integrations.import_module", fake_import)
    model = OpenAIEssayModel(ModelConfig())
    ChatModel.response = ExternalMessage(content=["text"])
    assert model.complete("system", "user") == "['text']"
    ChatModel.structured_response = {"queries": ["one", "two"]}
    assert model.queries("system", "user") == ("one", "two")


@pytest.mark.parametrize(
    "result",
    [None, {}, {"queries": "wrong"}, {"queries": ["ok", 2]}],
)
def test_essay_model_validates_queries(
    monkeypatch: pytest.MonkeyPatch, result: object
) -> None:
    monkeypatch.setattr("langgraph_agents.integrations.import_module", fake_import)
    ChatModel.structured_response = result
    with pytest.raises(ModelError):
        OpenAIEssayModel(ModelConfig()).queries("system", "user")


def test_tavily_client_and_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("langgraph_agents.integrations.import_module", fake_import)
    TavilyClient.response = {"results": [{"content": " one "}, {"content": "two"}]}
    client = TavilySearchClient("credential")
    assert client.search("query", max_results=2) == [" one ", "two"]
    tool = TavilySearchTool(client, max_results=2)
    assert tool.name == "web_search"
    assert tool.invoke({"query": " query "}) == " one \n\ntwo"
    assert TavilyClient.arguments == {"query": "query", "max_results": 2}


def test_tavily_validates_inputs_and_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("langgraph_agents.integrations.import_module", fake_import)
    with pytest.raises(SearchError, match="API key"):
        TavilySearchClient("")
    client = TavilySearchClient("credential")
    with pytest.raises(SearchError, match="query"):
        client.search("", max_results=1)
    with pytest.raises(SearchError, match="max_results"):
        client.search("query", max_results=0)
    for response in (None, {}, {"results": "bad"}, {"results": [{}]}):
        TavilyClient.response = response
        with pytest.raises(SearchError, match="malformed|missing"):
            client.search("query", max_results=1)


def test_search_tool_validates_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("langgraph_agents.integrations.import_module", fake_import)
    client = TavilySearchClient("credential")
    with pytest.raises(ValueError, match="max_results"):
        TavilySearchTool(client, max_results=0)
    tool = TavilySearchTool(client, max_results=1)
    with pytest.raises(SearchError, match="non-empty"):
        tool.invoke({})
    with pytest.raises(SearchError, match="non-empty"):
        tool.invoke({"query": 4})


def test_missing_optional_dependency_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr("langgraph_agents.integrations.import_module", missing)
    with pytest.raises(IntegrationUnavailableError, match=r"\.\[agents\]"):
        OpenAIEssayModel(ModelConfig())
