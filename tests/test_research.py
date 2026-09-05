from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from langgraph_agents.errors import ModelError
from langgraph_agents.messages import Message, ToolCall
from langgraph_agents.research import ResearchAgent


class Model:
    def __init__(self, result: Message | Exception) -> None:
        self.result = result
        self.messages: Sequence[Message] = ()

    def respond(self, messages: Sequence[Message]) -> Message:
        self.messages = messages
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class Tool:
    def __init__(self, name: str = "search", result: str | Exception = "result") -> None:
        self._name = name
        self.result = result
        self.arguments: Mapping[str, object] = {}

    @property
    def name(self) -> str:
        return self._name

    def invoke(self, arguments: Mapping[str, object]) -> str:
        self.arguments = arguments
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_agent_validates_prompt_and_unique_tools() -> None:
    model = Model(Message(role="assistant", content="ok"))
    with pytest.raises(ValueError, match="prompt"):
        ResearchAgent(model, [], system_prompt=" ")
    with pytest.raises(ValueError, match="unique"):
        ResearchAgent(model, [Tool(), Tool()])


def test_call_model_adds_system_prompt() -> None:
    model = Model(Message(role="assistant", content="answer"))
    result = ResearchAgent(model, [], system_prompt="system").call_model(
        {"messages": [Message(role="user", content="question")]}
    )
    assert result["messages"][0].content == "answer"
    assert model.messages[0] == Message(role="system", content="system")


@pytest.mark.parametrize(
    ("model_result", "message"),
    [
        (RuntimeError("offline"), "research model failed"),
        (ModelError("known"), "known"),
        (Message(role="user", content="wrong"), "assistant message"),
    ],
)
def test_call_model_normalizes_failures(
    model_result: Message | Exception, message: str
) -> None:
    agent = ResearchAgent(Model(model_result), [])
    with pytest.raises(ModelError, match=message):
        agent.call_model({"messages": [Message(role="user", content="question")]})


def test_call_model_requires_messages() -> None:
    agent = ResearchAgent(Model(Message(role="assistant", content="answer")), [])
    with pytest.raises(ModelError, match="at least one"):
        agent.call_model({"messages": []})


def test_tool_call_detection_handles_empty_state() -> None:
    assert not ResearchAgent.has_tool_calls({"messages": []})
    assert not ResearchAgent.has_tool_calls(
        {"messages": [Message(role="assistant", content="done")]}
    )
    assert ResearchAgent.has_tool_calls(
        {
            "messages": [
                Message(
                    role="assistant",
                    content="",
                    tool_calls=(ToolCall("1", "search", {}),),
                )
            ]
        }
    )


def test_take_actions_runs_known_tool_and_reports_unknown_and_failure() -> None:
    good = Tool(result="evidence")
    bad = Tool(name="broken", result=RuntimeError("offline"))
    agent = ResearchAgent(Model(Message(role="assistant", content="")), [good, bad])
    state = {
        "messages": [
            Message(
                role="assistant",
                content="",
                tool_calls=(
                    ToolCall("1", "search", {"query": "topic"}),
                    ToolCall("2", "missing", {}),
                    ToolCall("3", "broken", {}),
                ),
            )
        ]
    }
    messages = agent.take_actions(state)["messages"]
    assert [message.content for message in messages] == [
        "evidence",
        "Unknown tool 'missing'; choose one of ['broken', 'search'].",
        "Tool 'broken' failed: offline",
    ]
    assert messages[0].tool_call_id == "1"
    assert good.arguments == {"query": "topic"}


def test_take_actions_requires_messages() -> None:
    with pytest.raises(ModelError, match="at least one"):
        ResearchAgent(Model(Message(role="assistant", content="")), []).take_actions(
            {"messages": []}
        )
