from __future__ import annotations

from collections.abc import Sequence

import pytest

from langgraph_agents.config import EssayConfig, SearchConfig
from langgraph_agents.errors import ModelError, SearchError, StateError
from langgraph_agents.essay import EssayModel, EssayState, EssayWriter


class Model:
    def __init__(
        self,
        completions: Sequence[str | Exception] = ("text",),
        query_results: Sequence[str] | Exception = ("query",),
    ) -> None:
        self.completions = list(completions)
        self.query_results = query_results
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        result = self.completions.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def queries(self, system_prompt: str, user_prompt: str) -> Sequence[str]:
        self.calls.append((system_prompt, user_prompt))
        if isinstance(self.query_results, Exception):
            raise self.query_results
        return self.query_results


class Search:
    def __init__(
        self, values: dict[str, Sequence[str] | Exception] | None = None
    ) -> None:
        self.values = values or {"query": ("evidence",)}
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, max_results: int) -> Sequence[str]:
        self.calls.append((query, max_results))
        result = self.values[query]
        if isinstance(result, Exception):
            raise result
        return result


def writer(model: EssayModel | None = None, search: Search | None = None) -> EssayWriter:
    return EssayWriter(
        model or Model(),
        search or Search(),
        SearchConfig(max_results=2, max_queries=2),
        EssayConfig(max_revisions=3),
    )


def state(**updates: object) -> EssayState:
    value: EssayState = {
        "task": "topic",
        "plan": "plan",
        "draft": "draft",
        "critique": "critique",
        "content": ["existing"],
        "revision_number": 1,
        "max_revisions": 3,
    }
    value.update(updates)  # type: ignore[typeddict-item]
    return value


def test_initial_state_is_clean_and_validated() -> None:
    result = writer().initial_state(" topic ")
    assert result == {
        "task": "topic",
        "plan": "",
        "draft": "",
        "critique": "",
        "content": [],
        "revision_number": 0,
        "max_revisions": 3,
    }
    with pytest.raises(StateError, match="topic"):
        writer().initial_state(" ")


def test_plan_and_reflection_nodes() -> None:
    model = Model((" plan ", " critique "))
    agent = writer(model)
    assert agent.plan(state()) == {"plan": "plan"}
    assert agent.reflect(state()) == {"critique": "critique"}


def test_generation_includes_content_plan_and_optional_critique() -> None:
    model = Model(("draft one", "draft two"))
    agent = writer(model)
    first = agent.generate(state(critique=""))
    assert first == {"draft": "draft one", "revision_number": 2}
    assert "existing" in model.calls[0][0]
    assert "Critique to address" not in model.calls[0][1]
    second = agent.generate(state())
    assert second["draft"] == "draft two"
    assert "critique" in model.calls[1][1]


def test_research_nodes_copy_content_limit_queries_and_skip_empty() -> None:
    model = Model(query_results=(" first ", "", "second", "ignored"))
    search = Search({"first": (" one ", ""), "second": ("two",)})
    agent = writer(model, search)
    original = state()
    plan_result = agent.research_plan(original)
    assert plan_result == {"content": ["existing", "one", "two"]}
    assert original["content"] == ["existing"]
    assert search.calls == [("first", 2), ("second", 2)]
    critique_result = agent.research_critique(original)
    assert critique_result["content"][-2:] == ["one", "two"]


@pytest.mark.parametrize(
    ("current", "maximum", "expected"),
    [(1, 3, "reflect"), (3, 3, "end"), (4, 3, "end")],
)
def test_revision_limit_is_exact(current: int, maximum: int, expected: str) -> None:
    assert (
        EssayWriter.should_continue(state(revision_number=current, max_revisions=maximum))
        == expected
    )


def test_revision_limit_rejects_invalid_state() -> None:
    with pytest.raises(StateError, match="positive"):
        EssayWriter.should_continue(state(max_revisions=0))


@pytest.mark.parametrize("field", ["task", "plan", "draft", "critique"])
def test_nodes_require_input_fields(field: str) -> None:
    value = state()
    value[field] = ""  # type: ignore[literal-required]
    agent = writer()
    operation = {
        "task": agent.plan,
        "plan": agent.generate,
        "draft": agent.reflect,
        "critique": agent.research_critique,
    }[field]
    with pytest.raises(StateError, match=field):
        operation(value)


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (RuntimeError("offline"), "essay model failed"),
        (ModelError("known"), "known"),
        (" ", "empty text"),
    ],
)
def test_completion_failures(result: str | Exception, message: str) -> None:
    with pytest.raises(ModelError, match=message):
        writer(Model((result,))).plan(state())


def test_query_generation_failures() -> None:
    with pytest.raises(ModelError, match="generation failed"):
        writer(Model(query_results=RuntimeError("offline"))).research_plan(state())
    with pytest.raises(ModelError, match="no usable"):
        writer(Model(query_results=("", " "))).research_plan(state())


def test_search_failures_are_normalized_or_preserved() -> None:
    with pytest.raises(SearchError, match="search failed"):
        writer(search=Search({"query": RuntimeError("offline")})).research_plan(state())
    with pytest.raises(SearchError, match="known"):
        writer(search=Search({"query": SearchError("known")})).research_plan(state())
