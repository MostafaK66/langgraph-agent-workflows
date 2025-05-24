import operator
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from uuid import uuid4


def reduce_messages(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
    for msg in right:
        if not msg.id:
            msg.id = str(uuid4())
    merged = left.copy()
    for msg in right:
        for i, existing in enumerate(merged):
            if existing.id == msg.id:
                merged[i] = msg
                break
        else:
            merged.append(msg)
    return merged

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], reduce_messages]