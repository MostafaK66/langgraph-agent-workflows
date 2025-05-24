from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from agent_state import AgentState
from agent_tools import AgentTools
from prompts import prompt
from persistence import Persistence


class Agent:
    def __init__(self, model, tools, system: str = "", checkpointer=None):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm", self.exists_action, {True: "action", False: END}
        )
        graph.add_edge(start_key="action", end_key="llm")
        graph.set_entry_point("llm")
        if checkpointer is not None:
            print("i found check pointer")
            self.graph = graph.compile(checkpointer=checkpointer,
                                       interrupt_before=["action"])

        else:
            self.graph = graph.compile()
        self.graph = graph.compile()
        self.model = model.bind_tools(tools)
        self.tools = {t.name: t for t in tools}

    def exists_action(self, state: AgentState) -> bool:
        return bool(state["messages"][-1].tool_calls)

    def call_openai(self, state: AgentState):
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        ai_msg = self.model.invoke(messages)
        return {"messages": [ai_msg]}

    def take_action(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for tc in tool_calls:
            name, args = tc["name"], tc["args"]
            if name not in self.tools:
                result = "bad tool name, retry"
            else:
                result = self.tools[name].invoke(args)
            results.append(
                ToolMessage(
                    tool_call_id=tc["id"],
                    name=name,
                    content=str(result),
                )
            )
        return {"messages": results}

    @classmethod
    def from_defaults(cls):
        model = ChatOpenAI(model="gpt-3.5-turbo")
        tools = AgentTools().get_known_actions()
        saver = Persistence.synchronous(":memory:")
        return cls(model=model, tools=tools, system=prompt, checkpointer=saver)




