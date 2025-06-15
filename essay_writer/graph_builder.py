from langgraph.graph import StateGraph, END
from persistence import Persistence
from agent_state import AgentState
from utility import EssayWriterAgent


class EssayWriterGraph:
    def __init__(self, agent: EssayWriterAgent, checkpointer=None):
        """
        Build and compile the LangGraph for the essay‐writing workflow.

        :param agent: an instance of EssayWriterAgent, whose node methods we register
        :param checkpointer: optional LangGraph saver; if None, uses an in‐memory SQLite saver
        """
        builder = StateGraph(AgentState)

        builder.add_node("planner", agent.plan_node)
        builder.add_node("research_plan", agent.research_plan_node)
        builder.add_node("generate", agent.generation_node)
        builder.add_node("reflect", agent.reflection_node)
        builder.add_node("research_critique", agent.research_critique_node)

        builder.set_entry_point("planner")

        builder.add_edge("planner", "research_plan")
        builder.add_edge("research_plan", "generate")

        builder.add_conditional_edges(
            "generate",
            agent.should_continue,
            {END: END,
             "reflect": "reflect"}
        )

        builder.add_edge("reflect", "research_critique")
        builder.add_edge("research_critique", "generate")

        if checkpointer is None:
            memory = Persistence.synchronous(":memory:")
        else:
            memory = checkpointer

        self.graph = builder.compile(checkpointer=memory)


