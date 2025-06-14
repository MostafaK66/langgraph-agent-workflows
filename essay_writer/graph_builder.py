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
        # 1) Create a builder over your AgentState TypedDict
        builder = StateGraph(AgentState)

        # 2) Register nodes, binding to this agent’s methods
        builder.add_node("planner", agent.plan_node)
        builder.add_node("research_plan", agent.research_plan_node)
        builder.add_node("generate", agent.generation_node)
        builder.add_node("reflect", agent.reflection_node)
        builder.add_node("research_critique", agent.research_critique_node)

        # 3) Entry point of the graph
        builder.set_entry_point("planner")

        # 4) Linear edges for the “happy path”
        builder.add_edge("planner", "research_plan")
        builder.add_edge("research_plan", "generate")

        # 5) After “generate” we decide whether to loop or finish
        builder.add_conditional_edges(
            "generate",
            agent.should_continue,  # returns either END or "reflect"
            {END: END,
             "reflect": "reflect"}
        )

        # 6) The reflection sub‐loop
        builder.add_edge("reflect", "research_critique")
        builder.add_edge("research_critique", "generate")

        # 7) Set up the checkpointer (defaults to in‐memory SQLite)
        if checkpointer is None:
            memory = Persistence.synchronous(":memory:")
        else:
            memory = checkpointer

        # 8) Compile into an executable graph
        self.graph = builder.compile(checkpointer=memory)
