"""
LangGraph-based business consulting agent.
Coordinates LLM reasoning and tool usage.
"""

from typing import TypedDict, Sequence, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from core.config import OPENAI_API_KEY
from agents.tools import tools

# LLM bound to tool definitions
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
    openai_api_key=OPENAI_API_KEY
).bind_tools(tools)

class AgentState(TypedDict):
    """
    Conversation state shared across graph nodes.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]

def agent(state: AgentState):
    """
    Main agent logic node.
    """
    response = llm.invoke([
        SystemMessage(
            content="You are a helpful business consulting agent."
        ),
        *state["messages"]
    ])
    return {"messages": [response]}

def should_continue(state: AgentState) -> Literal["continue", "end"]:
    """
    Determines whether tool execution is required.
    """
    return "continue" if state["messages"][-1].tool_calls else "end"

def build_agent():
    """
    Builds and compiles the LangGraph agent.
    """
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"continue": "tools", "end": END}
    )

    graph.add_edge("tools", "agent")

    return graph.compile()
