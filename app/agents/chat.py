#from langchain_core.messages import HumanMessage
#from langgraph.graph import END, START, StateGraph

#from app.models.search import GraphState


#def respond(state: GraphState) -> dict[str, str]:
#    """Echo the user message through a single LangGraph node."""
#    user_message = HumanMessage(content=state["message"])
#    return {"reply": f"Received: {user_message.content}"}


#def build_graph():
#    graph = StateGraph(GraphState)
#    graph.add_node("respond", respond)
#    graph.add_edge(START, "respond")
#    graph.add_edge("respond", END)
#    return graph.compile()


#agent = build_graph()
