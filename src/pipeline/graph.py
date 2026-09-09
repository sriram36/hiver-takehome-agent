from langgraph.graph import StateGraph, END
from .state import AgentState
from .classify_node import classify_intent
from .retrieve import Retriever
from .draft import draft_reply
from .route import route_decision

def build_graph():
    workflow = StateGraph(AgentState)
    
    retriever = Retriever()
    
    workflow.add_node("classify", classify_intent)
    workflow.add_node("retrieve", retriever.retrieve)
    workflow.add_node("draft", draft_reply)
    workflow.add_node("route", route_decision)
    
    workflow.set_entry_point("classify")
    workflow.add_edge("classify", "retrieve")
    workflow.add_edge("retrieve", "draft")
    workflow.add_edge("draft", "route")
    workflow.add_edge("route", END)
    
    return workflow.compile()

def run_agent(text: str) -> AgentState:
    app = build_graph()
    result = app.invoke({"customer_text": text})
    return result
