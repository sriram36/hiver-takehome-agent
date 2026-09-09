from .state import AgentState

def route_decision(state: AgentState) -> AgentState:
    reasons = []
    
    intent = state.get("predicted_intent", "other")
    if intent in ["refund_billing", "account_access", "service_complaint"]:
        reasons.append(f"Intent '{intent}' is high-risk")
        
    conf = state.get("classifier_confidence", 1.0)
    if conf < 0.6:
        reasons.append(f"Classifier confidence too low ({conf:.2f})")
        
    sim = state.get("max_similarity", 1.0)
    if sim < 0.3:
        reasons.append(f"No highly similar historical resolution found (sim={sim:.2f})")
        
    text = state.get("customer_text", "").lower()
    if any(word in text for word in ["human", "person", "supervisor", "manager"]):
        reasons.append("Explicit request for human/supervisor")
        
    if reasons:
        state["escalation_decision"] = "escalate"
    else:
        state["escalation_decision"] = "auto"
        
    state["escalation_reasons"] = reasons
    return state
