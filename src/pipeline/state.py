from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    customer_text: str
    predicted_intent: str
    classifier_confidence: float
    retrieved_exemplars: List[Dict[str, str]]
    max_similarity: float
    draft_reply: str
    escalation_decision: str
    escalation_reasons: List[str]
