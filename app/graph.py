from langgraph.graph import StateGraph, START, END
from .types import State
from .nodes import classify, retrieval, draft, review, refine, escalate

def _route_after_review(state: State) -> str:
    rv = state.get("review", {}) or {}
    if rv.get("approved"):
        return "finish"
    if int(state.get("attempts", 0)) >= 2:
        return "escalate"
    return "retry"

def build_graph():
    workflow = StateGraph(State)

    workflow.add_node("classify", classify)
    workflow.add_node("retrieve", retrieval)
    workflow.add_node("draft", draft)
    workflow.add_node("review", review)
    workflow.add_node("refine", refine)
    workflow.add_node("escalate", escalate)

    workflow.add_edge(START, "classify")
    workflow.add_edge("classify", "retrieve")
    workflow.add_edge("retrieve", "draft")
    workflow.add_edge("draft", "review")

    workflow.add_conditional_edges(
        "review", _route_after_review,
        {"finish": END, "retry": "refine", "escalate": "escalate"}
    )

    workflow.add_edge("refine", "draft")
    workflow.add_edge("escalate", END)

    return workflow.compile()

graph = build_graph()
