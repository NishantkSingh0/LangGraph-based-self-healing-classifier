from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from .nodes import State, inference_node, confidence_check_node, fallback_node, should_fallback

def build_graph():
    graph=StateGraph(State)
    graph.add_node("inference", RunnableLambda(inference_node))
    graph.add_node("confidence_check", RunnableLambda(confidence_check_node))
    graph.add_node("fallback", RunnableLambda(fallback_node))

    graph.set_entry_point("inference")
    graph.add_edge("inference", "confidence_check")
    graph.add_conditional_edges("confidence_check", should_fallback, {
        "fallback": "fallback",
        "end": END
    })
    graph.add_edge("fallback", END)
    
    return graph.compile()
