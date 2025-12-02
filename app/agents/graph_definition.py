from langgraph.graph import StateGraph, START, END

from app.agents.orchestrator import (
    entry_point,
    manage_unsecure,
    summarize_conversation,
    continue_operation,
    wait_to_analyze,
    verify_malicious_content,
    should_summarize,
    intention_finder,
)
from app.agents.financial_plan import financial_plan_graph
from app.agents.offer_value import offer_value_graph
from app.agents.car_catalog import car_catalog_graph
from app.agents.models import MainOrchestratorState
from app.depends import get_memory


memory = get_memory()


# Main orchestrator graph

workflow = StateGraph(MainOrchestratorState)
workflow.add_node(entry_point)
workflow.add_node(manage_unsecure)
workflow.add_node(summarize_conversation)
workflow.add_node("financial_plan", financial_plan_graph.compile(checkpointer=memory))
workflow.add_node("offer_value", offer_value_graph.compile(checkpointer=memory))
workflow.add_node("car_catalog", car_catalog_graph.compile(checkpointer=memory))
workflow.add_node(continue_operation)
workflow.add_node(wait_to_analyze)

workflow.add_edge(START, "entry_point")
workflow.add_conditional_edges("entry_point", verify_malicious_content)
workflow.add_conditional_edges("wait_to_analyze", should_summarize)
workflow.add_conditional_edges("continue_operation", intention_finder)
workflow.add_edge("summarize_conversation", "continue_operation")
workflow.add_edge("financial_plan", END)
workflow.add_edge("car_catalog", END)
workflow.add_edge("offer_value", END)
workflow.add_edge("manage_unsecure", END)

orchestrator_graph = workflow.compile(checkpointer=memory)
