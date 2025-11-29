from operator import add
import logging
from typing import Annotated, Literal, TypedDict


from langgraph.graph import StateGraph, START
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END


import sqlite3

# In memory
db_path = "/Users/jesusalbino/Projects/kavak/commercial-ai-agent/example.db"
conn = sqlite3.connect(db_path, check_same_thread=False)

memory = SqliteSaver(conn)

log = logging.getLogger(__name__)

SUB_AGENTS = Literal["offer_value", "car_catalog", "financial_plan"]

orchestrator_agent = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", temperature=0
)


class MainOrchestratorState(TypedDict):
    messages: Annotated[list, add]
    summary: str
    current_action: str


def financial_plan(state: MainOrchestratorState) -> dict:
    print(f"FINANCIAL PLAN state: {state}")
    return {"financial_plan": "coso"}


def offer_value(state: MainOrchestratorState) -> dict:
    print(f"offer_value state: {state}")
    return {"offer_value": "coso"}


def car_catalog(state: MainOrchestratorState) -> dict:
    print(f"car_catalog, state: {state}")
    return {"car_catalog": "coso"}


def intention_finder(state: MainOrchestratorState) -> SUB_AGENTS:
    INTENTION_PROMPT = """Encuentra la decision que un usuario desea realizar basada en la siguiente entrada:
    {user_input}
    # REQUISITOS:
    * OBLIGATORIO: Deber elegir alguno de los siguientes escenarios:
        1. El usuario busca conocer alguna respuesta relacionada con la empresa, saber que hace, donde se origino, como funciona, todo acerca de la propuesta de valor, responde "offer_value".
        2. El usuario desea conocer cualquier tema relacionado algun vehiculo como marca, modelo, año, detalles del mismo, response "car_catalog".
        3. El usuario planea comprar un vehiculo o desea conocer informacion sobre como comprarlo. Ya conoce que vehiculo desea comprar o busca uno para comprar de forma inmediata. Response "financial_plan".
    * solo response con una de las tres posibles opciones posibles: "offer_value", "car_catalog", "financial_plan"
    * Piensa antes de responder
    """
    user_input = state["messages"][-1].content
    response = orchestrator_agent.invoke(INTENTION_PROMPT.format(user_input=user_input))
    return response.content


def entry_point(state: MainOrchestratorState) -> MainOrchestratorState:
    state["current_action"] = "analyzing"
    return state


# Define a new graph
workflow = StateGraph(MainOrchestratorState)
workflow.add_node(entry_point)
workflow.add_node(intention_finder)
workflow.add_node(financial_plan)
workflow.add_node(offer_value)
workflow.add_node(car_catalog)


workflow.add_edge(START, "entry_point")
workflow.add_conditional_edges("entry_point", intention_finder)
workflow.add_edge("intention_finder", "financial_plan")
workflow.add_edge("intention_finder", "car_catalog")
workflow.add_edge("intention_finder", "offer_value")
workflow.add_edge("financial_plan", END)
workflow.add_edge("car_catalog", END)
workflow.add_edge("offer_value", END)

# Compile
orchestrator_graph = workflow.compile(checkpointer=memory)
