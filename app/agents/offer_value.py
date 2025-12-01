import logging
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END

from app.config import Configuration
from app.agents.models import MainOrchestratorState

log = logging.getLogger(__name__)
conf = Configuration()

DATA_PATH = os.path.join(os.getcwd(), "data")
COMPANY_DATA = open(os.path.join(DATA_PATH, "company_info.md"), "r").read()

offer_value_agent = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", temperature=conf.temperature
)


def search_data(state: MainOrchestratorState) -> dict:
    question = state["message_to_analyze"]

    SYSTEM_PROMPT = f"""
    Eres un experto en datos de la empresa, responde a cualquier pregunta relacionada con los datos de la empresa.
    No inventes ninguna información.
    No respondas preguntas que no sean relacionadas con los datos de la empresa.
    Planifica y luego responde a la pregunta.
    Responde de una manera corta, concisa, gentil y clara.
    Usa estos datos de la empresa para responder la pregunta:
    {COMPANY_DATA}
    """
    USER_PROMPT = f"""Responde la siguiente pregunta: {question}"""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=USER_PROMPT),
    ]
    response = offer_value_agent.invoke(messages)
    return {"response": response.content, messages: [response]}


def entry_point(state: MainOrchestratorState) -> dict:
    return {"current_action": "offer_value"}


# Define a new graph
offer_value_graph = StateGraph(MainOrchestratorState)
offer_value_graph.add_node(entry_point)
offer_value_graph.add_node(search_data)
offer_value_graph.add_edge(START, "entry_point")
offer_value_graph.add_edge("entry_point", "search_data")
offer_value_graph.add_edge("search_data", END)
