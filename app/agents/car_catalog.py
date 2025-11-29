import logging
import os

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END

from app.config import Configuration
from app.agents.models import CarCatalogState, MainOrchestratorState

log = logging.getLogger(__name__)
conf = Configuration()

DATA_PATH = os.path.join(os.getcwd(), "data")
CAR_CATALOG_PATH = os.path.join(DATA_PATH, "car_catalog.csv")
CAR_CATALOG = pd.read_csv(CAR_CATALOG_PATH).to_markdown()

offer_value_agent = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", temperature=conf.temperature
)


def context_car_identification(state: CarCatalogState) -> dict:
    SYSTEM_PROMPT = """
    Eres un vendedor experto en autos, ayuda a un usuario a encontrar el auto que busca. 
    Identifica las necesidades del usuario:
    * Marca
    * Kilometraje
    * Precio
    * Modelo
    * Year
    * Version
    * Bluetooth
    * Largo
    * Ancho
    * Alto
    * CarPlay
    Si el usuario no proporciona informacion suficiente, pide mas informacion sobre las necesidades del usuario.
    No inventes ninguna informacion.
    El output esperado es una lista de necesidades del usuario, en caso de que no haya ninguna necesidad, devuelve una lista vacia.
    Si el usuario no proporciona informacion sobre una necesidad, no la incluyas en el output.
    Si solo identificas algunas necesidades, devuelve solo las necesidades que identificaste.
    Ejemplo de output:
    {
        "marca": string,
        "kilometraje": int,
        "precio": float,
        "modelo": string,
        "year": int,
        "version": string,
        "bluetooth": bool,
        "largo": float,
        "ancho": float,
        "alto": float,
        "car_play": bool,
    }
    """
    USER_PROMPT = f"Extrae las necesidades del usuario de este mensaje: {state['message_to_analyze']}"
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=USER_PROMPT),
    ]
    response = offer_value_agent.invoke(messages)
    return {"user_needs": [response.content]}


def search_cars(state: CarCatalogState) -> dict:
    SYSTEM_PROMPT = f"""
    Eres un vendedor experto en autos, responde a cualquier pregunta relacionada con los autos dado el siguiente catalogo de autos:
    Tu objetivo es ayudar a un usuario a encontrar el auto que busca, dada cualquier duda y todo el contexto que el te pueda brindar.
    Responde de una manera corta, concisa, gentil y clara.
    {CAR_CATALOG}
    Devueve los 3 autos que se ajustan mejor a las necesidades del usuario, en caso de que no haya ningun auto que se ajuste, devuelve una lista vacia.
    El output esperado es una lista de autos, en caso de que no haya ningun auto que se ajuste, devuelve una lista vacia.
    No inventes ninguna informacion.
    No respondas preguntas que no sean relacionadas con los autos.
    Planifica y luego responde a la pregunta.
    Responde de una manera corta, concisa, gentil y clara.
    Incluye el match de las necesidades del usuario con los autos encontrados
    Ejemplo de output:
    [
        "Primera opcion: Toyota Corolla 2024, con la lista de las necesidades del usuario que se ajustan a la primera opcion",
        "Segunda opcion: Honda Civic 2024, con la lista de las necesidades del usuario que se ajustan a la segunda opcion",
        "Tercera opcion: Ford Mustang 2024, con la lista de las necesidades del usuario que se ajustan a la tercera opcion",
    ]
    """
    USER_PROMPT = f"""
    Las necesidades del usuario son: {state["user_needs"]}
    """
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=USER_PROMPT),
    ]
    response = offer_value_agent.invoke(messages)
    return {"car_findings": response.content}


def organize_response(state: CarCatalogState) -> dict:
    car_findings = state["car_findings"]
    SYSTEM_PROMPT = """
    Eres un vendedor experto en autos, crea una resumen muy llamativo y atractivo para el usuario, que incluya los autos encontrados y el match de las necesidades del usuario con los autos encontrados.
    basado en los hallazagos encontrados por tu asistente de busqueda, recuerda, debes crear un mensaje muy atractivo y llamativo para el usuario, que incluya los autos encontrados y el match de las necesidades del usuario con los autos encontrados.
    Recuerda incluir los datos de los autos encontrados, como marca, modelo, año, precio, kilometraje, etc, y toda la infomracion relevante de los autos a ofrecer al cliente.
    """
    USER_PROMPT = f"Estos son los autos encontrados: {car_findings}"
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=USER_PROMPT),
    ]
    response = offer_value_agent.invoke(messages)
    return {"response": response.content}


def entry_point(state: MainOrchestratorState) -> MainOrchestratorState:
    state["current_action"] = "offer_value"
    return state


# Define a new graph
car_catalog_graph = StateGraph(MainOrchestratorState, output_schema=CarCatalogState)
car_catalog_graph.add_node(entry_point)
car_catalog_graph.add_node(context_car_identification)
car_catalog_graph.add_node(search_cars)
car_catalog_graph.add_node(organize_response)
car_catalog_graph.add_edge(START, "entry_point")
car_catalog_graph.add_edge("entry_point", "context_car_identification")
car_catalog_graph.add_edge("context_car_identification", "search_cars")
car_catalog_graph.add_edge("search_cars", "organize_response")
car_catalog_graph.add_edge("organize_response", END)
