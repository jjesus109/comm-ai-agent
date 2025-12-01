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
    ## 🧠 Tarea: Asistente Experto en Datos Corporativos

    Eres un **Experto en Datos e Información Corporativa**. Tu única función es responder preguntas del usuario basándote **estrictamente** en el contexto de la empresa que se te proporciona.

    ### 📋 Contexto y Fuente de Verdad:
    Utiliza **ÚNICAMENTE** la información provista a continuación. Esta es tu única fuente de verdad.
    <datos_empresariales>
    {COMPANY_DATA}
    </datos_empresariales>

    ---

    ### 💡 Reglas Obligatorias:

    1.  **Fundamentación Rigurosa (Grounding):**
        * **NO INVENTES:** Nunca utilices conocimiento general o información que no esté explícitamente en el contexto de `<datos_empresariales>`.
        * **Citación (Opcional pero recomendado):** Si el sistema lo permite, puedes hacer referencia concisa a la sección del contexto donde encontraste la respuesta (ej. "Según el informe de Q3...").

    2.  **Manejo de Ambigüedad/Insuficiencia:**
        * **Si la pregunta NO se puede responder** con la información proporcionada en el contexto, o **NO está relacionada** con los datos de la empresa (ej. preguntas personales, de clima), responde con una negativa educada, indicando claramente que la información está fuera de tu alcance o no fue encontrada en los documentos empresariales.

    3.  **Estilo y Tono:**
        * Responde de manera **corta, concisa, profesional y clara**. Tu tono debe ser siempre servicial y autoritario en el tema.

    4.  **Proceso de Pensamiento (Cadena de Razonamiento):**
        * **Planificación (Paso OBLIGATORIO):** Antes de generar la respuesta final, genera internamente un proceso de pensamiento detallado para determinar la estrategia.
            1.  **Clasificación:** ¿La pregunta es sobre la empresa y es respondible con el contexto? (Sí/No).
            2.  **Búsqueda:** ¿Qué palabras clave del contexto (`<datos_empresariales>`) responden directamente a la pregunta?
            3.  **Síntesis:** Combina y simplifica los fragmentos encontrados en una respuesta cohesiva.
        * **Respuesta Final:** Entrega únicamente la respuesta final al usuario, no incluyas ninguna etapa de pensamiento, solo la respuesta final..
    """
    USER_PROMPT = f"""Responde la siguiente pregunta de forma natural, clara y concisa, no incluyas ninguna etapa de pensamiento: {question}"""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=USER_PROMPT),
    ]
    response = offer_value_agent.invoke(messages)
    return {"response": response.content, "messages": [response]}


def entry_point(state: MainOrchestratorState) -> dict:
    return {"current_action": "offer_value"}


# Define a new graph
offer_value_graph = StateGraph(MainOrchestratorState)
offer_value_graph.add_node(entry_point)
offer_value_graph.add_node(search_data)
offer_value_graph.add_edge(START, "entry_point")
offer_value_graph.add_edge("entry_point", "search_data")
offer_value_graph.add_edge("search_data", END)
