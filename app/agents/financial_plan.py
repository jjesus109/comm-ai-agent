import json
import logging
import os
import re

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END

from app.config import Configuration
from app.agents.models import MainOrchestratorState

log = logging.getLogger(__name__)
conf = Configuration()

DATA_PATH = os.path.join(os.getcwd(), "data")
CAR_CATALOG_PATH = os.path.join(DATA_PATH, "car_catalog.csv")
CAR_CATALOG = pd.read_csv(CAR_CATALOG_PATH).to_markdown()
ANNUAL_RATE = 0.10

financial_plan_agent = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", temperature=conf.temperature
)


def context_financial_identification(state: MainOrchestratorState) -> dict:
    SYSTEM_PROMPT = """
    Eres un experto en **Extracción de Información y Ventas de Automóviles**. Tu tarea es analizar la última intervención del cliente y ayudarle a definir sus necesidades de financiamiento.

    ### 🎯 Tareas Clave:
    1.  **Extraer Campos:** Identifica los valores para **Años de financiamiento** y **Pago inicial** a partir de la respuesta del cliente.
    2.  **Validación y Pregunta:** Si un campo es **ausente o ambiguo**, no lo incluyas en el diccionario de salida y genera un **único mensaje** conciso dirigido al cliente para solicitar específicamente la información faltante.

    ### 📋 Reglas de Extracción:
    * **Años de financiamiento (`years`):** Debe ser un número entero (int) que represente la duración del préstamo en años. (Ej. 3, 5, 7).
    * **Pago inicial (`down_payment`):** Debe ser un número decimal (float) que represente el monto total de enganche. (Ej. 10000.00, 5000.50).
    * **NO INVENTES NINGÚN VALOR.** Si la información no está clara, omite el campo en la salida.

    ### 📝 Formato de Salida Requerido (JSON):
    Tu respuesta **debe ser únicamente un objeto JSON**.

    * Incluye **solo** los campos (`years` o `down_payment`) que hayas podido extraer.
    * Si **falta alguna información** requerida (años o pago inicial), debes incluir la clave `"user_response"` con el mensaje de seguimiento.
    * Si se han extraído **TODOS** los campos, el campo `"user_response"` debe estar **OMITIDO**.

    **Ejemplo 1 (Falta información):**
    ```json
    {
        "years": 5,
        "user_response": "Gracias por indicar 5 años. Para continuar, ¿de cuánto sería tu pago inicial o enganche?"
    }
    ```

    **Ejemplo 2 (Información completa):**
    ```json
    {
        "years": 3,
        "down_payment": 10000.00
    }
    ```

    **Ejemplo 3 (Información ambigua):**
    ```json
    {
        "user_response": "Para obtener un financiamiento, necesitamos saber cuántos años estará el préstamo y cuánto sería el enganche. ¿Puedes indicar estos detalles?"
    }
    """
    USER_PROMPT = f"Extrae las necesidades de financiamiento del usuario de este mensaje: {state['message_to_analyze']}"
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=USER_PROMPT),
    ]
    response = financial_plan_agent.invoke(messages)
    pattern = re.compile(r"\{(.*)\}", re.DOTALL)

    # Buscar y extraer el contenido
    match = pattern.search(response.content)
    response_json = {}
    if match:
        contenido_json = "{" + match.group(1).strip() + "}"
        response_json = json.loads(contenido_json)
    return {
        "years": response_json.get("years"),
        "down_payment": response_json.get("down_payment"),
        "user_response": response_json.get("user_response"),
        "current_action": "context_financial_identification",
    }


def financial_calculator(state: MainOrchestratorState) -> dict:
    """
    This functions calculates the financial plan for a given car based on price, interest rate, months and down payment
    """
    months = state["years"] * 12
    down_payment = state["down_payment"]
    value_to_finance = state["selected_car"]["price"] - down_payment
    monthly_rate = ANNUAL_RATE / 12
    monthly_payment = (
        value_to_finance * monthly_rate * (1 + monthly_rate) ** months
    ) / ((1 + monthly_rate) ** months - 1)

    return {
        "monthly_payment": monthly_payment,
        "current_action": "financial_calculator",
    }


def organize_response(state: MainOrchestratorState) -> dict:
    selected_car = state["selected_car"]
    years = state["years"]
    down_payment = state["down_payment"]
    monthly_payment = state["monthly_payment"]
    value_to_finance = state["selected_car"]["price"]
    SYSTEM_PROMPT = f"""
    Eres un experto vendedor de autos, crea una respuesta muy atractiva y llamativa para el usuario, que incluya los datos del auto seleccionado, el plazo de financiamiento, el pago mensual y el valor a financiar.
    No inventes ninguna informacion.
    No respondas preguntas que no sean relacionadas con el financiamiento de un auto.
    Planifica y luego responde a la pregunta.
    Responde de una manera corta, concisa, gentil y clara.
    Ejemplo de output:
    "El auto {selected_car["brand"]} {selected_car["model"]} {selected_car["year"]} cuesta {selected_car["price"]} pesos, puedes financiarlo en {years} años con un pago mensual de {monthly_payment} pesos, y el valor a financiar es de {value_to_finance} pesos."
    """
    USER_PROMPT = f"""Estos son los datos que usaras para crear la respuesta:
    * Auto: {selected_car}
    * Años de financiamiento: {years}
    * Pago inicial: {down_payment}
    * Pago mensual: {monthly_payment}
    * Valor a financiar: {value_to_finance}
    * Tasa de interes: {ANNUAL_RATE}
    """
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=USER_PROMPT),
    ]
    response = financial_plan_agent.invoke(messages)
    return {
        "response": response.content,
        "current_action": "organize_response",
        "messages": [response],
    }


def decision_node(state: MainOrchestratorState) -> str:
    """
    This function decides what to do next based on the user's response
    """
    if state.get("user_response"):
        return END
    if state.get("years") and state.get("down_payment"):
        return "financial_calculator"
    else:
        return "context_financial_identification"


def entry_point(state: MainOrchestratorState) -> dict:
    # Ask user to complete the missing information
    if state.get("user_response"):
        return {"response": state.get("user_response")}
    return {"current_action": "financial_plan"}


# Define a new graph
financial_plan_graph = StateGraph(MainOrchestratorState)
financial_plan_graph.add_node(entry_point)
financial_plan_graph.add_node(context_financial_identification)
financial_plan_graph.add_node(financial_calculator)
financial_plan_graph.add_node(organize_response)
financial_plan_graph.add_edge(START, "entry_point")
financial_plan_graph.add_conditional_edges("entry_point", decision_node)
financial_plan_graph.add_edge("context_financial_identification", "entry_point")
financial_plan_graph.add_edge("financial_calculator", "organize_response")
financial_plan_graph.add_edge("organize_response", END)
