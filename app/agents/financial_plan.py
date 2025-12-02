import os
import re
import json
import logging
from typing import Literal

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
SUB_NODES = Literal[
    "context_financial_identification", "financial_calculator", "select_car", "__end__"
]
NODE_NAMES = ["context_financial_identification", "financial_calculator", "select_car"]
financial_plan_agent = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", temperature=conf.temperature
)


def context_financial_identification(state: MainOrchestratorState) -> dict:
    """
    Extracts the financial plan details from the user message
    and returns the financial plan details to the state.
    Args:
        state (MainOrchestratorState): The state of the orchestrator.
    Returns:
        dict: financial plan details to include in the state.
    """
    SYSTEM_PROMPT = """
    Eres un experto en **Extracción de Información y Ventas de Automóviles**. Tu tarea es analizar la última intervención del cliente y ayudarle a definir sus necesidades de financiamiento.
    ### Contexto:
    El usuario ha seleccionado un auto y está iniciando el proceso de cotización o financiación.


    ### 🎯 Tareas Clave:
    1.  **Extraer Campos:** Identifica los valores para **Años de financiamiento** y **Pago inicial** a partir de la respuesta del cliente.
    2.  **Validación y Pregunta:** Si un campo es **ausente o ambiguo**, no lo incluyas en el diccionario de salida y genera un **único mensaje** conciso dirigido al cliente para solicitar específicamente la información faltante.

    ### 📋 Reglas de Extracción:
    * **Años de financiamiento (`years`):** Debe ser un número entero (int) que represente la duración del préstamo en años. (Ej. 3, 5, 7).
    * **Pago inicial (`down_payment`):** Debe ser un número decimal (float) que represente el monto total de enganche. (Ej. 10000.00, 5000.50).
    * **NO INVENTES NINGÚN VALOR.** Si la información no está clara, omite el campo en la salida.
    * **Si ya tiene alguno de los valores necesarios, no solicites el mismo valor nuevamente, y confirma al usuario si desea continuar con el proceso para mostrarle su cotizacion en el campo "user_response"*
    * **Si ya tiene todos los valores, es posible que desee cambiar algun valor, confirma regresando el valor y confirmando en un mensaje el valor al usuario.**

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
        
        "user_response": "Para obtener un financiamiento, necesitamos saber en cuantos años deseas pagar el auto y cuánto sería el enganche. ¿Puedes indicarme estos detalles?"
    }
    **Ejemplo 4 (Información completa):**
    ```json
    {
        "years": 3,
        "down_payment": 10000.00,
        "user_response": "Muy bien!, ya tengo los datos necesarios para calcular tu cotizacion. Continuaremos con el proceso"
    }
    """
    financial_plan_values = {
        "years": state.get("years"),
        "down_payment": state.get("down_payment"),
    }
    USER_PROMPT = f"""Extrae las necesidades de financiamiento del usuario de este mensaje: {state["message_to_analyze"]}
    Aqui esta el listado de valores necesarios para el proceso de financiación:
    <financial_plan_values>
    {financial_plan_values}
    </financial_plan_values>
    """
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
    Calculates the financial plan for a given car based on price, interest rate, months and down payment
    Args:
        state (MainOrchestratorState): The state of the orchestrator.
    Returns:
        dict: financial plan details to include in the state.
    """
    months = float(state["years"]) * 12
    down_payment = float(state["down_payment"])
    value_to_finance = float(state["selected_car"]["price"]) - down_payment
    monthly_rate = float(ANNUAL_RATE / 12)
    monthly_payment = (
        value_to_finance * monthly_rate * (1 + monthly_rate) ** months
    ) / ((1 + monthly_rate) ** months - 1)

    return {
        "monthly_payment": monthly_payment,
        "current_action": "financial_calculator",
    }


def organize_response(state: MainOrchestratorState) -> dict:
    """
    Organizes the response to the user based on the financial plan details.
    Args:
        state (MainOrchestratorState): The state of the orchestrator.
    Returns:
        dict: response to the user to include in the state.
    """
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


def select_car(state: MainOrchestratorState) -> dict:
    """
    Selects the car based on the car findings and the user message.
    Args:
        state (MainOrchestratorState): The state of the orchestrator.
    Returns:
        dict: selected car and current action to include in the state.
    """
    if not state.get("car_findings"):
        return {
            "user_response": "No hemos buscado un auto todavia, por favor, dime alguna caracteristica del auto que deseas buscar.",
            "current_action": "select_car",
        }
    car_findings = state["car_findings"]
    selected_car = state["message_to_analyze"]
    SYSTEM_PROMPT = """
    Basado en la seleccion del usuario, elie el auto al que se refiere el usuario.
    Si el auto seleccionado no existe, devuelve un mensaje piendo al usuario que eliga un auto de los encontrados.
    El output esperado es un json con los datos del auto seleccionado.
    No inventes ninguna informacion.
    No respondas preguntas que no sean relacionadas con el auto seleccionado.
    Planifica y luego responde a la pregunta.
    Ejemplo de output:
    ```json
        {
            "brand": str,
            "model": str,
            "year": int,
            "price": float,
            "stock_id": str,
        }
    ```
    """
    USER_PROMPT = f"""
    Esta es la lista de los autos ofrecidos al usuario: 
    ```json
    {car_findings}
    ```
    Esta es la seleccion del usuario: {selected_car}
    """
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=USER_PROMPT),
    ]
    response = financial_plan_agent.invoke(messages)

    pattern = re.compile(r"\{(.*)\}", re.DOTALL)

    # Buscar y extraer el contenido
    match = pattern.search(response.content)
    selected_car = {}  # type: ignore
    if match:
        contenido_json = "{" + match.group(1).strip() + "}"
        selected_car = json.loads(contenido_json)
    log.debug(f"Este es el auto seleccionado: {selected_car}")
    return {"selected_car": selected_car, "current_action": "select_car"}


def router_node(state: MainOrchestratorState) -> SUB_NODES:
    """
    Decides the next action to take based on the current action and the user needs.
    This node is used to decide the next action to take based on the current action and the user needs.
    Args:
        state (MainOrchestratorState): The state of the orchestrator.
    Returns:
        SUB_NODES: The next action to take.
    """
    current_action = state["current_action"]
    # To avoid infinite loop, if the current action is a sub node, return END
    if current_action in NODE_NAMES:
        return END
    INTENTION_PROMPT = """## 🤖 Tarea: Enrutador de Etapa de Compra
    Eres un enrutador de LangGraph experto en el proceso de compra y financiación de vehículos. Tu única función es determinar la **etapa exacta** de la cotización en la que se encuentra el usuario, basándote en su último mensaje.

    ### 📝 CONTEXTO:
    Se asume que el usuario ya ha seleccionado un vehículo y está iniciando el proceso de cotización o financiación.
    Aqui esta el listado de valores necesarios para el proceso de financiación:
    <financial_plan_values>
    {financial_plan_values}
    </financial_plan_values>
    

    ### 💡 INSTRUCCIONES CLAVE Y REQUISITOS:
    1.  **Analiza la Intención:** Clasifica el mensaje del usuario en una de las tres acciones específicas del proceso de financiación.
    2.  **Respuesta OBLIGATORIA:** Responde **únicamente** con el nombre de la acción (el nombre de la ruta o nodo siguiente), sin explicaciones, encabezados, o texto adicional.
    3.  **Si existen todos los campos necesarios, elige la ruta "financial_calculator", en caso contrario, elige la ruta "context_financial_identification".**

    ### ⚙️ POSIBLES RUTAS (ESCOGE SÓLO UNA):

    | Ruta | Definición y Propósito | Ejemplos de Entrada |
    | :--- | :--- | :--- |
    | **"select_car"** | El usuario confirma el vehículo elegido o solicita el **siguiente paso lógico** (cotización inicial) **sin dar cifras** de enganche o plazos. Actúa como el *punto de entrada* al subgrafo de financiación. | "Quiero cotizar este auto.", "Procedamos con el [Nombre de Auto].", "Dime más sobre la financiación.", "¿Cómo puedo comprarlo?". |
    | **"context_financial_identification"** | El usuario **proporciona datos numéricos** esenciales para el cálculo, como el **monto de enganche (down payment)**, el **plazo (meses/años)** o **actualiza** alguno de estos valores. | "Mi enganche será de 50,000 pesos.", "Quiero pagarlo a 36 meses.", "Dime cuánto es la mensualidad si doy $100,000 de enganche y 48 meses.", "Cambia mi plazo a 60 meses.". |
    | **"financial_calculator"** | El usuario ha terminado de dar los datos o simplemente **solicita ver los planes/cálculos finales** disponibles para el vehículo seleccionado. Esta es la instrucción para ejecutar la lógica de cálculo. | "Calcula la mensualidad final ahora.", "¿Cuáles son los planes de financiamiento disponibles?", "Ya tengo todos los datos, dame el plan de pago.", "Dame la cotización completa.". |

    ---

    **MENSAJE DEL USUARIO A CLASIFICAR:**
    """
    financial_plan_values = {
        "years": state.get("years"),
        "down_payment": state.get("down_payment"),
    }
    system_message = f"""Este es el resumen de la conversacion hasta el momento: {state.get("summary", "")} \n\n
        {INTENTION_PROMPT.format(car_characteristics=state.get("user_needs"), financial_plan_values=financial_plan_values)}
    """
    messages = [SystemMessage(content=system_message)] + state["messages"]
    response = financial_plan_agent.invoke(messages)
    selected_action = response.content
    log.debug(
        f"Este es la seleccion de financial plan elegida por el enrutador: {selected_action}"
    )
    return selected_action


def entry_point(state: MainOrchestratorState) -> dict:
    """
    Entry point of the financial plan graph.
    This node is used to return the response to the user.
    Args:
        state (MainOrchestratorState): The state of the orchestrator.
    Returns:
        dict: response to the user to include in the state.
    """
    response = state.get("user_response")
    return {"response": response}


# Define a new graph
financial_plan_graph = StateGraph(MainOrchestratorState)
financial_plan_graph.add_node(entry_point)
financial_plan_graph.add_node(context_financial_identification)
financial_plan_graph.add_node(financial_calculator)
financial_plan_graph.add_node(organize_response)
financial_plan_graph.add_node(select_car)
financial_plan_graph.add_edge(START, "entry_point")
financial_plan_graph.add_conditional_edges("entry_point", router_node)
financial_plan_graph.add_edge("select_car", "context_financial_identification")
financial_plan_graph.add_edge("context_financial_identification", "entry_point")
financial_plan_graph.add_edge("financial_calculator", "organize_response")
financial_plan_graph.add_edge("organize_response", END)
