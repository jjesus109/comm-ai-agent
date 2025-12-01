import os
import re
import json
import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END

from app.config import Configuration
from app.depends import car_catalog_db_conn
from app.agents.models import MainOrchestratorState

log = logging.getLogger(__name__)
conf = Configuration()

DATA_PATH = os.path.join(os.getcwd(), "data")
CAR_CATALOG_PATH = os.path.join(DATA_PATH, "car_catalog.csv")
NODE_NAMES = [
    "select_car",
    "context_car_identification",
    "clear_car_context",
    "text_to_sql",
]
SUB_NODES = Literal[NODE_NAMES]  # type: ignore

car_catalog_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", temperature=conf.temperature
)


def select_car(state: MainOrchestratorState) -> dict:
    if not state.get("car_findings"):
        return {
            "response": "No hemos buscado un auto todavia, por favor, defina algunas caracteristicas del auto que desea buscar y seleccione uno para poderlo ayudar",
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
    response = car_catalog_llm.invoke(messages)

    pattern = re.compile(r"\{(.*)\}", re.DOTALL)

    # Buscar y extraer el contenido
    match = pattern.search(response.content)
    selected_car = {}  # type: ignore
    if match:
        contenido_json = "{" + match.group(1).strip() + "}"
        selected_car = json.loads(contenido_json)
    log.debug(f"Este es el auto seleccionado: {selected_car}")
    return {"selected_car": selected_car, "current_action": "select_car"}


def context_car_identification(state: MainOrchestratorState) -> dict:
    SYSTEM_PROMPT = """
    Eres un experto en **Extracción de Características y Asesoría de Automóviles**. Tu tarea es analizar la última intervención del cliente para extraer sus requisitos específicos de búsqueda y, al finalizar, solicitarle cualquier otro requisito que no haya mencionado.

    ### 🎯 Tareas Clave:
    1.  **Extracción Estricta:** Identifica y extrae **solo** los valores para las características listadas a continuación.
    2.  **Generación de Pregunta Abierta:** Al finalizar la extracción, genera un **mensaje de seguimiento** dirigido al cliente para preguntarle sobre **cualquier otra característica o requisito** que no haya mencionado y que considere importante. Este mensaje debe ir en el campo `user_response`.

    ### 📋 Características a Extraer (y Tipos de Datos):
    | Característica | Clave de Salida | Tipo de Dato | Notas de Extracción |
    | :--- | :--- | :--- | :--- |
    | Marca(s) | `marca` | `List[str]` | Una lista de marcas mencionadas. |
    | Kilometraje | `kilometraje` | `int` | Kilometraje máximo o un rango (extrae el valor singular o el máximo). |
    | Precio | `precio_minimo`, `precio_maximo` | `float` | Extrae el rango de precios si es posible. |
    | Modelo(s) | `modelo` | `List[str]` | Una lista de modelos mencionados. |
    | Año | `year_minimo`, `year_maximo` | `int` | Extrae el rango de años si es posible. |
    | Versión Específica | `version` | `Optional[str]` | Si menciona una versión como "S", "GT", etc. |
    | Requisito Bluetooth | `bluetooth` | `bool` | `True` si lo menciona como deseado. |
    | Largo del Auto | `largo` | `float` | Valor numérico del largo. |
    | Ancho del Auto | `ancho` | `float` | Valor numérico del ancho. |
    | Alto del Auto | `alto` | `float` | Valor numérico del alto. |
    | Requisito CarPlay | `car_play` | `bool` | `True` si lo menciona como deseado. |

    ### 📝 Reglas de Salida:
    * **NO INVENTES NINGUNA INFORMACIÓN O VALOR.**
    * Si el cliente no proporciona información para una necesidad, **OMITE** la clave de salida correspondiente (no uses `null`, `None`, o `0` por defecto).
    * Tu respuesta **debe ser únicamente un objeto JSON**.
    * Si el cliente no tiene ninguna necesidad, debes incluir la clave `"user_response"` con el mensaje de seguimiento para preguntar por OTRAS características de forma amable y resaltar sobre la utilidad de dar la mayor cantidad de caracteristicas posibles.

    ### 🔑 Estructura de Salida Requerida:
    Tu objeto JSON **debe** incluir la clave `"user_response"`.

    ```json
    {
        "marca": List[str],
        "kilometraje": int,
        "precio_minimo": float,
        "precio_maximo": float,
        "modelo": List[str],
        "year_minimo": int,
        "year_maximo": int,
        "version": str, 
        "bluetooth": bool,
        "largo": float,
        "ancho": float,
        "alto": float,
        "car_play": bool,
        "user_response": str // Mensaje de seguimiento para preguntar por OTRAS características.
    }
    """

    USER_PROMPT = f"Extrae las necesidades del usuario de este mensaje: {state['message_to_analyze']}"
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=USER_PROMPT),
    ]
    response = car_catalog_llm.invoke(messages)
    pattern = re.compile(r"\{(.*)\}", re.DOTALL)

    # Buscar y extraer el contenido
    match = pattern.search(response.content)
    user_needs = {}
    if match:
        contenido_json = "{" + match.group(1).strip() + "}"
        user_needs = json.loads(contenido_json)
    # Update new findings in user needs with old user needs
    existing_user_needs = state.get("user_needs")
    updated_user_needs = existing_user_needs.copy() if existing_user_needs else {}  # type: ignore
    for key, value in user_needs.items():
        if value:
            updated_user_needs[key] = value  # type: ignore
    log.debug(f"Este es el contexto del auto identificado: {updated_user_needs}")
    return {
        "user_needs": updated_user_needs,
        "current_action": "context_car_identification",
        "user_response": user_needs.get("user_response"),
    }


def text_to_sql(state: MainOrchestratorState) -> dict:
    if not state.get("user_needs"):
        return {
            "messages": [
                SystemMessage(
                    content="Primero defina algunas caracteristicas del auto que desea buscar"
                )
            ]
        }
    user_needs = state["user_needs"]
    SYSTEM_PROMPT = """
    Eres un experto en SQL, crea una consulta en SQL para buscar los autos que se ajustan a las necesidades del usuario.
    El output esperado es una consulta en SQL.
    No inventes ninguna informacion.
    No respondas preguntas que no sean relacionadas con la consulta en SQL.
    Planifica y luego responde a la pregunta.
    Tienes una tabla llamada "cars" con las siguientes columnas:
    stock_id,km,price,make,model,year,version,bluetooth,largo,ancho,altura,car_play
    Ejemplo de output:
    "SELECT km,price,make,model,year,version,bluetooth,largo,ancho,altura,car_play
    FROM car_catalog WHERE brand = 'Toyota' AND model = 'Corolla' AND year = 2024"
    """
    USER_PROMPT = f"Crea una consulta en pandas para buscar los autos que se ajustan a las necesidades del usuario: {user_needs}"
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=USER_PROMPT),
    ]
    response = car_catalog_llm.invoke(messages)
    pattern = re.compile(r"```sql\n(.*)```", re.DOTALL)
    # Buscar y extraer el contenido
    match = pattern.search(response.content)
    clean_query = match.group(1).strip()  # type: ignore
    log.debug(f"Esta es la consulta en SQL: {clean_query}")
    return {"query": clean_query, "current_action": "text_to_sql"}


def dict_factory(cursor, row):
    fields = [column[0] for column in cursor.description]
    return {key: value for key, value in zip(fields, row)}


def search_cars(state: MainOrchestratorState) -> dict:
    query = state["query"]
    car_catalog_db_conn.row_factory = dict_factory
    try:
        cars_filtered = car_catalog_db_conn.execute(query).fetchall()
        return {"car_findings": cars_filtered}
    except Exception as e:
        return {"error": str(e)}


def organize_response(state: MainOrchestratorState) -> dict:
    car_findings = state["car_findings"]
    SYSTEM_PROMPT = """
    Eres un vendedor experto en autos, crea una resumen muy llamativo y atractivo para el usuario, que incluya los autos encontrados y el match de las necesidades del usuario con los autos encontrados.
    basado en los hallazagos encontrados por tu asistente de busqueda, recuerda, debes crear un mensaje muy atractivo y llamativo para el usuario, que incluya los autos encontrados y el match de las necesidades del usuario con los autos encontrados.
    Recuerda incluir los datos de los autos encontrados, como marca, modelo, año, precio, kilometraje, etc, y toda la infomracion relevante de los autos a ofrecer al cliente.
    Limita la respuesta a 3 autos encontrados.
    """
    USER_PROMPT = f"Estos son los autos encontrados: {car_findings}"
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=USER_PROMPT),
    ]
    response = car_catalog_llm.invoke(messages)
    log.debug(f"Este es el resumen de los autos encontrados: {response.content}")
    return {"response": response.content}


def clear_car_context(state: MainOrchestratorState) -> dict:
    return {"user_needs": {}, "current_action": "clear_car_context"}


def router_node(state: MainOrchestratorState) -> SUB_NODES:
    current_action = state["current_action"]
    # To avoid infinite loop, if the current action is a sub node, return END
    if current_action in NODE_NAMES:
        return END
    user_input = state["message_to_analyze"]
    INTENTION_PROMPT = """Encuentra la decision que un usuario desea realizar basado en las caracteristicas del auto que eligio el usuario
    <car_characteristics>
    {car_characteristics}
    </car_characteristics>
    Si el usuario no tiene ninguna caracteristica, vendra con un lista vacia, por lo que tendra intencion de buscar un vehiculo.
    * Piensa antes de responder
    # REQUISITOS:
    * OBLIGATORIO: Deber elegir alguno de los siguientes escenarios:
        1. Si el usuario selecciona un auto, responde "select_car".
        Ejemplo de entrada:
            * "Me gusta la chevrolet chayenne 2019"
            * "Me interesa el audio a3 2010"
            * "Quiero el toyota 2018"
            * "Quiero la camioneta ford f150 2020"
            * "Me encanta el mercedes benz c200 2021"
        2. Si el usuario da caracteristicas de un vehiculo, responde "context_car_identification".
        Ejemplo de entrada:
            * "Quiero un auto de la marca Toyota"
            * "Quiero un auto de la marca Toyota, modelo Corolla, año 2024, version 1.6, bluetooth, dimensiones 4.5x1.8x1.5, categoria de vehiculo sedan"º
        3. Si el usuario desea comenzar con una nueva busqueda de vehiculos, responde "clear_car_context".
        Ejemplo de entrada:
            *"Quiero buscar un nuevo vehiculo"
            *"Probemos con otras caracteristicas"
        4. Si el usuario desea ver los resultados de la busqueda de vehiculos:
        Ejemplo de entrada:
            * "Quiero ver los resultados de la busqueda de vehiculos"
            * "Muestra que autos encontraste"
            * "A ver que autos encontraste"
            * "Dame todos los autos encontrados"
            * "Que vehiculos tienes con esas caracteristicas"
            responde "text_to_sql".
    * Solo response con una de las cuatro posibles opciones posibles: "select_car", "context_car_identification", "clear_car_context", "text_to_sql"
    
    """
    messages = [
        SystemMessage(
            content=INTENTION_PROMPT.format(car_characteristics=state.get("user_needs"))
        ),
        HumanMessage(content=f"Este es el mensaje del usuario: {user_input}"),
    ]
    response = car_catalog_llm.invoke(messages)
    selected_action = response.content
    log.debug(f"Este es la accion seleccionada: {selected_action}")
    return selected_action


def orchestrator_node(state: MainOrchestratorState) -> dict:
    """
    This function is the entry point of the car catalog graph, all the nodes return to this node to continue the flow
    and respond to the user with the appropriate message.
    According to the user input, it will call the appropriate node to continue the flow.
    And returns the proper message to the user based on the user input.
    """
    response = ""
    if state["current_action"] == "context_car_identification":
        response = state["user_response"]
    elif state["current_action"] == "text_to_sql":
        response = state["response"]
    elif state["current_action"] == "select_car":
        response = f"Genial, Amaras tu {state['selected_car']['brand']} {state['selected_car']['model']} {state['selected_car']['year']} ¿Deseas un plan de financiamiento para este vehiculo?"
    elif state["current_action"] == "clear_car_context":
        response = "Perfecto, ¡Iniciaremos una nueva busqueda de tu auto ideal!"

    return {"response": response}


# Define a new graph
car_catalog_graph = StateGraph(MainOrchestratorState)
car_catalog_graph.add_node(orchestrator_node)
car_catalog_graph.add_node(context_car_identification)
car_catalog_graph.add_node(search_cars)
car_catalog_graph.add_node(organize_response)
car_catalog_graph.add_node(select_car)
car_catalog_graph.add_node(clear_car_context)
car_catalog_graph.add_node(text_to_sql)
car_catalog_graph.add_edge(START, "orchestrator_node")
car_catalog_graph.add_conditional_edges("orchestrator_node", router_node)
car_catalog_graph.add_edge("select_car", "orchestrator_node")
car_catalog_graph.add_edge("clear_car_context", "orchestrator_node")
car_catalog_graph.add_edge("text_to_sql", "search_cars")
car_catalog_graph.add_edge("search_cars", "organize_response")
car_catalog_graph.add_edge("organize_response", "orchestrator_node")
car_catalog_graph.add_edge("context_car_identification", "orchestrator_node")
car_catalog_graph.add_edge("orchestrator_node", END)
