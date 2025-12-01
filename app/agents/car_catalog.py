import os
import re
import json
import logging
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END
import psycopg

from app.config import Configuration
from app.depends import get_car_catalog_db_conn
from app.agents.models import MainOrchestratorState

log = logging.getLogger(__name__)
conf = Configuration()

DATA_PATH = os.path.join(os.getcwd(), "data")
CAR_CATALOG_PATH = os.path.join(DATA_PATH, "car_catalog.csv")
NODE_NAMES = [
    "context_car_identification",
    "clear_car_context",
    "text_to_sql",
]
SUB_NODES = Literal[
    "context_car_identification",
    "clear_car_context",
    "text_to_sql",
    "__end__",  # Instead of use END from lang graph, to avoid mypy error
]

car_catalog_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", temperature=conf.temperature
)


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
        return {"error": "No user needs found", "current_action": "text_to_sql"}
    user_needs = state["user_needs"]
    SYSTEM_PROMPT = """
    Eres un experto en **Generación de Consultas SQL (SQL Generator)**. Tu única tarea es crear una consulta SQL estándar (SQL ANSI, compatible con PostgreSQL/MySQL) para buscar vehículos.

    ### 🎯 Tarea y Requisitos:

    1.  **Genera una consulta SELECT** que devuelva todas las columnas del catálogo de autos.
    2.  **Aplica filtros** (`WHERE`) basándote en las necesidades de búsqueda proporcionadas por el usuario.
    3.  **OBLIGATORIO: Implementa Búsqueda *Case-Insensitive***. Para todas las columnas de texto (`make`, `model`, `version`), debes utilizar la función `LOWER()` sobre la columna de la tabla y comparar contra el valor de búsqueda convertido a minúsculas (`LOWER('Valor de Búsqueda')`). Esto asegura que la búsqueda no dependa de si el usuario escribió con mayúsculas o minúsculas.
    4.  **Columnas `bluetooth` y `car_play`**: Estas columnas son de tipo **STRING** en la base de datos. Cuando el usuario requiera estas características (cuando `bluetooth` o `car_play` sean `True` en las necesidades del usuario), debes filtrar buscando vehículos donde el valor de la columna sea igual a `'Si'` (string) o `NULL`. Utiliza la condición: `(bluetooth = 'Si' OR bluetooth IS NULL)` para bluetooth y `(car_play = 'Si' OR car_play IS NULL)` para car_play.
    5.  **No inventes información.** Solo usa los campos y valores proporcionados.

    ### 📋 Estructura de la Base de Datos:

    * **Tabla:** `cars`
    * **Columnas:** 
    - `stock_id` (string)
    - `km` (numérico)
    - `price` (numérico)
    - `make` (string)
    - `model` (string)
    - `year` (numérico)
    - `version` (string)
    - `bluetooth` (string) - Valores posibles: `'Si'` o `NULL`
    - `largo` (numérico)
    - `ancho` (numérico)
    - `altura` (numérico)
    - `car_play` (string) - Valores posibles: `'Si'` o `NULL`

    ### 📝 Formato de Salida Requerido:

    Tu respuesta **debe ser únicamente la consulta SQL completa**, sin texto introductorio, explicaciones o código adicional.

    **Ejemplo de Búsqueda *Case-Insensitive* para Marca y Modelo (asume que el usuario busca 'Toyota' y 'Corolla'):**
    SELECT stock_id, km, price, make, model, year, version, bluetooth, largo, ancho, altura, car_play
    FROM cars
    WHERE LOWER(make) = LOWER('Toyota') AND LOWER(model) = LOWER('Corolla');**Ejemplo de Búsqueda con Bluetooth y CarPlay (asume que el usuario requiere bluetooth y car_play):**
    SELECT stock_id, km, price, make, model, year, version, bluetooth, largo, ancho, altura, car_play
    FROM cars
    WHERE (bluetooth = 'Si' OR bluetooth IS NULL) AND (car_play = 'Si' OR car_play IS NULL);

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
    if not state.get("query"):
        return {
            "user_response": "Primero, empezemos con definir algunas caracteristicas del auto que deseas buscar"
        }
    query = state["query"]
    try:
        # Conexión a la base de datos
        conn = get_car_catalog_db_conn()
        cursor = conn.cursor()
        cursor.execute(query)
        column_names = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        results_list_of_dicts = []
        for row in rows:
            results_list_of_dicts.append(dict(zip(column_names, row)))
        formatted_results = str(results_list_of_dicts[:5])
        return {"car_findings": formatted_results}

    except psycopg.errors.Error as e:
        error_message = f"Error de PostgreSQL: {e}"
        log.error(f"Database error: {error_message}")
        return {"error": error_message}
    finally:
        if conn:
            conn.close()


def organize_response(state: MainOrchestratorState) -> dict:
    if not state.get("car_findings"):
        return {
            "error": "No cars found",
            "user_response": "Primero, empezemos con definir algunas caracteristicas del auto que deseas buscar",
        }
    car_findings = state["car_findings"]
    SYSTEM_PROMPT = """
    Eres un vendedor experto en autos, crea una resumen muy llamativo y atractivo para el usuario, que incluya los autos encontrados y el match de las necesidades del usuario con los autos encontrados.
    basado en los hallazagos encontrados por tu asistente de busqueda, recuerda, debes crear un mensaje muy atractivo y llamativo para el usuario, que incluya los autos encontrados y el match de las necesidades del usuario con los autos encontrados.
    Recuerda incluir los datos de los autos encontrados, como marca, modelo, año, precio, kilometraje, etc, y toda la infomracion relevante de los autos a ofrecer al cliente.
    Limita la respuesta a 3 autos encontrados.
    En caso de que no haya autos encontrados, responde con un mensaje de que no se encontraron autos que se ajusten a las necesidades del usuario.
    Recuerda que debes mostrar al usuario las caracteristicas que definio de una manera amigable y clara.
    Estas son las caracteristicas que busco el usuario: {user_needs}
    Ejemplo de output:
    "Estos son los autos encontrados: 'Lista de autos encontrados'"
    "No se encontraron autos que se ajusten a tus necesidades, prueba modificando las caracteristicas del auto que deseas buscar"
    
    """
    USER_PROMPT = f"Estos son los autos encontrados: {car_findings}"
    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(user_needs=state.get("user_needs"))),
        HumanMessage(content=USER_PROMPT),
    ]
    response = car_catalog_llm.invoke(messages)
    log.debug(f"Este es el resumen de los autos encontrados: {response.content}")
    return {"user_response": response.content, "messages": [response]}


def clear_car_context(state: MainOrchestratorState) -> dict:
    return {
        "user_needs": {},
        "current_action": "clear_car_context",
        "user_response": "Perfecto, ¡Iniciaremos una nueva busqueda de tu auto ideal!",
        "messages": [
            AIMessage(
                content="Perfecto, ¡Iniciaremos una nueva busqueda de tu auto ideal!"
            )
        ],
    }


def router_node(state: MainOrchestratorState) -> SUB_NODES:
    current_action = state["current_action"]
    # To avoid infinite loop, if the current action is a sub node, return END
    if current_action in NODE_NAMES:
        return END
    INTENTION_PROMPT = """## 🤖 Tarea: Clasificador de Intención de Búsqueda de Vehículos
    Eres un clasificador de intención experto y tu única función es determinar el **objetivo principal** del último mensaje del usuario en un contexto de búsqueda de vehículos.

    ### 📝 CONTEXTO DE ENTRADA:
    El sistema te proporciona el contexto actual de la búsqueda (dentro de <car_characteristics>), pero la decisión de acción debe basarse primariamente en el **nuevo mensaje del usuario y el resumen de la conversacion hasta el momento**.

    <car_characteristics>
    {car_characteristics}
    </car_characteristics>

    ### 💡 INSTRUCCIONES CLAVE Y REQUISITOS:
    1.  **Analiza la Intención:** Clasifica el nuevo mensaje del usuario en una de las cuatro categorías definidas.
    2.  **Respuesta OBLIGATORIA:** Responde **únicamente** con el nombre de la acción (e.g., "select_car"). No incluyas explicaciones, encabezados, o texto adicional.

    ### ⚙️ POSIBLES ACCIONES (ESCOGE SÓLO UNA):

    | Acción | Definición y Lógica | Ejemplos de Entrada |
    | :--- | :--- | :--- |
    | **"select_car"** | El usuario ha **elegido un vehículo específico** (por marca, modelo y/o año) con la intención de **obtener detalles adicionales, iniciar cotización/financiamiento, o reservarlo**. | "Me gusta la Chevrolet Cheyenne 2019", "Quiero el Toyota 2018", "Me interesa el audio a3 2010", "Dame más detalles de ese Mercedes Benz c200 2021". |
    | **"context_car_identification"** | El usuario está **definiendo, agregando o modificando criterios de búsqueda** (filtros, rangos, características). El sistema debe actualizar el contexto de búsqueda con esta información (incluso si la lista de características actual está vacía). | "Quiero un auto de la marca Toyota","Un mercedez","una rav4" , "un mustang","Debe ser modelo Corolla, año 2024", "Que tenga bluetooth y CarPlay", "Busco un sedan más barato que 200,000 pesos". |
    | **"text_to_sql"** | El usuario solicita **ejecutar la búsqueda actual** (basada en el contexto existente) y **ver los resultados** encontrados por el sistema. | "Quiero ver los resultados de la busqueda de vehiculos", "Muestra que autos encontraste", "¿Qué opciones tienes?", "A ver que autos encontraste", "Que vehiculos tienes con esas caracteristicas". |
    | **"clear_car_context"** | El usuario pide **borrar** la búsqueda actual y **comenzar de cero** con un nuevo set de filtros. | "Quiero buscar un nuevo vehiculo", "Probemos con otras caracteristicas", "Borra mi búsqueda actual", "Empecemos de nuevo". |

    ---
    **MENSAJE DEL USUARIO A CLASIFICAR:**
    """
    system_message = f"""Este es el resumen de la conversacion hasta el momento: {state.get("summary", "")} \n\n
        {INTENTION_PROMPT.format(car_characteristics=state.get("user_needs"))}
    """
    messages = [SystemMessage(content=system_message)] + state["messages"]
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
    response = state.get("user_response", "")
    return {"response": response}


# Define a new graph
car_catalog_graph = StateGraph(MainOrchestratorState)
car_catalog_graph.add_node(orchestrator_node)
car_catalog_graph.add_node(context_car_identification)
car_catalog_graph.add_node(search_cars)
car_catalog_graph.add_node(organize_response)
car_catalog_graph.add_node(clear_car_context)
car_catalog_graph.add_node(text_to_sql)
car_catalog_graph.add_edge(START, "orchestrator_node")
car_catalog_graph.add_conditional_edges("orchestrator_node", router_node)
car_catalog_graph.add_edge("clear_car_context", "orchestrator_node")
car_catalog_graph.add_edge("text_to_sql", "search_cars")
car_catalog_graph.add_edge("search_cars", "organize_response")
car_catalog_graph.add_edge("organize_response", "orchestrator_node")
car_catalog_graph.add_edge("context_car_identification", "orchestrator_node")
car_catalog_graph.add_edge("orchestrator_node", END)
