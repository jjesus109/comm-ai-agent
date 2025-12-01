from typing_extensions import Annotated, Optional, TypedDict
from typing import List

from langgraph.graph.message import add_messages


class SelectedCar(TypedDict):
    brand: str
    model: str
    year: int
    price: float
    stock_id: str


class UserNeeds(TypedDict):
    marca: Optional[str]
    kilometraje: Optional[int]
    precio_minimo: Optional[float]
    precio_maximo: Optional[float]
    modelo: Optional[str]
    year_minimo: Optional[int]
    year_maximo: Optional[int]
    version: Optional[str]
    bluetooth: Optional[bool]
    largo: Optional[float]
    ancho: Optional[float]
    alto: Optional[float]
    car_play: Optional[bool]


class MainOrchestratorState(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str
    current_action: str
    secure_input: bool
    secure_output: bool
    message_to_analyze: str
    response: str
    selected_car: SelectedCar
    price: float
    user_needs: UserNeeds
    query: str
    errors: str
    car_findings: List[dict]
    years: str
    down_payment: str
    monthly_payment: float
    user_response: str
