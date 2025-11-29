from operator import add
from typing import Annotated, TypedDict
from typing import List


class MainOrchestratorState(TypedDict):
    messages: Annotated[list, add]
    summary: str
    current_action: str
    secure_input: bool
    secure_output: bool
    message_to_analyze: str
    response: str
    user_needs: Annotated[List[str], add]


class CarCatalogState(TypedDict):
    message_to_analyze: str
    current_action: str
    user_needs: Annotated[List[str], add]
    car_findings: List[str]
    response: str
