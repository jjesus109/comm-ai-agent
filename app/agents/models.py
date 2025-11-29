from operator import add
from typing import Annotated, TypedDict


class MainOrchestratorState(TypedDict):
    messages: Annotated[list, add]
    summary: str
    current_action: str
    secure_input: bool
    secure_output: bool
    message_to_analyze: str
