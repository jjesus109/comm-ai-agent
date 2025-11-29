import uuid
from typing import Optional

from pydantic import BaseModel


class MessageModel(BaseModel):
    conversation_id: Optional[uuid.UUID] = None
    message: str


class ResponseModel(BaseModel):
    response: str
    conversation_id: uuid.UUID
