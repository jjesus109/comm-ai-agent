import logging
from uuid import uuid4

import fastapi
from langchain_core.messages import HumanMessage
import uvicorn


from app.models import MessageModel, ResponseModel

from app.utils import configure_logger
from app.config import Configuration
from app.agents.orchestrator import orchestrator_graph


responses = {
    "400": {"description": "Problems with request"},
    "404": {"description": "Conversation not found"},
    "409": {"description": "Conflict the message received from the user"},
    "500": {"description": "Problems with other services"},
}


log = logging.getLogger(__name__)
app = fastapi.FastAPI(
    description="Commercial AI Agent",
    version="0.0.1",
    openapi_url="/api/openapi.json",
)


@app.post("/api/chat/", response_model=ResponseModel, responses=responses)
async def send_messages(message: MessageModel) -> ResponseModel:
    # Create a thread
    conversation_id = message.conversation_id
    if conversation_id is None:
        conversation_id = uuid4()
    config = {"configurable": {"thread_id": conversation_id}}
    input_message = HumanMessage(content=message.message)
    responses = orchestrator_graph.invoke({"messages": [input_message]}, config)
    final_response = responses["messages"]
    print(f"final_response: {final_response}")
    return ResponseModel(
        response=final_response.__str__(), conversation_id=conversation_id
    )


if __name__ == "__main__":
    conf = Configuration()
    configure_logger()
    config = uvicorn.Config(app="app.main:app", port=conf.port, host=conf.host)
    server = uvicorn.Server(config)
    server.run()
