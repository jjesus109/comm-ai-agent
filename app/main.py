import logging
from typing import Annotated

import uvicorn
from twilio.rest import Client
from fastapi import FastAPI, Request, Form
from starlette.middleware.base import BaseHTTPMiddleware
from app.config import Configuration
from app.utils import configure_logger, generate_correlation_id, set_correlation_id
from langchain_core.messages import HumanMessage
from app.agents.graph_definition import orchestrator_graph


conf = Configuration()
account_sid = conf.twilio_account_sid
auth_token = conf.twilio_auth_token
client = Client(account_sid, auth_token)


responses = {
    "400": {"description": "Problems with request"},
    "404": {"description": "Conversation not found"},
    "409": {"description": "Conflict the message received from the user"},
    "500": {"description": "Problems with other services"},
}


log = logging.getLogger(__name__)


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware that generates and sets a unique correlation ID for each request.
    """

    async def dispatch(self, request: Request, call_next):
        correlation_id = generate_correlation_id()
        set_correlation_id(correlation_id)
        response = await call_next(request)
        return response


app = FastAPI(
    title="Commercial AI Agent",
    description="Commercial AI Agent that respond user doubts about Kavak company and its products",
    version="0.0.1",
    openapi_url="/api/openapi.json",
)

app.add_middleware(CorrelationIDMiddleware)


@app.post("/api/chat/", responses=responses)
async def send_messages(
    Body: Annotated[str, Form()],
    WaId: Annotated[str, Form()],
    From: Annotated[str, Form()],
    To: Annotated[str, Form()],
):
    # Manage all logic for twilio requests, it could be in another service,
    # but to simplyfy the code, we will keep it here.
    message = Body
    thread_id = WaId
    from_number = From
    to_number = To
    config = {"configurable": {"thread_id": thread_id}}
    input_message = HumanMessage(content=message)
    try:
        responses = orchestrator_graph.invoke({"messages": [input_message]}, config)
        final_response = responses["response"]
    except Exception as e:
        final_response = "Lo siento, no puedo procesar tu mensaje en este momento. Por favor, intenta nuevamente."
        log.error(f"Error invoking orchestrator graph: {e}")
    log.info(f"final_response: {final_response}")
    try:
        client.messages.create(
            from_=to_number,
            body=final_response,
            to=from_number,
        )
    except Exception as e:
        log.error(f"Error sending message to twilio: {e}")
        return {"status": "error"}
    return {"status": "ok"}


if __name__ == "__main__":
    configure_logger()
    config = uvicorn.Config(app="app.main:app", port=conf.port, host=conf.host)
    server = uvicorn.Server(config)
    server.run()
