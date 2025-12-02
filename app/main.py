import logging


import uvicorn
from twilio.rest import Client
from fastapi import FastAPI, Request
from app.config import Configuration
from app.utils import configure_logger
from langchain_core.messages import HumanMessage
from app.agents.orchestrator import orchestrator_graph


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
app = FastAPI(
    description="Commercial AI Agent",
    version="0.0.1",
    openapi_url="/api/openapi.json",
)


@app.middleware("http")
async def custom_middleware(request: Request, call_next):
    log.info(f"Incoming request: {request.method} {request.url}")
    log.info(f"Form: {request.form()} and headers {request.headers}")
    log.info(f"All request: {request}")
    response = await call_next(request)

    print(f"Outgoing response status: {response.status_code}")
    return response


@app.post("/api/chat/", responses=responses)
async def send_messages(request: Request):
    form_data = await request.form()
    message = form_data.get("Body", "")
    thread_id = form_data.get("WaId", "")
    from_number = form_data.get("From", "")
    to_number = form_data.get("To", "")
    config = {"configurable": {"thread_id": thread_id}}
    input_message = HumanMessage(content=message)
    responses = orchestrator_graph.invoke({"messages": [input_message]}, config)
    final_response = responses["response"]
    log.info(f"final_response: {final_response}")
    client.messages.create(
        from_=to_number,
        body=final_response,
        to=from_number,
    )
    return {"status": "ok"}


if __name__ == "__main__":
    configure_logger()
    config = uvicorn.Config(app="app.main:app", port=conf.port, host=conf.host)
    server = uvicorn.Server(config)
    server.run()
