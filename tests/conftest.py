"""Shared fixtures and mocks for all tests."""

import os

import pytest

from langchain_core.messages import HumanMessage

from app.agents.models import MainOrchestratorState

os.environ["GOOGLE_API_KEY"] = "test"
os.environ["PORT"] = "1000"
os.environ["HOST"] = "test"
os.environ["LOG_LEVEL"] = "test"
os.environ["DB_HOST"] = "test"
os.environ["DB_PORT"] = "0"
os.environ["DB_USER"] = "test"
os.environ["DB_PASSWORD"] = "test"
os.environ["DB_NAME"] = "test"
os.environ["CATALOG_DB_HOST"] = "test"
os.environ["CATALOG_DB_PORT"] = "0"
os.environ["CATALOG_DB_USER"] = "test"
os.environ["CATALOG_DB_PASSWORD"] = "test"
os.environ["CATALOG_DB_NAME"] = "test"
os.environ["TWILIO_ACCOUNT_SID"] = "0"
os.environ["TWILIO_AUTH_TOKEN"] = "0"


@pytest.fixture(scope="function")
def mock_state() -> MainOrchestratorState:
    """Create a mock state for testing."""
    return {
        "messages": [HumanMessage(content="test message")],
        "summary": "",
        "current_action": "",
        "secure_input": True,
        "secure_output": True,
        "message_to_analyze": "test message",
        "response": "",
        "selected_car": {},  # type: ignore
        "price": 0.0,
        "user_needs": {},  # type: ignore
        "query": "",
        "errors": "",
        "car_findings": [],
        "years": "",
        "down_payment": "",
        "monthly_payment": 0.0,
        "user_response": "",
    }
