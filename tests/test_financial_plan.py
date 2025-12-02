"""Tests for the financial plan agent."""

import json
from unittest.mock import MagicMock, patch

from langgraph.graph import END
from langchain_core.messages import HumanMessage

from app.agents.financial_plan import (
    context_financial_identification,
    financial_calculator,
    organize_response,
    select_car,
    router_node,
    entry_point,
)


class TestContextFinancialIdentification:
    """Test cases for context_financial_identification function."""

    @patch("app.agents.financial_plan.financial_plan_agent")
    def test_context_financial_identification_extracts_years(
        self, mock_llm: MagicMock, mock_state: MagicMock
    ) -> None:
        """Test that financial identification extracts years correctly."""
        response_json = {
            "years": 5,
            "user_response": "¿De cuánto sería tu pago inicial?",
        }

        mock_response = MagicMock()
        mock_response.content = json.dumps(response_json)
        mock_llm.invoke.return_value = mock_response

        state = mock_state
        state["messages"] = [HumanMessage(content="I want to finance for 5 years")]
        state["message_to_analyze"] = "I want to finance for 5 years"
        state["selected_car"] = {
            "brand": "Toyota",
            "model": "Corolla",
            "year": 2020,
            "price": 250000.0,
            "stock_id": "ST001",
        }

        result = context_financial_identification(state)

        assert result["years"] == 5
        assert "user_response" in result
        assert result["current_action"] == "context_financial_identification"

    @patch("app.agents.financial_plan.financial_plan_agent")
    def test_context_financial_identification_extracts_down_payment(
        self, mock_llm: MagicMock, mock_state: MagicMock
    ) -> None:
        """Test that financial identification extracts down payment correctly."""
        response_json = {
            "down_payment": 10000.00,
            "user_response": "Perfecto, continuemos con el monto de tu pago inicial",
        }

        mock_response = MagicMock()
        mock_response.content = json.dumps(response_json)
        mock_llm.invoke.return_value = mock_response

        state = mock_state
        state["messages"] = [HumanMessage(content="My down payment is 10000 pesos")]
        state["message_to_analyze"] = "My down payment is 10000 pesos"
        state["selected_car"] = {
            "brand": "Toyota",
            "model": "Corolla",
            "year": 2020,
            "price": 250000.0,
            "stock_id": "ST001",
        }
        state["years"] = 3

        result = context_financial_identification(state)

        assert result["down_payment"] == 10000.00
        assert "user_response" in result.keys()

    @patch("app.agents.financial_plan.financial_plan_agent")
    def test_context_financial_identification_complete_info(
        self, mock_llm: MagicMock, mock_state: MagicMock
    ) -> None:
        """Test that financial identification handles complete information."""
        response_json = {
            "years": 3,
            "down_payment": 10000.00,
        }

        mock_response = MagicMock()
        mock_response.content = json.dumps(response_json)
        mock_llm.invoke.return_value = mock_response

        state = mock_state
        state["messages"] = [HumanMessage(content="3 years and 10000 down payment")]
        state["message_to_analyze"] = "3 years and 10000 down payment"
        state["selected_car"] = {
            "brand": "Toyota",
            "model": "Camry",
            "year": 2029,
            "price": 350000.0,
            "stock_id": "ST0201",
        }

        result = context_financial_identification(state)

        assert result["years"] == 3
        assert result["down_payment"] == 10000.00
        # When all info is present, user_response may not be included
        assert "user_response" not in result or result.get("user_response") is None


class TestFinancialCalculator:
    """Test cases for financial_calculator function."""

    def test_financial_calculator_calculates_monthly_payment(
        self, mock_state: MagicMock
    ) -> None:
        """Test that monthly payment is calculated correctly."""
        state = mock_state
        state["messages"] = []
        state["message_to_analyze"] = ""
        state["selected_car"] = {
            "brand": "Toyota",
            "model": "Corolla",
            "year": 2020,
            "price": 250000.0,
            "stock_id": "ST001",
        }
        state["years"] = "3"
        state["down_payment"] = "50000"

        result = financial_calculator(state)

        assert "monthly_payment" in result
        assert result["monthly_payment"] > 0
        # Verify calculation: (250000 - 50000) = 200000 to finance
        # At 10% annual rate (0.1/12 monthly) for 36 months
        # Expected monthly payment should be around 6449.23
        assert 6000 < result["monthly_payment"] < 7000
        assert result["current_action"] == "financial_calculator"


class TestOrganizeResponse:
    """Test cases for organize_response function."""

    @patch("app.agents.financial_plan.financial_plan_agent")
    def test_organize_response_creates_message(
        self, mock_llm: MagicMock, mock_state: MagicMock
    ) -> None:
        """Test that response is organized correctly."""
        mock_response = MagicMock()
        mock_response.content = "Your monthly payment will be $6449.23"
        mock_llm.invoke.return_value = mock_response

        state = mock_state
        state["messages"] = []
        state["message_to_analyze"] = ""
        state["selected_car"] = {
            "brand": "Toyota",
            "model": "Corolla",
            "year": 2020,
            "price": 250000.0,
            "stock_id": "ST001",
        }
        state["years"] = "3"
        state["down_payment"] = "50000"
        state["monthly_payment"] = 6449.23

        result = organize_response(state)

        assert result["response"] == "Your monthly payment will be $6449.23"
        assert result["current_action"] == "organize_response"
        assert "messages" in result
        mock_llm.invoke.assert_called_once()


class TestSelectCar:
    """Test cases for select_car function."""

    @patch("app.agents.financial_plan.financial_plan_agent")
    def test_select_car_success(
        self, mock_llm: MagicMock, mock_state: MagicMock
    ) -> None:
        """Test that car selection works correctly."""
        selected_car_json = {
            "brand": "Toyota",
            "model": "Corolla",
            "year": 2020,
            "price": 250000.0,
            "stock_id": "ST001",
        }

        mock_response = MagicMock()
        mock_response.content = json.dumps(selected_car_json)
        mock_llm.invoke.return_value = mock_response

        state = mock_state
        state["messages"] = [HumanMessage(content="I want the Toyota Corolla 2020")]
        state["message_to_analyze"] = "I want the Toyota Corolla 2020"
        state["car_findings"] = (
            "[{'make': 'Toyota', 'model': 'Corolla', 'year': 2020, 'price': 250000.0, 'stock_id': 'ST001'}]"
        )

        result = select_car(state)

        assert "selected_car" in result
        assert result["selected_car"]["brand"] == "Toyota"
        assert result["selected_car"]["model"] == "Corolla"
        assert result["selected_car"]["year"] == 2020
        assert result["selected_car"]["price"] == 250000.0
        assert result["current_action"] == "select_car"

    @patch("app.agents.financial_plan.financial_plan_agent")
    def test_select_car_no_car_findings(
        self, mock_llm: MagicMock, mock_state: MagicMock
    ) -> None:
        """Test that error is returned when no car findings are present."""
        state = mock_state
        state["messages"] = []
        state["message_to_analyze"] = ""

        result = select_car(state)

        assert "user_response" in result
        assert "No hemos buscado" in result["user_response"]
        assert result["current_action"] == "select_car"
        mock_llm.invoke.assert_not_called()


class TestRouterNode:
    """Test cases for router_node function."""

    @patch("app.agents.financial_plan.financial_plan_agent")
    def test_router_node_select_car(
        self, mock_llm: MagicMock, mock_state: MagicMock
    ) -> None:
        """Test routing to select_car node."""
        mock_response = MagicMock()
        mock_response.content = "select_car"
        mock_llm.invoke.return_value = mock_response

        state = mock_state
        state["messages"] = [HumanMessage(content="I want to finance this car")]
        state["current_action"] = "entry_point"
        state["message_to_analyze"] = ""

        result = router_node(state)

        assert result == "select_car"

    @patch("app.agents.financial_plan.financial_plan_agent")
    def test_router_node_context_financial_identification(
        self, mock_llm: MagicMock, mock_state: MagicMock
    ) -> None:
        """Test routing to context_financial_identification node."""
        mock_response = MagicMock()
        mock_response.content = "context_financial_identification"
        mock_llm.invoke.return_value = mock_response

        state = mock_state
        state["messages"] = [HumanMessage(content="My down payment is 50000")]
        state["current_action"] = "entry_point"
        state["message_to_analyze"] = ""
        state["selected_car"] = {
            "brand": "Toyota",
            "model": "Corolla",
            "year": 2020,
            "price": 250000.0,
            "stock_id": "ST001",
        }

        result = router_node(state)

        assert result == "context_financial_identification"

    @patch("app.agents.financial_plan.financial_plan_agent")
    def test_router_node_financial_calculator(
        self, mock_llm: MagicMock, mock_state: MagicMock
    ) -> None:
        """Test routing to financial_calculator node."""
        mock_response = MagicMock()
        mock_response.content = "financial_calculator"
        mock_llm.invoke.return_value = mock_response

        state = mock_state
        state["messages"] = [HumanMessage(content="Calculate the payment")]
        state["current_action"] = "entry_point"
        state["message_to_analyze"] = ""
        state["selected_car"] = {
            "brand": "Toyota",
            "model": "Corolla",
            "year": 2020,
            "price": 250000.0,
            "stock_id": "ST001",
        }
        state["years"] = "3"
        state["down_payment"] = "50000"

        result = router_node(state)

        assert result == "financial_calculator"

    def test_router_node_returns_end_when_current_action_in_nodes(
        self, mock_state: MagicMock
    ) -> None:
        """Test that router returns END when current_action is in node names."""
        state = mock_state
        state["messages"] = []
        state["current_action"] = "context_financial_identification"
        state["message_to_analyze"] = ""

        result = router_node(state)

        assert result == END


class TestEntryPoint:
    """Test cases for entry_point function."""

    def test_entry_point_returns_response(self, mock_state: MagicMock) -> None:
        """Test that entry_point returns user_response."""
        state = mock_state
        state["messages"] = []
        state["message_to_analyze"] = ""
        state["user_response"] = "Please provide financing details"

        result = entry_point(state)

        assert result["response"] == "Please provide financing details"
