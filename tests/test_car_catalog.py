"""Tests for the car catalog agent."""

import json

from langgraph.graph import END
from unittest.mock import MagicMock, patch, Mock
from langchain_core.messages import HumanMessage, AIMessage

from app.agents.car_catalog import (
    context_car_identification,
    text_to_sql,
    search_cars,
    organize_response,
    clear_car_context,
    router_node,
    orchestrator_node,
)


class TestContextCarIdentification:
    """Test cases for context_car_identification function."""

    @patch("app.agents.car_catalog.car_catalog_llm")
    def test_context_car_identification_extracts_needs(
        self, mock_llm: MagicMock, mock_state: MagicMock
    ) -> None:
        """Test that car identification extracts user needs correctly."""
        user_needs_json = {
            "marca": ["Toyota"],
            "modelo": ["Corolla"],
            "year_minimo": 2020,
            "user_response": "¿Te gustaría algún otro detalle específico?",
        }

        mock_response = MagicMock()
        mock_response.content = json.dumps(user_needs_json)
        mock_llm.invoke.return_value = mock_response

        state = mock_state
        state["messages"] = [HumanMessage(content="I want a Toyota Corolla from 2020")]
        state["message_to_analyze"] = "I want a Toyota Corolla from 2020"

        result = context_car_identification(state)

        assert "user_needs" in result
        assert result["user_needs"]["marca"] == ["Toyota"]
        assert result["user_needs"]["modelo"] == ["Corolla"]
        assert result["user_needs"]["year_minimo"] == 2020
        assert result["current_action"] == "context_car_identification"
        assert "user_response" in result

    @patch("app.agents.car_catalog.car_catalog_llm")
    def test_context_car_identification_updates_existing_needs(
        self, mock_llm: MagicMock, mock_state: MagicMock
    ) -> None:
        """Test that car identification updates existing user needs."""
        new_needs_json = {
            "kilometraje": 50000,
            "user_response": "Perfecto",
        }

        mock_response = MagicMock()
        mock_response.content = json.dumps(new_needs_json)
        mock_llm.invoke.return_value = mock_response

        state = mock_state
        state["messages"] = [HumanMessage(content="Less than 50000 km")]
        state["message_to_analyze"] = "Less than 50000 km"
        state["user_needs"] = {"marca": ["Toyota"], "modelo": ["Corolla"]}

        result = context_car_identification(state)

        # Should merge existing and new needs
        assert result["user_needs"]["marca"] == ["Toyota"]
        assert result["user_needs"]["modelo"] == ["Corolla"]
        assert result["user_needs"]["kilometraje"] == 50000


car_catalog_llm = MagicMock()
car_catalog_llm.invoke.return_value = "dummy test llm value"


class TestTextToSql:
    """Test cases for text_to_sql function."""

    @patch("app.agents.car_catalog.car_catalog_llm")
    def test_text_to_sql_generates_query(
        self, mock_llm: MagicMock, mock_state: MagicMock
    ) -> None:
        """Test that SQL query is generated correctly."""
        sql_query = "SELECT * FROM cars WHERE LOWER(make) = LOWER('Toyota')"
        mock_response = MagicMock()
        mock_response.content = f"```sql\n{sql_query}\n```"
        mock_llm.invoke.return_value = mock_response

        state = mock_state
        state["messages"] = []
        state["message_to_analyze"] = ""
        state["user_needs"] = {"marca": ["Toyota"]}

        result = text_to_sql(state)

        assert result["query"] == sql_query
        assert result["current_action"] == "text_to_sql"

    @patch("app.agents.car_catalog.car_catalog_llm")
    def test_text_to_sql_no_user_needs(
        self, mock_llm: MagicMock, mock_state: MagicMock
    ) -> None:
        """Test that error is returned when no user needs are present."""
        state = mock_state
        state["messages"] = []
        state["message_to_analyze"] = ""

        result = text_to_sql(state)

        assert "error" in result
        assert result["current_action"] == "text_to_sql"
        mock_llm.invoke.assert_not_called()


class TestSearchCars:
    """Test cases for search_cars function."""

    @patch("app.agents.car_catalog.car_catalog_db_conn")
    def test_search_cars_success(
        self, mock_db_conn: MagicMock, mock_state: MagicMock
    ) -> None:
        """Test successful car search."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        # Mock the connection context manager
        mock_conn_context = MagicMock()
        mock_conn_context.__enter__ = Mock(return_value=mock_conn)
        mock_conn_context.__exit__ = Mock(return_value=None)

        # Mock the cursor context manager
        mock_cursor_context = MagicMock()
        mock_cursor_context.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor_context.__exit__ = Mock(return_value=None)

        # Set up the connection pool mock
        mock_db_conn.connection.return_value = mock_conn_context
        mock_conn.cursor.return_value = mock_cursor_context

        # Mock database results
        mock_cursor.description = [
            ("stock_id",),
            ("make",),
            ("model",),
            ("year",),
            ("price",),
        ]
        mock_cursor.fetchall.return_value = [
            ("ST001", "Toyota", "Corolla", 2020, 250000),
            ("ST002", "Toyota", "Camry", 2021, 300000),
        ]

        state = mock_state
        state["messages"] = []
        state["message_to_analyze"] = ""
        state["query"] = "SELECT * FROM cars WHERE make = 'Toyota'"

        result = search_cars(state)

        assert "car_findings" in result
        assert "ST001" in result["car_findings"]
        mock_cursor.execute.assert_called_once_with(state["query"])

    def test_search_cars_no_query(self, mock_state: MagicMock) -> None:
        """Test search_cars when no query is present."""
        state = mock_state
        state["messages"] = []
        state["message_to_analyze"] = ""
        state["query"] = ""

        result = search_cars(state)

        assert "user_response" in result
        assert "Primero" in result["user_response"]

    @patch("app.agents.car_catalog.log")
    @patch("app.agents.car_catalog.car_catalog_db_conn")
    def test_search_cars_database_error(
        self, mock_db_conn: MagicMock, mock_state: MagicMock
    ) -> None:
        """Test handling of database errors."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        # Mock the connection context manager
        mock_conn_context = MagicMock()
        mock_conn_context.__enter__ = Mock(return_value=mock_conn)
        mock_conn_context.__exit__ = Mock(return_value=None)

        # Mock the cursor context manager
        mock_cursor_context = MagicMock()
        mock_cursor_context.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor_context.__exit__ = Mock(return_value=None)

        # Set up the connection pool mock
        mock_db_conn.connection.return_value = mock_conn_context
        mock_conn.cursor.return_value = mock_cursor_context

        import psycopg

        mock_cursor.execute.side_effect = psycopg.errors.Error("Database error")

        state = mock_state
        state["messages"] = []
        state["message_to_analyze"] = ""
        state["query"] = "SELECT * FROM cars"

        result = search_cars(state)

        assert "error" in result


class TestOrganizeResponse:
    """Test cases for organize_response function."""

    @patch("app.agents.car_catalog.car_catalog_llm")
    def test_organize_response_success(
        self, mock_llm: MagicMock, mock_state: MagicMock
    ) -> None:
        """Test that response is organized correctly."""
        mock_response = MagicMock()
        mock_response.content = "Here are the cars found: Toyota Corolla 2020"
        mock_llm.invoke.return_value = mock_response

        state = mock_state
        state["messages"] = []
        state["message_to_analyze"] = ""
        state["user_needs"] = {"marca": ["Toyota"]}
        state["car_findings"] = "[{'make': 'Toyota', 'model': 'Corolla', 'year': 2020}]"

        result = organize_response(state)

        assert "user_response" in result
        assert result["user_response"] == "Here are the cars found: Toyota Corolla 2020"
        assert "messages" in result

    @patch("app.agents.car_catalog.car_catalog_llm")
    def test_organize_response_no_cars_found(
        self, mock_llm: MagicMock, mock_state: MagicMock
    ) -> None:
        """Test response when no cars are found."""
        state = mock_state
        state["messages"] = []
        state["message_to_analyze"] = ""

        result = organize_response(state)

        assert "error" in result
        assert "user_response" in result
        mock_llm.invoke.assert_not_called()


class TestClearCarContext:
    """Test cases for clear_car_context function."""

    def test_clear_car_context(self, mock_state: MagicMock) -> None:
        """Test that car context is cleared correctly."""
        state = mock_state
        state["messages"] = []
        state["message_to_analyze"] = ""
        state["user_needs"] = {"marca": ["Toyota"]}
        state["query"] = "SELECT * FROM cars"
        state["car_findings"] = "some findings"

        result = clear_car_context(state)

        assert result["user_needs"] == {}
        assert result["current_action"] == "clear_car_context"
        assert "Perfecto" in result["user_response"]
        assert "messages" in result
        assert isinstance(result["messages"][0], AIMessage)


class TestRouterNode:
    """Test cases for router_node function."""

    @patch("app.agents.car_catalog.car_catalog_llm")
    def test_router_node_select_car(
        self, mock_llm: MagicMock, mock_state: MagicMock
    ) -> None:
        """Test routing to select_car node."""
        mock_response = MagicMock()
        mock_response.content = "select_car"
        mock_llm.invoke.return_value = mock_response

        state = mock_state
        state["messages"] = [HumanMessage(content="I want the Toyota Corolla 2020")]
        state["current_action"] = "orchestrator_node"
        state["message_to_analyze"] = ""
        state["user_needs"] = {"marca": ["Toyota"]}
        state["car_findings"] = "[{'make': 'Toyota', 'model': 'Corolla'}]"

        result = router_node(state)

        assert result == "select_car"

    @patch("app.agents.car_catalog.car_catalog_llm")
    def test_router_node_context_car_identification(
        self, mock_llm: MagicMock, mock_state: MagicMock
    ) -> None:
        """Test routing to context_car_identification node."""
        mock_response = MagicMock()
        mock_response.content = "context_car_identification"
        mock_llm.invoke.return_value = mock_response

        state = mock_state
        state["messages"] = [HumanMessage(content="I want a Toyota")]
        state["current_action"] = "orchestrator_node"
        state["message_to_analyze"] = ""

        result = router_node(state)

        assert result == "context_car_identification"

    @patch("app.agents.car_catalog.car_catalog_llm")
    def test_router_node_text_to_sql(
        self, mock_llm: MagicMock, mock_state: MagicMock
    ) -> None:
        """Test routing to text_to_sql node."""
        mock_response = MagicMock()
        mock_response.content = "text_to_sql"
        mock_llm.invoke.return_value = mock_response

        state = mock_state
        state["messages"] = [HumanMessage(content="Show me the results")]
        state["current_action"] = "orchestrator_node"
        state["message_to_analyze"] = ""
        state["user_needs"] = {"marca": ["Toyota"]}

        result = router_node(state)

        assert result == "text_to_sql"

    @patch("app.agents.car_catalog.car_catalog_llm")
    def test_router_node_clear_car_context(
        self, mock_llm: MagicMock, mock_state: MagicMock
    ) -> None:
        """Test routing to clear_car_context node."""
        mock_response = MagicMock()
        mock_response.content = "clear_car_context"
        mock_llm.invoke.return_value = mock_response

        state = mock_state
        state["messages"] = [HumanMessage(content="Let's start over")]
        state["current_action"] = "orchestrator_node"
        state["message_to_analyze"] = ""
        state["user_needs"] = {"marca": ["Toyota"]}

        result = router_node(state)

        assert result == "clear_car_context"

    def test_router_node_returns_end_when_current_action_in_nodes(
        self, mock_state: MagicMock
    ) -> None:
        """Test that router returns END when current_action is in node names."""

        state = mock_state
        state["messages"] = []
        state["current_action"] = "context_car_identification"
        state["message_to_analyze"] = ""

        result = router_node(state)

        assert result == END


class TestOrchestratorNode:
    """Test cases for orchestrator_node function."""

    def test_orchestrator_node_returns_response(self, mock_state: MagicMock) -> None:
        """Test that orchestrator_node returns user_response."""
        state = mock_state
        state["messages"] = []
        state["message_to_analyze"] = ""
        state["user_response"] = "Test response"

        result = orchestrator_node(state)

        assert result["response"] == "Test response"
