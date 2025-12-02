"""Tests for the offer value agent."""

from unittest.mock import MagicMock, patch, mock_open

from langchain_core.messages import HumanMessage

from app.agents.offer_value import search_data, entry_point


class TestSearchData:
    """Test cases for search_data function."""

    @patch("app.agents.offer_value.offer_value_agent")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="# Company Info\n\n## About\nTest company",
    )
    def test_search_data_success(
        self, mock_file: MagicMock, mock_llm: MagicMock, mock_state: MagicMock
    ) -> None:
        """Test that search_data responds correctly."""
        mock_response = MagicMock()
        mock_response.content = "We are a test company that sells cars"
        mock_llm.invoke.return_value = mock_response

        state = mock_state
        state["messages"] = [HumanMessage(content="What does your company do?")]
        state["message_to_analyze"] = "What does your company do?"

        result = search_data(state)

        assert "response" in result
        assert result["response"] == "We are a test company that sells cars"
        assert "messages" in result
        mock_llm.invoke.assert_called_once()


class TestEntryPoint:
    """Test cases for entry_point function."""

    def test_entry_point_sets_current_action(self, mock_state: MagicMock) -> None:
        """Test that entry_point sets the correct current_action."""
        state = mock_state
        state["messages"] = []
        state["message_to_analyze"] = ""

        result = entry_point(state)

        assert result["current_action"] == "offer_value"
