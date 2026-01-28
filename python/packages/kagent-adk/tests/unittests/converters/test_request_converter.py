"""Tests for request_converter module."""

from kagent.adk.converters.request_converter import (
    ADK_REQUEST_CONFIRMATION_NAME,
    convert_tool_decision_to_adk_function_response,
)
from kagent.core.a2a import (
    KAGENT_HITL_DECISION_TYPE_APPROVE,
    KAGENT_HITL_DECISION_TYPE_DENY,
    ToolDecision,
)


class TestConvertToolDecisionToAdkFunctionResponse:
    """Tests for convert_tool_decision_to_adk_function_response function."""

    def test_converts_approved_decision(self):
        """Test converting an approved decision to ADK function response."""
        decision = ToolDecision(
            decision_type=KAGENT_HITL_DECISION_TYPE_APPROVE,
            tool_id="adk-123-456",
        )

        response = convert_tool_decision_to_adk_function_response(decision)

        assert response.role == "user"
        assert len(response.parts) == 1
        func_response = response.parts[0].function_response
        assert func_response.id == "adk-123-456"
        assert func_response.name == ADK_REQUEST_CONFIRMATION_NAME
        assert func_response.response == {"confirmed": True}

    def test_converts_denied_decision(self):
        """Test converting a denied decision to ADK function response."""
        decision = ToolDecision(
            decision_type=KAGENT_HITL_DECISION_TYPE_DENY,
            tool_id="adk-789-abc",
        )

        response = convert_tool_decision_to_adk_function_response(decision)

        assert response.role == "user"
        assert len(response.parts) == 1
        func_response = response.parts[0].function_response
        assert func_response.id == "adk-789-abc"
        assert func_response.name == ADK_REQUEST_CONFIRMATION_NAME
        assert func_response.response == {"confirmed": False}

    def test_returns_none_for_missing_tool_id(self):
        """Test that missing tool_id returns None."""
        decision = ToolDecision(
            decision_type=KAGENT_HITL_DECISION_TYPE_APPROVE,
            tool_id=None,
        )

        result = convert_tool_decision_to_adk_function_response(decision)

        assert result is None
