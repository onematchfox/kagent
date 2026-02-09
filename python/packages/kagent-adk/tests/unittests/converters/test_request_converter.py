"""Tests for request_converter module."""

from types import SimpleNamespace

from kagent.adk._consts import PENDING_TOOL_CONFIRMATION_KEY
from kagent.adk.converters.request_converter import (
    ADK_REQUEST_CONFIRMATION_NAME,
    convert_tool_decision_to_adk_function_response,
)
from kagent.core.a2a import (
    KAGENT_HITL_DECISION_TYPE_APPROVE,
    KAGENT_HITL_DECISION_TYPE_DENY,
    ToolDecision,
)


def _make_session(adk_confirmation_id_by_tool_id):
    """Mock session with state containing pending tool confirmation (same_agent entries)."""
    pending = {
        tid: {"type": "same_agent", "adk_confirmation_id": cid} for tid, cid in adk_confirmation_id_by_tool_id.items()
    }
    return SimpleNamespace(state={PENDING_TOOL_CONFIRMATION_KEY: pending})


def _make_session_empty():
    """Mock session with no confirmation id mapping."""
    return SimpleNamespace(state={})


class TestConvertToolDecisionToAdkFunctionResponse:
    """Tests for convert_tool_decision_to_adk_function_response function."""

    def test_converts_approved_decision(self):
        """Test converting an approved decision to ADK function response (lookup from session)."""
        decision = ToolDecision(
            decision_type=KAGENT_HITL_DECISION_TYPE_APPROVE,
            tool_id="call_tool_xyz",
        )
        session = _make_session({"call_tool_xyz": "adk-confirmation-call-123"})

        response = convert_tool_decision_to_adk_function_response(session, decision)

        assert response.role == "user"
        assert len(response.parts) == 1
        func_response = response.parts[0].function_response
        assert func_response.id == "adk-confirmation-call-123"
        assert func_response.name == ADK_REQUEST_CONFIRMATION_NAME
        assert func_response.response == {"confirmed": True}

    def test_returns_none_when_adk_confirmation_id_missing_in_session(self):
        """When session has no adk_confirmation_id for tool_id, returns None."""
        decision = ToolDecision(
            decision_type=KAGENT_HITL_DECISION_TYPE_APPROVE,
            tool_id="call_tool_xyz",
        )
        session = _make_session_empty()

        result = convert_tool_decision_to_adk_function_response(session, decision)

        assert result is None

    def test_converts_denied_decision(self):
        """Test converting a denied decision to ADK function response."""
        decision = ToolDecision(
            decision_type=KAGENT_HITL_DECISION_TYPE_DENY,
            tool_id="call_789",
        )
        session = _make_session({"call_789": "adk-conf-789"})

        response = convert_tool_decision_to_adk_function_response(session, decision)

        assert response.role == "user"
        assert len(response.parts) == 1
        func_response = response.parts[0].function_response
        assert func_response.id == "adk-conf-789"
        assert func_response.name == ADK_REQUEST_CONFIRMATION_NAME
        assert func_response.response == {"confirmed": False}

    def test_returns_none_for_missing_tool_id(self):
        """Test that missing tool_id returns None."""
        decision = ToolDecision(
            decision_type=KAGENT_HITL_DECISION_TYPE_APPROVE,
            tool_id=None,
        )
        session = _make_session({"call_other": "adk-call-123"})

        result = convert_tool_decision_to_adk_function_response(session, decision)

        assert result is None
