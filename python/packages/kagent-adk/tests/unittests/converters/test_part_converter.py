"""Tests for part_converter module."""

from __future__ import annotations

import pytest
from a2a.types import DataPart
from google.genai import types as genai_types

from kagent.adk.converters.part_converter import (
    ADK_REQUEST_CONFIRMATION_NAME,
    convert_genai_part_to_a2a_part,
)
from kagent.core.a2a import (
    A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE,
    A2A_DATA_PART_METADATA_TYPE_KEY,
    KAGENT_HITL_INTERRUPT_TYPE_TOOL_APPROVAL,
    get_kagent_metadata_key,
)


class TestConvertFunctionResponsePart:
    """Tests for converting FunctionResponse parts."""

    def test_converts_regular_function_response(self):
        """Test that regular function responses are converted correctly."""
        func_response = genai_types.FunctionResponse(
            name="test_tool",
            id="call-123",
            response={"result": "success"},
        )
        part = genai_types.Part(function_response=func_response)

        result = convert_genai_part_to_a2a_part(part)

        assert result is not None
        assert isinstance(result.root, DataPart)
        assert result.root.data["name"] == "test_tool"
        assert result.root.data["id"] == "call-123"
        assert (
            result.root.metadata[get_kagent_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)]
            == A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE
        )

    def test_filters_out_confirmation_placeholder_response(self):
        """Test that confirmation placeholder responses are filtered out."""
        func_response = genai_types.FunctionResponse(
            name="some_tool",
            id="call-456",
            response={"error": "Tool some_tool requires confirmation before execution"},
        )
        part = genai_types.Part(function_response=func_response)

        result = convert_genai_part_to_a2a_part(part)

        assert result is None

    def test_pending_response_is_passed_through(self):
        """Test that pending responses are passed through as regular function responses."""
        func_response = genai_types.FunctionResponse(
            name="some_async_tool",
            id="call-999",
            response={
                "status": "pending",
                "task_id": "async-task-123",
            },
        )
        part = genai_types.Part(function_response=func_response)

        result = convert_genai_part_to_a2a_part(part)

        assert result is not None
        assert isinstance(result.root, DataPart)
        # Should be regular function response
        assert (
            result.root.metadata[get_kagent_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)]
            == A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE
        )

    def test_function_response_with_tool_approval_normalized_to_unified_format(self):
        """Tool approval inside a function_response is emitted as same shape as direct tool_approval."""
        func_response = genai_types.FunctionResponse(
            name="kagent__NS__sub_agent",
            id="call_xluaxLv6APpcdqA0nzgdrG27",
            response={
                "result": '{"interrupt_type": "tool_approval", "action_requests": ['
                '{"name": "datetime_get_current_time", "args": {}, "id": "call_R0MD5cZYzdn3hfBeigNvX4zN"}'
                "]}"
            },
        )
        part = genai_types.Part(function_response=func_response)

        result = convert_genai_part_to_a2a_part(part)

        assert result is not None
        assert isinstance(result.root, DataPart)
        assert result.root.data["interrupt_type"] == KAGENT_HITL_INTERRUPT_TYPE_TOOL_APPROVAL
        assert "action_requests" in result.root.data
        action_requests = result.root.data["action_requests"]
        assert len(action_requests) == 1
        assert action_requests[0]["name"] == "datetime_get_current_time"
        assert action_requests[0]["id"] == "call_R0MD5cZYzdn3hfBeigNvX4zN"
        assert (
            result.root.metadata[get_kagent_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)]
            == KAGENT_HITL_INTERRUPT_TYPE_TOOL_APPROVAL
        )
        assert result.root.metadata.get(get_kagent_metadata_key("interrupt_agent_name")) is None
        assert result.root.metadata[get_kagent_metadata_key("function_call_id")] == "call_xluaxLv6APpcdqA0nzgdrG27"
        assert result.root.metadata[get_kagent_metadata_key("function_call_name")] == "kagent__NS__sub_agent"


class TestConvertAdkRequestConfirmationPart:
    """Tests for converting adk_request_confirmation function calls."""

    def test_converts_adk_request_confirmation_to_tool_approval(self):
        """Test that adk_request_confirmation calls are converted to HITL format."""
        func_call = genai_types.FunctionCall(
            name=ADK_REQUEST_CONFIRMATION_NAME,
            id="adk-confirm-123",
            args={
                "originalFunctionCall": {
                    "name": "dangerous_tool",
                    "args": {"target": "production"},
                    "id": "original-call-456",
                }
            },
        )
        part = genai_types.Part(function_call=func_call)

        result = convert_genai_part_to_a2a_part(part)

        assert result is not None
        assert isinstance(result.root, DataPart)

        # Check it's converted to HITL format
        assert result.root.data["interrupt_type"] == KAGENT_HITL_INTERRUPT_TYPE_TOOL_APPROVAL
        assert (
            result.root.metadata[get_kagent_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)]
            == KAGENT_HITL_INTERRUPT_TYPE_TOOL_APPROVAL
        )

        # Check action_requests structure; confirmation_id is not exposed (backend stores in session)
        action_requests = result.root.data["action_requests"]
        assert len(action_requests) == 1
        assert action_requests[0]["name"] == "dangerous_tool"
        assert action_requests[0]["args"] == {"target": "production"}
        assert action_requests[0]["id"] == "original-call-456"
        assert action_requests[0].get("metadata") is None or "confirmation_id" not in (
            action_requests[0].get("metadata") or {}
        )
