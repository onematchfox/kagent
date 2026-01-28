"""Tests for HITL utility functions."""

import pytest
from a2a.types import DataPart, Message, Part, Task, TaskState, TaskStatus, TextPart

from kagent.core.a2a import (
    KAGENT_HITL_DECISION_TYPE_APPROVE,
    KAGENT_HITL_DECISION_TYPE_DENY,
    KAGENT_HITL_DECISION_TYPE_KEY,
    KAGENT_HITL_INTERRUPT_TYPE_TOOL_APPROVAL,
    ToolApprovalRequest,
    ToolDecision,
    escape_markdown_backticks,
    extract_decision_from_message,
    extract_tool_requests_from_message,
    find_pending_tool_request,
    format_tool_approval_text_parts,
    is_input_required_task,
)


def test_escape_markdown_backticks():
    """Test backtick escaping for all cases."""
    assert escape_markdown_backticks("foo`bar") == "foo\\`bar"
    assert escape_markdown_backticks("`code` and `more`") == "\\`code\\` and \\`more\\`"
    assert escape_markdown_backticks("plain text") == "plain text"
    assert escape_markdown_backticks("") == ""


def test_is_input_required_task():
    """Test is_input_required_task() for various states."""
    assert is_input_required_task(TaskState.input_required) is True
    assert is_input_required_task(TaskState.working) is False
    assert is_input_required_task(TaskState.completed) is False
    assert is_input_required_task(None) is False


def test_extract_decision_datapart_global():
    """Test DataPart decision extraction with global format (decision_type key)."""
    # Approve - global format returns ToolDecision with tool_id=None
    message = Message(
        role="user",
        message_id="test",
        task_id="task1",
        context_id="ctx1",
        parts=[Part(DataPart(data={KAGENT_HITL_DECISION_TYPE_KEY: KAGENT_HITL_DECISION_TYPE_APPROVE}))],
    )
    assert extract_decision_from_message(message) == ToolDecision(
        decision_type=KAGENT_HITL_DECISION_TYPE_APPROVE, tool_id=None
    )

    # Deny - global format
    message = Message(
        role="user",
        message_id="test",
        task_id="task1",
        context_id="ctx1",
        parts=[Part(DataPart(data={KAGENT_HITL_DECISION_TYPE_KEY: KAGENT_HITL_DECISION_TYPE_DENY}))],
    )
    assert extract_decision_from_message(message) == ToolDecision(
        decision_type=KAGENT_HITL_DECISION_TYPE_DENY, tool_id=None
    )


def test_extract_decision_datapart_per_tool():
    """Test DataPart decision extraction with per-tool format (decision_type + tool_id)."""
    # Approve - per-tool format returns ToolDecision with tool_id
    message = Message(
        role="user",
        message_id="test",
        task_id="task1",
        context_id="ctx1",
        parts=[
            Part(
                DataPart(data={KAGENT_HITL_DECISION_TYPE_KEY: KAGENT_HITL_DECISION_TYPE_APPROVE, "tool_id": "call_123"})
            )
        ],
    )
    assert extract_decision_from_message(message) == ToolDecision(
        decision_type=KAGENT_HITL_DECISION_TYPE_APPROVE, tool_id="call_123"
    )

    # Deny - per-tool format
    message = Message(
        role="user",
        message_id="test",
        task_id="task1",
        context_id="ctx1",
        parts=[
            Part(DataPart(data={KAGENT_HITL_DECISION_TYPE_KEY: KAGENT_HITL_DECISION_TYPE_DENY, "tool_id": "call_456"}))
        ],
    )
    assert extract_decision_from_message(message) == ToolDecision(
        decision_type=KAGENT_HITL_DECISION_TYPE_DENY, tool_id="call_456"
    )


def test_extract_decision_with_and_without_tool_id():
    """Test that tool_id presence determines per-tool vs global extraction."""
    # With tool_id - returns ToolDecision with tool_id
    message = Message(
        role="user",
        message_id="test",
        task_id="task1",
        context_id="ctx1",
        parts=[
            Part(
                DataPart(
                    data={
                        KAGENT_HITL_DECISION_TYPE_KEY: KAGENT_HITL_DECISION_TYPE_APPROVE,
                        "tool_id": "call_123",
                    }
                )
            )
        ],
    )
    assert extract_decision_from_message(message) == ToolDecision(
        decision_type=KAGENT_HITL_DECISION_TYPE_APPROVE, tool_id="call_123"
    )

    # Without tool_id - returns ToolDecision with tool_id=None
    message = Message(
        role="user",
        message_id="test",
        task_id="task1",
        context_id="ctx1",
        parts=[
            Part(
                DataPart(
                    data={
                        KAGENT_HITL_DECISION_TYPE_KEY: KAGENT_HITL_DECISION_TYPE_APPROVE,
                    }
                )
            )
        ],
    )
    assert extract_decision_from_message(message) == ToolDecision(
        decision_type=KAGENT_HITL_DECISION_TYPE_APPROVE, tool_id=None
    )


def test_extract_decision_textpart():
    """Test TextPart keyword extraction (returns ToolDecision with None tool_id)."""
    # Approve keyword
    message = Message(
        role="user",
        message_id="test",
        task_id="task1",
        context_id="ctx1",
        parts=[Part(TextPart(text="I have approved this action"))],
    )
    assert extract_decision_from_message(message) == ToolDecision(
        decision_type=KAGENT_HITL_DECISION_TYPE_APPROVE, tool_id=None
    )

    # Deny keyword
    message = Message(
        role="user",
        message_id="test",
        task_id="task1",
        context_id="ctx1",
        parts=[Part(TextPart(text="Request denied, do not proceed"))],
    )
    assert extract_decision_from_message(message) == ToolDecision(
        decision_type=KAGENT_HITL_DECISION_TYPE_DENY, tool_id=None
    )

    # Case insensitive
    message = Message(
        role="user",
        message_id="test",
        task_id="task1",
        context_id="ctx1",
        parts=[Part(TextPart(text="APPROVED"))],
    )
    assert extract_decision_from_message(message) == ToolDecision(
        decision_type=KAGENT_HITL_DECISION_TYPE_APPROVE, tool_id=None
    )


def test_extract_decision_priority():
    """Test DataPart takes priority over TextPart."""
    message = Message(
        role="user",
        message_id="test",
        task_id="task1",
        context_id="ctx1",
        parts=[
            Part(TextPart(text="approved")),  # Would detect as approve
            Part(DataPart(data={KAGENT_HITL_DECISION_TYPE_KEY: KAGENT_HITL_DECISION_TYPE_DENY})),  # But deny wins
        ],
    )
    assert extract_decision_from_message(message) == ToolDecision(
        decision_type=KAGENT_HITL_DECISION_TYPE_DENY, tool_id=None
    )


def test_extract_decision_edge_cases():
    """Test edge cases: empty message, no parts, no decision."""
    # Empty message
    assert extract_decision_from_message(None) is None

    # No parts
    message = Message(role="user", message_id="test", task_id="task1", context_id="ctx1", parts=[])
    assert extract_decision_from_message(message) is None

    # No decision found
    message = Message(
        role="user",
        message_id="test",
        task_id="task1",
        context_id="ctx1",
        parts=[Part(TextPart(text="This is just a comment"))],
    )
    assert extract_decision_from_message(message) is None

    # Invalid decision value in per-tool format
    message = Message(
        role="user",
        message_id="test",
        task_id="task1",
        context_id="ctx1",
        parts=[Part(DataPart(data={"tool_id": "call_123", "decision": "invalid"}))],
    )
    assert extract_decision_from_message(message) is None

    # Missing tool_id in per-tool format (falls through, no global format either)
    message = Message(
        role="user",
        message_id="test",
        task_id="task1",
        context_id="ctx1",
        parts=[Part(DataPart(data={"decision": KAGENT_HITL_DECISION_TYPE_APPROVE}))],
    )
    assert extract_decision_from_message(message) is None

    # Empty tool_id in per-tool format
    message = Message(
        role="user",
        message_id="test",
        task_id="task1",
        context_id="ctx1",
        parts=[Part(DataPart(data={"tool_id": "", "decision": KAGENT_HITL_DECISION_TYPE_APPROVE}))],
    )
    assert extract_decision_from_message(message) is None


def test_format_tool_approval_text_parts():
    """Test formatting tool approval requests with all edge cases."""
    requests = [
        ToolApprovalRequest(name="search", args={"query": "test"}),
        ToolApprovalRequest(name="run`code`", args={"cmd": "echo `test`"}),
        ToolApprovalRequest(name="reset", args={}),
    ]
    parts = format_tool_approval_text_parts(requests)

    # Convert to text
    text_content = ""
    for p in parts:
        if hasattr(p, "root") and hasattr(p.root, "text"):
            text_content += p.root.text

    # Check structure and content
    assert "Approval Required" in text_content
    assert "search" in text_content
    assert "reset" in text_content
    # Check backticks are escaped
    assert "\\`" in text_content


class TestExtractToolRequestsFromMessage:
    """Tests for extract_tool_requests_from_message function."""

    def test_extract_from_valid_interrupt_message(self):
        """Test extracting tool requests from a properly formatted interrupt message."""
        interrupt_data = {
            "interrupt_type": KAGENT_HITL_INTERRUPT_TYPE_TOOL_APPROVAL,
            "action_requests": [
                {"name": "search", "args": {"query": "test"}, "id": "call_123"},
                {"name": "delete", "args": {"file": "foo.txt"}, "id": "call_456"},
            ],
        }
        message = Message(
            role="agent",
            message_id="test",
            task_id="task1",
            context_id="ctx1",
            parts=[Part(DataPart(data=interrupt_data))],
        )

        tool_requests = extract_tool_requests_from_message(message)

        assert len(tool_requests) == 2
        assert tool_requests[0].name == "search"
        assert tool_requests[0].args == {"query": "test"}
        assert tool_requests[0].id == "call_123"
        assert tool_requests[1].name == "delete"
        assert tool_requests[1].args == {"file": "foo.txt"}
        assert tool_requests[1].id == "call_456"

    def test_extract_returns_empty_for_no_message(self):
        """Test that None message returns empty list."""
        assert extract_tool_requests_from_message(None) == []

    def test_extract_returns_empty_for_no_parts(self):
        """Test that message with no parts returns empty list."""
        message = Message(
            role="agent",
            message_id="test",
            task_id="task1",
            context_id="ctx1",
            parts=[],
        )
        assert extract_tool_requests_from_message(message) == []

    def test_extract_returns_empty_for_wrong_interrupt_type(self):
        """Test that messages with different interrupt types return empty list."""
        interrupt_data = {
            "interrupt_type": "some_other_type",
            "action_requests": [
                {"name": "search", "args": {"query": "test"}, "id": "call_123"},
            ],
        }
        message = Message(
            role="agent",
            message_id="test",
            task_id="task1",
            context_id="ctx1",
            parts=[Part(DataPart(data=interrupt_data))],
        )
        assert extract_tool_requests_from_message(message) == []

    def test_extract_returns_empty_for_text_only_message(self):
        """Test that text-only messages return empty list."""
        message = Message(
            role="agent",
            message_id="test",
            task_id="task1",
            context_id="ctx1",
            parts=[Part(TextPart(text="Please approve the following actions"))],
        )
        assert extract_tool_requests_from_message(message) == []

    def test_extract_handles_missing_id(self):
        """Test that tool requests without ID are handled correctly."""
        interrupt_data = {
            "interrupt_type": KAGENT_HITL_INTERRUPT_TYPE_TOOL_APPROVAL,
            "action_requests": [
                {"name": "search", "args": {"query": "test"}},  # No ID
            ],
        }
        message = Message(
            role="agent",
            message_id="test",
            task_id="task1",
            context_id="ctx1",
            parts=[Part(DataPart(data=interrupt_data))],
        )

        tool_requests = extract_tool_requests_from_message(message)

        assert len(tool_requests) == 1
        assert tool_requests[0].name == "search"
        assert tool_requests[0].id is None

    def test_extract_finds_data_part_among_multiple_parts(self):
        """Test that DataPart is found even when mixed with other parts."""
        interrupt_data = {
            "interrupt_type": KAGENT_HITL_INTERRUPT_TYPE_TOOL_APPROVAL,
            "action_requests": [
                {"name": "search", "args": {"query": "test"}, "id": "call_123"},
            ],
        }
        message = Message(
            role="agent",
            message_id="test",
            task_id="task1",
            context_id="ctx1",
            parts=[
                Part(TextPart(text="**Approval Required**")),
                Part(DataPart(data=interrupt_data)),
            ],
        )

        tool_requests = extract_tool_requests_from_message(message)

        assert len(tool_requests) == 1
        assert tool_requests[0].name == "search"


def _make_tool_approval_message(action_requests: list[dict]) -> Message:
    """Helper to create a message with tool approval requests."""
    return Message(
        role="agent",
        message_id="msg_1",
        task_id="task1",
        context_id="ctx1",
        parts=[
            Part(
                DataPart(
                    data={
                        "interrupt_type": KAGENT_HITL_INTERRUPT_TYPE_TOOL_APPROVAL,
                        "action_requests": action_requests,
                    }
                )
            )
        ],
    )


def _make_task(
    status_message: Message | None = None,
    history: list[Message] | None = None,
) -> Task:
    """Helper to create a Task with optional status message and history."""
    return Task(
        id="task_1",
        context_id="ctx_1",
        status=TaskStatus(state=TaskState.input_required, message=status_message),
        history=history,
    )


class TestFindPendingToolRequest:
    """Tests for find_pending_tool_request function."""

    def test_returns_none_for_none_task(self):
        """Test that None task returns None."""
        assert find_pending_tool_request(None, "call_123") is None

    def test_returns_none_for_task_with_no_message_and_no_history(self):
        """Test that task with no message and no history returns None."""
        task = _make_task()
        assert find_pending_tool_request(task, "call_123") is None

    def test_finds_request_in_status_message_by_id(self):
        """Test finding a tool request in status.message by ID."""
        message = _make_tool_approval_message(
            [
                {"name": "search", "args": {"query": "test"}, "id": "call_123"},
                {"name": "delete", "args": {"file": "foo.txt"}, "id": "call_456"},
            ]
        )
        task = _make_task(status_message=message)

        result = find_pending_tool_request(task, "call_456")

        assert result is not None
        assert result.name == "delete"
        assert result.id == "call_456"

    def test_finds_request_in_history_when_status_message_empty(self):
        """Test finding a tool request in history when status.message has no requests."""
        # Status message with no tool requests
        status_message = Message(
            role="agent",
            message_id="msg_status",
            task_id="task1",
            context_id="ctx1",
            parts=[Part(TextPart(text="Some text"))],
        )
        # History message with tool requests
        history_message = _make_tool_approval_message(
            [
                {"name": "search", "args": {"query": "test"}, "id": "call_123"},
            ]
        )
        task = _make_task(status_message=status_message, history=[history_message])

        result = find_pending_tool_request(task, "call_123")

        assert result is not None
        assert result.name == "search"
        assert result.id == "call_123"

    def test_finds_request_in_history_when_no_status_message(self):
        """Test finding a tool request in history when status has no message."""
        history_message = _make_tool_approval_message(
            [
                {"name": "search", "args": {"query": "test"}, "id": "call_123"},
            ]
        )
        task = _make_task(history=[history_message])

        result = find_pending_tool_request(task, "call_123")

        assert result is not None
        assert result.name == "search"

    def test_searches_history_in_reverse_order(self):
        """Test that history is searched from most recent to oldest."""
        old_message = _make_tool_approval_message(
            [
                {"name": "old_tool", "args": {}, "id": "call_old"},
            ]
        )
        new_message = _make_tool_approval_message(
            [
                {"name": "new_tool", "args": {}, "id": "call_new"},
            ]
        )
        # old_message is first (oldest), new_message is last (most recent)
        task = _make_task(history=[old_message, new_message])

        # Should find in the most recent message first
        result = find_pending_tool_request(task, "call_new")
        assert result is not None
        assert result.name == "new_tool"

    def test_returns_none_when_no_match_found(self):
        """Test that None is returned when tool_id doesn't match any request."""
        message = _make_tool_approval_message(
            [
                {"name": "search", "args": {"query": "test"}, "id": "call_123"},
            ]
        )
        task = _make_task(status_message=message)

        result = find_pending_tool_request(task, "call_nonexistent")

        assert result is None

    def test_returns_none_for_none_tool_id(self):
        """Test behavior when tool_id is None."""
        message = _make_tool_approval_message(
            [
                {"name": "search", "args": {"query": "test"}, "id": "call_123"},
            ]
        )
        task = _make_task(status_message=message)

        # None tool_id won't match any request (t.id == None is False, t.name == None is False)
        result = find_pending_tool_request(task, None)

        assert result is None

    def test_preserves_metadata_in_result(self):
        """Test that metadata field is preserved in the returned request."""
        message = _make_tool_approval_message(
            [
                {
                    "name": "search",
                    "args": {"query": "test"},
                    "id": "call_123",
                    "metadata": {"confirmation_id": "adk-conf-456"},
                },
            ]
        )
        task = _make_task(status_message=message)

        result = find_pending_tool_request(task, "call_123")

        assert result is not None
        assert result.metadata == {"confirmation_id": "adk-conf-456"}

    def test_prefers_status_message_over_history(self):
        """Test that status.message is checked before history."""
        status_message = _make_tool_approval_message(
            [
                {"name": "status_tool", "args": {}, "id": "call_status"},
            ]
        )
        history_message = _make_tool_approval_message(
            [
                {"name": "history_tool", "args": {}, "id": "call_history"},
            ]
        )
        task = _make_task(status_message=status_message, history=[history_message])

        # Should find in status message, not history
        result = find_pending_tool_request(task, "call_status")
        assert result is not None
        assert result.name == "status_tool"

        # History tool shouldn't be found since status has requests
        result = find_pending_tool_request(task, "call_history")
        assert result is None
