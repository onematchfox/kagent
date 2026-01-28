"""Human-in-the-Loop (HITL) support for kagent executors.

This module provides types, utilities, and handlers for implementing
human-in-the-loop workflows in kagent agent executors using A2A protocol primitives.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskStore
from a2a.types import (
    DataPart,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)

from ._consts import (
    KAGENT_HITL_DECISION_TYPE_APPROVE,
    KAGENT_HITL_DECISION_TYPE_DENY,
    KAGENT_HITL_DECISION_TYPE_KEY,
    KAGENT_HITL_DECISION_TYPE_REJECT,
    KAGENT_HITL_INTERRUPT_TYPE_TOOL_APPROVAL,
    KAGENT_HITL_RESUME_KEYWORDS_APPROVE,
    KAGENT_HITL_RESUME_KEYWORDS_DENY,
    get_kagent_metadata_key,
)

logger = logging.getLogger(__name__)

# Type definitions

DecisionType = Literal["approve", "deny", "reject"]
"""Represents a user decision in HITL workflows."""


@dataclass
class ToolDecision:
    """Type for user decisions in HITL workflows.

    Represents both the decision type and, optionally, the tool to which the decision applies.

    Attributes:
        decision_type: The type of decision (approve, deny, reject)
        tool_id: The ID of the tool to which the decision applies, or None if the decision applies to all tools
    """

    decision_type: DecisionType
    tool_id: str | None


@dataclass
class ToolApprovalRequest:
    """Generic structure for a tool call requiring approval.

    Any agent framework can map their tool calls to this structure.

    Attributes:
        name: The name of the tool/function being called
        args: Dictionary of arguments to pass to the tool
        id: Optional unique identifier for this specific tool call
        metadata: Optional framework-specific data
    """

    name: str
    args: dict[str, Any]
    id: str | None = None
    metadata: dict[str, Any] | None = None


# Utility functions


def escape_markdown_backticks(text: str) -> str:
    """Escape backticks in text to prevent markdown formatting issues.

    Used when displaying code, tool names, or arguments in markdown-formatted
    approval messages.

    Args:
        text: Text that may contain backticks

    Returns:
        Text with all backticks escaped with backslash

    Examples:
        >>> escape_markdown_backticks("function `foo`")
        'function \\`foo\\`'
    """
    return str(text).replace("`", "\\`")


def is_input_required_task(task_state: TaskState | None) -> bool:
    """Check if task state indicates waiting for user input.

    Args:
        task_state: Current task state, or None if no task

    Returns:
        True if task is in input_required state
    """
    return task_state == TaskState.input_required


def _is_valid_decision(decision: str | None) -> bool:
    """Check if a decision value is valid."""
    return decision in (
        KAGENT_HITL_DECISION_TYPE_APPROVE,
        KAGENT_HITL_DECISION_TYPE_DENY,
        KAGENT_HITL_DECISION_TYPE_REJECT,
    )


def extract_decision_from_data_part(data: dict) -> ToolDecision | None:
    """Extract decision from structured DataPart.

    Supports two formats using the same decision_type key:
    1. Global format: {decision_type: "approve"} - applies to all tools
    2. Per-tool format: {decision_type: "approve", tool_id: "call_123"} - specific tool

    Args:
        data: DataPart.data dictionary

    Returns:
        ToolDecision if found and valid, None otherwise.
        tool_id is None for global decisions.
    """
    decision = data.get(KAGENT_HITL_DECISION_TYPE_KEY)
    if not _is_valid_decision(decision):
        return None

    return ToolDecision(decision_type=decision, tool_id=data.get("tool_id"))


def extract_decision_from_text(text: str) -> DecisionType | None:
    """Extract decision from text using keyword matching.

    Searches for approval or denial keywords in the text (case-insensitive).
    Denial keywords take priority if both are present (to avoid accidental approval).

    Args:
        text: User input text

    Returns:
        "deny" if denial keywords found, "approve" if approval keywords found,
        None if no keywords found
    """
    text_lower = text.lower()

    # Check deny keywords first (safer - prevents accidental approval)
    if any(keyword in text_lower for keyword in KAGENT_HITL_RESUME_KEYWORDS_DENY):
        return KAGENT_HITL_DECISION_TYPE_DENY

    # Check approve keywords
    if any(keyword in text_lower for keyword in KAGENT_HITL_RESUME_KEYWORDS_APPROVE):
        return KAGENT_HITL_DECISION_TYPE_APPROVE

    return None


def extract_decision_from_message(message: Message | None) -> ToolDecision | None:
    """Extract decision from A2A message using two-tier detection.

    Priority:
    1. Structured DataPart with decision fields (most reliable)
    2. Keyword matching in TextPart (fallback for human input)

    DataPart is checked across all parts first before falling back to TextPart,
    ensuring structured decisions always take precedence.

    Args:
        message: A2A message from user

    Returns:
        ToolDecision if found, None otherwise.
        tool_id is None for global decisions or text-based decisions.
    """
    if not message or not message.parts:
        return None

    # Priority 1: Scan all parts for DataPart with decision (most reliable)
    for part in message.parts:
        # Access .root for RootModel union types
        if not hasattr(part, "root"):
            continue

        inner = part.root

        if isinstance(inner, DataPart):
            result = extract_decision_from_data_part(inner.data)
            if result:
                logger.info(f"Extracted decision from DataPart: {inner.data}")
                return result

    # Priority 2: Fallback to TextPart keyword matching (no tool_id)
    for part in message.parts:
        if not hasattr(part, "root"):
            continue

        inner = part.root

        if isinstance(inner, TextPart):
            if inner.text and isinstance(inner.text, str):
                decision = extract_decision_from_text(inner.text)
                if decision:
                    logger.info(f"Extracted decision from TextPart: {inner.text}")
                    return ToolDecision(decision_type=decision, tool_id=None)

    return None


def extract_tool_requests_from_message(message: Message | None) -> list[ToolApprovalRequest]:
    """Extract tool approval requests from an input_required task message.

    Args:
        message: The task status message from an input_required task

    Returns:
        List of ToolApprovalRequest objects, empty if none found
    """
    if not message or not message.parts:
        return []

    for _i, part in enumerate(message.parts):
        if not hasattr(part, "root"):
            continue

        inner = part.root

        if isinstance(inner, DataPart) and inner.data:
            if inner.data.get("interrupt_type") == KAGENT_HITL_INTERRUPT_TYPE_TOOL_APPROVAL:
                action_requests = inner.data.get("action_requests", [])
                return [
                    ToolApprovalRequest(
                        name=req.get("name", ""),
                        args=req.get("args", {}),
                        id=req.get("id"),
                        metadata=req.get("metadata"),
                    )
                    for req in action_requests
                ]

    return []


def find_pending_tool_request(
    task: Task | None,
    tool_id: str | None,
) -> ToolApprovalRequest | None:
    """Find a pending tool approval request matching the given tool_id.

    Searches the task's status message first, then history (most recent first).

    Args:
        task: The current A2A task
        tool_id: The tool ID to match (matches against request.id)

    Returns:
        The matching ToolApprovalRequest, or None if not found
    """
    if not task:
        return None

    tool_requests = []

    # First check the task's current status message
    task_message = task.status.message if task.status else None
    if task_message:
        tool_requests = extract_tool_requests_from_message(task_message)

    # If no requests found in status.message, search task history
    if not tool_requests and task.history:
        for msg in reversed(task.history):
            tool_requests = extract_tool_requests_from_message(msg)
            if tool_requests:
                break

    if not tool_requests:
        return None

    return next(
        (t for t in tool_requests if (t.id and t.id == tool_id)),
        None,
    )


def format_tool_approval_text_parts(
    action_requests: list[ToolApprovalRequest],
) -> list[Part]:
    """Format tool approval requests as human-readable TextParts.

    Creates a formatted approval message listing all tools and their arguments
    with proper markdown escaping to prevent rendering issues.

    Args:
        action_requests: List of tool approval request objects

    Returns:
        List of Part objects containing formatted approval message
    """
    parts = []

    # Add header
    parts.append(Part(TextPart(text="**Approval Required**\n\n")))
    parts.append(Part(TextPart(text="The following actions require your approval:\n\n")))

    # List each action
    for action in action_requests:
        tool_name = action.name
        tool_args = action.args

        # Escape backticks to prevent markdown breaking
        escaped_tool_name = escape_markdown_backticks(tool_name)
        parts.append(Part(TextPart(text=f"**Tool**: `{escaped_tool_name}`\n")))
        parts.append(Part(TextPart(text="**Arguments**:\n")))

        for key, value in tool_args.items():
            escaped_key = escape_markdown_backticks(key)
            escaped_value = escape_markdown_backticks(value)
            parts.append(Part(TextPart(text=f"  • {escaped_key}: `{escaped_value}`\n")))

        parts.append(Part(TextPart(text="\n")))

    return parts


# High-level handlers


async def handle_tool_approval_interrupt(
    action_requests: list[ToolApprovalRequest],
    task_id: str,
    context_id: str,
    event_queue: EventQueue,
    task_store: TaskStore,
    app_name: str | None = None,
    review_configs: list[dict[str, Any]] | None = None,
) -> None:
    """Send input_required event for tool approval.

    This is a framework-agnostic handler that any executor can call when
    it needs user approval for tool calls. It formats an approval message,
    sends an input_required event, and waits for the task to be saved.

    Args:
        action_requests: List of tool calls requiring approval
        task_id: A2A task ID
        context_id: A2A context ID
        event_queue: Event queue for publishing events
        task_store: Task store for synchronization
        app_name: Optional application name for metadata
        review_configs: Optional framework-specific review configurations

    Raises:
        TimeoutError: If task save doesn't complete within 5 seconds (logged as warning)
    """
    # Build human-readable message
    text_parts = format_tool_approval_text_parts(action_requests)

    # Build structured DataPart for machine processing (client can parse this)
    interrupt_data: dict[str, Any] = {
        "interrupt_type": KAGENT_HITL_INTERRUPT_TYPE_TOOL_APPROVAL,
        "action_requests": [{"name": req.name, "args": req.args, "id": req.id} for req in action_requests],
    }

    if review_configs:
        interrupt_data["review_configs"] = review_configs

    data_part = Part(
        DataPart(
            data=interrupt_data,
            metadata={get_kagent_metadata_key("type"): "interrupt_data"},
        )
    )

    # Combine message parts
    message_parts = text_parts + [data_part]

    # Build event metadata
    event_metadata = {"interrupt_type": KAGENT_HITL_INTERRUPT_TYPE_TOOL_APPROVAL}
    if app_name:
        event_metadata["app_name"] = app_name

    # Send input_required event
    await event_queue.enqueue_event(
        TaskStatusUpdateEvent(
            task_id=task_id,
            status=TaskStatus(
                state=TaskState.input_required,
                timestamp=datetime.now(UTC).isoformat(),
                message=Message(
                    message_id=str(uuid.uuid4()),
                    role=Role.agent,
                    parts=message_parts,
                ),
            ),
            context_id=context_id,
            final=False,  # Not final - waiting for user input
            metadata=event_metadata,
        )
    )

    logger.info(f"Interrupt detected, sent input_required event for task {task_id} with {len(action_requests)} actions")

    # Wait for the event consumer to persist the task (event-based sync)
    # This prevents race condition where approval arrives before task is saved
    try:
        await task_store.wait_for_save(task_id, timeout=5.0)
    except TimeoutError:
        logger.warning("Task save event timeout, proceeding anyway")
