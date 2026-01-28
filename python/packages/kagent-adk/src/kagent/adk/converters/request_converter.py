import logging
from typing import Any

from a2a.server.agent_execution import RequestContext
from google.adk.agents.run_config import StreamingMode
from google.adk.runners import RunConfig
from google.genai import types as genai_types

from kagent.core.a2a import KAGENT_HITL_DECISION_TYPE_APPROVE, ToolDecision

from .part_converter import convert_a2a_part_to_genai_part

# ADK's function name for confirmation requests
ADK_REQUEST_CONFIRMATION_NAME = "adk_request_confirmation"

logger = logging.getLogger("kagent_adk." + __name__)


def _get_user_id(request: RequestContext) -> str:
    # Get user from call context if available (auth is enabled on a2a server)
    if request.call_context and request.call_context.user and request.call_context.user.user_name:
        return request.call_context.user.user_name

    # Get user from context id
    return f"A2A_USER_{request.context_id}"


def convert_a2a_request_to_adk_run_args(
    request: RequestContext,
    stream: bool = False,
) -> dict[str, Any]:
    if not request.message:
        raise ValueError("Request message cannot be None")

    return {
        "user_id": _get_user_id(request),
        "session_id": request.context_id,
        "new_message": genai_types.Content(
            role="user",
            parts=[convert_a2a_part_to_genai_part(part) for part in request.message.parts],
        ),
        "run_config": RunConfig(streaming_mode=StreamingMode.SSE if stream else StreamingMode.NONE),
    }


def convert_tool_decision_to_adk_function_response(
    decision: ToolDecision,
) -> genai_types.Content | None:
    """Convert a ToolDecision to ADK FunctionResponse Content.

    This converts the `kagent-core` ToolDecision format back to ADK's native format.

    ADK expects a FunctionResponse with:
    - name='adk_request_confirmation' (ADK filters by this name)
    - id matching the adk_request_confirmation FunctionCall.id
    - response containing a ToolConfirmation-like structure with 'confirmed' field

    Args:
        decision: The user's decision containing decision_type (approve/deny)
                 and tool_id (the ADK confirmation ID from ToolApprovalRequest.metadata)

    Returns:
        genai_types.Content with a single FunctionResponse part, or None if tool_id is missing
    """
    if not decision.tool_id:
        logger.warning("Tool ID is required for confirmation response - ignoring decision")
        return None

    approved = decision.decision_type == KAGENT_HITL_DECISION_TYPE_APPROVE

    part = genai_types.Part(
        function_response=genai_types.FunctionResponse(
            name=ADK_REQUEST_CONFIRMATION_NAME,
            id=decision.tool_id,
            response={
                "confirmed": approved,
            },
        )
    )
    return genai_types.Content(parts=[part], role="user")
