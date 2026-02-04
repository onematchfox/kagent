import logging
from typing import Any

from a2a.server.agent_execution import RequestContext
from google.adk.agents.run_config import StreamingMode
from google.adk.runners import RunConfig
from google.genai import types as genai_types

from kagent.core.a2a import KAGENT_HITL_DECISION_TYPE_APPROVE, ToolDecision

from .part_converter import convert_a2a_part_to_genai_part
from .._consts import ADK_CONFIRMATION_ID_BY_TOOL_ID_KEY, ADK_REQUEST_CONFIRMATION_NAME

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
    session: Any,
    decision: ToolDecision,
) -> genai_types.Content | None:
    """Convert a ToolDecision to ADK FunctionResponse Content.

    This converts the `kagent-core` ToolDecision format back to ADK's native
    format.

    Args:
        session: The session object containing the session state. Used to lookup
                 mapping between decision.tool_id and adk_confirmation_id.
        decision: The user's decision containing decision_type (approve/deny)
                  and tool_id (action_requests[].id from the tool approval
                  request).

    Returns:
        genai_types.Content with a single FunctionResponse part, or None if
        tool_id or adk_confirmation_id is missing
    """

    if not decision.tool_id:
        logger.warning("Tool ID is required for confirmation response - ignoring decision")
        return None

    adk_confirmation_id = None

    if session and session.state:
        by_tool_id = session.state.get(ADK_CONFIRMATION_ID_BY_TOOL_ID_KEY) or {}
        adk_confirmation_id = by_tool_id.get(decision.tool_id)

    if not adk_confirmation_id:
        logger.warning(
            f"Failed to lookup adk_confirmation_id for HITL resume for tool_id={decision.tool_id} - ignoring decision"
        )
        return None

    logger.info(
        f"HITL resume: {decision.decision_type} for tool_id={decision.tool_id}, "
        f"adk_confirmation_id={adk_confirmation_id}"
    )

    approved = decision.decision_type == KAGENT_HITL_DECISION_TYPE_APPROVE

    part = genai_types.Part(
        function_response=genai_types.FunctionResponse(
            name=ADK_REQUEST_CONFIRMATION_NAME,
            id=adk_confirmation_id,
            response={
                "confirmed": approved,
            },
        )
    )
    return genai_types.Content(parts=[part], role="user")
