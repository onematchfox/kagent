from __future__ import annotations

import inspect
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

import httpx
from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import (
    Artifact,
    DataPart,
    Message,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from a2a.types import Message as A2AMessage
from a2a.types import Part as A2APart
from google.adk.events import Event, EventActions
from google.adk.runners import Runner
from google.adk.utils.context_utils import Aclosing
from google.genai import types as genai_types
from pydantic import BaseModel
from typing_extensions import override

from kagent.core.a2a import (
    KAGENT_HITL_DECISION_TYPE_APPROVE,
    KAGENT_HITL_DECISION_TYPE_KEY,
    KAGENT_HITL_INTERRUPT_TYPE_TOOL_APPROVAL,
    TaskResultAggregator,
    ToolDecision,
    extract_decision_from_message,
    find_pending_tool_request,
    get_kagent_metadata_key,
)
from kagent.core.tracing._span_processor import (
    clear_kagent_span_attributes,
    set_kagent_span_attributes,
)

from .converters.event_converter import convert_event_to_a2a_events
from .converters.request_converter import (
    convert_a2a_request_to_adk_run_args,
    convert_tool_decision_to_adk_function_response,
)

logger = logging.getLogger("kagent_adk." + __name__)

# Session state key for storing pending child agent HITL contexts
PENDING_CHILD_HITL_KEY = "kagent_pending_child_hitl"


def _extract_content_from_part(part: Any) -> str | None:
    """Extract string content from an A2A part, handling various part types.

    Handles TextPart, DataPart (extracts JSON), and nested structures.

    Args:
        part: An A2A message part (may have .root for RootModel types)

    Returns:
        String content if found, None otherwise
    """
    # Handle RootModel wrapping
    inner = part.root if hasattr(part, "root") else part

    # TextPart
    if hasattr(inner, "text") and inner.text:
        return inner.text

    # DataPart - convert to JSON string
    if hasattr(inner, "data") and inner.data:
        try:
            return json.dumps(inner.data) if isinstance(inner.data, dict) else str(inner.data)
        except (TypeError, ValueError):
            return str(inner.data)

    # FilePart or other types - skip
    return None


class A2aAgentExecutorConfig(BaseModel):
    """Configuration for the A2aAgentExecutor."""

    stream: bool = False


# This class is a copy of the A2aAgentExecutor class in the ADK sdk,
# with the following changes:
# - The runner is ALWAYS a callable that returns a Runner instance
# - The runner is cleaned up at the end of the execution
class A2aAgentExecutor(AgentExecutor):
    """An AgentExecutor that runs an ADK Agent against an A2A request and
    publishes updates to an event queue.
    """

    def __init__(
        self,
        *,
        runner: Callable[..., Runner | Awaitable[Runner]],
        config: Optional[A2aAgentExecutorConfig] = None,
    ):
        super().__init__()
        self._runner = runner
        self._config = config

    async def _resolve_runner(self) -> Runner:
        """Resolve the runner, handling cases where it's a callable that returns a Runner."""
        if callable(self._runner):
            # Call the function to get the runner
            result = self._runner()

            # Handle async callables
            if inspect.iscoroutine(result):
                resolved_runner = await result
            else:
                resolved_runner = result

            # Ensure we got a Runner instance
            if not isinstance(resolved_runner, Runner):
                raise TypeError(f"Callable must return a Runner instance, got {type(resolved_runner)}")

            return resolved_runner

        raise TypeError(
            f"Runner must be a Runner instance or a callable that returns a Runner, got {type(self._runner)}"
        )

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """Cancel the execution."""
        # TODO: Implement proper cancellation logic if needed
        raise NotImplementedError("Cancellation is not supported")

    @override
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ):
        """Executes an A2A request and publishes updates to the event queue
        specified. It runs as following:
        * Takes the input from the A2A request
        * Convert the input to ADK input content, and runs the ADK agent
        * Collects output events of the underlying ADK Agent
        * Converts the ADK output events into A2A task updates
        * Publishes the updates back to A2A server via event queue
        """
        if not context.message:
            raise ValueError("A2A request must have a message")

        # Convert the a2a request to ADK run args
        stream = self._config.stream if self._config is not None else False
        run_args = convert_a2a_request_to_adk_run_args(context, stream=stream)

        # Prepare span attributes.
        span_attributes = {}
        if run_args.get("user_id"):
            span_attributes["kagent.user_id"] = run_args["user_id"]
        if context.task_id:
            span_attributes["gen_ai.task.id"] = context.task_id
        if run_args.get("session_id"):
            span_attributes["gen_ai.conversation.id"] = run_args["session_id"]

        # Set kagent span attributes for all spans in context.
        context_token = set_kagent_span_attributes(span_attributes)
        try:
            # for new task, create a task submitted event
            if not context.current_task:
                await event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        task_id=context.task_id,
                        status=TaskStatus(
                            state=TaskState.submitted,
                            message=context.message,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                        ),
                        context_id=context.context_id,
                        final=False,
                    )
                )

            # Handle the request and publish updates to the event queue
            runner = await self._resolve_runner()
            try:
                await self._handle_request(context, event_queue, runner, run_args)
            except Exception as e:
                logger.error("Error handling A2A request: %s", e, exc_info=True)

                # Check if this is a LiteLLM JSON parsing error (common with Ollama models that don't support function calling)
                error_message = str(e)
                if (
                    "JSONDecodeError" in error_message
                    or "Unterminated string" in error_message
                    or "APIConnectionError" in error_message
                ):
                    # Check if it's related to function calling
                    if "function_call" in error_message.lower() or "json.loads" in error_message:
                        error_message = (
                            "The model does not support function calling properly. "
                            "This error typically occurs when using Ollama models with tools. "
                            "Please either:\n"
                            "1. Remove tools from the agent configuration, or\n"
                            "2. Use a model that supports function calling (e.g., OpenAI, Anthropic, or Gemini models)."
                        )
                # Publish failure event
                try:
                    await event_queue.enqueue_event(
                        TaskStatusUpdateEvent(
                            task_id=context.task_id,
                            status=TaskStatus(
                                state=TaskState.failed,
                                timestamp=datetime.now(timezone.utc).isoformat(),
                                message=Message(
                                    message_id=str(uuid.uuid4()),
                                    role=Role.agent,
                                    parts=[Part(TextPart(text=error_message))],
                                ),
                            ),
                            context_id=context.context_id,
                            final=True,
                        )
                    )
                except Exception as enqueue_error:
                    logger.error("Failed to publish failure event: %s", enqueue_error, exc_info=True)
        finally:
            clear_kagent_span_attributes(context_token)
            # close the runner which cleans up the mcptoolsets
            # since the runner is created for each a2a request
            # and the mcptoolsets are not shared between requests
            # this is necessary to gracefully handle mcp toolset connections
            await runner.close()

    async def _handle_request(
        self,
        context: RequestContext,
        event_queue: EventQueue,
        runner: Runner,
        run_args: dict[str, Any],
    ):
        # ensure the session exists
        session = await self._prepare_session(context, run_args, runner)

        # set request headers to session state
        headers = context.call_context.state.get("headers", {})
        state_changes = {
            "headers": headers,
        }

        actions_with_update = EventActions(state_delta=state_changes)
        system_event = Event(
            invocation_id="header_update",
            author="system",
            actions=actions_with_update,
        )

        await runner.session_service.append_event(session, system_event)

        if context.current_task:
            tool_decision = extract_decision_from_message(context.message)
            if tool_decision:
                # Check if this is a decision for a child agent's tool (multi-agent HITL)
                if tool_decision.child_agent_name:
                    logger.info(
                        f"Child agent tool decision: {tool_decision.decision_type} for "
                        f"tool {tool_decision.tool_id} on agent {tool_decision.child_agent_name}"
                    )
                    # Forward the decision directly to the child agent
                    child_response, child_response_text = await self._forward_decision_to_child(
                        runner,
                        tool_decision.child_agent_name,
                        tool_decision,
                        session,
                    )
                    if child_response:
                        run_args["new_message"] = child_response

                        # Emit an event with the child's response so the UI can update the tool call display
                        # This makes the child agent's result visible to the user before the parent processes it
                        if child_response_text:
                            # Get the function_call_id from the child_response
                            function_call_id = None
                            if child_response.parts:
                                for part in child_response.parts:
                                    if part.function_response:
                                        function_call_id = part.function_response.id
                                        break

                            # Create an ADK event for persistence
                            # This ensures the child's response is available when loading from history
                            child_response_event = Event(
                                invocation_id=f"child_response_{function_call_id or uuid.uuid4()}",
                                author=tool_decision.child_agent_name,
                                content=child_response,
                            )
                            await runner.session_service.append_event(session, child_response_event)

                            await event_queue.enqueue_event(
                                TaskStatusUpdateEvent(
                                    task_id=context.task_id,
                                    status=TaskStatus(
                                        state=TaskState.working,
                                        timestamp=datetime.now(timezone.utc).isoformat(),
                                        message=Message(
                                            message_id=str(uuid.uuid4()),
                                            role=Role.agent,
                                            parts=[
                                                Part(
                                                    DataPart(
                                                        data={
                                                            "id": function_call_id,
                                                            "name": tool_decision.child_agent_name,
                                                            "response": {"result": child_response_text},
                                                        },
                                                        metadata={get_kagent_metadata_key("type"): "function_response"},
                                                    )
                                                )
                                            ],
                                        ),
                                    ),
                                    context_id=context.context_id,
                                    final=False,
                                )
                            )
                    else:
                        # Failed to forward - DO NOT run the parent agent as it will likely
                        # make a new call to the child instead of processing the decision
                        logger.error(
                            f"Failed to forward decision to child agent '{tool_decision.child_agent_name}'. "
                            "NOT running parent agent to avoid duplicate child calls."
                        )
                        # Publish an error event to inform the user
                        await event_queue.enqueue_event(
                            TaskStatusUpdateEvent(
                                task_id=context.task_id,
                                status=TaskStatus(
                                    state=TaskState.failed,
                                    timestamp=datetime.now(timezone.utc).isoformat(),
                                    message=Message(
                                        message_id=str(uuid.uuid4()),
                                        role=Role.agent,
                                        parts=[
                                            Part(
                                                TextPart(
                                                    text=(
                                                        f"Failed to process tool approval for child agent '{tool_decision.child_agent_name}'. "
                                                        "The approval context may have been lost. Please try your request again."
                                                    )
                                                )
                                            )
                                        ],
                                    ),
                                ),
                                context_id=context.context_id,
                                final=True,
                            )
                        )
                        return
                else:
                    # Single-agent tool approval - use ADK's built-in confirmation mechanism
                    confirmation_message = self._prepare_confirmation_message(context, tool_decision)
                    if confirmation_message:
                        run_args["new_message"] = confirmation_message

        # create invocation context
        invocation_context = runner._new_invocation_context(
            session=session,
            new_message=run_args["new_message"],
            run_config=run_args["run_config"],
        )

        # publish the task working event
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                status=TaskStatus(
                    state=TaskState.working,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
                context_id=context.context_id,
                final=False,
                metadata={
                    get_kagent_metadata_key("app_name"): runner.app_name,
                    get_kagent_metadata_key("user_id"): run_args["user_id"],
                    get_kagent_metadata_key("session_id"): run_args["session_id"],
                },
            )
        )

        task_result_aggregator = TaskResultAggregator()
        child_hitl_detected = False
        # Preserve session_name before entering the loop - ADK may modify session.state
        original_session_name = session.state.get("session_name")
        async with Aclosing(runner.run_async(**run_args)) as agen:
            async for adk_event in agen:
                # Check for child agent HITL data and store context for later forwarding
                child_hitl = self._extract_child_hitl_from_event(adk_event)
                if child_hitl:
                    self._store_child_hitl_context(session, child_hitl)
                    child_hitl_detected = True

                for a2a_event in convert_event_to_a2a_events(
                    adk_event, invocation_context, context.task_id, context.context_id
                ):
                    # Only aggregate non-partial events to avoid duplicates from streaming chunks
                    # Partial events are sent to frontend for display but not accumulated
                    if not adk_event.partial:
                        task_result_aggregator.process_event(a2a_event)
                    await event_queue.enqueue_event(a2a_event)

                # When child HITL is detected, break out of the loop to stop processing
                # This prevents the parent's LLM from seeing the tool_approval JSON and
                # trying to respond to it. The TaskResultAggregator has already captured
                # the input_required state from the converted event.
                if child_hitl_detected:
                    logger.info(
                        f"Child agent HITL detected, breaking runner loop to await user decision. "
                        f"Aggregator state: {task_result_aggregator.task_state}"
                    )
                    # Persist the session state so child HITL context is available
                    # when the user sends their approval decision
                    # Ensure session_name is preserved (ADK may have modified session.state)
                    state_to_persist = dict(session.state)
                    if original_session_name and "session_name" not in state_to_persist:
                        state_to_persist["session_name"] = original_session_name
                    await runner.session_service.create_session(
                        app_name=runner.app_name,
                        user_id=run_args["user_id"],
                        session_id=run_args["session_id"],
                        state=state_to_persist,
                    )
                    break

                # Break out of runner loop when input is required (tool confirmation,
                # child agent HITL, or other scenarios needing external input).
                # This ensures we stop processing and wait for user approval before
                # continuing with tool execution or child agent delegation.
                if task_result_aggregator.task_state == TaskState.input_required:
                    logger.info("Breaking runner loop: input_required state detected")
                    break

        # publish the task result event - this is final
        if (
            task_result_aggregator.task_state == TaskState.working
            and task_result_aggregator.task_status_message is not None
            and task_result_aggregator.task_status_message.parts
        ):
            # if task is still working properly, publish the artifact update event as
            # the final result according to a2a protocol.
            await event_queue.enqueue_event(
                TaskArtifactUpdateEvent(
                    task_id=context.task_id,
                    last_chunk=True,
                    context_id=context.context_id,
                    artifact=Artifact(
                        artifact_id=str(uuid.uuid4()),
                        parts=task_result_aggregator.task_status_message.parts,
                    ),
                )
            )
            # publish the final status update event
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=context.task_id,
                    status=TaskStatus(
                        state=TaskState.completed,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ),
                    context_id=context.context_id,
                    final=True,
                )
            )
        else:
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=context.task_id,
                    status=TaskStatus(
                        state=task_result_aggregator.task_state,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        message=task_result_aggregator.task_status_message,
                    ),
                    context_id=context.context_id,
                    final=True,
                )
            )

    async def _prepare_session(self, context: RequestContext, run_args: dict[str, Any], runner: Runner):
        session_id = run_args["session_id"]
        # create a new session if not exists
        user_id = run_args["user_id"]
        session = await runner.session_service.get_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id,
        )

        if session is None:
            # Extract session name from the first TextPart (like the UI does)
            session_name = None
            if context.message and context.message.parts:
                for part in context.message.parts:
                    # A2A parts have a .root property that contains the actual part (TextPart, FilePart, etc.)
                    if isinstance(part, Part):
                        root_part = part.root
                        if isinstance(root_part, TextPart) and root_part.text:
                            # Take first 20 chars + "..." if longer (matching UI behavior)
                            text = root_part.text.strip()
                            session_name = text[:20] + ("..." if len(text) > 20 else "")
                            break

            session = await runner.session_service.create_session(
                app_name=runner.app_name,
                user_id=user_id,
                state={"session_name": session_name},
                session_id=session_id,
            )

            # Update run_args with the new session_id
            run_args["session_id"] = session.id

        return session

    def _prepare_confirmation_message(
        self,
        context: RequestContext,
        tool_decision: ToolDecision,
    ) -> genai_types.Content | None:
        """Prepare confirmation message for HITL resume.

        Finds the pending tool request matching the user's decision and extracts
        the confirmation_id from its metadata to create an ADK FunctionResponse.

        Args:
            context: The request context containing the current task
            tool_decision: The user's decision (approve/deny) and tool ID

        Returns:
            The confirmation Content to send to ADK, or None if no match found
        """
        matched_request = find_pending_tool_request(context.current_task, tool_decision.tool_id)
        if not matched_request:
            logger.error(f"No pending tool request found for tool_id: {tool_decision.tool_id}")
            return None

        # Get the ADK confirmation ID from the request's metadata
        confirmation_id = matched_request.metadata.get("confirmation_id") if matched_request.metadata else None
        if not confirmation_id:
            logger.error(f"No confirmation_id in metadata for tool request: {matched_request.id}")
            return None

        logger.info(
            f"HITL resume: {tool_decision.decision_type} for {matched_request.name} (confirmation_id={confirmation_id})"
        )

        # Create a ToolDecision with the ADK confirmation ID for the response
        decision_with_confirmation_id = ToolDecision(
            decision_type=tool_decision.decision_type,
            tool_id=confirmation_id,
        )
        return convert_tool_decision_to_adk_function_response(decision_with_confirmation_id)

    def _extract_child_hitl_from_event(self, event: Event) -> dict[str, Any] | None:
        """Extract child agent HITL data from an ADK event.

        When a RemoteA2aAgent (child) returns input_required with tool_approval data,
        the parent receives it as a function_response. This method detects and extracts
        the relevant context for forwarding the approval later.

        Args:
            event: The ADK event to check

        Returns:
            Dict with child HITL context if found, None otherwise
        """
        if not event.content or not event.content.parts:
            return None

        for part in event.content.parts:
            if not part.function_response:
                continue

            response = part.function_response.response
            if not isinstance(response, (dict, str)):
                continue

            # Parse string responses (child might return JSON string)
            if isinstance(response, str):
                try:
                    response = json.loads(response)
                except (json.JSONDecodeError, TypeError):
                    continue

            if not isinstance(response, dict):
                continue

            # RemoteA2aAgent wraps the child's response in {"result": "..."}
            # The tool_approval data may be inside response["result"] as a JSON string
            tool_approval_data = None
            if response.get("interrupt_type") == KAGENT_HITL_INTERRUPT_TYPE_TOOL_APPROVAL:
                tool_approval_data = response
            elif "result" in response:
                # Check if the result contains tool_approval data
                result = response.get("result")
                if isinstance(result, str):
                    try:
                        parsed_result = json.loads(result)
                        if (
                            isinstance(parsed_result, dict)
                            and parsed_result.get("interrupt_type") == KAGENT_HITL_INTERRUPT_TYPE_TOOL_APPROVAL
                        ):
                            tool_approval_data = parsed_result
                    except (json.JSONDecodeError, TypeError):
                        pass
                elif (
                    isinstance(result, dict)
                    and result.get("interrupt_type") == KAGENT_HITL_INTERRUPT_TYPE_TOOL_APPROVAL
                ):
                    tool_approval_data = result

            if not tool_approval_data:
                continue

            action_requests = tool_approval_data.get("action_requests", [])
            if not action_requests:
                continue

            # Extract child context from action_requests metadata
            # The child agent includes a2a_context_id and a2a_task_id in the tool_approval event
            child_context_id = None
            child_task_id = None
            if action_requests:
                first_request = action_requests[0]
                request_metadata = first_request.get("metadata", {})
                child_context_id = request_metadata.get("a2a_context_id")
                child_task_id = request_metadata.get("a2a_task_id")

            logger.info(
                f"Detected child agent HITL in event: agent={part.function_response.name}, "
                f"action_requests={len(action_requests)}, "
                f"child_context_id={child_context_id}, child_task_id={child_task_id}"
            )

            # Get the agent name from the function_response (it's the AgentTool name)
            agent_name = part.function_response.name

            return {
                "agent_name": agent_name,
                "context_id": child_context_id,
                "task_id": child_task_id,
                "action_requests": action_requests,
                "function_call_id": part.function_response.id,
            }

        return None

    def _store_child_hitl_context(
        self,
        session: Any,
        child_hitl: dict[str, Any],
    ) -> None:
        """Store child HITL context in parent's session state for later retrieval.

        State is persisted via KAgentSessionService when the runner loop breaks
        (create_session with state_to_persist), so any pod can serve the next request.

        Args:
            session: The parent's session
            child_hitl: The extracted child HITL context
        """
        if PENDING_CHILD_HITL_KEY not in session.state:
            session.state[PENDING_CHILD_HITL_KEY] = {}

        agent_name = child_hitl["agent_name"]
        context_data = {
            "context_id": child_hitl["context_id"],
            "task_id": child_hitl["task_id"],
            "action_requests": child_hitl["action_requests"],
            "function_call_id": child_hitl["function_call_id"],
        }
        session.state[PENDING_CHILD_HITL_KEY][agent_name] = context_data

        logger.info(
            f"Stored child HITL context for agent '{agent_name}': "
            f"context_id={child_hitl['context_id']}, task_id={child_hitl['task_id']}, "
            f"function_call_id={child_hitl.get('function_call_id')}, session_id={session.id}"
        )

    async def _forward_decision_to_child(
        self,
        runner: Runner,
        child_agent_name: str,
        tool_decision: ToolDecision,
        session: Any,
    ) -> tuple[genai_types.Content | None, str | None]:
        """Forward a tool decision to a child agent via direct A2A call.

        Args:
            runner: The parent's runner (to access child agent tools)
            child_agent_name: Name of the child agent
            tool_decision: The user's decision
            session: The parent's session (to get stored child context)

        Returns:
            A tuple of (Content object with the child's response, response text) or (None, None) if failed
        """
        # Get stored child context from session state (persisted via session service on break)
        logger.info(
            f"Looking up child HITL context: agent='{child_agent_name}', session_id='{session.id}'"
        )

        pending_child = session.state.get(PENDING_CHILD_HITL_KEY, {}).get(child_agent_name)
        if not pending_child:
            session_state_agents = list(session.state.get(PENDING_CHILD_HITL_KEY, {}).keys())
            logger.error(
                f"No pending child HITL context found for agent '{child_agent_name}'. "
                f"Session state agents: {session_state_agents}"
            )
            return None, None

        child_context_id = pending_child.get("context_id")
        child_task_id = pending_child.get("task_id")
        action_requests = pending_child.get("action_requests", [])

        if not child_context_id:
            logger.error(f"No context_id in stored child HITL context for agent '{child_agent_name}'")
            return None, None

        # Find the child agent's RemoteA2aAgent in the runner's tools
        child_agent = None
        from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
        from google.adk.tools.agent_tool import AgentTool

        for tool in runner.agent.tools:
            if isinstance(tool, AgentTool) and tool.agent.name == child_agent_name:
                if isinstance(tool.agent, RemoteA2aAgent):
                    child_agent = tool.agent
                    break

        if not child_agent:
            logger.error(f"Could not find RemoteA2aAgent for '{child_agent_name}'")
            return None, None

        # Ensure the child agent is resolved (has HTTP client and agent card)
        try:
            await child_agent._ensure_resolved()
        except Exception as e:
            logger.error(f"Failed to resolve child agent '{child_agent_name}': {e}")
            return None, None

        # Build the decision message to send to child
        # The child expects a message with the tool decision in its data part
        decision_parts: list[A2APart] = []

        # Add the decision as a DataPart
        for action_request in action_requests:
            # Use the LLM-assigned tool call ID (e.g., "call_...")
            # This is what find_pending_tool_request() matches against
            # The child's _prepare_confirmation_message will look up the ADK
            # confirmation ID from the matched request's metadata
            tool_id = action_request.get("id")

            decision_data = {
                KAGENT_HITL_DECISION_TYPE_KEY: tool_decision.decision_type,
                "tool_id": tool_id,  # Use LLM call ID, not ADK confirmation_id
            }
            decision_parts.append(A2APart(DataPart(data=decision_data)))

        # Add a text summary
        decision_text = (
            f"Tool decision: {tool_decision.decision_type}"
            if tool_decision.decision_type == KAGENT_HITL_DECISION_TYPE_APPROVE
            else "Tool execution denied"
        )
        decision_parts.append(A2APart(TextPart(text=decision_text)))

        # Create the A2A message
        decision_message = A2AMessage(
            message_id=str(uuid.uuid4()),
            role=Role.user,
            parts=decision_parts,
            context_id=child_context_id,
            task_id=child_task_id,
        )

        logger.info(
            f"Forwarding decision to child agent '{child_agent_name}': "
            f"{tool_decision.decision_type}, context_id={child_context_id}"
        )

        # Send the message via the child's A2A client
        try:
            response_content = None
            final_state = None
            async for a2a_response in child_agent._a2a_client.send_message(request=decision_message):
                logger.debug(f"Received A2A response: type={type(a2a_response)}")
                # Process the response - we want the final result
                # The response could be a tuple (task, update) or an A2AMessage
                if isinstance(a2a_response, tuple):
                    task, update = a2a_response
                    if task and task.status:
                        final_state = task.status.state
                        logger.debug(
                            f"Tuple response: state={final_state}, has_message={task.status.message is not None}, has_artifacts={bool(getattr(task, 'artifacts', None))}"
                        )

                        # Try to extract content from various locations
                        response_parts = []

                        # 1. Check task.status.message (primary location for streaming updates)
                        if task.status.message and task.status.message.parts:
                            for part in task.status.message.parts:
                                content = _extract_content_from_part(part)
                                if content:
                                    response_parts.append(content)

                        # 2. Check task.artifacts (location for completed task results)
                        if not response_parts and hasattr(task, "artifacts") and task.artifacts:
                            for artifact in task.artifacts:
                                if hasattr(artifact, "parts") and artifact.parts:
                                    for part in artifact.parts:
                                        content = _extract_content_from_part(part)
                                        if content:
                                            response_parts.append(content)

                        # 3. Check task.history for the last agent message
                        if not response_parts and hasattr(task, "history") and task.history:
                            for msg in reversed(task.history):
                                if hasattr(msg, "role") and msg.role == Role.agent and hasattr(msg, "parts"):
                                    for part in msg.parts:
                                        content = _extract_content_from_part(part)
                                        if content:
                                            response_parts.append(content)
                                    if response_parts:
                                        break

                        if response_parts:
                            candidate_content = "\n".join(response_parts)
                            # Only update if this looks like actual content (not just status)
                            if len(candidate_content) > 50 or final_state == TaskState.completed:
                                response_content = candidate_content
                                logger.info(
                                    f"Extracted response content from child: {len(response_content)} chars, state={final_state}"
                                )
                        elif final_state == TaskState.completed:
                            # Log what we have for debugging
                            logger.warning(
                                f"Task completed but no content found. "
                                f"message_parts={len(task.status.message.parts) if task.status.message and task.status.message.parts else 0}, "
                                f"artifacts={len(task.artifacts) if hasattr(task, 'artifacts') and task.artifacts else 0}, "
                                f"history={len(task.history) if hasattr(task, 'history') and task.history else 0}"
                            )
                elif hasattr(a2a_response, "parts"):
                    response_parts = []
                    for part in a2a_response.parts:
                        content = _extract_content_from_part(part)
                        if content:
                            response_parts.append(content)
                    if response_parts:
                        response_content = "\n".join(response_parts)
                        logger.info(f"Extracted response content from message: {len(response_content)} chars")

            # Clear the pending child HITL context and persist so any pod sees updated state
            if PENDING_CHILD_HITL_KEY in session.state:
                session.state[PENDING_CHILD_HITL_KEY].pop(child_agent_name, None)
            await runner.session_service.create_session(
                app_name=runner.app_name,
                user_id=session.user_id,
                session_id=session.id,
                state=dict(session.state),
            )

            if response_content:
                logger.info(f"Successfully got response from child agent '{child_agent_name}'")
                # Create a FunctionResponse to return to the parent's LLM
                # Note: FunctionResponse.response must be a dict, not a string
                function_response = genai_types.FunctionResponse(
                    name=child_agent_name,
                    id=pending_child.get("function_call_id"),
                    response={"result": response_content},
                )
                return genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(function_response=function_response)],
                ), response_content

            logger.warning(f"No response content from child agent '{child_agent_name}'")
            return None, None

        except Exception as e:
            logger.error(f"Failed to forward decision to child agent '{child_agent_name}': {e}")
            return None, None
