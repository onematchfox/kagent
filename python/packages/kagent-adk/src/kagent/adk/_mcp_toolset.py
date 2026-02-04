"""
MCP toolset with optional human-in-the-loop (HITL) confirmation.

Internal module: used when building agent tools from config (types.py).
Extends ADK's McpToolset to support require_confirmation. When True, tools
are wrapped so that request_confirmation() propagates correctly through
agent hierarchies (unlike ADK's built-in require_confirmation on McpToolset).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.mcp_tool import McpToolset as _AdkMcpToolset
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types

logger = logging.getLogger(__name__)


class McpTool(BaseTool):
    """Wraps an MCP tool with confirmation (HITL) for multi-agent propagation.

    Sets is_long_running=True and implements request_confirmation() / run on
    approval so that confirmation propagates correctly through agent hierarchies.
    """

    def __init__(self, mcp_tool: BaseTool):
        super().__init__(
            name=mcp_tool.name,
            description=mcp_tool.description,
            is_long_running=True,
        )
        self._mcp_tool = mcp_tool

    def _get_declaration(self) -> Optional[genai_types.FunctionDeclaration]:
        return self._mcp_tool._get_declaration()

    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        tool_name = self.name
        tool_confirmation = tool_context.tool_confirmation
        if tool_confirmation:
            approved = tool_confirmation.confirmed
            if approved:
                logger.info(f"MCP Tool '{tool_name}' approved, executing...")
                try:
                    result = await self._mcp_tool.run_async(args=args, tool_context=tool_context)
                    return result
                except Exception as e:
                    logger.error(f"MCP Tool '{tool_name}' execution failed: {e}")
                    return f"Error executing tool: {str(e)}"
            else:
                logger.info(f"MCP Tool '{tool_name}' denied by user")
                return f"Tool '{tool_name}' execution was denied by user."

        logger.info(f"MCP Tool '{tool_name}' requires confirmation.")
        tool_context.request_confirmation(
            hint=f"Tool '{tool_name}' requires approval before execution.",
            payload={"name": tool_name, "args": args},
        )
        return f"Tool '{tool_name}' is awaiting user approval."


class McpToolset(_AdkMcpToolset):
    """MCP toolset that optionally wraps tools with confirmation (HITL).

    Extends ADK's McpToolset. When require_confirmation is True, get_tools()
    returns each tool wrapped in McpTool so confirmation propagates correctly
    through agent hierarchies.
    """

    def __init__(self, *, require_confirmation: bool = False, **kwargs: Any):
        # Do not pass require_confirmation to ADK; we handle it in get_tools().
        kwargs = {k: v for k, v in kwargs.items() if k != "require_confirmation"}
        super().__init__(**kwargs)
        self._require_confirmation = require_confirmation

    async def get_tools(
        self,
        readonly_context: Any = None,
    ) -> list[BaseTool]:
        tools = await super().get_tools(readonly_context)
        if self._require_confirmation:
            return [McpTool(t) for t in tools]
        return tools
