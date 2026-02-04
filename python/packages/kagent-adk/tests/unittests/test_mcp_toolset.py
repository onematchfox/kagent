"""Tests for the MCP toolset module (extends ADK McpToolset, optional HITL)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from google.adk.tools.mcp_tool import McpToolset as AdkMcpToolset

from kagent.adk._mcp_toolset import McpTool, McpToolset


class TestMcpTool:
    """Tests for the McpTool class (confirmation wrapper)."""

    def test_preserves_schema(self):
        from google.genai import types as genai_types

        mock_mcp_tool = MagicMock()
        mock_mcp_tool.name = "query_documentation"
        mock_mcp_tool.description = "Query documentation"
        mock_mcp_tool._get_declaration.return_value = genai_types.FunctionDeclaration(
            name="query_documentation",
            description="Query documentation",
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "queryText": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        description="The query text",
                    ),
                    "productName": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        description="The product name",
                    ),
                },
                required=["queryText", "productName"],
            ),
        )

        tool = McpTool(mock_mcp_tool)
        declaration = tool._get_declaration()

        assert declaration is not None
        assert declaration.name == "query_documentation"
        assert declaration.parameters is not None
        assert "queryText" in declaration.parameters.properties
        assert "productName" in declaration.parameters.properties

    def test_is_long_running(self):
        mock_mcp_tool = MagicMock()
        mock_mcp_tool.name = "test_tool"
        mock_mcp_tool.description = "A test tool"
        tool = McpTool(mock_mcp_tool)
        assert tool.is_long_running is True

    def test_preserves_description(self):
        mock_mcp_tool = MagicMock()
        mock_mcp_tool.name = "test_tool"
        mock_mcp_tool.description = "Original description"
        tool = McpTool(mock_mcp_tool)
        assert tool.description == "Original description"

    @pytest.mark.asyncio
    async def test_requests_confirmation_on_first_call(self):
        mock_mcp_tool = MagicMock()
        mock_mcp_tool.name = "test_tool"
        mock_mcp_tool.description = "A test tool"
        mock_tool_context = MagicMock()
        mock_tool_context.tool_confirmation = None

        tool = McpTool(mock_mcp_tool)
        result = await tool.run_async(
            args={"queryText": "test", "productName": "my-product"},
            tool_context=mock_tool_context,
        )

        mock_tool_context.request_confirmation.assert_called_once()
        assert "awaiting" in result.lower()

    @pytest.mark.asyncio
    async def test_executes_on_approval(self):
        mock_mcp_tool = MagicMock()
        mock_mcp_tool.name = "test_tool"
        mock_mcp_tool.description = "A test tool"
        mock_mcp_tool.run_async = AsyncMock(return_value="tool_result")
        mock_tool_context = MagicMock()
        mock_tool_context.tool_confirmation = MagicMock()
        mock_tool_context.tool_confirmation.confirmed = True

        tool = McpTool(mock_mcp_tool)
        result = await tool.run_async(
            args={"queryText": "test"},
            tool_context=mock_tool_context,
        )

        mock_mcp_tool.run_async.assert_called_once()
        assert result == "tool_result"

    @pytest.mark.asyncio
    async def test_returns_denied_on_rejection(self):
        mock_mcp_tool = MagicMock()
        mock_mcp_tool.name = "test_tool"
        mock_mcp_tool.description = "A test tool"
        mock_tool_context = MagicMock()
        mock_tool_context.tool_confirmation = MagicMock()
        mock_tool_context.tool_confirmation.confirmed = False

        tool = McpTool(mock_mcp_tool)
        result = await tool.run_async(
            args={"queryText": "test"},
            tool_context=mock_tool_context,
        )

        assert "denied" in result.lower()
        mock_mcp_tool.run_async.assert_not_called()


class TestMcpToolset:
    """Tests for the McpToolset class (extends ADK, optional confirmation)."""

    @pytest.mark.asyncio
    async def test_require_confirmation_true_wraps_tools(self):
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tool1.description = "Tool 1"
        mock_tool2 = MagicMock()
        mock_tool2.name = "tool2"
        mock_tool2.description = "Tool 2"
        mock_tools = [mock_tool1, mock_tool2]

        with patch.object(AdkMcpToolset, "get_tools", new_callable=AsyncMock, return_value=mock_tools):
            toolset = McpToolset(
                require_confirmation=True,
                connection_params=MagicMock(),
                tool_filter=None,
                header_provider=None,
            )
            tools = await toolset.get_tools()

        assert len(tools) == 2
        assert all(isinstance(t, McpTool) for t in tools)
        assert all(t.is_long_running for t in tools)
        assert {t.name for t in tools} == {"tool1", "tool2"}

    @pytest.mark.asyncio
    async def test_require_confirmation_false_returns_raw_tools(self):
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tool2 = MagicMock()
        mock_tool2.name = "tool2"
        mock_tools = [mock_tool1, mock_tool2]

        with patch.object(AdkMcpToolset, "get_tools", new_callable=AsyncMock, return_value=mock_tools):
            toolset = McpToolset(
                require_confirmation=False,
                connection_params=MagicMock(),
                tool_filter=None,
                header_provider=None,
            )
            tools = await toolset.get_tools()

        assert len(tools) == 2
        assert tools[0] is mock_tool1
        assert tools[1] is mock_tool2
        assert not any(isinstance(t, McpTool) for t in tools)

    @pytest.mark.asyncio
    async def test_close_inherited(self):
        with patch.object(AdkMcpToolset, "close", new_callable=AsyncMock) as mock_close:
            toolset = McpToolset(
                require_confirmation=False,
                connection_params=MagicMock(),
                tool_filter=None,
                header_provider=None,
            )
            await toolset.close()
            mock_close.assert_called_once()
