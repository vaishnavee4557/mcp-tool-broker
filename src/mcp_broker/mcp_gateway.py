from __future__ import annotations

from typing import Any

from fastmcp import Client  # type: ignore[import]

from mcp_broker.models import ToolDescriptor


class MCPGateway:
    """
    MCP client layer.

    Connects the broker to MCP servers, discovers available tools,
    calls selected tools, and normalizes their responses.
    """

    def __init__(
        self,
        servers: dict[str, str],
    ) -> None:
        self.servers = servers

    def _get_server_url(
        self,
        server_name: str,
    ) -> str:
        server_url = self.servers.get(server_name)

        if not server_url:
            raise ValueError(
                f"MCP server '{server_name}' is not configured"
            )

        return server_url

    async def list_tools(
        self,
        server_name: str,
    ) -> list[ToolDescriptor]:
        """
        Get all tools available on one MCP server and convert them
        into the broker's ToolDescriptor model.
        """

        server_url = self._get_server_url(server_name)

        async with Client(server_url) as client:
            result = await client.list_tools()

        tools: list[ToolDescriptor] = []

        for tool in result:
            descriptor_data: dict[str, Any] = {
                "server_name": server_name,
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }

            # Add title only when ToolDescriptor defines that field.
            if "title" in ToolDescriptor.model_fields:
                descriptor_data["title"] = getattr(
                    tool,
                    "title",
                    None,
                )

            descriptor = ToolDescriptor.model_validate(
                descriptor_data
            )

            tools.append(descriptor)

        return tools

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """
        Call one MCP tool and return normal Python data.
        """

        server_url = self._get_server_url(server_name)

        async with Client(server_url) as client:
            result = await client.call_tool(
                tool_name,
                arguments,
                raise_on_error=False,
            )

        if result.is_error:
            error_messages: list[str] = []

            for block in result.content:
                text = getattr(block, "text", None)

                if text:
                    error_messages.append(text)

            error_text = " ".join(error_messages)

            raise RuntimeError(
                error_text
                or f"MCP tool '{tool_name}' failed"
            )

        # FastMCP's normalized Python result.
        if result.data is not None:
            data = result.data

            if hasattr(data, "model_dump"):
                return data.model_dump(mode="json")

            return data

        # Standard MCP structured JSON result.
        if result.structured_content is not None:
            return result.structured_content

        # Fallback for text or other content blocks.
        return [
            getattr(block, "text", str(block))
            for block in result.content
        ]



















# from __future__ import annotations

# from typing import Any


# from fastmcp import Client  # type: ignore[import]



# class MCPGateway:
#     """
#     MCP client layer.

#     It connects the broker to MCP servers, lists available tools,
#     calls selected tools, and normalizes their responses.
#     """

#     def __init__(
#         self,
#         servers: dict[str, str],
#     ) -> None:
#         self.servers = servers

#     def _get_server_url(
#         self,
#         server_name: str,
#     ) -> str:
#         server_url = self.servers.get(server_name)

#         if not server_url:
#             raise ValueError(
#                 f"MCP server '{server_name}' is not configured"
#             )

#         return server_url

#     async def list_tools(
#         self,
#         server_name: str,
#     ) -> list[dict[str, Any]]:
#         """
#         Get all tools available on one MCP server.
#         """

#         server_url = self._get_server_url(server_name)

#         async with Client(server_url) as client:
#             result = await client.list_tools()

#             return [
#                 {
#                     "server_name": server_name,
#                     "name": tool.name,
#                     "title": tool.title,
#                     "description": tool.description,
#                     "input_schema": tool.inputSchema,
#                 }
#                 for tool in result
#             ]

#     async def call_tool(
#         self,
#         server_name: str,
#         tool_name: str,
#         arguments: dict[str, Any],
#     ) -> Any:
#         """
#         Call one MCP tool and return normal Python data.
#         """

#         server_url = self._get_server_url(server_name)

#         async with Client(server_url) as client:
#             result = await client.call_tool(
#                 tool_name,
#                 arguments,
#             )

#         if result.is_error:
#             error_messages = []

#             for block in result.content:
#                 text = getattr(block, "text", None)

#                 if text:
#                     error_messages.append(text)

#             error_text = " ".join(error_messages)

#             raise RuntimeError(
#                 error_text or f"MCP tool '{tool_name}' failed"
#             )

#         if result.structured_content is not None:
#             return result.structured_content

#         return [
#             getattr(block, "text", str(block))
#             for block in result.content
#         ]