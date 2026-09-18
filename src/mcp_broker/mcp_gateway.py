from __future__ import annotations

from typing import Any

from fastmcp import Client  # type: ignore[import]

from mcp_broker.models import ToolDescriptor


class MCPGateway:
    """
    MCP client boundary.

    Responsibilities:
    - connect to configured MCP servers
    - discover the LIVE tool catalog
    - normalize MCP tool metadata into ToolDescriptor
    - execute the selected tool on the correct MCP server
    - normalize MCP results into normal Python data

    This layer contains no question-specific or API-specific routing logic.
    """

    def __init__(
        self,
        servers: dict[str, str],
    ) -> None:
        cleaned_servers = {
            str(name).strip():
                str(url).strip()
            for name, url in servers.items()
            if (
                str(name).strip()
                and str(url).strip()
            )
        }

        if not cleaned_servers:
            raise ValueError(
                "At least one MCP server must be configured."
            )

        self.servers = cleaned_servers

    # ========================================================
    # SERVER LOOKUP
    # ========================================================

    def _get_server_url(
        self,
        server_name: str,
    ) -> str:
        server_name = str(
            server_name or ""
        ).strip()

        server_url = self.servers.get(
            server_name
        )

        if not server_url:
            raise ValueError(
                f"MCP server {server_name!r} is not configured. "
                f"Configured servers: {sorted(self.servers)}"
            )

        return server_url

    # ========================================================
    # GENERIC OBJECT HELPERS
    # ========================================================

    @staticmethod
    def _model_dump(
        value: Any,
    ) -> dict[str, Any]:
        """
        Convert Pydantic/dataclass-like MCP objects to a dict when possible.
        """

        if value is None:
            return {}

        if isinstance(
            value,
            dict,
        ):
            return dict(value)

        if hasattr(
            value,
            "model_dump",
        ):
            try:
                dumped = value.model_dump(
                    mode="python"
                )

                if isinstance(
                    dumped,
                    dict,
                ):
                    return dumped
            except Exception:
                pass

        if hasattr(
            value,
            "dict",
        ):
            try:
                dumped = value.dict()

                if isinstance(
                    dumped,
                    dict,
                ):
                    return dumped
            except Exception:
                pass

        return {}

    @classmethod
    def _first_value(
        cls,
        source: Any,
        *names: str,
        default: Any = None,
    ) -> Any:
        """
        Read equivalent snake_case/camelCase fields robustly.
        """

        source_dict = cls._model_dump(
            source
        )

        for name in names:
            if name in source_dict:
                value = source_dict[
                    name
                ]

                if value is not None:
                    return value

            try:
                value = getattr(
                    source,
                    name,
                )
            except Exception:
                value = None

            if value is not None:
                return value

        return default

    @classmethod
    def _normalize_schema(
        cls,
        value: Any,
    ) -> dict[str, Any] | None:
        if value is None:
            return None

        if isinstance(
            value,
            dict,
        ):
            return value

        dumped = cls._model_dump(
            value
        )

        if dumped:
            return dumped

        return None

    @classmethod
    def _normalize_annotations(
        cls,
        value: Any,
    ) -> dict[str, Any] | None:
        if value is None:
            return None

        if isinstance(
            value,
            dict,
        ):
            return value

        dumped = cls._model_dump(
            value
        )

        return dumped or None

    # ========================================================
    # TOOL DISCOVERY
    # ========================================================

    async def list_tools(
        self,
        server_name: str,
    ) -> list[ToolDescriptor]:
        """
        Discover all live tools from one MCP server.

        Preserve as much generic metadata as the MCP implementation exposes,
        because registry -> tool_store -> embeddings -> pgvector depends on it.
        """

        server_url = self._get_server_url(
            server_name
        )

        async with Client(
            server_url
        ) as client:
            result = await client.list_tools()

        if result is None:
            return []

        tools: list[
            ToolDescriptor
        ] = []

        for tool in result:
            name = str(
                self._first_value(
                    tool,
                    "name",
                    default="",
                )
                or ""
            ).strip()

            if not name:
                # A nameless MCP tool cannot be called safely.
                continue

            description = str(
                self._first_value(
                    tool,
                    "description",
                    default="",
                )
                or ""
            ).strip()

            input_schema = (
                self._normalize_schema(
                    self._first_value(
                        tool,
                        "inputSchema",
                        "input_schema",
                        default={},
                    )
                )
                or {}
            )

            output_schema = (
                self._normalize_schema(
                    self._first_value(
                        tool,
                        "outputSchema",
                        "output_schema",
                        default=None,
                    )
                )
            )

            response_schema = (
                self._normalize_schema(
                    self._first_value(
                        tool,
                        "responseSchema",
                        "response_schema",
                        default=None,
                    )
                )
            )

            annotations = (
                self._normalize_annotations(
                    self._first_value(
                        tool,
                        "annotations",
                        default=None,
                    )
                )
            )

            # Generic transport/OpenAPI metadata may or may not be exposed
            # by a particular FastMCP version/adapter.
            base_url = self._first_value(
                tool,
                "base_url",
                "baseUrl",
                default=None,
            )

            http_method = self._first_value(
                tool,
                "http_method",
                "httpMethod",
                "method",
                default=None,
            )

            api_path = self._first_value(
                tool,
                "api_path",
                "apiPath",
                "path",
                default=None,
            )

            descriptor_data: dict[
                str,
                Any,
            ] = {
                "server_name":
                    str(server_name),
                "name":
                    name,
                "description":
                    description,
                "input_schema":
                    input_schema,
                "output_schema":
                    output_schema,
                "response_schema":
                    response_schema,
                "annotations":
                    annotations,
                "base_url":
                    (
                        str(base_url).strip()
                        if base_url is not None
                        else None
                    ),
                "http_method":
                    (
                        str(http_method).strip()
                        if http_method is not None
                        else None
                    ),
                "api_path":
                    (
                        str(api_path).strip()
                        if api_path is not None
                        else None
                    ),
            }

            descriptor = (
                ToolDescriptor.model_validate(
                    descriptor_data
                )
            )

            tools.append(
                descriptor
            )

        return tools

    # ========================================================
    # TOOL EXECUTION
    # ========================================================

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """
        Call one MCP tool on the requested server and return normalized data.

        Important for multi-server operation:
        the caller must pass the selected ToolDescriptor.server_name rather
        than a global hardcoded server name.
        """

        server_url = self._get_server_url(
            server_name
        )

        tool_name = str(
            tool_name or ""
        ).strip()

        if not tool_name:
            raise ValueError(
                "MCP tool name cannot be empty."
            )

        if not isinstance(
            arguments,
            dict,
        ):
            raise TypeError(
                "MCP tool arguments must be a dictionary."
            )

        async with Client(
            server_url
        ) as client:
            result = await client.call_tool(
                tool_name,
                arguments,
                raise_on_error=False,
            )

        if result is None:
            raise RuntimeError(
                f"MCP tool {tool_name!r} returned no result object."
            )

        is_error = bool(
            self._first_value(
                result,
                "is_error",
                "isError",
                default=False,
            )
        )

        if is_error:
            error_text = (
                self._extract_error_text(
                    result
                )
            )

            raise RuntimeError(
                error_text
                or (
                    f"MCP tool {tool_name!r} "
                    f"failed on server {server_name!r}."
                )
            )

        # ----------------------------------------------------
        # 1. FastMCP normalized Python result
        # ----------------------------------------------------

        data = self._first_value(
            result,
            "data",
            default=None,
        )

        if data is not None:
            return self._to_python(
                data
            )

        # ----------------------------------------------------
        # 2. Standard MCP structured result
        # ----------------------------------------------------

        structured = self._first_value(
            result,
            "structured_content",
            "structuredContent",
            default=None,
        )

        if structured is not None:
            return self._to_python(
                structured
            )

        # ----------------------------------------------------
        # 3. Fallback content blocks
        # ----------------------------------------------------

        content = self._first_value(
            result,
            "content",
            default=[],
        )

        if not isinstance(
            content,
            (list, tuple),
        ):
            content = [
                content
            ]

        normalized_blocks = [
            self._content_block_to_python(
                block
            )
            for block in content
        ]

        # Common case: one text block.
        if len(normalized_blocks) == 1:
            return normalized_blocks[0]

        return normalized_blocks

    # ========================================================
    # RESULT NORMALIZATION
    # ========================================================

    @classmethod
    def _to_python(
        cls,
        value: Any,
    ) -> Any:
        if value is None:
            return None

        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
            ),
        ):
            return value

        if isinstance(
            value,
            dict,
        ):
            return {
                str(key):
                    cls._to_python(
                        child
                    )
                for key, child
                in value.items()
            }

        if isinstance(
            value,
            (list, tuple),
        ):
            return [
                cls._to_python(
                    child
                )
                for child in value
            ]

        if hasattr(
            value,
            "model_dump",
        ):
            try:
                return value.model_dump(
                    mode="json"
                )
            except Exception:
                try:
                    return value.model_dump()
                except Exception:
                    pass

        return value

    @classmethod
    def _content_block_to_python(
        cls,
        block: Any,
    ) -> Any:
        if block is None:
            return None

        text = cls._first_value(
            block,
            "text",
            default=None,
        )

        if text is not None:
            return str(text)

        dumped = cls._model_dump(
            block
        )

        if dumped:
            return dumped

        return str(
            block
        )

    @classmethod
    def _extract_error_text(
        cls,
        result: Any,
    ) -> str:
        messages: list[
            str
        ] = []

        content = cls._first_value(
            result,
            "content",
            default=[],
        )

        if not isinstance(
            content,
            (list, tuple),
        ):
            content = [
                content
            ]

        for block in content:
            text = cls._first_value(
                block,
                "text",
                default=None,
            )

            if text is not None:
                clean = str(
                    text
                ).strip()

                if clean:
                    messages.append(
                        clean
                    )

        # Some MCP clients expose an error/message field outside content.
        for field in (
            "error",
            "message",
        ):
            value = cls._first_value(
                result,
                field,
                default=None,
            )

            if value is None:
                continue

            clean = str(
                value
            ).strip()

            if clean:
                messages.append(
                    clean
                )

        # Stable de-duplication.
        return " ".join(
            dict.fromkeys(
                messages
            )
        ).strip()



















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