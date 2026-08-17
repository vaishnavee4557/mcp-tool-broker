from __future__ import annotations

import asyncio
import copy
import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx
import jsonschema
import mcp.types as types
from mcp.server import Server, ServerRequestContext


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENAPI_URL = os.getenv(
    "OPENAPI_URL",
    "http://151.185.44.114:30020/openapi.json",
)

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://151.185.44.114:30020",
).rstrip("/")

API_BEARER_TOKEN = os.getenv(
    "API_BEARER_TOKEN",
    "",
).strip()

HTTP_TIMEOUT_SECONDS = float(
    os.getenv("HTTP_TIMEOUT_SECONDS", "30")
)

SUPPORTED_METHODS = {
    "get",
    "post",
    "put",
    "patch",
    "delete",
}


# ---------------------------------------------------------------------------
# Internal models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class APIParameter:
    name: str
    location: str


@dataclass(frozen=True)
class APIOperation:
    """
    One Swagger/OpenAPI operation represented as one MCP tool.
    """

    tool_name: str
    method: str
    path: str
    description: str
    input_schema: dict[str, Any]
    parameters: tuple[APIParameter, ...]
    has_json_body: bool


# ---------------------------------------------------------------------------
# OpenAPI catalogue
# ---------------------------------------------------------------------------

class SwaggerToolCatalog:
    """
    Downloads openapi.json once and creates MCP tools automatically.
    """

    def __init__(self) -> None:
        self._loaded = False
        self._lock = asyncio.Lock()
        self._operations: dict[str, APIOperation] = {}

    async def ensure_loaded(self) -> None:
        if self._loaded:
            return

        async with self._lock:
            if self._loaded:
                return

            specification = await self._download_openapi()
            operations = self._parse_operations(specification)

            self._operations = {
                operation.tool_name: operation
                for operation in operations
            }

            self._loaded = True

            print(
                f"Loaded {len(self._operations)} API operations "
                f"from {OPENAPI_URL}"
            )

    async def list_mcp_tools(self) -> list[types.Tool]:
        await self.ensure_loaded()

        return [
            types.Tool(
            name=operation.tool_name,
            title=operation.tool_name,
            description=operation.description,
            input_schema=operation.input_schema,
            )
            for operation in self._operations.values()
        ]

    async def get_operation(
        self,
        tool_name: str,
    ) -> APIOperation | None:
        await self.ensure_loaded()
        return self._operations.get(tool_name)

    async def _download_openapi(self) -> dict[str, Any]:
        headers = {
            "Accept": "application/json",
        }

        if API_BEARER_TOKEN:
            headers["Authorization"] = (
                f"Bearer {API_BEARER_TOKEN}"
            )

        async with httpx.AsyncClient(
            timeout=HTTP_TIMEOUT_SECONDS,
            headers=headers,
            follow_redirects=False,
        ) as client:
            response = await client.get(OPENAPI_URL)
            response.raise_for_status()
            specification = response.json()

        if not isinstance(specification, dict):
            raise RuntimeError(
                "openapi.json must return a JSON object"
            )

        if not isinstance(
            specification.get("paths"),
            dict,
        ):
            raise RuntimeError(
                "OpenAPI document has no valid 'paths' object"
            )

        return specification

    def _parse_operations(
        self,
        specification: dict[str, Any],
    ) -> list[APIOperation]:
        operations: list[APIOperation] = []
        used_names: set[str] = set()

        paths = specification["paths"]

        for api_path, raw_path_item in paths.items():
            if not isinstance(raw_path_item, dict):
                continue

            path_item = self._resolve_ref(
                specification,
                raw_path_item,
            )

            path_parameters = path_item.get(
                "parameters",
                [],
            )

            for method, raw_operation in path_item.items():
                method_lower = method.lower()

                if method_lower not in SUPPORTED_METHODS:
                    continue

                if not isinstance(raw_operation, dict):
                    continue

                operation = self._resolve_ref(
                    specification,
                    raw_operation,
                )

                operation_id = str(
                    operation.get("operationId")
                    or f"{method_lower}_{api_path}"
                )

                tool_name = self._make_tool_name(
                    operation_id=operation_id,
                    method=method_lower,
                    api_path=api_path,
                    used_names=used_names,
                )

                description = str(
                    operation.get("summary")
                    or operation.get("description")
                    or (
                        f"{method_lower.upper()} "
                        f"{api_path}"
                    )
                )

                (
                    input_schema,
                    parameters,
                    has_json_body,
                ) = self._build_input_schema(
                    specification=specification,
                    path_parameters=path_parameters,
                    operation=operation,
                )

                operations.append(
                    APIOperation(
                        tool_name=tool_name,
                        method=method_lower.upper(),
                        path=api_path,
                        description=description,
                        input_schema=input_schema,
                        parameters=tuple(parameters),
                        has_json_body=has_json_body,
                    )
                )

        return operations

    def _build_input_schema(
        self,
        *,
        specification: dict[str, Any],
        path_parameters: list[dict[str, Any]],
        operation: dict[str, Any],
    ) -> tuple[
        dict[str, Any],
        list[APIParameter],
        bool,
    ]:
        properties: dict[str, Any] = {}
        required: list[str] = []
        parameters: list[APIParameter] = []

        combined_parameters = [
            *path_parameters,
            *operation.get("parameters", []),
        ]

        for raw_parameter in combined_parameters:
            if not isinstance(raw_parameter, dict):
                continue

            parameter = self._resolve_ref(
                specification,
                raw_parameter,
            )

            name = str(
                parameter.get("name", "")
            ).strip()

            location = str(
                parameter.get("in", "query")
            ).strip()

            if not name:
                continue

            raw_schema = parameter.get(
                "schema",
                {"type": "string"},
            )

            schema = self._resolve_schema(
                specification,
                raw_schema,
            )

            description = parameter.get(
                "description"
            )

            if description:
                schema["description"] = description

            # This custom field tells the executor where the
            # argument belongs in the HTTP request.
            schema["x-openapi-in"] = location

            properties[name] = schema
            parameters.append(
                APIParameter(
                    name=name,
                    location=location,
                )
            )

            if (
                parameter.get("required")
                or location == "path"
            ):
                required.append(name)

        has_json_body = False
        request_body = operation.get("requestBody")

        if isinstance(request_body, dict):
            request_body = self._resolve_ref(
                specification,
                request_body,
            )

            content = request_body.get(
                "content",
                {},
            )

            media = self._find_json_media(content)

            if media is not None:
                raw_body_schema = media.get(
                    "schema",
                    {"type": "object"},
                )

                body_schema = self._resolve_schema(
                    specification,
                    raw_body_schema,
                )

                properties["body"] = body_schema
                has_json_body = True

                if request_body.get("required"):
                    required.append("body")

        input_schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }

        if required:
            input_schema["required"] = sorted(
                set(required)
            )

        return (
            input_schema,
            parameters,
            has_json_body,
        )

    def _resolve_ref(
        self,
        specification: dict[str, Any],
        value: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Resolve an OpenAPI object reference such as:
        #/components/parameters/MachineId
        """

        current = copy.deepcopy(value)
        visited: set[str] = set()

        while isinstance(current, dict):
            reference = current.get("$ref")

            if not isinstance(reference, str):
                break

            if not reference.startswith("#/"):
                raise RuntimeError(
                    "External OpenAPI references are not "
                    f"supported: {reference}"
                )

            if reference in visited:
                raise RuntimeError(
                    f"Circular OpenAPI reference: {reference}"
                )

            visited.add(reference)
            current = copy.deepcopy(
                self._lookup_reference(
                    specification,
                    reference,
                )
            )

        return current

    def _resolve_schema(
        self,
        specification: dict[str, Any],
        value: Any,
        visited: set[str] | None = None,
    ) -> Any:
        """
        Recursively inline local schema references.

        This makes the JSON Schema self-contained for the MCP client.
        """

        if visited is None:
            visited = set()

        if isinstance(value, list):
            return [
                self._resolve_schema(
                    specification,
                    item,
                    set(visited),
                )
                for item in value
            ]

        if not isinstance(value, dict):
            return value

        reference = value.get("$ref")

        if isinstance(reference, str):
            if not reference.startswith("#/"):
                raise RuntimeError(
                    "External schema references are not "
                    f"supported: {reference}"
                )

            if reference in visited:
                # Stop recursion for circular schemas.
                return {"type": "object"}

            new_visited = set(visited)
            new_visited.add(reference)

            resolved = self._lookup_reference(
                specification,
                reference,
            )

            return self._resolve_schema(
                specification,
                copy.deepcopy(resolved),
                new_visited,
            )

        return {
            key: self._resolve_schema(
                specification,
                child,
                set(visited),
            )
            for key, child in value.items()
        }

    @staticmethod
    def _lookup_reference(
        specification: dict[str, Any],
        reference: str,
    ) -> Any:
        current: Any = specification

        for part in reference[2:].split("/"):
            current = current[part]

        return current

    @staticmethod
    def _find_json_media(
        content: Any,
    ) -> dict[str, Any] | None:
        if not isinstance(content, dict):
            return None

        application_json = content.get(
            "application/json"
        )

        if isinstance(application_json, dict):
            return application_json

        for media_type, media in content.items():
            if (
                isinstance(media_type, str)
                and media_type.endswith("+json")
                and isinstance(media, dict)
            ):
                return media

        return None

    @staticmethod
    def _make_tool_name(
        *,
        operation_id: str,
        method: str,
        api_path: str,
        used_names: set[str],
    ) -> str:
        clean_name = re.sub(
            r"[^a-zA-Z0-9_-]",
            "_",
            operation_id,
        ).strip("_")

        if not clean_name:
            clean_name = "api_operation"

        clean_name = clean_name[:64]

        if clean_name not in used_names:
            used_names.add(clean_name)
            return clean_name

        suffix = abs(
            hash(f"{method}:{api_path}")
        ) % 100000

        unique_name = (
            f"{clean_name[:57]}_{suffix}"
        )

        used_names.add(unique_name)
        return unique_name


# ---------------------------------------------------------------------------
# Real API execution
# ---------------------------------------------------------------------------

async def execute_api_operation(
    operation: APIOperation,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert MCP arguments into path/query/header/body values
    and call the actual Swagger API.
    """

    jsonschema.validate(
        instance=arguments,
        schema=operation.input_schema,
    )

    path = operation.path
    query_params: list[tuple[str, Any]] = []
    request_headers: dict[str, str] = {}
    cookies: dict[str, str] = {}

    for parameter in operation.parameters:
        if parameter.name not in arguments:
            continue

        value = arguments[parameter.name]

        if parameter.location == "path":
            path = path.replace(
                f"{{{parameter.name}}}",
                quote(
                    str(value),
                    safe="",
                ),
            )

        elif parameter.location == "header":
            request_headers[parameter.name] = (
                str(value)
            )

        elif parameter.location == "cookie":
            cookies[parameter.name] = str(value)

        else:
            # Query arrays become repeated parameters:
            # ?parameters=temperature&parameters=voltage
            if isinstance(value, list):
                query_params.extend(
                    (parameter.name, item)
                    for item in value
                )
            else:
                query_params.append(
                    (parameter.name, value)
                )

    body = (
        arguments.get("body")
        if operation.has_json_body
        else None
    )

    common_headers = {
        "Accept": "application/json",
    }

    if API_BEARER_TOKEN:
        common_headers["Authorization"] = (
            f"Bearer {API_BEARER_TOKEN}"
        )

    common_headers.update(request_headers)

    async with httpx.AsyncClient(
        timeout=HTTP_TIMEOUT_SECONDS,
        headers=common_headers,
        follow_redirects=False,
    ) as client:
        response = await client.request(
            method=operation.method,
            url=f"{API_BASE_URL}{path}",
            params=query_params,
            cookies=cookies,
            json=body,
        )

    content_type = response.headers.get(
        "content-type",
        "",
    ).lower()

    if (
        "application/json" in content_type
        or "+json" in content_type
    ):
        try:
            response_data: Any = response.json()
        except ValueError:
            response_data = {
                "text": response.text,
            }
    else:
        response_data = {
            "text": response.text,
        }

    if response.is_error:
        raise RuntimeError(
            f"API returned HTTP {response.status_code}: "
            f"{json.dumps(response_data, default=str)[:1000]}"
        )

    return {
        "status_code": response.status_code,
        "method": operation.method,
        "path": operation.path,
        "data": response_data,
    }


# ---------------------------------------------------------------------------
# MCP handlers
# ---------------------------------------------------------------------------

catalog = SwaggerToolCatalog()


async def handle_list_tools(
    ctx: ServerRequestContext,
    params: types.PaginatedRequestParams | None,
) -> types.ListToolsResult:
    """
    Called when the MCP client asks which tools are available.
    """

    tools = await catalog.list_mcp_tools()

    return types.ListToolsResult(
        tools=tools
    )


async def handle_call_tool(
    ctx: ServerRequestContext,
    params: types.CallToolRequestParams,
) -> types.CallToolResult:
    """
    Called when the broker's MCP client selects one Swagger tool.
    """

    operation = await catalog.get_operation(
        params.name
    )

    if operation is None:
        return tool_error(
            f"Unknown MCP tool: {params.name}"
        )

    arguments = params.arguments or {}

    try:
        result = await execute_api_operation(
            operation=operation,
            arguments=arguments,
        )

    except jsonschema.ValidationError as exc:
        return tool_error(
            f"Invalid tool arguments: {exc.message}"
        )

    except httpx.HTTPError as exc:
        return tool_error(
            f"Could not call the API: {exc}"
        )

    except Exception as exc:
        return tool_error(str(exc))

    text_result = json.dumps(
        result,
        ensure_ascii=False,
        default=str,
    )

    return types.CallToolResult(
        content=[
            types.TextContent(
                type="text",
                text=text_result,
            )
        ]structured_content=result,
    )


def tool_error(
    message: str,
) -> types.CallToolResult:
    return types.CallToolResult(
        content=[
            types.TextContent(
                type="text",
                text=message,
            )
        ],
        structured_content={
            "error": message,
        },
        is_error=True,
    )


# ---------------------------------------------------------------------------
# MCP application
# ---------------------------------------------------------------------------

mcp_server = Server(
    "Factigent Swagger MCP",
    on_list_tools=handle_list_tools,
    on_call_tool=handle_call_tool,
)

# Run with:
# uvicorn swagger_mcp_server:app --host 127.0.0.1 --port 8000
app = mcp_server.streamable_http_app()