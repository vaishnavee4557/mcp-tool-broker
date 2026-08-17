from __future__ import annotations

import re
from typing import Any

import httpx

from mcp_broker.models import (
    Operation,
    ToolDescriptor,
    ToolPolicy,
)


HTTP_METHODS = {
    "get",
    "post",
    "put",
    "patch",
    "delete",
}


class OpenAPIToolLoader:
    def __init__(
        self,
        *,
        openapi_url: str,
        api_base_url: str,
        server_name: str = "factigent_api",
    ) -> None:
        self.openapi_url = openapi_url
        self.api_base_url = api_base_url.rstrip("/")
        self.server_name = server_name

    async def load_tools(self) -> list[ToolDescriptor]:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(self.openapi_url)
            response.raise_for_status()
            specification = response.json()

        return self._create_tools(specification)

    def _create_tools(
        self,
        specification: dict[str, Any],
    ) -> list[ToolDescriptor]:
        tools: list[ToolDescriptor] = []

        for api_path, path_data in specification.get("paths", {}).items():
            if not isinstance(path_data, dict):
                continue

            for method, operation in path_data.items():
                if method.lower() not in HTTP_METHODS:
                    continue

                if not isinstance(operation, dict):
                    continue

                operation_id = (
                    operation.get("operationId")
                    or self._make_name(method, api_path)
                )

                summary = (
                    operation.get("summary")
                    or operation.get("description")
                    or operation_id
                )

                tags = operation.get("tags", [])

                input_schema = self._create_input_schema(
                    operation
                )

                tools.append(
                    ToolDescriptor(
                        server_name=self.server_name,
                        name=operation_id,
                        description=summary,
                        input_schema=input_schema,
                        base_url=self.api_base_url,
                        http_method=method.upper(),
                        api_path=api_path,
                        policy=ToolPolicy(
                            domain=tags[0].lower() if tags else "general",
                            operation=self._get_operation(method),
                            summary=summary,
                            keywords=[
                                *tags,
                                *input_schema.get(
                                    "properties",
                                    {},
                                ).keys(),
                            ],
                        ),
                    )
                )

        return tools

    def _create_input_schema(
        self,
        operation: dict[str, Any],
    ) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required: list[str] = []

        for parameter in operation.get("parameters", []):
            name = parameter.get("name")

            if not name:
                continue

            schema = parameter.get(
                "schema",
                {"type": "string"},
            )

            properties[name] = {
                **schema,
                "description": parameter.get(
                    "description",
                    "",
                ),
                "x-parameter-location": parameter.get(
                    "in",
                    "query",
                ),
            }

            if parameter.get("required"):
                required.append(name)

        result: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }

        if required:
            result["required"] = required

        return result

    @staticmethod
    def _get_operation(method: str) -> Operation:
        method = method.lower()

        if method == "get":
            return Operation.READ

        if method == "delete":
            return Operation.ADMIN

        return Operation.WRITE

    @staticmethod
    def _make_name(
        method: str,
        api_path: str,
    ) -> str:
        clean_path = re.sub(
            r"[^a-zA-Z0-9]+",
            "_",
            api_path,
        ).strip("_")

        return f"{method.lower()}_{clean_path}"