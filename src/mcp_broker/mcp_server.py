from __future__ import annotations

from mcp_broker.openapi_loader import (
    API_BASE_URL,
    MCP_HOST,
    MCP_PORT,
    MCP_TRANSPORT,
    SOURCE_NAME,
    count_openapi_operations,
    mcp,
    openapi_spec,
)


def main() -> None:
    """
    Start one MCP process for the OpenAPI source selected by configuration.

    Source selection, API base URL, authentication headers, OpenAPI loading,
    and FastMCP tool generation all live in openapi_loader.py.

    This file is intentionally only the runtime entry point so that the same
    MCP code can serve different Factigent API sources without duplicating
    loader logic or adding endpoint-specific Python code.
    """

    operation_count = count_openapi_operations(
        openapi_spec
    )

    print()
    print(
        "========== FACTIGENT MCP SERVER =========="
    )
    print(
        "OpenAPI source:",
        SOURCE_NAME,
    )
    print(
        "API base URL:",
        API_BASE_URL,
    )
    print(
        "OpenAPI operations exposed:",
        operation_count,
    )
    print(
        "Local MCP endpoint:",
        f"http://{MCP_HOST}:{MCP_PORT}/mcp",
    )
    print(
        "Transport:",
        MCP_TRANSPORT,
    )
    print()

    if operation_count <= 0:
        raise RuntimeError(
            "The selected OpenAPI specification contains no "
            "supported API operations, so no MCP tools can be exposed."
        )

    mcp.run(
        transport=MCP_TRANSPORT,
        host=MCP_HOST,
        port=MCP_PORT,
    )


if __name__ == "__main__":
    main()
    
# from __future__ import annotations

# import os
# import uuid
# from datetime import UTC, datetime
# from pathlib import Path
# from typing import Any

# import httpx
# from dotenv import load_dotenv
# from fastmcp import FastMCP  # type: ignore[import]


# # --------------------------------------------------
# # Load environment variables
# # --------------------------------------------------

# ENV_FILE = Path(__file__).resolve().parent / ".env"

# load_dotenv(
#     dotenv_path=ENV_FILE,
#     override=True,
# )


# # --------------------------------------------------
# # MCP server
# # --------------------------------------------------

# mcp = FastMCP("Factigent MCP Server")


# # --------------------------------------------------
# # Factigent API configuration
# # --------------------------------------------------

# FACTIGENT_API_BASE_URL = (
#     os.getenv("FACTIGENT_API_BASE_URL")
#     or os.getenv("API_BASE_URL")
#     or ""
# ).rstrip("/")

# if not FACTIGENT_API_BASE_URL:
#     raise RuntimeError(
#         "FACTIGENT_API_BASE_URL or API_BASE_URL "
#         "is missing from the .env file."
#     )


# FACTIGENT_API_TOKEN = (
#     os.getenv("FACTIGENT_API_TOKEN")
#     or os.getenv("API_BEARER_TOKEN")
#     or ""
# ).strip()


# LATEST_READINGS_PATH = os.getenv(
#     "LATEST_READINGS_PATH",
#     "/api/data/latest-readings",
# )

# ANOMALIES_PATH = os.getenv(
#     "ANOMALIES_PATH",
#     "/api/anomalies",
# )

# MACHINES_PATH = os.getenv(
#     "MACHINES_PATH",
#     "/api/data/machines",
# )

# MAINTENANCE_TICKET_PATH = os.getenv(
#     "MAINTENANCE_TICKET_PATH",
#     "/api/maintenance/tickets",
# )


# API_TIMEOUT_SECONDS = float(
#     os.getenv("FACTIGENT_API_TIMEOUT_SECONDS")
#     or os.getenv("HTTP_TIMEOUT_SECONDS")
#     or "30"
# )

# MAX_RESPONSE_BYTES = int(
#     os.getenv(
#         "MAX_RESPONSE_BYTES",
#         "1000000",
#     )
# )


# # --------------------------------------------------
# # Common HTTP helpers
# # --------------------------------------------------

# def _build_headers(
#     extra_headers: dict[str, str] | None = None,
# ) -> dict[str, str]:
#     """
#     Create common headers for Factigent API requests.
#     """

#     headers = {
#         "Accept": "application/json",
#         "Content-Type": "application/json",
#     }

#     if FACTIGENT_API_TOKEN:
#         headers["Authorization"] = (
#             f"Bearer {FACTIGENT_API_TOKEN}"
#         )

#     if extra_headers:
#         headers.update(extra_headers)

#     return headers


# async def _call_factigent_api(
#     method: str,
#     path: str,
#     *,
#     params: Any = None,
#     json_body: dict[str, Any] | None = None,
#     extra_headers: dict[str, str] | None = None,
# ) -> Any:
#     """
#     Call the Factigent backend API.

#     MCP tools use this function instead of directly
#     connecting to ClickHouse or another database.
#     """

#     if not path.startswith("/"):
#         path = f"/{path}"

#     url = f"{FACTIGENT_API_BASE_URL}{path}"

#     request_options: dict[str, Any] = {
#         "method": method.upper(),
#         "url": url,
#         "params": params,
#     }

#     if json_body is not None:
#         request_options["json"] = json_body

#     try:
#         async with httpx.AsyncClient(
#             timeout=API_TIMEOUT_SECONDS,
#             headers=_build_headers(
#                 extra_headers
#             ),
#         ) as client:
#             response = await client.request(
#                 **request_options
#             )

#             response.raise_for_status()

#     except httpx.TimeoutException as exc:
#         raise RuntimeError(
#             f"Factigent API timed out while calling {url}"
#         ) from exc

#     except httpx.HTTPStatusError as exc:
#         status_code = exc.response.status_code
#         response_text = exc.response.text[:1000]

#         raise RuntimeError(
#             "Factigent API returned an error. "
#             f"Status={status_code}, "
#             f"URL={url}, "
#             f"Response={response_text}"
#         ) from exc

#     except httpx.RequestError as exc:
#         raise RuntimeError(
#             f"Could not connect to Factigent API at {url}. "
#             f"Original error: {exc!r}"
#         ) from exc

#     if len(response.content) > MAX_RESPONSE_BYTES:
#         raise RuntimeError(
#             "Factigent API response exceeded the configured "
#             f"maximum size of {MAX_RESPONSE_BYTES} bytes. "
#             f"URL={url}"
#         )

#     try:
#         data = response.json()
#     except ValueError as exc:
#         raise RuntimeError(
#             f"Factigent API returned invalid JSON from {url}. "
#             f"Response={response.text[:1000]}"
#         ) from exc

#     if not isinstance(
#         data,
#         (dict, list),
#     ):
#         raise RuntimeError(
#             "Factigent API response must be a JSON "
#             "object or JSON list, "
#             f"but received {type(data).__name__}. "
#             f"URL={url}"
#         )

#     return data


# # --------------------------------------------------
# # Validation helpers
# # --------------------------------------------------

# def _validate_machine_id(
#     machine_id: str,
# ) -> str:
#     machine_id = machine_id.strip()

#     if not machine_id:
#         raise ValueError(
#             "machine_id cannot be empty"
#         )

#     if len(machine_id) > 100:
#         raise ValueError(
#             "machine_id cannot contain more "
#             "than 100 characters"
#         )

#     return machine_id


# def _clean_parameters(
#     parameters: list[str] | None,
# ) -> list[str] | None:
#     if parameters is None:
#         return None

#     cleaned: list[str] = []

#     for parameter in parameters:
#         name = parameter.strip().lower()

#         if name and name not in cleaned:
#             cleaned.append(name)

#     return cleaned or None


# # --------------------------------------------------
# # MCP tools
# # --------------------------------------------------

# @mcp.tool()
# async def get_latest_readings(
#     machine_id: str,
#     parameters: list[str] | None = None,
# ) -> dict[str, Any]:
#     """
#     Get the latest selected telemetry readings for one machine.

#     Use this tool when the user asks for current or latest
#     temperature, vibration, voltage, humidity, power,
#     frequency, level, or other machine telemetry.
#     """

#     machine_id = _validate_machine_id(
#         machine_id
#     )

#     parameters = _clean_parameters(
#         parameters
#     )

#     query_params: list[tuple[str, str]] = [
#         ("machine_id", machine_id),
#     ]

#     if parameters:
#         for parameter in parameters:
#             query_params.append(
#                 ("parameters", parameter)
#             )

#     api_data = await _call_factigent_api(
#         method="GET",
#         path=LATEST_READINGS_PATH,
#         params=query_params,
#     )

#     if not isinstance(api_data, dict):
#         raise RuntimeError(
#             "Latest-readings API must return "
#             "a JSON object."
#         )

#     readings = api_data.get("readings")

#     if not isinstance(readings, dict):
#         raise RuntimeError(
#             "Latest-readings API response must "
#             "contain a 'readings' JSON object."
#         )

#     if parameters:
#         readings = {
#             parameter: readings[parameter]
#             for parameter in parameters
#             if parameter in readings
#         }

#     return {
#         "machine_id": api_data.get(
#             "machine_id",
#             machine_id,
#         ),
#         "timestamp": api_data.get(
#             "timestamp"
#         ),
#         "readings": readings,
#         "source": "factigent_api",
#         "received_at": (
#             datetime.now(UTC).isoformat()
#         ),
#     }


# @mcp.tool()
# async def get_anomalies(
#     machine_id: str,
#     hours: int = 24,
#     limit: int = 5,
# ) -> dict[str, Any]:
#     """
#     Get recent anomaly detections for one machine.

#     Use this tool when the user asks about anomalies,
#     abnormal readings, alerts, or unusual machine behaviour.
#     """

#     machine_id = _validate_machine_id(
#         machine_id
#     )

#     if hours < 1 or hours > 24 * 90:
#         raise ValueError(
#             "hours must be between 1 and 2160"
#         )

#     if limit < 1 or limit > 100:
#         raise ValueError(
#             "limit must be between 1 and 100"
#         )

#     api_data = await _call_factigent_api(
#         method="GET",
#         path=ANOMALIES_PATH,
#         params={
#             "machine_id": machine_id,
#             "hours": hours,
#             "limit": limit,
#         },
#     )

#     if not isinstance(api_data, dict):
#         raise RuntimeError(
#             "Anomalies API must return "
#             "a JSON object."
#         )

#     anomalies = api_data.get(
#         "anomalies",
#         [],
#     )

#     if not isinstance(anomalies, list):
#         raise RuntimeError(
#             "Anomalies API response must contain "
#             "an 'anomalies' JSON list."
#         )

#     anomalies = anomalies[:limit]

#     return {
#         "machine_id": api_data.get(
#             "machine_id",
#             machine_id,
#         ),
#         "window_hours": api_data.get(
#             "window_hours",
#             hours,
#         ),
#         "count": len(anomalies),
#         "anomalies": anomalies,
#         "source": "factigent_api",
#         "received_at": (
#             datetime.now(UTC).isoformat()
#         ),
#     }


# @mcp.tool()
# async def get_machines() -> Any:
#     """
#     Return all available Factigent machines.

#     Use this tool when the user asks to list, show,
#     identify, count, or view all available machines.
#     Do not use it for machine readings, anomalies,
#     risk analysis, or maintenance tickets.
#     """

#     return await _call_factigent_api(
#         method="GET",
#         path=MACHINES_PATH,
#     )


# @mcp.tool()
# async def create_maintenance_ticket(
#     machine_id: str,
#     summary: str,
# ) -> dict[str, Any]:
#     """
#     Create a maintenance ticket for a machine problem.

#     Use this tool only when the user explicitly asks to create,
#     raise, open, or submit a maintenance ticket.
#     """

#     machine_id = _validate_machine_id(
#         machine_id
#     )

#     summary = summary.strip()

#     if not summary:
#         raise ValueError(
#             "summary cannot be empty"
#         )

#     if len(summary) > 1_000:
#         raise ValueError(
#             "summary cannot contain more "
#             "than 1000 characters"
#         )

#     idempotency_key = str(
#         uuid.uuid4()
#     )

#     api_data = await _call_factigent_api(
#         method="POST",
#         path=MAINTENANCE_TICKET_PATH,
#         json_body={
#             "machine_id": machine_id,
#             "summary": summary,
#             "source": "mcp_tool_broker",
#         },
#         extra_headers={
#             "Idempotency-Key": (
#                 idempotency_key
#             ),
#         },
#     )

#     if not isinstance(api_data, dict):
#         raise RuntimeError(
#             "Maintenance API must return "
#             "a JSON object."
#         )

#     ticket_id = (
#         api_data.get("ticket_id")
#         or api_data.get("number")
#         or api_data.get("id")
#     )

#     if not ticket_id:
#         raise RuntimeError(
#             "Maintenance API returned a response "
#             "but did not provide ticket_id, number, or id."
#         )

#     return {
#         "ticket_id": str(ticket_id),
#         "machine_id": api_data.get(
#             "machine_id",
#             machine_id,
#         ),
#         "summary": api_data.get(
#             "summary",
#             summary,
#         ),
#         "status": api_data.get(
#             "status",
#             "created",
#         ),
#         "idempotency_key": (
#             idempotency_key
#         ),
#         "source": "maintenance_api",
#     }


# # --------------------------------------------------
# # Run MCP server
# # --------------------------------------------------

# if __name__ == "__main__":
#     print(
#         "Factigent backend API:",
#         FACTIGENT_API_BASE_URL,
#     )

#     print(
#         "MCP endpoint:",
#         "http://127.0.0.1:8001/mcp",
#     )

#     mcp.run(
#         transport="http",
#         host="127.0.0.1",
#         port=8001,
#     )