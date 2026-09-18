from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

# PowerShell process variables must override root .env values.
load_dotenv(
    dotenv_path=ENV_FILE,
    override=False,
)


def _json_object_env(name: str) -> dict[str, Any]:
    raw = str(os.getenv(name, "") or "").strip()

    if not raw:
        return {}

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{name} must contain valid JSON."
        ) from exc

    if not isinstance(value, dict):
        raise RuntimeError(
            f"{name} must contain a JSON object."
        )

    return value


def _resolve_file(configured_path: str) -> Path:
    path = Path(configured_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


SOURCE_NAME = str(
    os.getenv(
        "OPENAPI_SOURCE_NAME",
        "predictive_maintenance",
    )
    or "predictive_maintenance"
).strip()

sources = _json_object_env(
    "OPENAPI_SOURCES_JSON"
)

source_config: dict[str, Any] = {}

if sources:
    selected = sources.get(SOURCE_NAME)

    if selected is None:
        raise RuntimeError(
            f"OPENAPI_SOURCE_NAME={SOURCE_NAME!r} was not found "
            "inside OPENAPI_SOURCES_JSON."
        )

    if not isinstance(selected, dict):
        raise RuntimeError(
            f"OpenAPI source {SOURCE_NAME!r} must be a JSON object."
        )

    source_config = selected


OPENAPI_URL = str(
    source_config.get(
        "openapi_url",
        os.getenv("OPENAPI_URL", ""),
    )
    or ""
).strip()

OPENAPI_FILE = str(
    source_config.get(
        "openapi_file",
        os.getenv("OPENAPI_FILE", ""),
    )
    or ""
).strip()

API_BASE_URL = str(
    source_config.get(
        "api_base_url",
        os.getenv(
            "FACTIGENT_API_BASE_URL",
            os.getenv("API_BASE_URL", ""),
        ),
    )
    or ""
).strip().rstrip("/")

if not API_BASE_URL:
    raise RuntimeError(
        f"No API base URL is configured for OpenAPI source "
        f"{SOURCE_NAME!r}."
    )


MCP_HOST = str(
    os.getenv("MCP_HOST", "127.0.0.1")
    or "127.0.0.1"
).strip()

MCP_PORT = int(
    os.getenv("MCP_PORT", "8001")
)

MCP_TRANSPORT = str(
    os.getenv("MCP_TRANSPORT", "http")
    or "http"
).strip()

HTTP_TIMEOUT_SECONDS = float(
    os.getenv(
        "HTTP_TIMEOUT_SECONDS",
        "30",
    )
)


API_BEARER_TOKEN = str(
    os.getenv(
        "API_BEARER_TOKEN",
        os.getenv("FACTIGENT_API_TOKEN", ""),
    )
    or ""
).strip()

API_HEADERS_JSON = _json_object_env(
    "API_HEADERS_JSON"
)

OPENAPI_HEADERS_JSON = _json_object_env(
    "OPENAPI_HEADERS_JSON"
)


def build_api_headers() -> dict[str, str]:
    headers: dict[str, str] = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    if API_BEARER_TOKEN:
        headers["Authorization"] = (
            f"Bearer {API_BEARER_TOKEN}"
        )

    for key, value in API_HEADERS_JSON.items():
        if value is None:
            continue

        headers[str(key)] = str(value)

    return headers


def build_openapi_headers() -> dict[str, str]:
    headers: dict[str, str] = {
        "Accept": "application/json",
    }

    if API_BEARER_TOKEN:
        headers["Authorization"] = (
            f"Bearer {API_BEARER_TOKEN}"
        )

    for key, value in OPENAPI_HEADERS_JSON.items():
        if value is None:
            continue

        headers[str(key)] = str(value)

    return headers


def load_openapi_spec() -> dict[str, Any]:
    if OPENAPI_FILE:
        spec_path = _resolve_file(
            OPENAPI_FILE
        )

        if not spec_path.exists():
            raise RuntimeError(
                f"OPENAPI_FILE was not found: {spec_path}"
            )

        try:
            spec = json.loads(
                spec_path.read_text(
                    encoding="utf-8"
                )
            )
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid JSON in OpenAPI file: {spec_path}"
            ) from exc

        source_label = str(spec_path)

    elif OPENAPI_URL:
        try:
            response = httpx.get(
                OPENAPI_URL,
                headers=build_openapi_headers(),
                timeout=HTTP_TIMEOUT_SECONDS,
                follow_redirects=True,
            )

            response.raise_for_status()

        except httpx.TimeoutException as exc:
            raise RuntimeError(
                "Timed out while downloading OpenAPI from "
                f"{OPENAPI_URL}"
            ) from exc

        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                "OpenAPI server returned an error. "
                f"Status={exc.response.status_code}, "
                f"URL={OPENAPI_URL}, "
                f"Response={exc.response.text[:1000]}"
            ) from exc

        except httpx.RequestError as exc:
            raise RuntimeError(
                f"Could not connect to OpenAPI URL "
                f"{OPENAPI_URL}: {exc!r}"
            ) from exc

        try:
            spec = response.json()
        except ValueError as exc:
            raise RuntimeError(
                "OpenAPI URL did not return valid JSON."
            ) from exc

        source_label = OPENAPI_URL

    else:
        raise RuntimeError(
            f"No openapi_url/openapi_file is configured for "
            f"source {SOURCE_NAME!r}."
        )

    if not isinstance(spec, dict):
        raise RuntimeError(
            "OpenAPI specification must be a JSON object."
        )

    paths = spec.get("paths")

    if not isinstance(paths, dict):
        raise RuntimeError(
            "OpenAPI specification has no valid 'paths' object."
        )

    print(
        f"OpenAPI source selected: {SOURCE_NAME}"
    )
    print(
        f"OpenAPI specification loaded from: {source_label}"
    )

    return spec


def count_openapi_operations(
    spec: dict[str, Any],
) -> int:
    supported_methods = {
        "get",
        "post",
        "put",
        "patch",
        "delete",
        "options",
        "head",
    }

    paths = spec.get("paths", {})

    if not isinstance(paths, dict):
        return 0

    return sum(
        1
        for path_definition in paths.values()
        if isinstance(path_definition, dict)
        for method in path_definition.keys()
        if str(method).casefold()
        in supported_methods
    )


openapi_spec = load_openapi_spec()

api_client = httpx.AsyncClient(
    base_url=API_BASE_URL,
    headers=build_api_headers(),
    timeout=HTTP_TIMEOUT_SECONDS,
    follow_redirects=True,
)

mcp = FastMCP.from_openapi(
    openapi_spec=openapi_spec,
    client=api_client,
    name=f"Factigent {SOURCE_NAME} MCP Server",
)
