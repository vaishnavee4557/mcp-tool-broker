from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from mcp_broker.async_runtime import run_async
from mcp_broker.conversation_store import ConversationStore
from mcp_broker.db import Database
from mcp_broker.llm import OpenAICompatibleChatModel
from mcp_broker.mcp_gateway import MCPGateway
from mcp_broker.models import (
    ChatTurn,
    Operation,
    ToolDescriptor,
    UserContext,
)
from mcp_broker.pg_retriever import PgVectorToolRetriever
from mcp_broker.request_context import TrustedRequestContext
from mcp_broker.registry import ToolRegistry
from mcp_broker.retrieval import PgHybridToolRetriever
from mcp_broker.settings import get_settings


# ============================================================
# CONFIGURATION
# ============================================================

DEBUG_RETRIEVER = True

RETRIEVAL_LIMIT = 20
LLM_SHORTLIST_LIMIT = 8
MAX_TOOL_STEPS = 4
MAX_HISTORY_TURNS = 10


# ============================================================
# GENERIC QUERY NORMALIZATION
# ============================================================

def normalize_query(
    query: str,
) -> str:
    """
    Preserve the user's wording and identifiers.

    Typo tolerance/case-insensitive matching belongs in retrieval.py and is
    learned from live MCP/OpenAPI metadata. The CLI must not contain
    machine-, parameter-, or question-specific rewrite rules.
    """

    return " ".join(
        str(query or "").strip().split()
    )


# ============================================================
# SETTINGS / POLICY
# ============================================================

def resolve_policy_path(
    settings: Any,
) -> Path:
    configured_path = getattr(
        settings,
        "tool_policies_path",
        None,
    )

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    if configured_path:
        policy_path = Path(
            configured_path
        )

        if not policy_path.is_absolute():
            policy_path = (
                project_root
                / policy_path
            )
    else:
        policy_path = (
            project_root
            / "config"
            / "tool_policies.json"
        )

    return policy_path.resolve()


def read_secret(
    value: Any,
) -> str:
    if value is None:
        return ""

    if hasattr(
        value,
        "get_secret_value",
    ):
        return str(
            value.get_secret_value()
        ).strip()

    return str(value).strip()


def resolve_mcp_servers(
    settings: Any,
) -> dict[str, str]:
    """
    Use the parsed multi-server setting when available.

    Example:
        {
          "predictive_maintenance": "http://127.0.0.1:8001/mcp",
          "factigent_platform": "http://127.0.0.1:8002/mcp"
        }
    """

    parsed = getattr(
        settings,
        "mcp_servers",
        None,
    )

    if isinstance(
        parsed,
        dict,
    ) and parsed:
        return {
            str(name).strip():
                str(url).strip()
            for name, url in parsed.items()
            if (
                str(name).strip()
                and str(url).strip()
            )
        }

    raise RuntimeError(
        "No MCP servers are configured. "
        "Set MCP_SERVERS_JSON in the root .env."
    )


# ============================================================
# READ-ONLY SAFETY
# ============================================================

_GENERIC_WRITE_MARKERS = (
    "create",
    "update",
    "delete",
    "remove",
    "start",
    "stop",
    "submit",
    "capture",
    "assign",
    "unassign",
    "set",
    "enable",
    "disable",
    "upload",
    "install",
    "uninstall",
    "revoke",
)


def is_read_only_tool(
    tool: ToolDescriptor,
) -> bool:
    """
    Metadata-first informational-tool classification.

    POST is NOT automatically write: reports/searches/read operations may
    legitimately use POST.
    """

    explicit = getattr(
        tool,
        "effective_read_only",
        None,
    )

    if isinstance(
        explicit,
        bool,
    ):
        return explicit

    policy = getattr(
        tool,
        "policy",
        None,
    )

    if policy is not None:
        policy_read_only = getattr(
            policy,
            "read_only",
            None,
        )

        if isinstance(
            policy_read_only,
            bool,
        ):
            return policy_read_only

        operation = getattr(
            policy,
            "operation",
            Operation.READ,
        )

        if operation in {
            Operation.WRITE,
            Operation.ADMIN,
        }:
            return False

    method = str(
        getattr(
            tool,
            "http_method",
            "",
        )
        or ""
    ).strip().upper()

    if method in {
        "GET",
        "HEAD",
        "OPTIONS",
    }:
        return True

    name = str(
        getattr(
            tool,
            "name",
            "",
        )
        or ""
    ).casefold()

    description = str(
        getattr(
            tool,
            "description",
            "",
        )
        or ""
    ).casefold()

    text = (
        name.replace("_", " ")
        + " "
        + description
    )

    # Conservative generic fallback only.
    tokens = set(
        re.findall(
            r"[a-z0-9]+",
            text,
        )
    )

    if any(
        marker in tokens
        for marker in _GENERIC_WRITE_MARKERS
    ):
        return False

    # Policy defaults are READ in this informational chatbot.
    return True


# ============================================================
# SCHEMA-DRIVEN ARGUMENT VALIDATION
# ============================================================

def validate_arguments(
    tool: ToolDescriptor,
    arguments: dict[str, Any],
) -> tuple[
    dict[str, Any],
    list[str],
]:
    """
    Keep only values defined by the selected tool's live input schema.

    A field is missing only when the schema itself marks it required.
    Optional site/zone/line/cell/etc. are never forced.
    """

    schema = (
        tool.input_schema
        or {}
    )

    properties = schema.get(
        "properties",
        {},
    )

    if not isinstance(
        properties,
        dict,
    ):
        properties = {}

    required_fields = schema.get(
        "required",
        [],
    )

    if not isinstance(
        required_fields,
        list,
    ):
        required_fields = []

    valid_arguments = {
        key: value
        for key, value
        in (arguments or {}).items()
        if key in properties
    }

    missing_fields = [
        str(field)
        for field in required_fields
        if (
            field not in valid_arguments
            or valid_arguments[field]
            in (
                None,
                "",
            )
        )
    ]

    return (
        valid_arguments,
        missing_fields,
    )


def canonicalize_schema_values(
    tool: ToolDescriptor,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """
    Canonicalize only explicit schema enum values.

    Example:
        user/LLM value "cutting"
        schema enum value "Cutting"
        -> "Cutting"

    No machine/site/zone/etc. values are hardcoded.
    """

    result = dict(
        arguments
    )

    properties = (
        tool.input_schema
        or {}
    ).get(
        "properties",
        {},
    )

    if not isinstance(
        properties,
        dict,
    ):
        return result

    for (
        field_name,
        field_schema,
    ) in properties.items():
        if (
            field_name not in result
            or not isinstance(
                field_schema,
                dict,
            )
        ):
            continue

        enum_values = field_schema.get(
            "enum"
        )

        current = result.get(
            field_name
        )

        if (
            not isinstance(
                current,
                str,
            )
            or not isinstance(
                enum_values,
                list,
            )
        ):
            continue

        for enum_value in enum_values:
            if (
                isinstance(
                    enum_value,
                    str,
                )
                and enum_value.casefold()
                == current.casefold()
            ):
                result[
                    field_name
                ] = enum_value
                break

    return result


def apply_required_schema_defaults(
    tool: ToolDescriptor,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """
    Apply a schema default ONLY when that field is genuinely required.

    Optional defaults are not sent automatically because sending an optional
    hierarchy/filter field can change API semantics.
    """

    result = dict(
        arguments
    )

    schema = (
        tool.input_schema
        or {}
    )

    properties = schema.get(
        "properties",
        {},
    )

    required = schema.get(
        "required",
        [],
    )

    if (
        not isinstance(
            properties,
            dict,
        )
        or not isinstance(
            required,
            list,
        )
    ):
        return result

    for field_name in required:
        if field_name in result:
            continue

        field_schema = properties.get(
            field_name
        )

        if (
            isinstance(
                field_schema,
                dict,
            )
            and "default"
            in field_schema
            and field_schema.get(
                "default"
            )
            is not None
        ):
            result[
                field_name
            ] = field_schema[
                "default"
            ]

    return result


# ============================================================
# VALIDATED CONVERSATION CONTEXT
# ============================================================

_SENSITIVE_CONTEXT_KEYS = {
    "authorization",
    "token",
    "accesstoken",
    "refreshtoken",
    "password",
    "secret",
    "apikey",
    "cookie",
}

_NON_IDENTIFIER_META_KEYS = {
    "status",
    "success",
    "message",
    "error",
    "code",
    "count",
    "total",
    "totalcount",
    "timestamp",
    "createdat",
    "updatedat",
}


def normalize_field_name(
    value: str,
) -> str:
    """
    Match snake_case/camelCase/kebab-case field names generically while
    preserving the original API value.
    """

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value or "").casefold(),
    )


def _is_scalar_context_value(
    value: Any,
) -> bool:
    return (
        value is None
        or isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
            ),
        )
    )


def _remember_context_value(
    context: dict[
        str,
        list[Any],
    ],
    key: str,
    value: Any,
) -> None:
    normalized_key = (
        normalize_field_name(
            key
        )
    )

    if (
        not normalized_key
        or normalized_key
        in _SENSITIVE_CONTEXT_KEYS
        or normalized_key
        in _NON_IDENTIFIER_META_KEYS
        or value in (
            None,
            "",
        )
        or not _is_scalar_context_value(
            value
        )
    ):
        return

    values = context.setdefault(
        normalized_key,
        [],
    )

    identity = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )

    existing = {
        json.dumps(
            item,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        for item in values
    }

    if identity not in existing:
        values.append(
            value
        )

    if len(values) > 50:
        del values[
            :-50
        ]


def _collect_context_values(
    value: Any,
    context: dict[
        str,
        list[Any],
    ],
    *,
    depth: int = 0,
) -> None:
    """
    Collect exact structured fields from successful API/MCP data.

    This does not infer identifiers from assistant prose.
    """

    if depth > 10:
        return

    if isinstance(
        value,
        dict,
    ):
        for key, child in value.items():
            if _is_scalar_context_value(
                child
            ):
                _remember_context_value(
                    context,
                    str(key),
                    child,
                )
            else:
                _collect_context_values(
                    child,
                    context,
                    depth=depth + 1,
                )

    elif isinstance(
        value,
        list,
    ):
        for child in value[
            :200
        ]:
            _collect_context_values(
                child,
                context,
                depth=depth + 1,
            )


def update_validated_context(
    context: dict[
        str,
        list[Any],
    ],
    wrapped_result: dict[str, Any],
) -> None:
    """
    Store exact fields only from successful validated calls.

    These values may later satisfy a genuinely required schema field.
    Optional hierarchy fields are never auto-added.
    """

    validation = wrapped_result.get(
        "validation",
        {},
    )

    if (
        not wrapped_result.get(
            "success"
        )
        or not isinstance(
            validation,
            dict,
        )
        or not validation.get(
            "valid"
        )
    ):
        return

    _collect_context_values(
        wrapped_result.get(
            "arguments",
            {},
        ),
        context,
    )

    _collect_context_values(
        wrapped_result.get(
            "result",
            {},
        ),
        context,
    )


def resolve_required_from_context(
    tool: ToolDescriptor,
    arguments: dict[str, Any],
    context: dict[
        str,
        list[Any],
    ],
) -> tuple[
    dict[str, Any],
    list[str],
    list[str],
]:
    """
    Fill only REQUIRED schema fields, and only when one unambiguous exact
    structured value is known.

    If multiple values exist, do not guess: the workflow should use a
    context/discovery tool.
    """

    result = dict(
        arguments
    )

    _, missing_fields = (
        validate_arguments(
            tool,
            result,
        )
    )

    resolved_fields: list[
        str
    ] = []

    for field in missing_fields:
        normalized = (
            normalize_field_name(
                field
            )
        )

        values = context.get(
            normalized,
            [],
        )

        unique_values: list[
            Any
        ] = []

        seen: set[
            str
        ] = set()

        for value in values:
            identity = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )

            if identity in seen:
                continue

            seen.add(
                identity
            )

            unique_values.append(
                value
            )

        if len(
            unique_values
        ) == 1:
            result[
                field
            ] = unique_values[0]

            resolved_fields.append(
                field
            )

    result = canonicalize_schema_values(
        tool,
        result,
    )

    result, missing_fields = (
        validate_arguments(
            tool,
            result,
        )
    )

    return (
        result,
        missing_fields,
        resolved_fields,
    )


# ============================================================
# TOOL LOOKUP
# ============================================================

def build_candidate_map(
    tools: list[
        ToolDescriptor
    ],
) -> dict[
    str,
    ToolDescriptor,
]:
    """
    Prefer collision-safe llm_name/registry_id mappings.

    Raw tool.name is accepted only when it is unique in this shortlist.
    """

    mapping: dict[
        str,
        ToolDescriptor,
    ] = {}

    name_counts = Counter(
        tool.name
        for tool in tools
    )

    for tool in tools:
        mapping[
            tool.registry_id
        ] = tool

        mapping[
            tool.llm_name
        ] = tool

        if (
            tool.name
            and name_counts[
                tool.name
            ] == 1
        ):
            mapping[
                tool.name
            ] = tool

    return mapping


# ============================================================
# RETRIEVER
# ============================================================

async def build_retriever(
    db: Database,
    read_only_tools: list[
        ToolDescriptor
    ],
) -> PgHybridToolRetriever:
    pg_vector_retriever = (
        PgVectorToolRetriever(
            db=db,
        )
    )

    retriever = (
        PgHybridToolRetriever(
            pg_retriever=(
                pg_vector_retriever
            ),
        )
    )

    await retriever.rebuild(
        read_only_tools
    )

    return retriever


# ============================================================
# RESULT COMPACTION
# ============================================================

def compact_value(
    value: Any,
    *,
    depth: int = 0,
) -> Any:
    """
    Bound large nested payloads without destroying useful scalar lists such
    as parameter names, machine names, IDs, tags, zones, lines, or cells.
    """

    if depth >= 8:
        return (
            "<nested data removed>"
        )

    if isinstance(
        value,
        dict,
    ):
        result: dict[
            str,
            Any,
        ] = {}

        for index, (
            key,
            child,
        ) in enumerate(
            value.items()
        ):
            if index >= 60:
                result[
                    "_truncated_keys"
                ] = (
                    len(value)
                    - 60
                )
                break

            result[
                str(key)
            ] = compact_value(
                child,
                depth=depth + 1,
            )

        return result

    if isinstance(
        value,
        list,
    ):
        scalar_list = all(
            item is None
            or isinstance(
                item,
                (
                    str,
                    int,
                    float,
                    bool,
                ),
            )
            for item in value
        )

        limit = (
            300
            if scalar_list
            else 30
        )

        items = [
            compact_value(
                item,
                depth=depth + 1,
            )
            for item
            in value[:limit]
        ]

        if len(value) > limit:
            items.append(
                {
                    "_notice":
                        (
                            f"{len(value) - limit} "
                            "additional items removed"
                        )
                }
            )

        return items

    if (
        isinstance(
            value,
            str,
        )
        and len(value) > 3000
    ):
        return (
            value[:3000]
            + "...<truncated>"
        )

    return value


def build_retrieval_query(
    *,
    original_query: str,
    tool_context: list[
        dict[str, Any]
    ],
) -> str:
    """
    Subsequent retrieval steps include a small amount of the most recent
    MCP/workflow context so discovery -> data-call chains can emerge
    generically.
    """

    if not tool_context:
        return original_query

    last_result = (
        tool_context[-1]
    )

    context_text = json.dumps(
        last_result,
        ensure_ascii=False,
        default=str,
    )

    if len(
        context_text
    ) > 2200:
        context_text = (
            context_text[:2200]
            + "..."
        )

    return (
        f"{original_query}\n\n"
        "Previous MCP/workflow context:\n"
        f"{context_text}"
    )


def print_candidates(
    candidates: list[Any],
    step: int,
) -> None:
    if not DEBUG_RETRIEVER:
        return

    print(
        "\n========== RETRIEVER SHORTLIST "
        f"(STEP {step}) =========="
    )

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):
        print(
            f"{index}. "
            f"{candidate.tool.server_name}/"
            f"{candidate.tool.name} "
            f"(score={candidate.score:.4f})"
        )


# ============================================================
# DUPLICATE CALL GUARD
# ============================================================

def tool_call_signature(
    server_name: str,
    tool_name: str,
    arguments: dict[
        str,
        Any,
    ],
) -> str:
    return (
        server_name
        + "::"
        + tool_name
        + "::"
        + json.dumps(
            arguments,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
    )


# ============================================================
# RESPONSE VALIDATION
# ============================================================

_FAILURE_STATUSES = {
    "error",
    "failed",
    "failure",
    "fail",
    "invalid",
    "unauthorized",
    "forbidden",
}


def validate_mcp_response(
    response: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    """
    Validate the MCP execution wrapper generically.

    Do not reject a valid API solely because it returns a scalar/string or an
    unfamiliar success status value.
    """

    validation: dict[
        str,
        Any,
    ] = {
        "valid":
            True,
        "execution_success":
            False,
        "backend_status":
            None,
        "empty_data":
            False,
        "issues":
            [],
    }

    if not response.get(
        "success",
        False,
    ):
        validation[
            "valid"
        ] = False

        validation[
            "issues"
        ].append(
            "tool_execution_failed"
        )

        return validation

    validation[
        "execution_success"
    ] = True

    if "result" not in response:
        validation[
            "valid"
        ] = False

        validation[
            "issues"
        ].append(
            "missing_api_result"
        )

        return validation

    result = response.get(
        "result"
    )

    if result is None:
        validation[
            "empty_data"
        ] = True

        return validation

    if isinstance(
        result,
        dict,
    ):
        success_value = result.get(
            "success"
        )

        if (
            isinstance(
                success_value,
                bool,
            )
            and not success_value
        ):
            validation[
                "valid"
            ] = False

            validation[
                "issues"
            ].append(
                "backend_success_false"
            )

        backend_status = result.get(
            "status"
        )

        validation[
            "backend_status"
        ] = backend_status

        if isinstance(
            backend_status,
            str,
        ):
            normalized_status = (
                backend_status
                .strip()
                .casefold()
            )

            if normalized_status in (
                _FAILURE_STATUSES
            ):
                validation[
                    "valid"
                ] = False

                validation[
                    "issues"
                ].append(
                    "backend_status_failure"
                )

        if "data" in result:
            data = result.get(
                "data"
            )

            if data in (
                None,
                [],
                {},
                "",
            ):
                validation[
                    "empty_data"
                ] = True

    elif isinstance(
        result,
        list,
    ):
        if not result:
            validation[
                "empty_data"
            ] = True

    elif isinstance(
        result,
        str,
    ):
        if not result.strip():
            validation[
                "empty_data"
            ] = True

    elif not isinstance(
        result,
        (
            int,
            float,
            bool,
        ),
    ):
        validation[
            "valid"
        ] = False

        validation[
            "issues"
        ].append(
            "unexpected_result_type"
        )

    return validation


# ============================================================
# MCP EXECUTION
# ============================================================

async def safe_call(
    *,
    gateway: MCPGateway,
    tool: ToolDescriptor,
    arguments: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    """
    Execute the selected tool on the MCP server that actually owns it.
    """

    tool_name = str(
        tool.name
    ).strip()

    server_name = str(
        tool.server_name
    ).strip()

    if not server_name:
        raise RuntimeError(
            f"Tool {tool_name!r} has no server_name."
        )

    if (
        server_name
        not in gateway.servers
    ):
        raise RuntimeError(
            "Selected tool belongs to an unconfigured MCP server: "
            f"{server_name!r}"
        )

    print(
        "\n========== MCP TOOL CALL =========="
    )

    print(
        "Server:",
        server_name,
    )

    print(
        "Tool:",
        tool_name,
    )

    print(
        "Arguments:",
        json.dumps(
            arguments,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
    )

    try:
        raw_result = (
            await gateway.call_tool(
                server_name,
                tool_name,
                arguments,
            )
        )

        wrapped: dict[
            str,
            Any,
        ] = {
            "success":
                True,
            "server":
                server_name,
            "tool":
                tool_name,
            "arguments":
                arguments,
            "result":
                raw_result,
        }

    except Exception as exc:
        wrapped = {
            "success":
                False,
            "server":
                server_name,
            "tool":
                tool_name,
            "arguments":
                arguments,
            "error_type":
                type(
                    exc
                ).__name__,
            "error":
                str(
                    exc
                ),
        }

    wrapped[
        "validation"
    ] = validate_mcp_response(
        wrapped
    )

    print(
        "\n========== RESPONSE VALIDATION =========="
    )

    print(
        json.dumps(
            wrapped[
                "validation"
            ],
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )

    print(
        "\n========== MCP TOOL RESULT =========="
    )

    print(
        json.dumps(
            compact_value(
                wrapped
            ),
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )

    return wrapped


# ============================================================
# LLM ERROR HANDLING
# ============================================================

def is_invalid_api_key_error(
    exc: Exception,
) -> bool:
    text = str(
        exc
    ).casefold()

    return (
        "invalid_api_key"
        in text
        or (
            "401"
            in text
            and "api key"
            in text
        )
    )


def print_llm_auth_error(
) -> None:
    print(
        "\n=========================================="
    )
    print(
        "OPENAI AUTHENTICATION FAILED"
    )
    print(
        "=========================================="
    )
    print(
        "The LLM API key configured in .env is invalid."
    )
    print(
        "Update LLM_API_KEY with a valid key and restart the CLI."
    )


# ============================================================
# CONVERSATION SELECTION
# ============================================================

async def select_conversation(
    *,
    store: ConversationStore,
) -> tuple[
    str,
    list[ChatTurn],
]:
    """
    Create a new persistent conversation or resume an existing one.

    The returned history uses the same list[ChatTurn] structure already
    consumed by the existing LLM code, so the retrieval/LLM/MCP flow
    remains unchanged.
    """

    while True:
        print(
            "\n========== CONVERSATION =========="
        )

        print(
            "1. New conversation"
        )

        print(
            "2. Resume conversation"
        )

        choice = input(
            "Choose [1/2]: "
        ).strip().casefold()

        if choice in {
            "1",
            "new",
            "n",
        }:
            conversation_id = (
                await store.create_conversation()
            )

            print(
                "\nNew conversation created."
            )

            print(
                "Conversation ID:",
                conversation_id,
            )

            return (
                conversation_id,
                [],
            )

        if choice in {
            "2",
            "resume",
            "r",
        }:
            recent_conversations = (
                await store.list_recent_conversations(
                    limit=10
                )
            )

            conversation_id = ""

            if recent_conversations:
                print(
                    "\nRecent conversations:"
                )

                for index, item in enumerate(
                    recent_conversations,
                    start=1,
                ):
                    print(
                        f"\n{index}. "
                        f"{item['title']}"
                    )

                    print(
                        "   ID:",
                        item["id"],
                    )

                    print(
                        "   Updated:",
                        item["updated_at"],
                    )

                print(
                    "\n0. Enter conversation ID manually"
                )

                selected = input(
                    "Choose conversation number: "
                ).strip()

                try:
                    selected_number = int(
                        selected
                    )
                except ValueError:
                    print(
                        "Invalid selection. "
                        "Enter a conversation number."
                    )
                    continue

                if selected_number == 0:
                    conversation_id = input(
                        "Enter conversation ID: "
                    ).strip()

                elif (
                    1
                    <= selected_number
                    <= len(
                        recent_conversations
                    )
                ):
                    conversation_id = str(
                        recent_conversations[
                            selected_number - 1
                        ]["id"]
                    )

                else:
                    print(
                        "Conversation number is out of range."
                    )
                    continue

            else:
                print(
                    "\nNo recent conversations were found."
                )

                conversation_id = input(
                    "Enter conversation ID manually, "
                    "or press Enter to return: "
                ).strip()

                if not conversation_id:
                    continue

            if not conversation_id:
                print(
                    "Conversation ID cannot be empty."
                )
                continue

            try:
                exists = await store.exists(
                    conversation_id
                )
            except ValueError:
                print(
                    "Invalid conversation ID. "
                    "Enter a valid UUID."
                )
                continue

            if not exists:
                print(
                    "Conversation not found. "
                    "Enter another ID or choose a new conversation."
                )
                continue

            all_messages = (
                await store.get_all_messages(
                    conversation_id
                )
            )

            print_previous_conversation(
                all_messages
            )

            history = (
                await store.get_recent_turns(
                    conversation_id,
                    limit=MAX_HISTORY_TURNS,
                )
            )

            print(
                "\nConversation resumed."
            )

            print(
                "Conversation ID:",
                conversation_id,
            )

            print(
                "Restored history:",
                len(history) // 2,
                "turn(s)",
            )

            return (
                conversation_id,
                history,
            )

        print(
            "Invalid choice. Enter 1 for New "
            "or 2 for Resume."
        )


# ============================================================
# PREVIOUS CONVERSATION DISPLAY
# ============================================================

def print_previous_conversation(
    messages: list[dict[str, Any]],
) -> None:
    """
    Display the full persisted conversation in the terminal.

    This is display-only. The LLM still receives only the bounded
    `history` returned by get_recent_turns().
    """

    print(
        "\n========== PREVIOUS CONVERSATION =========="
    )

    if not messages:
        print(
            "\nNo previous messages were found."
        )

        print(
            "\n==========================================="
        )

        return

    for item in messages:
        role = str(
            item.get(
                "role",
                "",
            )
        ).strip().casefold()

        content = str(
            item.get(
                "content",
                "",
            )
        )

        if role == "user":
            label = "You"
        elif role == "assistant":
            label = "Assistant"
        else:
            label = role.title() or "Message"

        print()
        print(
            f"{label}:"
        )
        print(
            content
        )

    print()
    print(
        "==========================================="
    )


# ============================================================
# HISTORY
# ============================================================

def add_to_history(
    *,
    history: list[
        ChatTurn
    ],
    user_query: str,
    assistant_answer: str,
) -> None:
    history.append(
        ChatTurn(
            role="user",
            content=user_query,
        )
    )

    history.append(
        ChatTurn(
            role="assistant",
            content=assistant_answer,
        )
    )

    maximum_messages = (
        MAX_HISTORY_TURNS
        * 2
    )

    if len(
        history
    ) > maximum_messages:
        del history[
            :-maximum_messages
        ]


# ============================================================
# MAIN
# ============================================================

async def main(
) -> None:
    settings = get_settings()

    db = Database()
    db_opened = False

    llm: (
        OpenAICompatibleChatModel
        | None
    ) = None

    try:
        # ----------------------------------------------------
        # 1. DATABASE
        # ----------------------------------------------------

        await db.open()
        db_opened = True

        print(
            "PostgreSQL/pgvector connected."
        )

        # ----------------------------------------------------
        # 1A. PERSISTENT CONVERSATION
        # ----------------------------------------------------

        conversation_store = ConversationStore(
            db
        )

        await conversation_store.ensure_schema()

        (
            conversation_id,
            history,
        ) = await select_conversation(
            store=conversation_store,
        )

        # ----------------------------------------------------
        # 2. LLM
        # ----------------------------------------------------

        api_key = read_secret(
            settings.llm_api_key
        )

        if not api_key:
            raise RuntimeError(
                "LLM_API_KEY is missing from .env."
            )

        # ----------------------------------------------------
        # 3. ALL CONFIGURED MCP SERVERS
        # ----------------------------------------------------

        mcp_servers = resolve_mcp_servers(
            settings
        )

        gateway = MCPGateway(
            servers=mcp_servers
        )

        registry = ToolRegistry(
            gateway=gateway,
            policy_path=(
                resolve_policy_path(
                    settings
                )
            ),
        )

        print(
            "\nConfigured MCP servers:"
        )

        for (
            server_name,
            server_url,
        ) in mcp_servers.items():
            print(
                f"  {server_name}: "
                f"{server_url}"
            )

        # ----------------------------------------------------
        # 4. DISCOVER LIVE MCP TOOLS
        # ----------------------------------------------------

        tools = (
            await registry.refresh()
        )

        if not tools:
            print(
                "No MCP tools were discovered."
            )
            return

        print(
            f"\nLoaded {len(tools)} MCP tools."
        )

        if DEBUG_RETRIEVER:
            print(
                "\n========== LIVE MCP CATALOG =========="
            )

            for tool in sorted(
                tools,
                key=lambda item: (
                    item.server_name,
                    item.name,
                ),
            ):
                print(
                    f"{tool.server_name}/"
                    f"{tool.name}"
                )

        read_only_tools = [
            tool
            for tool in tools
            if is_read_only_tool(
                tool
            )
        ]

        if not read_only_tools:
            print(
                "No read-only MCP tools are available."
            )
            return

        print(
            "Read-only tools available:",
            len(
                read_only_tools
            ),
        )

        # ----------------------------------------------------
        # 5. RETRIEVER
        # ----------------------------------------------------

        retriever = await build_retriever(
            db,
            read_only_tools,
        )

        print(
            "\nPGVector hybrid retriever ready."
        )

        # ----------------------------------------------------
        # 6. LOCAL USER CONTEXT
        # ----------------------------------------------------

        user = UserContext(
            user_id=(
                "local-cli-user"
            ),
            roles={
                "admin",
            },
        )

        # Trusted API authorization/scope values are separate from
        # question-specific site/zone/line/machine arguments.
        trusted_request_context = (
            TrustedRequestContext.from_environment()
        )

        if trusted_request_context.configured_header_names:
            print(
                "Trusted API request context loaded:",
                ", ".join(
                    trusted_request_context.configured_header_names
                ),
            )
        else:
            print(
                "Trusted API request context: not configured "
                "(only required for tools that need user role/scope headers)."
            )

        # ----------------------------------------------------
        # 7. LLM CLIENT
        # ----------------------------------------------------

        llm = (
            OpenAICompatibleChatModel(
                base_url=(
                    settings.llm_base_url
                ),
                api_key=api_key,
                model=(
                    settings.llm_model
                ),
                timeout_seconds=60,
                max_retries=3,
            )
        )

        # `history` is empty for a new conversation or restored
        # from PostgreSQL for a resumed conversation.

        # Exact values from prior successful structured API calls.
        conversation_context: dict[
            str,
            list[Any],
        ] = {}

        print(
            "LLM implementation:",
            type(
                llm
            ).__name__,
        )

        print(
            "LLM model:",
            settings.llm_model,
        )

        # ----------------------------------------------------
        # 8. CHAT LOOP
        # ----------------------------------------------------

        while True:
            raw_query = input(
                "\nAsk a question, or type exit: "
            ).strip()

            if not raw_query:
                continue

            if raw_query.casefold() in {
                "exit",
                "quit",
                "stop",
            }:
                print(
                    "CLI stopped."
                )
                break

            await conversation_store.add_message(
                conversation_id,
                "user",
                raw_query,
            )

            query = normalize_query(
                raw_query
            )

            tool_context: list[
                dict[str, Any]
            ] = []

            executed_signatures: set[
                str
            ] = set()

            final_answer: (
                str
                | None
            ) = None

            last_tool_reference: (
                str
                | None
            ) = None

            llm_auth_failed = False

            # =================================================
            # MULTI-STEP TOOL WORKFLOW
            # =================================================

            for step in range(
                1,
                MAX_TOOL_STEPS + 1,
            ):
                retrieval_query = (
                    build_retrieval_query(
                        original_query=query,
                        tool_context=(
                            tool_context
                        ),
                    )
                )

                candidates = (
                    await retriever.retrieve(
                        query=(
                            retrieval_query
                        ),
                        user=user,
                        limit=(
                            RETRIEVAL_LIMIT
                        ),
                    )
                )

                candidates = [
                    candidate
                    for candidate
                    in candidates
                    if is_read_only_tool(
                        candidate.tool
                    )
                ][
                    :LLM_SHORTLIST_LIMIT
                ]

                if not candidates:
                    if tool_context:
                        break

                    print(
                        "\nNo relevant authorized "
                        "read-only tool was found."
                    )
                    break

                print_candidates(
                    candidates,
                    step,
                )

                candidate_tools = [
                    candidate.tool
                    for candidate
                    in candidates
                ]

                # --------------------------------------------
                # LLM TOOL SELECTION
                # --------------------------------------------

                try:
                    decision = (
                        await llm.choose_tool(
                            query=query,
                            history=history,
                            tools=(
                                candidate_tools
                            ),
                            tool_context=(
                                tool_context
                            ),
                        )
                    )

                except Exception as exc:
                    if is_invalid_api_key_error(
                        exc
                    ):
                        print_llm_auth_error()
                        llm_auth_failed = True
                        break

                    print(
                        "\nLLM tool selection failed:"
                    )
                    print(
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )
                    break

                if not decision.tool_name:
                    direct_answer = (
                        decision.direct_answer
                        or ""
                    ).strip()

                    if (
                        not tool_context
                        and direct_answer
                    ):
                        final_answer = (
                            direct_answer
                        )

                    break

                candidate_map = (
                    build_candidate_map(
                        candidate_tools
                    )
                )

                selected_tool = (
                    candidate_map.get(
                        decision.tool_name
                    )
                )

                if selected_tool is None:
                    tool_context.append(
                        {
                            "success":
                                False,
                            "stage":
                                "tool_selection",
                            "requested_tool":
                                decision.tool_name,
                            "error":
                                (
                                    "The tool selection was ambiguous "
                                    "or outside the authorized shortlist."
                                ),
                        }
                    )
                    continue

                # --------------------------------------------
                # SCHEMA-DRIVEN ARGUMENT HANDLING
                # --------------------------------------------

                (
                    trusted_arguments,
                    injected_request_fields,
                ) = trusted_request_context.apply_to_tool(
                    selected_tool,
                    decision.arguments
                    or {},
                )

                arguments, _ = (
                    validate_arguments(
                        selected_tool,
                        trusted_arguments,
                    )
                )

                arguments = (
                    apply_required_schema_defaults(
                        selected_tool,
                        arguments,
                    )
                )

                arguments = (
                    canonicalize_schema_values(
                        selected_tool,
                        arguments,
                    )
                )

                (
                    arguments,
                    missing_fields,
                    auto_resolved_fields,
                ) = (
                    resolve_required_from_context(
                        selected_tool,
                        arguments,
                        conversation_context,
                    )
                )

                print(
                    "\n========== LLM TOOL DECISION =========="
                )

                print(
                    "Step:",
                    step,
                )

                print(
                    "Server:",
                    selected_tool.server_name,
                )

                print(
                    "Tool:",
                    selected_tool.name,
                )

                if injected_request_fields:
                    print(
                        "Trusted request fields injected:",
                        ", ".join(
                            injected_request_fields
                        ),
                    )

                if auto_resolved_fields:
                    print(
                        "Auto-resolved required context:",
                        ", ".join(
                            auto_resolved_fields
                        ),
                    )

                print(
                    "Arguments:",
                    json.dumps(
                        arguments,
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    ),
                )

                # --------------------------------------------
                # MISSING REQUIRED CONTEXT
                # --------------------------------------------

                if missing_fields:
                    tool_context.append(
                        {
                            "success":
                                False,
                            "stage":
                                "argument_validation",
                            "server":
                                selected_tool.server_name,
                            "tool":
                                selected_tool.name,
                            "arguments":
                                arguments,
                            "missing_required_fields":
                                missing_fields,
                            "message":
                                (
                                    "The selected API genuinely requires "
                                    "additional identifiers. Choose an "
                                    "authorized read-only context/discovery/"
                                    "listing tool that can resolve them. "
                                    "Do not invent optional hierarchy fields."
                                ),
                        }
                    )

                    print(
                        "\nRequired context not known yet:",
                        ", ".join(
                            missing_fields
                        ),
                    )

                    continue

                # --------------------------------------------
                # DUPLICATE CALL GUARD
                # --------------------------------------------

                signature = (
                    tool_call_signature(
                        selected_tool.server_name,
                        selected_tool.name,
                        arguments,
                    )
                )

                if (
                    signature
                    in executed_signatures
                ):
                    tool_context.append(
                        {
                            "success":
                                False,
                            "stage":
                                "duplicate_call_guard",
                            "server":
                                selected_tool.server_name,
                            "tool":
                                selected_tool.name,
                            "arguments":
                                arguments,
                            "message":
                                (
                                    "The exact same MCP call was already "
                                    "executed in this request. Choose another "
                                    "tool or finish."
                                ),
                        }
                    )

                    print(
                        "\nDuplicate MCP tool call blocked."
                    )

                    continue

                executed_signatures.add(
                    signature
                )

                # --------------------------------------------
                # MCP EXECUTION
                # --------------------------------------------

                raw_result = await safe_call(
                    gateway=gateway,
                    tool=selected_tool,
                    arguments=arguments,
                )

                last_tool_reference = (
                    selected_tool.registry_id
                )

                update_validated_context(
                    conversation_context,
                    raw_result,
                )

                tool_context.append(
                    compact_value(
                        raw_result
                    )
                )

                # --------------------------------------------
                # CONTINUATION CHECK
                # --------------------------------------------

                if step < MAX_TOOL_STEPS:
                    try:
                        needs_more_tools = (
                            await llm.needs_more_tools(
                                query=query,
                                history=history,
                                tool_context=(
                                    tool_context
                                ),
                            )
                        )

                    except Exception as exc:
                        print(
                            "\nTool continuation check failed:"
                        )
                        print(
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        )
                        break

                    if not needs_more_tools:
                        break

            # =================================================
            # INVALID LLM KEY
            # =================================================

            if llm_auth_failed:
                break

            # =================================================
            # FINAL GROUNDED ANSWER
            # =================================================

            if final_answer is None:
                if not tool_context:
                    print(
                        "\nNo final answer could be generated."
                    )
                    continue

                try:
                    final_answer = (
                        await llm.answer(
                            query=query,
                            history=history,
                            tool_name=(
                                last_tool_reference
                                or "factigent_workflow"
                            ),
                            tool_result={
                                "tool_calls":
                                    tool_context
                            },
                            max_output_tokens=(
                                int(
                                    getattr(
                                        settings,
                                        "llm_max_output_tokens",
                                        2000,
                                    )
                                    or 2000
                                )
                            ),
                        )
                    )

                except Exception as exc:
                    if is_invalid_api_key_error(
                        exc
                    ):
                        print_llm_auth_error()
                        break

                    print(
                        "\nLLM final answer failed:"
                    )
                    print(
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )
                    continue

            await conversation_store.add_message(
                conversation_id,
                "assistant",
                final_answer,
            )

            print(
                "\n========== FINAL LLM ANSWER =========="
            )

            print(
                final_answer
            )

            add_to_history(
                history=history,
                user_query=(
                    raw_query
                ),
                assistant_answer=(
                    final_answer
                ),
            )

    finally:
        if llm is not None:
            await llm.close()

        if db_opened:
            await db.close()


if __name__ == "__main__":
    run_async(
        main()
    )
    