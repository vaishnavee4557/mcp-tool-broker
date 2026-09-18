from __future__ import annotations
import asyncio
import json
import re
from pathlib import Path
from typing import Any

from mcp_broker.llm import OpenAICompatibleChatModel
from mcp_broker.mcp_gateway import MCPGateway
from mcp_broker.models import (
    ChatTurn,
    ToolDescriptor,
    UserContext,
)
from mcp_broker.db import Database

from mcp_broker.pg_retriever import (
    PgVectorToolRetriever,
)

from mcp_broker.retrieval import (
    PgHybridToolRetriever,
)
from mcp_broker.registry import ToolRegistry
from mcp_broker.settings import get_settings



# ============================================================
# CONFIGURATION
# ============================================================

DEBUG_RETRIEVER = True

SERVER_NAME = "factigent_api"

DEFAULT_MCP_URL = (
    "http://127.0.0.1:8001/mcp"
)

# Retriever first searches a larger set.
RETRIEVAL_LIMIT = 12

# LLM receives only the best few tools.
LLM_SHORTLIST_LIMIT = 8

# One user question may require:
#
# parameter discovery
#       ↓
# reading/trend tool
#
# Therefore allow multiple MCP steps.
MAX_TOOL_STEPS = 4

# Keep limited in-memory history for the current CLI session.
# Persistent DB history can be added later.
MAX_HISTORY_TURNS = 10


# ============================================================
# KNOWN BACKEND CONTRACT REPAIRS
# ============================================================

DASHBOARD_TREND_TOOL = (
    "dashboard_trends_dashboard_trend_charts_get"
)

FEATURE_DATA_TOOL = (
    "feature_table_data_dashboard_feature_table_data_get"
)

DEFAULT_TREND_WINDOW_SIZE = 30
DEFAULT_READING_WINDOW_SIZE = 5
DEFAULT_TREND_INTERVAL_MINUTES = 60


# ============================================================
# READ-ONLY SAFETY
# ============================================================

MUTATING_MARKERS = (
    "create_",
    "remove_",
    "delete_",
    "start_",
    "stop_",
    "submit_",
    "_post",
    "_put",
    "_patch",
    "_delete",
)


# ============================================================
# QUERY NORMALIZATION
# ============================================================

QUERY_FIXES = {
    "temperture": "temperature",
    "temprature": "temperature",
    "tempereture": "temperature",
    "vibartion": "vibration",
    "viberation": "vibration",
    "paramater": "parameter",
    "paramters": "parameters",
}


def normalize_query(
    query: str,
) -> str:
    """
    Normalize common typing problems without changing
    the actual meaning of the user's question.
    """

    query = " ".join(
        query.strip().split()
    )

    # MACH -3 -> MACH-3
    query = re.sub(
        r"\bMACH\s*-\s*(\d+)\b",
        r"MACH-\1",
        query,
        flags=re.IGNORECASE,
    )

    # MAC -1 -> MAC1
    query = re.sub(
        r"\bMAC\s*-\s*(\d+)\b",
        r"MAC\1",
        query,
        flags=re.IGNORECASE,
    )

    lowered = query.lower()

    for wrong, correct in (
        QUERY_FIXES.items()
    ):
        lowered = lowered.replace(
            wrong,
            correct,
        )

    return lowered


# ============================================================
# SETTINGS / POLICY
# ============================================================

def resolve_policy_path(
    settings: Any,
) -> Path:
    """
    Resolve tool_policies.json location.
    """

    configured_path = getattr(
        settings,
        "tool_policies_path",
        None,
    )

    if configured_path:
        policy_path = Path(
            configured_path
        )

    else:
        # Project root/config/tool_policies.json
        policy_path = (
            Path(__file__)
            .resolve()
            .parents[2]
            / "config"
            / "tool_policies.json"
        )

    if not policy_path.is_absolute():
        policy_path = (
            Path.cwd()
            / policy_path
        )

    return policy_path.resolve()


def read_secret(
    value: Any,
) -> str:
    """
    Support either normal strings or Pydantic SecretStr.
    """

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


# ============================================================
# TOOL SAFETY
# ============================================================

def is_read_only_tool(
    tool: ToolDescriptor,
) -> bool:
    """
    Prevent write/delete/start/stop tools from being sent
    to the informational chatbot.
    """

    name = tool.name.lower()

    return not any(
        marker in name
        for marker in MUTATING_MARKERS
    )


def validate_arguments(
    tool: ToolDescriptor,
    arguments: dict[str, Any],
) -> tuple[
    dict[str, Any],
    list[str],
]:
    """
    Keep only arguments defined by the actual MCP schema.

    Returns:
        valid_arguments
        missing_required_fields
    """

    schema = (
        tool.input_schema
        or {}
    )

    properties = schema.get(
        "properties",
        {},
    )

    required_fields = schema.get(
        "required",
        [],
    )

    valid_arguments = {
        key: value
        for key, value
        in arguments.items()
        if key in properties
    }

    missing_fields = [
        field
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


def build_candidate_map(
    tools: list[ToolDescriptor],
) -> dict[
    str,
    ToolDescriptor,
]:
    """
    LLM may return either:
        tool.name

    or:
        tool.llm_name

    Support both safely.
    """

    mapping: dict[
        str,
        ToolDescriptor,
    ] = {}

    for tool in tools:

        if tool.name:
            mapping[
                tool.name
            ] = tool

        llm_name = getattr(
            tool,
            "llm_name",
            None,
        )

        if llm_name:
            mapping[
                llm_name
            ] = tool

    return mapping


# ============================================================
# LOCAL RETRIEVER
# ============================================================

async def build_retriever(
    db: Database,
    read_only_tools: list[ToolDescriptor],
) -> PgHybridToolRetriever:
    """
    Wire the DB-backed semantic retriever to the hybrid reranker.

    Important:
    - CLI does not know embedding model/dimension.
    - CLI does not calculate cosine similarity.
    - CLI does not keep tool embeddings in RAM.
    - Retrieval tuning/defaults stay centralized in retrieval.py.
    """

    pg_vector_retriever = PgVectorToolRetriever(
        db=db,
    )

    retriever = PgHybridToolRetriever(
        pg_retriever=pg_vector_retriever,
    )

    # This rebuild prepares only live MCP ToolDescriptor metadata
    # required for lexical/domain/access-policy reranking.
    # Stored tool vectors remain in PostgreSQL/pgvector.
    await retriever.rebuild(
        read_only_tools
    )

    return retriever

# ============================================================
# RETRIEVAL CONTEXT
# ============================================================

def compact_value(
    value: Any,
    *,
    depth: int = 0,
) -> Any:
    """
    Reduce API/MCP results before sending them back
    into retrieval or LLM context.

    This prevents massive API responses from consuming
    unnecessary tokens.
    """

    if depth >= 6:
        return "<nested data removed>"

    if isinstance(
        value,
        dict,
    ):
        result: dict[
            str,
            Any,
        ] = {}

        # Generic safety limit.
        for index, (
            key,
            child,
        ) in enumerate(
            value.items()
        ):
            if index >= 40:
                result[
                    "_truncated_keys"
                ] = (
                    len(value) - 40
                )
                break

            result[str(key)] = (
                compact_value(
                    child,
                    depth=depth + 1,
                )
            )

        return result

    if isinstance(
        value,
        list,
    ):
        items = [
            compact_value(
                item,
                depth=depth + 1,
            )
            for item
            in value[:12]
        ]

        if len(value) > 12:
            items.append(
                {
                    "_notice": (
                        f"{len(value) - 12} "
                        "additional items removed"
                    )
                }
            )

        return items

    if isinstance(
        value,
        str,
    ):
        if len(value) > 1500:
            return (
                value[:1500]
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
    On second/third MCP step, include a small amount
    of previous tool context in retrieval.

    Example:

    User:
        temperature of MACH-3

    Step 1:
        machine parameter API

    Step 2 retrieval:
        original question
        +
        discovered exact parameter identifiers

    This allows the correct reading/trend tool
    to rank higher.
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

    if len(context_text) > 1800:
        context_text = (
            context_text[:1800]
            + "..."
        )

    return (
        f"{original_query}\n\n"
        "Previous MCP step context:\n"
        f"{context_text}"
    )


def print_candidates(
    candidates: list[Any],
    step: int,
) -> None:
    """
    Debug output showing exactly what the retriever
    shortlisted before the LLM sees the tools.
    """

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
            f"{candidate.tool.name} "
            f"(score={candidate.score:.4f})"
        )


# ============================================================
# DUPLICATE TOOL CALL PROTECTION
# ============================================================

def tool_call_signature(
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    """
    Prevent LLM from calling the exact same tool
    with the exact same arguments forever.
    """

    return (
        tool_name
        + "::"
        + json.dumps(
            arguments,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
    )


# ============================================================
# BACKEND CONTRACT REPAIR
# ============================================================

def repair_allowed_numeric_argument(
    *,
    error_text: str,
    field_name: str,
    arguments: dict[str, Any],
    preferred: int | None = None,
) -> bool:
    """
    Repair errors such as:

        window_size must be 5, 15, or 30
    """

    pattern = (
        rf"{re.escape(field_name)}"
        rf"\s+must\s+be\s+([^'\"]+)"
    )

    match = re.search(
        pattern,
        error_text,
        flags=re.IGNORECASE,
    )

    if not match:
        return False

    allowed = [
        int(value)
        for value
        in re.findall(
            r"\d+",
            match.group(1),
        )
    ]

    if not allowed:
        return False

    if (
        preferred is not None
        and preferred in allowed
    ):
        chosen = preferred
    else:
        chosen = max(
            allowed
        )

    arguments[
        field_name
    ] = chosen

    return True


def apply_known_defaults(
    tool: ToolDescriptor,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """
    Apply only backend defaults we already know are
    stricter than the generated OpenAPI document.

    This does NOT choose the tool.
    LLM already selected the tool.
    """

    repaired = dict(
        arguments
    )

    properties = (
        tool.input_schema
        or {}
    ).get(
        "properties",
        {},
    )

    if (
        tool.name
        in {
            DASHBOARD_TREND_TOOL,
            FEATURE_DATA_TOOL,
        }
        and "window_size"
        in properties
    ):
        preferred = (
            DEFAULT_TREND_WINDOW_SIZE
            if tool.name
            == DASHBOARD_TREND_TOOL
            else DEFAULT_READING_WINDOW_SIZE
        )

        current = repaired.get(
            "window_size"
        )

        if current not in {
            5,
            15,
            30,
        }:
            repaired[
                "window_size"
            ] = preferred

    if (
        tool.name
        == DASHBOARD_TREND_TOOL
        and "interval_minutes"
        in properties
    ):
        repaired.setdefault(
            "interval_minutes",
            DEFAULT_TREND_INTERVAL_MINUTES,
        )

    return repaired


# ============================================================
# MCP EXECUTION
# ============================================================

async def safe_call(
    *,
    gateway: MCPGateway,
    tool: ToolDescriptor,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute the exact MCP tool selected by the LLM.

    MCP calls the real Factigent API.
    """

    tool_name = tool.name

    print(
        "\n========== MCP TOOL CALL =========="
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
                SERVER_NAME,
                tool_name,
                arguments,
            )
        )

        wrapped = {
            "success": True,
            "tool": tool_name,
            "arguments": arguments,
            "result": raw_result,
        }

    except Exception as exc:

        original_error = str(
            exc
        )

        repaired_arguments = dict(
            arguments
        )

        should_retry = False

        # -----------------------------------------------
        # Known trend/window backend mismatch recovery.
        # -----------------------------------------------

        if tool_name in {
            DASHBOARD_TREND_TOOL,
            FEATURE_DATA_TOOL,
        }:

            preferred_window = (
                DEFAULT_TREND_WINDOW_SIZE
                if tool_name
                == DASHBOARD_TREND_TOOL
                else DEFAULT_READING_WINDOW_SIZE
            )

            if (
                "window_size is required"
                in original_error
            ):
                repaired_arguments[
                    "window_size"
                ] = preferred_window

                should_retry = True

            if repair_allowed_numeric_argument(
                error_text=original_error,
                field_name="window_size",
                arguments=repaired_arguments,
                preferred=preferred_window,
            ):
                should_retry = True

        if (
            tool_name
            == DASHBOARD_TREND_TOOL
        ):
            if (
                "interval_minutes is required"
                in original_error
            ):
                repaired_arguments[
                    "interval_minutes"
                ] = (
                    DEFAULT_TREND_INTERVAL_MINUTES
                )

                should_retry = True

            if repair_allowed_numeric_argument(
                error_text=original_error,
                field_name="interval_minutes",
                arguments=repaired_arguments,
                preferred=(
                    DEFAULT_TREND_INTERVAL_MINUTES
                ),
            ):
                should_retry = True

        # -----------------------------------------------
        # Retry exactly once after repair.
        # -----------------------------------------------

        if should_retry:

            print(
                "\nBackend contract mismatch "
                "detected. Retrying with:"
            )

            print(
                json.dumps(
                    repaired_arguments,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
            )

            try:
                raw_result = (
                    await gateway.call_tool(
                        SERVER_NAME,
                        tool_name,
                        repaired_arguments,
                    )
                )

                wrapped = {
                    "success": True,
                    "tool": tool_name,
                    "arguments": (
                        repaired_arguments
                    ),
                    "result": raw_result,
                    (
                        "recovered_from_"
                        "backend_contract_mismatch"
                    ): True,
                }

            except Exception as retry_exc:

                wrapped = {
                    "success": False,
                    "tool": tool_name,
                    "arguments": (
                        repaired_arguments
                    ),
                    "error_type": (
                        type(
                            retry_exc
                        ).__name__
                    ),
                    "error": str(
                        retry_exc
                    ),
                }

        else:

            wrapped = {
                "success": False,
                "tool": tool_name,
                "arguments": arguments,
                "error_type": (
                    type(exc).__name__
                ),
                "error": (
                    original_error
                ),
            }

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
    """
    Detect OpenAI authentication errors so we can stop
    immediately instead of continuing with a broken LLM.
    """

    text = str(exc).lower()

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


def print_llm_auth_error() -> None:
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
        "The LLM API key configured in .env "
        "is invalid."
    )
    print(
        "Update LLM_API_KEY with a valid key "
        "and restart the CLI."
    )


# ============================================================
# CONVERSATION HISTORY
# ============================================================

def add_to_history(
    *,
    history: list[ChatTurn],
    user_query: str,
    assistant_answer: str,
) -> None:
    """
    Current CLI session memory.

    This is intentionally small.
    Persistent conversation storage can later replace this.
    """

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

    if (
        len(history)
        > maximum_messages
    ):
        del history[
            :-maximum_messages
        ]


# ============================================================
# MAIN CHATBOT
# ============================================================

async def main() -> None:
    """
    Factigent CLI orchestration.

    Stable boundary:
        CLI -> retriever.retrieve(...)

    The CLI does not care whether semantic retrieval is implemented
    with local vectors, pgvector, or another backend in the future.
    """

    settings = get_settings()

    db = Database()
    db_opened = False

    llm: OpenAICompatibleChatModel | None = None

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
        # 2. SETTINGS / LLM KEY
        # ----------------------------------------------------

        api_key = read_secret(
            settings.llm_api_key
        )

        if not api_key:
            raise RuntimeError(
                "LLM_API_KEY is missing from .env."
            )

        mcp_url = (
            getattr(
                settings,
                "mcp_server_url",
                None,
            )
            or getattr(
                settings,
                "mcp_url",
                None,
            )
            or DEFAULT_MCP_URL
        )

        # ----------------------------------------------------
        # 3. MCP CONNECTION
        # ----------------------------------------------------

        gateway = MCPGateway(
            servers={
                SERVER_NAME: mcp_url,
            }
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
            "Connecting to MCP server:",
            mcp_url,
        )

        # ----------------------------------------------------
        # 4. DISCOVER LIVE MCP TOOLS
        # ----------------------------------------------------

        tools = await registry.refresh()

        if not tools:
            print(
                "No MCP tools were discovered."
            )
            return

        print(
            f"\nLoaded {len(tools)} MCP tools."
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
            len(read_only_tools),
        )

        # ----------------------------------------------------
        # 5. BUILD PGVECTOR-BACKED HYBRID RETRIEVER
        # ----------------------------------------------------

        retriever = await build_retriever(
            db,
            read_only_tools,
        )

        print(
            "\nPGVector hybrid retriever ready."
        )

        # ----------------------------------------------------
        # 6. USER CONTEXT
        # ----------------------------------------------------

        user = UserContext(
            user_id="local-cli-user",
            roles={
                "admin",
            },
        )

        # ----------------------------------------------------
        # 7. LLM CLIENT
        # ----------------------------------------------------

        llm = OpenAICompatibleChatModel(
            base_url=(
                settings.llm_base_url
            ),
            api_key=api_key,
            model=settings.llm_model,
            timeout_seconds=60,
            max_retries=3,
        )

        history: list[
            ChatTurn
        ] = []

        print(
            "LLM implementation:",
            type(llm).__name__,
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

            if raw_query.lower() in {
                "exit",
                "quit",
                "stop",
            }:
                print(
                    "CLI stopped."
                )
                break

            query = normalize_query(
                raw_query
            )

            # Per-request workflow context.
            # This remains temporary RAM state.
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

            last_tool_name: (
                str
                | None
            ) = None

            llm_auth_failed = False

            # =================================================
            # 9. MULTI-STEP WORKFLOW
            #
            # Question
            #   -> retriever
            #   -> question embedding (temporary)
            #   -> pgvector tool similarity
            #   -> hybrid reranking
            #   -> LLM tool selection
            #   -> MCP
            #   -> repeat when more data is needed
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

                # Stable CLI/retriever boundary.
                # The implementation behind retrieve() may change later
                # without changing this chat workflow.
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

                # Extra safety boundary:
                # only read-only tools are exposed to the LLM.
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
                        # We already have MCP context, so generate
                        # the final answer from what was retrieved.
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
                # LLM CALL #1:
                # SELECT EXACT TOOL + ARGUMENTS
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

                # --------------------------------------------
                # LLM SAYS NO ADDITIONAL TOOL IS REQUIRED
                # --------------------------------------------

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

                    # When tool_context exists, ignore the selector's
                    # prose and generate the grounded final response
                    # using llm.answer() below.
                    break

                # --------------------------------------------
                # ENSURE LLM SELECTED A SHORTLISTED TOOL
                # --------------------------------------------

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
                            "success": False,
                            "stage": (
                                "tool_selection"
                            ),
                            "requested_tool": (
                                decision.tool_name
                            ),
                            "error": (
                                "LLM selected a tool "
                                "outside the authorized "
                                "retriever shortlist."
                            ),
                        }
                    )

                    continue

                # --------------------------------------------
                # VALIDATE + APPLY KNOWN BACKEND DEFAULTS
                # --------------------------------------------

                # First remove arguments that do not exist
                # in the live MCP schema.
                arguments, _ = (
                    validate_arguments(
                        selected_tool,
                        decision.arguments
                        or {},
                    )
                )

                # Apply centralized known contract repairs/defaults.
                arguments = (
                    apply_known_defaults(
                        selected_tool,
                        arguments,
                    )
                )

                # IMPORTANT:
                # Revalidate AFTER defaults were added.
                # Otherwise a field that was filled above may still
                # remain incorrectly listed in missing_fields.
                arguments, missing_fields = (
                    validate_arguments(
                        selected_tool,
                        arguments,
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
                    "Tool:",
                    selected_tool.name,
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
                # REQUIRED FIELD STILL MISSING
                # --------------------------------------------

                if missing_fields:

                    tool_context.append(
                        {
                            "success": False,
                            "stage": (
                                "argument_validation"
                            ),
                            "tool": (
                                selected_tool.name
                            ),
                            "arguments": (
                                arguments
                            ),
                            "missing_required_fields": (
                                missing_fields
                            ),
                            "message": (
                                "Required arguments are "
                                "missing. Choose a suitable "
                                "Factigent discovery/listing "
                                "tool first if it can provide "
                                "the missing identifiers."
                            ),
                        }
                    )

                    print(
                        "\nTool not executed. "
                        "Missing required fields:",
                        ", ".join(
                            missing_fields
                        ),
                    )

                    continue

                # --------------------------------------------
                # DUPLICATE TOOL CALL PROTECTION
                # --------------------------------------------

                signature = (
                    tool_call_signature(
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
                            "success": False,
                            "stage": (
                                "duplicate_call_guard"
                            ),
                            "tool": (
                                selected_tool.name
                            ),
                            "arguments": (
                                arguments
                            ),
                            "message": (
                                "The exact same MCP "
                                "tool call has already "
                                "been executed in this "
                                "request. Choose another "
                                "tool or finish."
                            ),
                        }
                    )

                    print(
                        "\nDuplicate MCP tool "
                        "call blocked."
                    )

                    continue

                executed_signatures.add(
                    signature
                )

                # --------------------------------------------
                # MCP EXECUTION
                # --------------------------------------------

                raw_result = (
                    await safe_call(
                        gateway=gateway,
                        tool=selected_tool,
                        arguments=(
                            arguments
                        ),
                    )
                )

                last_tool_name = (
                    selected_tool.name
                )

                compact_result = (
                    compact_value(
                        raw_result
                    )
                )

                tool_context.append(
                    compact_result
                )

            # =================================================
            # 10. INVALID LLM KEY = STOP CLI
            # =================================================

            if llm_auth_failed:
                break

            # =================================================
            # 11. LLM CALL #2: FINAL GROUNDED RESPONSE
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
                                last_tool_name
                                or "factigent_workflow"
                            ),
                            tool_result={
                                "tool_calls": (
                                    tool_context
                                )
                            },
                            max_output_tokens=max(
                                int(
                                    getattr(
                                        settings,
                                        "llm_max_output_tokens",
                                        2000,
                                    )
                                    or 2000
                                ),
                                2000,
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

                    # User requires final response from the LLM.
                    # Do not expose raw API data as a fake fallback.
                    continue

            # =================================================
            # 12. PRINT FINAL ANSWER
            # =================================================

            print(
                "\n========== FINAL LLM ANSWER =========="
            )

            print(
                final_answer
            )

            # =================================================
            # 13. CURRENT SESSION HISTORY (RAM)
            # =================================================

            add_to_history(
                history=history,
                user_query=raw_query,
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
    asyncio.run(
        main()
        
    )
