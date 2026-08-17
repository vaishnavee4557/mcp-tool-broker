from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp_broker.embeddings import OpenAICompatibleEmbeddingProvider
try:
    from mcp_broker.embeddings import HashEmbeddingProvider
except ImportError:
    HashEmbeddingProvider = None  # type: ignore

from mcp_broker.llm import OpenAICompatibleChatModel
from mcp_broker.mcp_gateway import MCPGateway
from mcp_broker.models import ChatTurn, ToolDescriptor, UserContext
from mcp_broker.registry import ToolRegistry
from mcp_broker.retrieval import HybridToolRetriever
from mcp_broker.settings import get_settings
DEBUG_RETRIEVER = True
SERVER_NAME = "factigent_api"
MCP_URL = "http://127.0.0.1:8001/mcp"

# Known read-only Factigent tools from your MCP discovery.
MACHINE_LIST_TOOL = "machine_names_dashboard_machine_names_get"
MACHINE_MAPPING_TOOL = "machine_mappings_api_inference_machine_mappings_get"
MACHINE_PARAMETERS_TOOL = "machine_parameters_api_pipeline_machine_parameters_get"
TOP_RISKY_TOOL = "top_risky_machines_dashboard_top_risky_machines_get"
HEALTH_STATUS_TOOL = "machine_health_status_dashboard_machine_health_status_ge"
HEALTH_TREND_TOOL = "health_trend_dashboard_health_trend_get"
MODEL_TOOL = "get_model_api_inference_model"
DASHBOARD_TREND_TOOL = "dashboard_trends_dashboard_trend_charts_get"
FEATURE_DATA_TOOL = "feature_table_data_dashboard_feature_table_data_get"
MACHINE_COUNT_TOOL = "machine_count_dashboard_machine_count_get"
PARAM_COUNT_TOOL = "machine_parameter_count_dashboard_machine_parameter_coun"

# Backend contract is stricter than the generated OpenAPI schema for trend.
#
# The live endpoint returns HTTP 400 unless window_size is supplied.

DEFAULT_TREND_WINDOW_SIZE = 30
DEFAULT_READING_WINDOW_SIZE = 5
DEFAULT_TREND_INTERVAL_MINUTES = 60

GENERIC_SENSORS = (
    "temperature",
    "vibration",
    "humidity",
    "power",
    "voltage",
    "frequency",
    "level",
    "pressure",
    "current",
    "speed",
)

QUERY_FIXES = {
    "temperture": "temperature",
    "temprature": "temperature",
    "tempereture": "temperature",
    "vibartion": "vibration",
    "viberation": "vibration",
    "paramater": "parameter",
    "paramters": "parameters",
    "mach -": "mach-",
    "mac -": "mac-",
}

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


@dataclass
class ConversationState:
    last_query: str | None = None
    last_machine_ref: str | None = None
    last_machine_name: str | None = None
    last_sensor: str | None = None
    last_intent: str | None = None
    last_result: Any | None = None
    pending_query: str | None = None
    machine_names_cache: list[str] = field(default_factory=list)


def normalize_query(query: str) -> str:
    q = " ".join(query.strip().split())

    # Normalize common machine ID spacing, e.g. "MACH -3" -> "MACH-3".
    q = re.sub(r"\bMACH\s*-\s*(\d+)\b", r"MACH-\1", q, flags=re.IGNORECASE)
    q = re.sub(r"\bMAC\s*-\s*(\d+)\b", r"MAC\1", q, flags=re.IGNORECASE)

    lowered = q.lower()
    for wrong, right in QUERY_FIXES.items():
        lowered = lowered.replace(wrong, right)

    return lowered



def repair_allowed_numeric_argument(
    error_text: str,
    field_name: str,
    arguments: dict[str, Any],
    preferred: int | None = None,
) -> bool:
    """
    Repair backend validation errors like:
      "window_size must be 5, 15, or 30"

    Returns True only when an allowed numeric value was found and applied.
    """
    pattern = rf"{re.escape(field_name)}\s+must\s+be\s+([^'\"]+)"
    match = re.search(pattern, error_text, flags=re.IGNORECASE)

    if not match:
        return False

    allowed = [
        int(value)
        for value in re.findall(r"\d+", match.group(1))
    ]

    if not allowed:
        return False

    if preferred is not None and preferred in allowed:
        chosen = preferred
    else:
        # Prefer the largest allowed window for a useful default trend.
        chosen = max(allowed)

    arguments[field_name] = chosen
    return True


def resolve_policy_path(settings: Any) -> Path:
    configured_path = getattr(settings, "tool_policies_path", None)

    if configured_path:
        policy_path = Path(configured_path)
    else:
        policy_path = Path(__file__).resolve().parent / "tool_policies.json"

    if not policy_path.is_absolute():
        policy_path = Path.cwd() / policy_path

    return policy_path.resolve()


def validate_arguments(
    tool: ToolDescriptor,
    arguments: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    schema = tool.input_schema or {}
    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])

    valid_arguments = {
        key: value
        for key, value in arguments.items()
        if key in properties
    }

    missing_fields = [
        field
        for field in required_fields
        if field not in valid_arguments
        or valid_arguments[field] in (None, "")
    ]

    return valid_arguments, missing_fields


def tool_map(tools: list[ToolDescriptor]) -> dict[str, ToolDescriptor]:
    return {tool.name: tool for tool in tools}


def is_read_only_tool(tool: ToolDescriptor) -> bool:
    name = tool.name.lower()
    return not any(marker in name for marker in MUTATING_MARKERS)


def extract_machine_reference(query: str) -> str | None:
    patterns = (
        r"\bMAC\d+\b",
        r"\bMACH-\d+\b",
        r"\bSTRILIZE\d+\b",
    )

    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return match.group(0).upper()

    return None


def detect_sensor(query: str) -> str | None:
    q = normalize_query(query)

    for sensor in GENERIC_SENSORS:
        if sensor in q:
            return sensor

    return None


def iter_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_dicts(child)


def collect_strings(value: Any) -> list[str]:
    result: list[str] = []

    if isinstance(value, str):
        result.append(value)
    elif isinstance(value, dict):
        for key, child in value.items():
            if isinstance(key, str):
                result.append(key)
            result.extend(collect_strings(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(collect_strings(child))

    return result


def resolve_machine_name_from_mapping(
    mapping_result: Any,
    machine_ref: str,
) -> str | None:
    target = machine_ref.lower()

    id_keys = (
        "machineId",
        "machine_id",
        "machineID",
        "id",
    )
    name_keys = (
        "machine_name",
        "machineName",
        "name",
    )

    for obj in iter_dicts(mapping_result):
        found_id = None

        for key in id_keys:
            value = obj.get(key)
            if value is not None:
                found_id = str(value)
                break

        if found_id is None or found_id.lower() != target:
            continue

        for key in name_keys:
            value = obj.get(key)
            if value:
                return str(value)

    return None


def extract_machine_names(result: Any) -> list[str]:
    if isinstance(result, dict):
        for key in ("machine_names", "machines", "data"):
            if key in result:
                names = extract_machine_names(result[key])
                if names:
                    return names

    if isinstance(result, list):
        names: list[str] = []

        for item in result:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict):
                for key in ("machine_name", "machineName", "name"):
                    if item.get(key):
                        names.append(str(item[key]))
                        break

        return names

    return []


def find_parameter_name(
    parameter_result: Any,
    sensor: str,
) -> str | None:
    sensor_lower = sensor.lower()
    candidates: list[str] = []

    for text in collect_strings(parameter_result):
        cleaned = text.strip()

        if not cleaned:
            continue

        if sensor_lower in cleaned.lower():
            candidates.append(cleaned)

    if not candidates:
        return None

    # Prefer values that look like exact parameter identifiers.
    def score(item: str) -> tuple[int, int, int]:
        return (
            0 if "_" in item else 1,
            0 if item.lower().endswith(sensor_lower) else 1,
            len(item),
        )

    candidates.sort(key=score)
    return candidates[0]


def is_yes(query: str) -> bool:
    return normalize_query(query) in {
        "yes",
        "y",
        "yeah",
        "yep",
        "haan",
        "ha",
        "ok",
        "okay",
        "do it",
    }


def is_machine_list_query(query: str) -> bool:
    q = normalize_query(query)
    return any(
        phrase in q
        for phrase in (
            "show all machines",
            "list all machines",
            "list machines",
            "machine names",
            "available machines",
            "what machines",
        )
    )


def is_parameter_list_query(query: str) -> bool:
    q = normalize_query(query)

    return (
        ("parameter" in q or "sensor" in q)
        and not any(word in q for word in ("trend", "history", "historical"))
    )


def is_risky_query(query: str) -> bool:
    q = normalize_query(query)
    return (
        "risky" in q
        or "highest risk" in q
        or "most risk" in q
        or "risk machine" in q
    )


def is_health_status_query(query: str) -> bool:
    q = normalize_query(query)
    return "health" in q and "trend" not in q


def is_health_trend_query(query: str) -> bool:
    q = normalize_query(query)
    return "health" in q and "trend" in q


def is_model_query(query: str) -> bool:
    q = normalize_query(query)
    return "model" in q and any(
        word in q for word in ("assigned", "using", "machine", "which")
    )


def is_trend_query(query: str) -> bool:
    q = normalize_query(query)
    return any(
        word in q
        for word in ("trend", "history", "historical", "over time")
    )


def is_current_sensor_query(query: str) -> bool:
    q = normalize_query(query)
    sensor = detect_sensor(q)

    if not sensor:
        return False

    if is_trend_query(q):
        return False

    # "temperature of MACH-3" should count as a reading request.
    return True


def is_count_query(query: str) -> bool:
    q = normalize_query(query)
    return any(
        phrase in q
        for phrase in (
            "how many machines",
            "machine count",
            "number of machines",
        )
    )


def previous_names_followup(query: str, state: ConversationState) -> bool:
    if state.last_result is None:
        return False

    q = normalize_query(query)

    return q in {
        "their name",
        "there name",
        "their names",
        "there names",
        "name",
        "names",
        "machine name",
        "machine names",
    }


def direct_previous_names_answer(state: ConversationState) -> str | None:
    names: list[str] = []

    for obj in iter_dicts(state.last_result):
        for key in ("machine_name", "machineName"):
            value = obj.get(key)
            if value:
                names.append(str(value))

    # Keep order while removing duplicates.
    unique = list(dict.fromkeys(names))

    if unique:
        return "Machine names: " + ", ".join(unique)

    return None


async def safe_call(
    gateway: MCPGateway,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    print("\n========== MCP TOOL CALL ==========")
    print("Tool:", tool_name)
    print(
        "Arguments:",
        json.dumps(arguments, indent=2, ensure_ascii=False, default=str),
    )

    try:
        raw_result = await gateway.call_tool(
            SERVER_NAME,
            tool_name,
            arguments,
        )

        wrapped = {
            "success": True,
            "tool": tool_name,
            "arguments": arguments,
            "result": raw_result,
        }

    except Exception as exc:
        error_text = str(exc)

        # The live trend endpoint currently requires fields that its
        # OpenAPI schema marks as optional. Repair that contract once
        # automatically instead of surfacing another avoidable 400.
        repaired_arguments = dict(arguments)
        should_retry = False

        if tool_name in {
            DASHBOARD_TREND_TOOL,
            FEATURE_DATA_TOOL,
        }:
            preferred_window = (
                DEFAULT_TREND_WINDOW_SIZE
                if tool_name == DASHBOARD_TREND_TOOL
                else DEFAULT_READING_WINDOW_SIZE
            )

            if "window_size is required" in error_text:
                repaired_arguments["window_size"] = preferred_window
                should_retry = True

            # Live backend may expose a stricter enum than OpenAPI.
            if repair_allowed_numeric_argument(
                error_text=error_text,
                field_name="window_size",
                arguments=repaired_arguments,
                preferred=preferred_window,
            ):
                should_retry = True

        if tool_name == DASHBOARD_TREND_TOOL:
            if "interval_minutes is required" in error_text:
                repaired_arguments["interval_minutes"] = (
                    DEFAULT_TREND_INTERVAL_MINUTES
                )
                should_retry = True

            if repair_allowed_numeric_argument(
                error_text=error_text,
                field_name="interval_minutes",
                arguments=repaired_arguments,
                preferred=DEFAULT_TREND_INTERVAL_MINUTES,
            ):
                should_retry = True

        if should_retry:
            print("\nBackend contract mismatch detected; retrying with:")
            print(
                json.dumps(
                    repaired_arguments,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
            )

            try:
                raw_result = await gateway.call_tool(
                    SERVER_NAME,
                    tool_name,
                    repaired_arguments,
                )

                arguments = repaired_arguments

                wrapped = {
                    "success": True,
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": raw_result,
                    "recovered_from_backend_contract_mismatch": True,
                }

            except Exception as retry_exc:
                wrapped = {
                    "success": False,
                    "tool": tool_name,
                    "arguments": repaired_arguments,
                    "error_type": type(retry_exc).__name__,
                    "error": str(retry_exc),
                }

        else:
            wrapped = {
                "success": False,
                "tool": tool_name,
                "arguments": arguments,
                "error_type": type(exc).__name__,
                "error": error_text,
            }

    print("\n========== MCP TOOL RESULT ==========")
    print(
        json.dumps(
            wrapped,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )

    return wrapped


def fallback_text_from_result(query: str, tool_result: Any) -> str:
    """
    Last-resort answer if the LLM final formatting call fails.
    This prevents a successful MCP/API result from becoming a user-visible error.
    """
    if isinstance(tool_result, dict):
        if tool_result.get("success") is False:
            return (
                "The Factigent API call failed: "
                + str(tool_result.get("error", "Unknown error"))
            )

        workflow = tool_result.get("workflow")
        if isinstance(workflow, list) and workflow:
            # Prefer the last successful call in the workflow.
            for item in reversed(workflow):
                if isinstance(item, dict) and item.get("success") is True:
                    return fallback_text_from_result(query, item)

        result = tool_result.get("result", tool_result)

        # Risk dashboard.
        if isinstance(result, dict):
            data = result.get("data")

            if isinstance(data, list) and data and all(
                isinstance(row, dict) for row in data
            ):
                lines = []
                for row in data:
                    name = (
                        row.get("machine_name")
                        or row.get("machineName")
                        or row.get("machineId")
                    )
                    if name:
                        details = []
                        for key in (
                            "machineId",
                            "severity",
                            "health_index",
                            "anomaly_score",
                            "total_alerts",
                            "last_updated",
                        ):
                            if key in row:
                                details.append(f"{key}={row[key]}")
                        lines.append(
                            f"- {name}"
                            + (f": {', '.join(details)}" if details else "")
                        )

                if lines:
                    return "\n".join(lines)

        names = extract_machine_names(result)
        if names:
            return "Machine names:\n- " + "\n- ".join(names)

        return (
            "Factigent returned data successfully:\n"
            + json.dumps(result, indent=2, ensure_ascii=False, default=str)[:5000]
        )

    return str(tool_result)


async def grounded_answer(
    llm: OpenAICompatibleChatModel,
    settings: Any,
    query: str,
    history: list[ChatTurn],
    tool_result: Any,
) -> str:
    try:
        return await llm.answer(
            query=query,
            history=history,
            tool_name="factigent_workflow",
            tool_result=tool_result,
            max_output_tokens=max(
                int(getattr(settings, "llm_max_output_tokens", 2000) or 2000),
                2000,
            ),
        )
    except Exception as exc:
        print(
            "\nLLM final formatting failed; using deterministic fallback:"
        )
        print(f"{type(exc).__name__}: {exc}")

        return fallback_text_from_result(
            query=query,
            tool_result=tool_result,
        )


async def resolve_machine_name(
    gateway: MCPGateway,
    tools: dict[str, ToolDescriptor],
    machine_ref: str,
) -> tuple[str, list[dict[str, Any]]]:
    context: list[dict[str, Any]] = []

    # MACH-1 / MACH-2 / MACH-3 are already machine names in your dashboard.
    if machine_ref.startswith("MACH-"):
        return machine_ref, context

    if MACHINE_MAPPING_TOOL not in tools:
        return machine_ref, context

    mapping_call = await safe_call(
        gateway,
        MACHINE_MAPPING_TOOL,
        {},
    )
    context.append(mapping_call)

    if not mapping_call.get("success"):
        return machine_ref, context

    resolved = resolve_machine_name_from_mapping(
        mapping_call.get("result"),
        machine_ref,
    )

    return resolved or machine_ref, context


async def get_parameters(
    gateway: MCPGateway,
    tools: dict[str, ToolDescriptor],
    machine_ref: str,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    machine_name, context = await resolve_machine_name(
        gateway,
        tools,
        machine_ref,
    )

    call = await safe_call(
        gateway,
        MACHINE_PARAMETERS_TOOL,
        {"machine_name": machine_name},
    )
    context.append(call)

    # Retry once with raw reference if mapping produced an unusable name.
    if not call.get("success") and machine_name != machine_ref:
        call = await safe_call(
            gateway,
            MACHINE_PARAMETERS_TOOL,
            {"machine_name": machine_ref},
        )
        context.append(call)
        machine_name = machine_ref

    return machine_name, call, context


async def handle_known_query(
    query: str,
    gateway: MCPGateway,
    tools: dict[str, ToolDescriptor],
    state: ConversationState,
) -> tuple[bool, Any]:
    """
    Deterministic routing for high-frequency Factigent questions.
    LLM is not allowed to randomly choose the wrong tool here.
    """

    q = normalize_query(query)

    # Follow-up like "their name" after risky-machine output.
    if previous_names_followup(q, state):
        direct = direct_previous_names_answer(state)
        if direct:
            return True, {"direct_answer": direct}

    # Follow-up "yes" means continue the pending operation.
    if is_yes(q):
        if state.pending_query:
            q = state.pending_query
            query = q
            state.pending_query = None
        else:
            return True, {
                "direct_answer": (
                    "There is no pending Factigent action to continue. "
                    "Please ask the machine/data question directly."
                )
            }

    if is_machine_list_query(q) and MACHINE_LIST_TOOL in tools:
        result = await safe_call(gateway, MACHINE_LIST_TOOL, {})
        state.last_intent = "machine_list"
        state.last_result = result

        if result.get("success"):
            state.machine_names_cache = extract_machine_names(
                result.get("result")
            )

        return True, result

    if is_risky_query(q) and TOP_RISKY_TOOL in tools:
        result = await safe_call(gateway, TOP_RISKY_TOOL, {})
        state.last_intent = "risky_machines"
        state.last_result = result
        return True, result

    if is_count_query(q) and MACHINE_COUNT_TOOL in tools:
        result = await safe_call(gateway, MACHINE_COUNT_TOOL, {})
        state.last_intent = "machine_count"
        state.last_result = result
        return True, result

    machine_ref = extract_machine_reference(q)

    # Use previous machine for short follow-ups when appropriate.
    if machine_ref is None and state.last_machine_ref:
        if any(
            word in q
            for word in (
                "temperature",
                "vibration",
                "humidity",
                "power",
                "voltage",
                "frequency",
                "level",
                "parameter",
                "sensor",
                "health",
                "trend",
                "model",
            )
        ):
            machine_ref = state.last_machine_ref

    if machine_ref:
        state.last_machine_ref = machine_ref

    if (
        is_parameter_list_query(q)
        and machine_ref
        and MACHINE_PARAMETERS_TOOL in tools
    ):
        machine_name, parameter_call, context = await get_parameters(
            gateway,
            tools,
            machine_ref,
        )

        state.last_machine_name = machine_name
        state.last_intent = "parameters"
        state.last_result = {"workflow": context}
        return True, {"workflow": context}

    if (
        is_health_trend_query(q)
        and machine_ref
        and HEALTH_TREND_TOOL in tools
    ):
        machine_name, mapping_context = await resolve_machine_name(
            gateway,
            tools,
            machine_ref,
        )

        result = await safe_call(
            gateway,
            HEALTH_TREND_TOOL,
            {"machine_name": machine_name},
        )

        workflow = mapping_context + [result]
        state.last_intent = "health_trend"
        state.last_machine_name = machine_name
        state.last_result = {"workflow": workflow}
        return True, {"workflow": workflow}

    if is_health_status_query(q) and HEALTH_STATUS_TOOL in tools:
        result = await safe_call(gateway, HEALTH_STATUS_TOOL, {})
        state.last_intent = "health_status"
        state.last_result = result
        return True, result

    if (
        is_model_query(q)
        and machine_ref
        and MODEL_TOOL in tools
    ):
        # This API schema uses machine_id.
        result = await safe_call(
            gateway,
            MODEL_TOOL,
            {"machine_id": machine_ref},
        )

        state.last_intent = "model"
        state.last_result = result
        return True, result

    sensor = detect_sensor(q)

    # Sensor trend: parameter discovery -> exact param -> trend.
    if (
        sensor
        and machine_ref
        and is_trend_query(q)
        and MACHINE_PARAMETERS_TOOL in tools
        and DASHBOARD_TREND_TOOL in tools
    ):
        machine_name, parameter_call, context = await get_parameters(
            gateway,
            tools,
            machine_ref,
        )

        state.last_machine_name = machine_name
        state.last_sensor = sensor

        if not parameter_call.get("success"):
            state.last_result = {"workflow": context}
            return True, {"workflow": context}

        exact_param = find_parameter_name(
            parameter_call.get("result"),
            sensor,
        )

        if not exact_param:
            error = {
                "success": False,
                "stage": "parameter_resolution",
                "machine_name": machine_name,
                "requested_sensor": sensor,
                "error": (
                    f"No exact parameter containing '{sensor}' "
                    "was found in the machine parameter response."
                ),
            }
            context.append(error)
            state.last_result = {"workflow": context}
            return True, {"workflow": context}

        trend_result = await safe_call(
            gateway,
            DASHBOARD_TREND_TOOL,
            {
                "machine_name": machine_name,
                "param_name": exact_param,
                "window_size": DEFAULT_TREND_WINDOW_SIZE,
                "interval_minutes": DEFAULT_TREND_INTERVAL_MINUTES,
            },
        )

        context.append(
            {
                "success": True,
                "stage": "resolved_identifiers",
                "machine_ref": machine_ref,
                "machine_name": machine_name,
                "sensor": sensor,
                "param_name": exact_param,
            }
        )
        context.append(trend_result)

        state.last_intent = "sensor_trend"
        state.last_result = {"workflow": context}
        return True, {"workflow": context}

    # Current/latest sensor reading:
    # parameter discovery -> exact param -> feature-data endpoint.
    if (
        sensor
        and machine_ref
        and is_current_sensor_query(q)
        and MACHINE_PARAMETERS_TOOL in tools
    ):
        machine_name, parameter_call, context = await get_parameters(
            gateway,
            tools,
            machine_ref,
        )

        state.last_machine_name = machine_name
        state.last_sensor = sensor

        if not parameter_call.get("success"):
            state.last_result = {"workflow": context}
            return True, {"workflow": context}

        exact_param = find_parameter_name(
            parameter_call.get("result"),
            sensor,
        )

        if not exact_param:
            error = {
                "success": False,
                "stage": "parameter_resolution",
                "machine_name": machine_name,
                "requested_sensor": sensor,
                "error": (
                    f"No exact parameter containing '{sensor}' "
                    "was found in the machine parameter response."
                ),
            }
            context.append(error)
            state.last_result = {"workflow": context}
            return True, {"workflow": context}

        # Prefer the feature-data endpoint because it accepts
        # machine_name + param_name. If unavailable, use dashboard trend.
        if FEATURE_DATA_TOOL in tools:
            reading_result = await safe_call(
                gateway,
                FEATURE_DATA_TOOL,
                {
                    "machine_name": machine_name,
                    "param_name": exact_param,
                    "window_size": DEFAULT_READING_WINDOW_SIZE,
                },
            )
        elif DASHBOARD_TREND_TOOL in tools:
            reading_result = await safe_call(
                gateway,
                DASHBOARD_TREND_TOOL,
                {
                    "machine_name": machine_name,
                    "param_name": exact_param,
                    "window_size": DEFAULT_READING_WINDOW_SIZE,
                },
            )
        else:
            reading_result = {
                "success": False,
                "stage": "reading_tool_resolution",
                "error": (
                    "No read-only tool accepting machine_name and "
                    "param_name is available for the current reading."
                ),
            }

        context.append(
            {
                "success": True,
                "stage": "resolved_identifiers",
                "machine_ref": machine_ref,
                "machine_name": machine_name,
                "sensor": sensor,
                "param_name": exact_param,
            }
        )
        context.append(reading_result)

        state.last_intent = "current_sensor"
        state.last_result = {"workflow": context}
        return True, {"workflow": context}

    return False, None


async def build_retriever(
    settings: Any,
    read_only_tools: list[ToolDescriptor],
) -> HybridToolRetriever:
    provider = OpenAICompatibleEmbeddingProvider(
        base_url=settings.embedding_base_url or settings.llm_base_url,
        api_key=settings.embedding_api_key or settings.llm_api_key,
        model=settings.embedding_model,
    )

    retriever = HybridToolRetriever(
        embedding_provider=provider,
        lexical_weight=0.55,
        embedding_weight=0.30,
        domain_weight=0.15,
    )

    try:
        await retriever.rebuild(read_only_tools)
        return retriever

    except Exception as exc:
        if HashEmbeddingProvider is None:
            raise

        print(
            "\nOpenAI embedding setup failed; "
            "falling back to local hash embeddings:"
        )
        print(f"{type(exc).__name__}: {exc}")

        local_provider = HashEmbeddingProvider(
            dimensions=256,
        )

        retriever = HybridToolRetriever(
            embedding_provider=local_provider,
            lexical_weight=0.65,
            embedding_weight=0.20,
            domain_weight=0.15,
        )

        await retriever.rebuild(read_only_tools)
        return retriever


async def main() -> None:
    settings = get_settings()

    gateway = MCPGateway(
        servers={
            SERVER_NAME: MCP_URL,
        }
    )

    registry = ToolRegistry(
        gateway=gateway,
        policy_path=resolve_policy_path(settings),
    )

    print("Connecting to MCP server:", MCP_URL)

    tools = await registry.refresh()

    if not tools:
        print("No MCP tools were discovered.")
        return

    print(f"\nLoaded {len(tools)} MCP tools.")

    available_tools = tool_map(tools)

    # Informational chatbot fallback sees read-only tools only.
    # This prevents accidental create/remove/start/stop selections.
    read_only_tools = [
        tool
        for tool in tools
        if is_read_only_tool(tool)
    ]

    print(
        f"Read-only tools available to Q&A fallback: "
        f"{len(read_only_tools)}"
    )

    retriever = await build_retriever(
        settings,
        read_only_tools,
    )

    print("\nHybrid retrieval index created.")

    user = UserContext(
        user_id="local-cli-user",
        roles={"admin"},
    )

    llm = OpenAICompatibleChatModel(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout_seconds=60,
        max_retries=3,
    )

    history: list[ChatTurn] = []
    state = ConversationState()

    print(
        "LLM implementation:",
        type(llm).__name__,
    )

    try:
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
                print("CLI stopped.")
                break

            query = normalize_query(raw_query)

            state.last_query = query

            # ---------------------------------------------
            # 1. High-confidence deterministic routing.
            # ---------------------------------------------
            handled, result = await handle_known_query(
                query=query,
                gateway=gateway,
                tools=available_tools,
                state=state,
            )

            if handled:
                if isinstance(result, dict) and "direct_answer" in result:
                    answer = str(result["direct_answer"])
                else:
                    answer = await grounded_answer(
                        llm=llm,
                        settings=settings,
                        query=query,
                        history=history,
                        tool_result=result,
                    )

                print(
                    "\n========== FINAL ANSWER =========="
                )
                print(answer)
                continue

            # ---------------------------------------------
            # 2. Read-only retriever + LLM fallback.
            # ---------------------------------------------
            enriched_query = query

            if state.last_machine_ref:
                enriched_query += (
                    f"\nKnown machine from conversation: "
                    f"{state.last_machine_ref}"
                )

            if state.last_sensor:
                enriched_query += (
                    f"\nKnown sensor from conversation: "
                    f"{state.last_sensor}"
                )

            candidates = await retriever.retrieve(
                query=enriched_query,
                user=user,
                limit=12,
            )

            if not candidates:
                print(
                    "\nNo relevant authorized read-only tool was found."
                )
                continue

            candidate_tools = [
                candidate.tool
                for candidate in candidates
                if is_read_only_tool(candidate.tool)
            ][:8]

            if not candidate_tools:
                print(
                    "\nNo safe read-only tool was found."
                )
                continue
            if DEBUG_RETRIEVER:
               print(
                "\n========== RETRIEVER SHORTLIST =========="
                )

            for index, candidate in enumerate(
                candidates[:8],
                 start=1,
             ):
                print(
                    f"{index}. {candidate.tool.name} "
                    f"(score={candidate.score:.4f})"
             )


            try:
                decision = await llm.choose_tool(
                    query=enriched_query,
                    history=history,
                    tools=candidate_tools,
                )
            except Exception as exc:
                print(
                    "\nLLM tool selection failed:"
                )
                print(f"{type(exc).__name__}: {exc}")
                continue

            if not decision.tool_name:
                direct = (decision.direct_answer or "").strip()
                print(
                    "\n========== FINAL ANSWER =========="
                )
                print(
                    direct
                    or "I could not determine a safe Factigent tool for that request."
                )
                continue

            candidate_map: dict[str, ToolDescriptor] = {}

            for tool in candidate_tools:
                candidate_map[tool.name] = tool
                llm_name = getattr(tool, "llm_name", None)
                if llm_name:
                    candidate_map[llm_name] = tool

            selected_tool = candidate_map.get(
                decision.tool_name
            )

            if selected_tool is None:
                print(
                    "\nLLM selected a tool outside the safe shortlist:",
                    decision.tool_name,
                )
                continue

            arguments, missing = validate_arguments(
                selected_tool,
                decision.arguments or {},
            )

            if missing:
                print(
                    "\nThe selected tool requires additional fields:",
                    ", ".join(missing),
                )
                continue

            # Repair known live-backend requirements before execution.
            if selected_tool.name in {
                DASHBOARD_TREND_TOOL,
                FEATURE_DATA_TOOL,
            }:
                preferred_window = (
                    DEFAULT_TREND_WINDOW_SIZE
                    if selected_tool.name == DASHBOARD_TREND_TOOL
                    else DEFAULT_READING_WINDOW_SIZE
                )

                if arguments.get("window_size") not in {5, 15, 30}:
                    arguments["window_size"] = preferred_window

            if selected_tool.name == DASHBOARD_TREND_TOOL:
                arguments.setdefault(
                    "interval_minutes",
                    DEFAULT_TREND_INTERVAL_MINUTES,
                )

            result = await safe_call(
                gateway,
                selected_tool.name,
                arguments,
            )

            state.last_result = result
            state.last_intent = "fallback"

            answer = await grounded_answer(
                llm=llm,
                settings=settings,
                query=query,
                history=history,
                tool_result=result,
            )

            print(
                "\n========== FINAL ANSWER =========="
            )
            print(answer)

    finally:
        await llm.close()


if __name__ == "__main__":
    asyncio.run(main())