from __future__ import annotations

import re
from typing import Any


# ============================================================
# 1. WORDS THAT ARE NOT VERY USEFUL FOR TOOL RETRIEVAL
# ============================================================

IGNORED_TOOL_WORDS = {
    "api",
    "get",
    "post",
    "put",
    "patch",
    "delete",
    "endpoint",
    "request",
    "response",
}


# ============================================================
# 2. MANUAL KNOWLEDGE OVERRIDES
#
# We only need to manually define tools whose OpenAPI names or
# descriptions are unclear.
#
# All other MCP tools will still get automatically generated
# knowledge.
# ============================================================

TOOL_KNOWLEDGE_OVERRIDES: dict[str, dict[str, Any]] = {

    # --------------------------------------------------------
    # MACHINE TREND TOOL
    # --------------------------------------------------------

    "trend_dashboard_trend": {

        "friendly_name": "get_machine_parameter_trend",

        "purpose": (
            "Get historical telemetry trend data for a specific "
            "machine parameter. Use this tool when the user asks "
            "about sensor history, historical readings, trends, "
            "temperature trend, vibration trend, power trend, "
            "humidity trend, voltage trend, frequency trend, "
            "level trend, or telemetry values over time."
        ),

        "keywords": [
            "trend",
            "history",
            "historical",
            "telemetry",
            "sensor",
            "readings",
            "over time",
            "temperature",
            "vibration",
            "power",
            "humidity",
            "voltage",
            "frequency",
            "level",
            "past values",
            "previous values",
            "last hour",
            "last 24 hours",
        ],

        "examples": [
            "Show temperature trend for MAC1",
            "Show vibration history for MAC1",
            "What was the power trend for MAC1?",
            "Show sensor readings for the last 24 hours",
            "How did temperature change over time?",
            "Show humidity trend for machine MAC1",
        ],
    },"machine_parameters_api_pipeline_machine_parameters_get": {
    "friendly_name": "get_machine_parameters",

    "purpose": (
        "List exact paramNames available for a machine. "
        "Use this before trend/history APIs when the user provides "
        "only a generic sensor name such as temperature or vibration."
    ),

    "keywords": [
        "parameter",
        "parameters",
        "paramName",
        "sensors",
        "machine parameters",
        "temperature parameter",
        "vibration parameter",
    ],

    "examples": [
        "What parameters does MAC1 have?",
        "Find the temperature parameter for MAC1",
        "List sensors for MAC1",
    ],
},

}
# ============================================================
# 3. CLEAN GENERATED MCP TOOL NAMES
#
# Example:
#
# machine_parameters_api_pipeline_machine_parameters_get
#
# becomes roughly:
#
# machine parameters pipeline machine parameters
# ============================================================

def clean_tool_name(name: str) -> str:
    """
    Convert a generated MCP/OpenAPI tool name into more readable text.

    Example:
        machine_parameters_api_pipeline_machine_parameters_get

    becomes:
        machine parameters pipeline machine parameters
    """

    if not name:
        return ""

    # Replace common separators with spaces.
    cleaned_name = re.sub(
        r"[_\-/]+",
        " ",
        name,
    )

    words = cleaned_name.split()

    useful_words: list[str] = []

    for word in words:
        normalized = word.strip().lower()

        if not normalized:
            continue

        if normalized in IGNORED_TOOL_WORDS:
            continue

        useful_words.append(normalized)

    return " ".join(useful_words)


# ============================================================
# 4. EXTRACT AUTOMATIC KEYWORDS
#
# For tools that do NOT have manual overrides, we extract useful
# words from their name and OpenAPI description.
# ============================================================

def extract_keywords(
    tool_name: str,
    description: str,
) -> list[str]:
    """
    Generate basic keywords automatically from tool name
    and description.
    """

    combined_text = f"{tool_name} {description}".lower()

    # Extract normal words.
    words = re.findall(
        r"[a-zA-Z0-9]+",
        combined_text,
    )

    keywords: list[str] = []

    for word in words:
        word = word.strip().lower()

        if len(word) < 3:
            continue

        if word in IGNORED_TOOL_WORDS:
            continue

        if word not in keywords:
            keywords.append(word)

    # Avoid creating excessively large retrieval text.
    return keywords[:40]


# ============================================================
# 5. SAFELY READ TOOL ATTRIBUTES
#
# Different MCP / FastMCP objects may expose schemas under
# slightly different attribute names.
# ============================================================

def get_tool_name(tool: Any) -> str:
    """
    Safely get tool name.
    """

    return str(
        getattr(
            tool,
            "name",
            "",
        )
        or ""
    )


def get_tool_description(tool: Any) -> str:
    """
    Safely get tool description.
    """

    return str(
        getattr(
            tool,
            "description",
            "",
        )
        or ""
    )


def get_tool_input_schema(tool: Any) -> dict[str, Any]:
    """
    Safely get tool input schema.

    Supports common attribute styles:
        input_schema
        inputSchema
    """

    schema = getattr(
        tool,
        "input_schema",
        None,
    )

    if schema is None:
        schema = getattr(
            tool,
            "inputSchema",
            None,
        )

    if isinstance(schema, dict):
        return schema

    return {}


# ============================================================
# 6. CREATE KNOWLEDGE FOR ONE MCP TOOL
#
# This is the main function.
#
# Every tool gets:
#
# - actual MCP tool name
# - friendly name
# - purpose
# - original description
# - keywords
# - examples
# - input schema
#
# If manual knowledge exists, it overrides automatically generated
# knowledge.
# ============================================================

def create_tool_knowledge(
    tool: Any,
) -> dict[str, Any]:
    """
    Create retrieval knowledge for one MCP tool.

    Manual overrides are used when available.

    Otherwise the knowledge is generated automatically from
    the MCP/OpenAPI tool metadata.
    """

    tool_name = get_tool_name(tool)

    description = get_tool_description(tool)

    input_schema = get_tool_input_schema(tool)

    # Check if we have manually written better knowledge.
    override = TOOL_KNOWLEDGE_OVERRIDES.get(
        tool_name,
        {},
    )

    # --------------------------------------------------------
    # Friendly name
    # --------------------------------------------------------

    friendly_name = override.get(
        "friendly_name",
    )

    if not friendly_name:
        friendly_name = clean_tool_name(
            tool_name
        )

    # --------------------------------------------------------
    # Purpose
    # --------------------------------------------------------

    purpose = override.get(
        "purpose",
    )

    if not purpose:
        purpose = description

    if not purpose:
        purpose = (
            f"Use the {friendly_name or tool_name} tool "
            f"to perform the operation represented by this MCP tool."
        )

    # --------------------------------------------------------
    # Keywords
    # --------------------------------------------------------

    keywords = override.get(
        "keywords",
    )

    if not keywords:
        keywords = extract_keywords(
            tool_name=tool_name,
            description=description,
        )

    # --------------------------------------------------------
    # Examples
    # --------------------------------------------------------

    examples = override.get(
        "examples",
        [],
    )

    return {
        "tool_name": tool_name,
        "friendly_name": friendly_name,
        "purpose": purpose,
        "description": description,
        "keywords": keywords,
        "examples": examples,
        "input_schema": input_schema,
    }


# ============================================================
# 7. CREATE KNOWLEDGE FOR ALL MCP TOOLS
#
# If MCP discovers 33 tools, this creates knowledge for all 33.
#
# If tomorrow Swagger has 40 tools, it automatically creates
# knowledge for all 40.
# ============================================================

def create_all_tool_knowledge(
    tools: list[Any],
) -> dict[str, dict[str, Any]]:
    """
    Build a knowledge dictionary for every discovered MCP tool.

    Output format:

    {
        "tool_name_1": {...},
        "tool_name_2": {...},
        ...
    }
    """

    all_knowledge: dict[str, dict[str, Any]] = {}

    for tool in tools:

        knowledge = create_tool_knowledge(
            tool
        )

        tool_name = knowledge[
            "tool_name"
        ]

        if not tool_name:
            continue

        all_knowledge[
            tool_name
        ] = knowledge

    return all_knowledge


# ============================================================
# 8. BUILD TEXT THAT WILL LATER BE USED BY THE RETRIEVER
#
# We are creating this now so Step 2 will be easy.
#
# This DOES NOT yet modify retrieval.py.
# ============================================================

def build_tool_knowledge_text(
    tool: Any,
) -> str:
    """
    Convert tool knowledge into rich text suitable for
    BM25 and embedding retrieval.
    """

    knowledge = create_tool_knowledge(
        tool
    )

    keywords = knowledge.get(
        "keywords",
        [],
    )

    examples = knowledge.get(
        "examples",
        [],
    )

    input_schema = knowledge.get(
        "input_schema",
        {},
    )

    keyword_text = ", ".join(
        str(keyword)
        for keyword in keywords
    )

    example_text = "\n".join(
        f"- {example}"
        for example in examples
    )

    return f"""
Actual MCP Tool Name:
{knowledge["tool_name"]}

Friendly Tool Name:
{knowledge["friendly_name"]}

Purpose:
{knowledge["purpose"]}

Original OpenAPI Description:
{knowledge["description"]}

Related Keywords:
{keyword_text}

Example User Questions:
{example_text}

Input Schema:
{input_schema}
""".strip()


# ============================================================
# 9. OPTIONAL DEBUG FUNCTION
#
# Useful while developing.
# ============================================================

def print_tool_knowledge(
    tool: Any,
) -> None:
    """
    Print tool knowledge in a readable format for debugging.
    """

    knowledge = create_tool_knowledge(
        tool
    )

    print(
        "\n========== TOOL KNOWLEDGE =========="
    )

    print(
        "Tool:",
        knowledge["tool_name"],
    )

    print(
        "Friendly name:",
        knowledge["friendly_name"],
    )

    print(
        "Purpose:",
        knowledge["purpose"],
    )

    print(
        "Keywords:",
        knowledge["keywords"],
    )

    print(
        "Examples:",
        knowledge["examples"],
    )

    print(
        "Input schema:",
        knowledge["input_schema"],
    )