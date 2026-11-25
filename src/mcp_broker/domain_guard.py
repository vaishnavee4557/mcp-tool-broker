
from __future__ import annotations

from dataclasses import dataclass

from mcp_broker.models import ToolDescriptor


@dataclass(frozen=True)
class DomainGuardResult:
    allowed: bool
    reason: str | None = None


def build_domain_guard_prompt(
    *,
    query: str,
    tools: list[ToolDescriptor],
) -> str:
    """
    Build a strict domain-classification prompt from the live
    authorized MCP/API tool catalog.

    The model must decide whether at least one supplied API/tool
    can meaningfully answer the user's question.
    """

    tool_lines: list[str] = []

    for tool in tools:
        schema = tool.input_schema or {}

        tool_lines.append(
            "\n".join(
                [
                    f"Tool: {tool.llm_name}",
                    f"Description: {tool.description or ''}",
                    f"Input schema: {schema}",
                ]
            )
        )

    tools_text = "\n\n".join(
        tool_lines
    )

    return f"""
You are a domain gate for an API-only enterprise chatbot.

The chatbot is allowed to answer ONLY questions that can be
answered using one or more of the supplied MCP/API tools.

USER QUESTION:
{query}

AVAILABLE AUTHORIZED TOOLS:
{tools_text}

Rules:

1. Return YES only when at least one supplied tool/API can
   meaningfully answer the user's request.

2. Return NO for general knowledge, unrelated questions,
   casual questions, weather, geography, politics, entertainment,
   or anything that is not supported by these tools.

3. Do not approve a question merely because it contains a word
   that also appears in an API.

4. Consider the meaning of the question and the capabilities of
   the tools.

5. Do not invent API capabilities.

6. Machine names, locations, zones, lines, sites, parameters,
   IDs, and other values may come from the user's question.
   Their presence alone does not automatically make the question valid.

Return exactly one word:

YES

or

NO
""".strip()
