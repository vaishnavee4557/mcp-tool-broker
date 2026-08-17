from __future__ import annotations

import re
from dataclasses import dataclass

from mcp_broker.models import ToolDescriptor


@dataclass(frozen=True)
class RetrievedTool:
    tool: ToolDescriptor
    score: float


class LocalToolRetriever:
    """
    Simple local retriever.

    It ranks MCP tools using their name, description,
    domain, and input-schema property names.
    """

    _synonyms: dict[str, set[str]] = {
        "temperature": {
            "temperature",
            "temp",
            "sensor",
            "reading",
            "readings",
            "telemetry",
        },
        "anomaly": {
            "anomaly",
            "anomalies",
            "abnormal",
            "alert",
            "alerts",
        },
        "machine": {
            "machine",
            "machines",
            "equipment",
            "device",
        },
        "parameter": {
            "parameter",
            "parameters",
            "sensor",
            "sensors",
        },
        "status": {
            "status",
            "health",
            "working",
            "condition",
            "system",
        },
        "latest": {
            "latest",
            "recent",
            "current",
            "newest",
        },
    }

    def search(
        self,
        query: str,
        tools: list[ToolDescriptor],
        top_k: int = 3,
    ) -> list[RetrievedTool]:
        query_tokens = self._expand_tokens(
            self._tokenize(query)
        )

        ranked_tools: list[RetrievedTool] = []

        for tool in tools:
            searchable_text = self._tool_text(tool)
            tool_tokens = self._tokenize(searchable_text)

            overlap = query_tokens.intersection(tool_tokens)

            score = float(len(overlap))

            normalized_query = query.lower()
            normalized_name = tool.name.lower()

            # Extra score when words occur directly in the tool name.
            for token in query_tokens:
                if token in normalized_name:
                    score += 2.0

            # Extra score for exact phrases.
            if normalized_name in normalized_query:
                score += 5.0

            # Small score for schema compatibility.
            properties = tool.input_schema.get(
                "properties",
                {},
            )

            if (
                "machine" in query_tokens
                and (
                    "machine_id" in properties
                    or "machineId" in properties
                )
            ):
                score += 1.5

            if (
                {"latest", "recent"}.intersection(query_tokens)
                and "limit" in properties
            ):
                score += 1.0

            if score > 0:
                ranked_tools.append(
                    RetrievedTool(
                        tool=tool,
                        score=score,
                    )
                )

        ranked_tools.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return ranked_tools[:top_k]

    def _tool_text(
        self,
        tool: ToolDescriptor,
    ) -> str:
        properties = tool.input_schema.get(
            "properties",
            {},
        )

        policy = getattr(tool, "policy", None)
        domain = getattr(policy, "domain", "") or ""

        values = [
            tool.name,
            getattr(tool, "llm_name", ""),
            tool.description or "",
            domain,
            " ".join(properties.keys()),
        ]

        return " ".join(
            str(value)
            for value in values
            if value
        )

    def _expand_tokens(
        self,
        tokens: set[str],
    ) -> set[str]:
        expanded = set(tokens)

        for token in list(tokens):
            for root_word, related_words in self._synonyms.items():
                if token == root_word or token in related_words:
                    expanded.add(root_word)
                    expanded.update(related_words)

        return expanded

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {
            token.lower()
            for token in re.findall(
                r"[A-Za-z0-9_]+",
                text,
            )
            if len(token) > 1
        }