from __future__ import annotations

import hashlib
import re
from typing import Any

from psycopg.types.json import Jsonb

from mcp_broker.db import Database
from mcp_broker.models import ToolDescriptor


_MUTATING_ACTION_MARKERS = (
    "create_",
    "update_",
    "delete_",
    "remove_",
    "start_",
    "stop_",
    "submit_",
    "capture_",
    "assign_",
    "unassign_",
    "set_",
    "enable_",
    "disable_",
    "upload_",
    "install_",
    "uninstall_",
    "revoke_",
    "_put",
    "_patch",
    "_delete",
)


_OPERATION_VOCABULARY: dict[str, str] = {
    "count": "count number quantity how many total count",
    "list": "list show display available all",
    "names": "names list identify which",
    "name": "name identify",
    "get": "get fetch retrieve show find",
    "search": "search find lookup locate",
    "create": "create add register make",
    "add": "add create register",
    "update": "update change modify edit",
    "delete": "delete remove",
    "remove": "remove delete",
    "average": "average mean avg",
    "avg": "average mean avg",
    "total": "total sum aggregate",
    "trend": "trend history historical time series",
    "history": "history historical previous past",
    "alert": "alert alarm warning event",
    "alerts": "alerts alarms warnings events",
    "health": "health condition status risk",
    "status": "status state condition",
    "parameter": "parameter sensor telemetry tag metric",
    "parameters": "parameters sensors telemetry tags metrics",
}


_SUBJECT_VOCABULARY: dict[str, str] = {
    "machine": "machine equipment asset",
    "machines": "machines equipment assets",
    "parameter": "parameter sensor telemetry tag metric",
    "parameters": "parameters sensors telemetry tags metrics",
    "user": "user account person",
    "users": "users accounts people",
    "role": "role permission access",
    "roles": "roles permissions access",
    "connection": "connection connector integration",
    "connections": "connections connectors integrations",
    "dataflow": "dataflow pipeline workflow",
    "alert": "alert alarm warning",
    "alerts": "alerts alarms warnings",
}


_SUBJECT_STOP_WORDS = {
    "get",
    "post",
    "put",
    "patch",
    "delete",
    "create",
    "add",
    "update",
    "remove",
    "fetch",
    "retrieve",
    "show",
    "dashboard",
    "api",
    "endpoint",
    "count",
    "list",
    "names",
    "name",
    "average",
    "avg",
    "total",
    "trend",
    "history",
}


def _explicit_read_only_hint(
    tool: ToolDescriptor,
) -> bool | None:
    """Return an explicit read-only value when metadata provides one."""

    for attr in (
        "is_read_only",
        "read_only",
    ):
        value = getattr(tool, attr, None)
        if isinstance(value, bool):
            return value

    policy = getattr(tool, "policy", None)

    if policy is not None:
        for attr in (
            "is_read_only",
            "read_only",
        ):
            if isinstance(policy, dict):
                value = policy.get(attr)
            else:
                value = getattr(policy, attr, None)

            if isinstance(value, bool):
                return value

    annotations = getattr(tool, "annotations", None)

    if annotations is not None and hasattr(annotations, "model_dump"):
        try:
            annotations = annotations.model_dump()
        except Exception:
            annotations = None

    if isinstance(annotations, dict):
        for key in (
            "readOnlyHint",
            "read_only",
            "is_read_only",
        ):
            value = annotations.get(key)
            if isinstance(value, bool):
                return value

    return None


def is_read_only_tool(
    tool: ToolDescriptor,
) -> bool:
    """
    Classify a tool as read-only using metadata first, then its name.

    ``_post`` is intentionally not considered mutating because reporting and
    search APIs often use POST only to carry a request body.
    """

    explicit = _explicit_read_only_hint(tool)

    if explicit is not None:
        return explicit

    name = str(getattr(tool, "name", "") or "").casefold()

    return not any(marker in name for marker in _MUTATING_ACTION_MARKERS)


def _policy_value(
    policy: object,
    name: str,
    default: Any,
) -> Any:
    if isinstance(policy, dict):
        return policy.get(name, default)

    return getattr(policy, name, default)


def _append_policy_search_metadata(
    tool: ToolDescriptor,
    parts: list[str],
) -> None:
    """Add retrieval-relevant domain and keyword metadata."""

    policy = getattr(tool, "policy", None)

    if policy is None:
        return

    domain = str(_policy_value(policy, "domain", "") or "").strip()

    if domain:
        parts.append(domain)

    keywords = _policy_value(policy, "keywords", [])

    if isinstance(keywords, (list, tuple, set)):
        parts.extend(
            str(item).strip()
            for item in keywords
            if str(item).strip()
        )


def _humanize_identifier(value: object) -> str:
    """Convert snake_case, paths and camelCase identifiers to plain words."""

    text = str(value or "").strip()

    if not text:
        return ""

    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
    return " ".join(text.split()).strip()


def _unique_words(chunks: list[str]) -> str:
    words: list[str] = []
    seen: set[str] = set()

    for chunk in chunks:
        for word in str(chunk).split():
            key = word.casefold()
            if key and key not in seen:
                seen.add(key)
                words.append(word)

    return " ".join(words)


def _build_intent_vocabulary(
    tool_name: str,
    description: str,
) -> str:
    combined = _humanize_identifier(
        f"{tool_name} {description}"
    ).lower()
    words = set(combined.split())
    vocabulary: list[str] = []

    for word, synonyms in _OPERATION_VOCABULARY.items():
        if word in words:
            vocabulary.append(synonyms)

    for word, synonyms in _SUBJECT_VOCABULARY.items():
        if word in words:
            vocabulary.append(synonyms)

    return _unique_words(vocabulary)


def _pluralize_subject(subject: str) -> str:
    words = subject.split()

    if not words:
        return subject

    last = words[-1]

    if not last.endswith("s"):
        if last.endswith("y") and len(last) > 1:
            last = last[:-1] + "ies"
        else:
            last += "s"

    words[-1] = last
    return " ".join(words)


def _build_question_examples(
    tool_name: str,
    description: str,
) -> str:
    readable_name = _humanize_identifier(description or tool_name).lower()
    words = readable_name.split()
    word_set = set(words)

    subject_words = [
        word
        for word in words
        if word not in _SUBJECT_STOP_WORDS
    ]
    subject = " ".join(subject_words).strip() or readable_name

    if "count" in word_set:
        plural_subject = _pluralize_subject(subject)
        return (
            f"How many {plural_subject} are there? "
            f"What is the number of {plural_subject}? "
            f"Give the total count of {plural_subject}."
        )

    if {"list", "names", "name"} & word_set:
        return (
            f"List {subject}. "
            f"Show all {subject}. "
            f"Which {subject} are available?"
        )

    if {"average", "avg"} & word_set:
        return (
            f"What is the average {subject}? "
            f"Calculate the mean {subject}."
        )

    if "trend" in word_set:
        return (
            f"Show the {subject} trend. "
            f"Give historical {subject} data."
        )

    if {"alert", "alerts"} & word_set:
        return (
            f"Show {subject} alerts. "
            f"Are there any {subject} warnings?"
        )

    if "create" in word_set or "add" in word_set:
        return f"Create {subject}. Add a new {subject}."

    if "update" in word_set:
        return f"Update {subject}. Change the existing {subject}."

    if "delete" in word_set or "remove" in word_set:
        return f"Delete {subject}. Remove the existing {subject}."

    return (
        f"Get {readable_name}. "
        f"Show {readable_name}. "
        f"Find information about {readable_name}."
    )


def _build_input_parameter_text(input_schema: object) -> str:
    """Build a small readable summary without embedding the complete JSON."""

    if not isinstance(input_schema, dict):
        return ""

    properties = input_schema.get("properties", {})

    if not isinstance(properties, dict):
        return ""

    required = input_schema.get("required", [])
    required_names = (
        {str(item) for item in required}
        if isinstance(required, list)
        else set()
    )
    parameters: list[str] = []

    for name, schema in list(properties.items())[:30]:
        readable_name = _humanize_identifier(name)

        if not isinstance(schema, dict):
            parameters.append(readable_name)
            continue

        value_type = str(schema.get("type", "") or "").strip()

        if value_type == "array":
            items = schema.get("items", {})
            if isinstance(items, dict):
                item_type = str(items.get("type", "") or "").strip()
                if item_type:
                    value_type = f"array of {item_type}"

        parameter_description = str(
            schema.get("description", "") or ""
        ).strip()
        pieces = [readable_name]

        if value_type:
            pieces.append(f"type {value_type}")

        pieces.append("required" if name in required_names else "optional")

        if parameter_description:
            pieces.append(" ".join(parameter_description.split())[:160])

        parameters.append(", ".join(pieces))

    return "; ".join(parameters)


def _build_output_schema_text(tool: ToolDescriptor) -> str:
    """Return only useful top-level response fields, avoiding schema noise."""

    summaries: list[str] = []

    for attr in ("output_schema", "response_schema"):
        schema = getattr(tool, attr, None)

        if not isinstance(schema, dict):
            continue

        description = str(schema.get("description", "") or "").strip()
        if description:
            summaries.append(" ".join(description.split())[:240])

        properties = schema.get("properties", {})
        if isinstance(properties, dict) and properties:
            fields = [
                _humanize_identifier(name)
                for name in list(properties)[:25]
            ]
            summaries.append("response fields " + ", ".join(fields))

    return "; ".join(dict.fromkeys(summaries))


def build_searchable_text(tool: ToolDescriptor) -> str:
    """
    Build concise semantic text for embedding and pgvector retrieval.

    The complete execution schema remains stored separately in ``input_schema``.
    This text focuses on purpose, intent, examples and meaningful parameters.
    """

    tool_name = str(getattr(tool, "name", "") or "").strip()
    description = str(getattr(tool, "description", "") or "").strip()
    readable_name = _humanize_identifier(tool_name)
    purpose = " ".join((description or readable_name or "Undescribed tool").split())

    parts: list[str] = [f"Purpose: {purpose}."]

    if readable_name:
        parts.append(f"Technical operation: {readable_name}.")

    intent_vocabulary = _build_intent_vocabulary(tool_name, description)
    if intent_vocabulary:
        parts.append(f"Intent vocabulary: {intent_vocabulary}.")

    question_examples = _build_question_examples(tool_name, description)
    if question_examples:
        parts.append(f"Typical user questions: {question_examples}")

    input_text = _build_input_parameter_text(
        getattr(tool, "input_schema", None)
    )
    parts.append(
        f"Supported input parameters: {input_text}."
        if input_text
        else "Supported input parameters: none."
    )

    output_text = _build_output_schema_text(tool)
    if output_text:
        parts.append(f"Response information: {output_text}.")

    policy_parts: list[str] = []
    _append_policy_search_metadata(tool, policy_parts)
    clean_policy = _unique_words(policy_parts)

    if clean_policy:
        parts.append(f"Domain and policy terms: {clean_policy}.")

    unique_parts: list[str] = []
    seen: set[str] = set()

    for part in parts:
        normalized = " ".join(str(part).split()).strip()
        key = normalized.casefold()

        if normalized and key not in seen:
            seen.add(key)
            unique_parts.append(normalized)

    return "\n".join(unique_parts)


def build_content_hash(searchable_text: str) -> str:
    return hashlib.sha256(searchable_text.encode("utf-8")).hexdigest()


class ToolStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def upsert_tools(
        self,
        *,
        server_name: str,
        tools: list[ToolDescriptor],
    ) -> int:
        """
        Store current MCP metadata.

        A searchable-text change invalidates the previous embedding so that
        ``embed_tools.py`` regenerates it. Unchanged embeddings are preserved.
        """

        if not tools:
            return 0

        saved_count = 0

        async with self.db.pool.connection() as conn:
            async with conn.cursor() as cur:
                for tool in tools:
                    searchable_text = build_searchable_text(tool)
                    content_hash = build_content_hash(searchable_text)

                    await cur.execute(
                        """
                        INSERT INTO mcp_tools (
                            server_name,
                            tool_name,
                            description,
                            input_schema,
                            searchable_text,
                            content_hash,
                            is_read_only,
                            updated_at
                        )
                        VALUES (
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            NOW()
                        )
                        ON CONFLICT (
                            server_name,
                            tool_name
                        )
                        DO UPDATE SET
                            description = EXCLUDED.description,
                            input_schema = EXCLUDED.input_schema,
                            searchable_text = EXCLUDED.searchable_text,
                            is_read_only = EXCLUDED.is_read_only,
                            embedding = CASE
                                WHEN mcp_tools.content_hash IS DISTINCT FROM
                                     EXCLUDED.content_hash
                                THEN NULL
                                ELSE mcp_tools.embedding
                            END,
                            content_hash = EXCLUDED.content_hash,
                            updated_at = NOW()
                        """,
                        (
                            str(server_name),
                            tool.name,
                            tool.description or "",
                            Jsonb(tool.input_schema or {}),
                            searchable_text,
                            content_hash,
                            is_read_only_tool(tool),
                        ),
                    )
                    saved_count += 1

            await conn.commit()

        return saved_count

    async def count_tools(self) -> int:
        async with self.db.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM mcp_tools
                    """
                )
                row = await cur.fetchone()

                if row is None:
                    return 0

                return int(row[0])

    async def list_tools(self) -> list[tuple]:
        async with self.db.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT
                        tool_name,
                        is_read_only,
                        embedding IS NOT NULL AS has_embedding
                    FROM mcp_tools
                    ORDER BY tool_name
                    """
                )
                rows = await cur.fetchall()
                return list(rows)
