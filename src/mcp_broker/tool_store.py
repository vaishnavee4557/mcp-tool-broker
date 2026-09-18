from __future__ import annotations

import hashlib
import json
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


def _explicit_read_only_hint(
    tool: ToolDescriptor,
) -> bool | None:
    """
    Prefer explicit MCP/tool/policy read-only metadata.

    Important: POST does not automatically mean mutation.
    Read/report/search APIs may legitimately use POST.
    """

    for attr in (
        "is_read_only",
        "read_only",
    ):
        value = getattr(
            tool,
            attr,
            None,
        )

        if isinstance(value, bool):
            return value

    policy = getattr(
        tool,
        "policy",
        None,
    )

    if policy is not None:
        for attr in (
            "is_read_only",
            "read_only",
        ):
            value = getattr(
                policy,
                attr,
                None,
            )

            if isinstance(value, bool):
                return value

    annotations = getattr(
        tool,
        "annotations",
        None,
    )

    if annotations is not None:
        if hasattr(
            annotations,
            "model_dump",
        ):
            try:
                annotations = annotations.model_dump()
            except Exception:
                annotations = None

        if isinstance(
            annotations,
            dict,
        ):
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
    Metadata-first read-only classification.

    `_post` is intentionally NOT a mutating marker because some
    data/report APIs use POST only to carry a request body.
    """

    explicit = _explicit_read_only_hint(
        tool
    )

    if explicit is not None:
        return explicit

    name = str(
        getattr(
            tool,
            "name",
            "",
        )
        or ""
    ).casefold()

    return not any(
        marker in name
        for marker in _MUTATING_ACTION_MARKERS
    )


def _collect_schema_text(
    value: Any,
    parts: list[str],
    *,
    depth: int = 0,
) -> None:
    """
    Flatten generic JSON-schema/OpenAPI metadata into searchable text.
    """

    if depth > 12:
        return

    if isinstance(value, dict):
        for field in (
            "title",
            "description",
            "type",
            "format",
        ):
            item = value.get(field)

            if (
                isinstance(item, str)
                and item.strip()
            ):
                parts.append(item.strip())

        required = value.get("required")

        if isinstance(required, list):
            for item in required:
                if isinstance(item, str):
                    parts.append(item)

        enum_values = value.get("enum")

        if isinstance(enum_values, list):
            for item in enum_values:
                if isinstance(
                    item,
                    (str, int, float, bool),
                ):
                    parts.append(str(item))

        const_value = value.get("const")

        if isinstance(
            const_value,
            (str, int, float, bool),
        ):
            parts.append(str(const_value))

        properties = value.get("properties")

        if isinstance(properties, dict):
            for (
                property_name,
                property_schema,
            ) in properties.items():
                parts.append(
                    str(property_name)
                )

                _collect_schema_text(
                    property_schema,
                    parts,
                    depth=depth + 1,
                )

        items = value.get("items")

        if items is not None:
            _collect_schema_text(
                items,
                parts,
                depth=depth + 1,
            )

        additional_properties = value.get(
            "additionalProperties"
        )

        if isinstance(
            additional_properties,
            (dict, list),
        ):
            _collect_schema_text(
                additional_properties,
                parts,
                depth=depth + 1,
            )

        for key in (
            "allOf",
            "anyOf",
            "oneOf",
            "prefixItems",
        ):
            children = value.get(key)

            if isinstance(children, list):
                for child in children:
                    _collect_schema_text(
                        child,
                        parts,
                        depth=depth + 1,
                    )

    elif isinstance(value, list):
        for child in value:
            _collect_schema_text(
                child,
                parts,
                depth=depth + 1,
            )


def _append_policy_search_metadata(
    tool: ToolDescriptor,
    parts: list[str],
) -> None:
    """
    Add retrieval-relevant policy metadata only.
    """

    policy = getattr(
        tool,
        "policy",
        None,
    )

    if policy is None:
        return

    domain = str(
        getattr(
            policy,
            "domain",
            "",
        )
        or ""
    ).strip()

    if domain:
        parts.append(domain)

    keywords = getattr(
        policy,
        "keywords",
        [],
    )

    if isinstance(
        keywords,
        (list, tuple, set),
    ):
        parts.extend(
            str(item).strip()
            for item in keywords
            if str(item).strip()
        )


def build_searchable_text(
    tool: ToolDescriptor,
) -> str:
    """
    Build rich, generic text used for embedding + pgvector retrieval.

    No machine/zone/line/cell/API-specific routing rules are added.
    """

    parts: list[str] = []

    for label, attr in (
        ("Registry ID", "registry_id"),
        ("Tool name", "name"),
        ("LLM name", "llm_name"),
        ("Description", "description"),
    ):
        value = getattr(
            tool,
            attr,
            None,
        )

        text = str(
            value or ""
        ).strip()

        if text:
            parts.append(
                f"{label}: {text}"
            )

    registry_searchable_text = str(
        getattr(
            tool,
            "searchable_text",
            "",
        )
        or ""
    ).strip()

    if registry_searchable_text:
        parts.append(
            "Search metadata: "
            + registry_searchable_text
        )

    input_schema = getattr(
        tool,
        "input_schema",
        None,
    )

    if isinstance(input_schema, dict):
        schema_parts: list[str] = []

        _collect_schema_text(
            input_schema,
            schema_parts,
        )

        if schema_parts:
            parts.append(
                "Input schema metadata: "
                + " ".join(schema_parts)
            )

        parts.append(
            "Input schema JSON: "
            + json.dumps(
                input_schema,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        )

    for label, attr in (
        (
            "Output schema metadata",
            "output_schema",
        ),
        (
            "Response schema metadata",
            "response_schema",
        ),
    ):
        schema = getattr(
            tool,
            attr,
            None,
        )

        if not isinstance(schema, dict):
            continue

        schema_parts: list[str] = []

        _collect_schema_text(
            schema,
            schema_parts,
        )

        if schema_parts:
            parts.append(
                f"{label}: "
                + " ".join(schema_parts)
            )

    _append_policy_search_metadata(
        tool,
        parts,
    )

    parts.append(
        "Read only: "
        + (
            "true"
            if is_read_only_tool(tool)
            else "false"
        )
    )

    unique_parts = list(
        dict.fromkeys(
            part
            for part in parts
            if str(part).strip()
        )
    )

    return "\n".join(
        unique_parts
    )


def build_content_hash(
    searchable_text: str,
) -> str:
    return hashlib.sha256(
        searchable_text.encode(
            "utf-8"
        )
    ).hexdigest()


class ToolStore:

    def __init__(
        self,
        db: Database,
    ) -> None:
        self.db = db

    async def upsert_tools(
        self,
        *,
        server_name: str,
        tools: list[ToolDescriptor],
    ) -> int:
        """
        Store current MCP metadata.

        If searchable metadata changes, the old embedding is invalidated
        so embed_tools.py regenerates it. If metadata is unchanged,
        the existing embedding is preserved.
        """

        if not tools:
            return 0

        saved_count = 0

        async with (
            self.db.pool.connection()
            as conn
        ):
            async with conn.cursor() as cur:

                for tool in tools:

                    searchable_text = (
                        build_searchable_text(
                            tool
                        )
                    )

                    content_hash = (
                        build_content_hash(
                            searchable_text
                        )
                    )

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
                            description =
                                EXCLUDED.description,

                            input_schema =
                                EXCLUDED.input_schema,

                            searchable_text =
                                EXCLUDED.searchable_text,

                            is_read_only =
                                EXCLUDED.is_read_only,

                            embedding =
                                CASE
                                    WHEN
                                        mcp_tools.content_hash
                                        IS DISTINCT FROM
                                        EXCLUDED.content_hash
                                    THEN NULL
                                    ELSE mcp_tools.embedding
                                END,

                            content_hash =
                                EXCLUDED.content_hash,

                            updated_at =
                                NOW()
                        """,
                        (
                            str(server_name),
                            tool.name,
                            tool.description or "",
                            Jsonb(
                                tool.input_schema
                                or {}
                            ),
                            searchable_text,
                            content_hash,
                            is_read_only_tool(
                                tool
                            ),
                        ),
                    )

                    saved_count += 1

            await conn.commit()

        return saved_count

    async def count_tools(
        self,
    ) -> int:

        async with (
            self.db.pool.connection()
            as conn
        ):
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

    async def list_tools(
        self,
    ) -> list[tuple]:

        async with (
            self.db.pool.connection()
            as conn
        ):
            async with conn.cursor() as cur:

                await cur.execute(
                    """
                    SELECT
                        tool_name,
                        is_read_only,
                        embedding IS NOT NULL
                            AS has_embedding
                    FROM mcp_tools
                    ORDER BY tool_name
                    """
                )

                rows = await cur.fetchall()

                return list(rows)
