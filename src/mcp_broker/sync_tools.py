from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from mcp_broker.async_runtime import run_async
from mcp_broker.db import Database
from mcp_broker.mcp_gateway import MCPGateway
from mcp_broker.models import ToolDescriptor
from mcp_broker.registry import ToolRegistry
from mcp_broker.settings import get_settings
from mcp_broker.tool_store import ToolStore


DEFAULT_SERVER_NAME = "factigent_api"
DEFAULT_MCP_URL = "http://127.0.0.1:8001/mcp"


# ============================================================
# SETTINGS HELPERS
# ============================================================

def _read_value(
    value: Any,
) -> Any:
    if value is None:
        return None

    if hasattr(
        value,
        "get_secret_value",
    ):
        return value.get_secret_value()

    return value


def resolve_policy_path(
    settings: Any,
) -> Path:
    """
    Resolve config/tool_policies.json from the project root when the
    configured path is relative.
    """

    configured_path = _read_value(
        getattr(
            settings,
            "tool_policies_path",
            None,
        )
    )

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    if configured_path:
        path = Path(
            str(configured_path)
        )

        if not path.is_absolute():
            path = (
                project_root
                / path
            )

    else:
        path = (
            project_root
            / "config"
            / "tool_policies.json"
        )

    return path.resolve()


def resolve_mcp_servers(
    settings: Any,
) -> dict[str, str]:
    """
    Resolve configured MCP servers generically.

    Supported configuration:
        MCP_SERVERS_JSON={
          "factigent_api":"http://127.0.0.1:8001/mcp",
          "factigent_platform":"http://127.0.0.1:8002/mcp"
        }

    Falls back to MCP_SERVER_NAME + MCP_SERVER_URL for a single server.

    This means adding another MCP server does not require changing this file.
    """

    raw_servers = _read_value(
        getattr(
            settings,
            "mcp_servers_json",
            None,
        )
    )

    parsed: Any = None

    if isinstance(
        raw_servers,
        dict,
    ):
        parsed = raw_servers

    elif isinstance(
        raw_servers,
        str,
    ) and raw_servers.strip():
        try:
            parsed = json.loads(
                raw_servers
            )
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "MCP_SERVERS_JSON is not valid JSON."
            ) from exc

    if isinstance(
        parsed,
        dict,
    ):
        servers = {
            str(name).strip():
                str(url).strip()
            for name, url in parsed.items()
            if (
                str(name).strip()
                and str(url).strip()
            )
        }

        if servers:
            return servers

    server_name = str(
        _read_value(
            getattr(
                settings,
                "mcp_server_name",
                None,
            )
        )
        or DEFAULT_SERVER_NAME
    ).strip()

    server_url = str(
        _read_value(
            getattr(
                settings,
                "mcp_server_url",
                None,
            )
        )
        or _read_value(
            getattr(
                settings,
                "mcp_url",
                None,
            )
        )
        or DEFAULT_MCP_URL
    ).strip()

    if not server_name:
        raise RuntimeError(
            "MCP server name is empty."
        )

    if not server_url:
        raise RuntimeError(
            "MCP server URL is empty."
        )

    return {
        server_name:
            server_url,
    }


# ============================================================
# TOOL GROUPING
# ============================================================

def group_tools_by_server(
    tools: list[ToolDescriptor],
    configured_servers: dict[str, str],
) -> dict[
    str,
    list[ToolDescriptor],
]:
    """
    Keep each discovered tool under its real MCP server.

    This is important once multiple MCP servers are configured. We must not
    save tools from different servers under one hardcoded server_name.
    """

    grouped: dict[
        str,
        list[ToolDescriptor],
    ] = defaultdict(list)

    single_server_name = (
        next(
            iter(
                configured_servers
            )
        )
        if len(
            configured_servers
        ) == 1
        else None
    )

    for tool in tools:
        server_name = str(
            getattr(
                tool,
                "server_name",
                "",
            )
            or ""
        ).strip()

        if not server_name:
            if single_server_name is None:
                raise RuntimeError(
                    "A discovered tool is missing server_name while "
                    "multiple MCP servers are configured. "
                    f"Tool={tool.name!r}"
                )

            server_name = (
                single_server_name
            )

        if (
            server_name
            not in configured_servers
        ):
            raise RuntimeError(
                "Tool was discovered for an unknown MCP server: "
                f"tool={tool.name!r}, server={server_name!r}"
            )

        grouped[
            server_name
        ].append(
            tool
        )

    return dict(
        grouped
    )


# ============================================================
# MAIN SYNC
# ============================================================

async def main() -> None:
    settings = get_settings()

    servers = resolve_mcp_servers(
        settings
    )

    policy_path = resolve_policy_path(
        settings
    )

    db = Database()
    db_opened = False

    try:
        await db.open()
        db_opened = True

        print(
            "Database connected."
        )

        print(
            "\nConfigured MCP servers:"
        )

        for (
            server_name,
            server_url,
        ) in servers.items():
            print(
                f"  {server_name}: {server_url}"
            )

        print(
            "Tool policy file:",
            policy_path,
        )

        gateway = MCPGateway(
            servers=servers
        )

        registry = ToolRegistry(
            gateway=gateway,
            policy_path=policy_path,
        )

        # ----------------------------------------------------
        # 1. DISCOVER THE LIVE MCP CATALOG
        # ----------------------------------------------------

        tools = await registry.refresh()

        if not tools:
            raise RuntimeError(
                "No MCP tools were discovered. "
                "Tool metadata was not changed."
            )

        print(
            "\nDiscovered MCP tools:",
            len(tools),
        )

        # ----------------------------------------------------
        # 2. GROUP BY REAL MCP SERVER
        # ----------------------------------------------------

        grouped_tools = (
            group_tools_by_server(
                tools,
                servers,
            )
        )

        # ----------------------------------------------------
        # 3. SAVE RICH METADATA INTO POSTGRESQL
        # ----------------------------------------------------

        store = ToolStore(
            db
        )

        saved_total = 0

        for server_name in servers:
            server_tools = (
                grouped_tools.get(
                    server_name,
                    [],
                )
            )

            if not server_tools:
                print(
                    f"\nWARNING: no tools discovered "
                    f"from {server_name!r}."
                )
                continue

            saved = await store.upsert_tools(
                server_name=server_name,
                tools=server_tools,
            )

            saved_total += saved

            print(
                f"{server_name}: "
                f"discovered={len(server_tools)}, "
                f"saved/updated={saved}"
            )

        total_in_db = (
            await store.count_tools()
        )

        # Metadata changes invalidate old embeddings in tool_store.py.
        tool_rows = (
            await store.list_tools()
        )

        embedding_ready = sum(
            1
            for row in tool_rows
            if (
                len(row) >= 3
                and bool(
                    row[2]
                )
            )
        )

        embedding_missing = (
            len(tool_rows)
            - embedding_ready
        )

        print()
        print(
            "========== TOOL SYNC =========="
        )
        print(
            "Tools discovered:",
            len(tools),
        )
        print(
            "Tools saved/updated:",
            saved_total,
        )
        print(
            "Tools currently in DB:",
            total_in_db,
        )
        print(
            "Tools with embeddings:",
            embedding_ready,
        )
        print(
            "Tools needing embeddings:",
            embedding_missing,
        )

        print()
        print(
            "TOOL METADATA SYNC OK"
        )

        if embedding_missing:
            print(
                "Run embed_tools.py next to generate/"
                "refresh missing embeddings."
            )

    finally:
        if db_opened:
            await db.close()


if __name__ == "__main__":
    run_async(
        main()
    )
