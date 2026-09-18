from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

# Keep the database configuration in the root project .env.
load_dotenv(
    dotenv_path=ENV_FILE,
    override=False,
)

DEFAULT_POOL_MIN_SIZE = 1
DEFAULT_POOL_MAX_SIZE = 5
DEFAULT_POOL_TIMEOUT_SECONDS = 30.0
EXPECTED_VECTOR_DIMENSIONS = 256


def _env_int(
    name: str,
    default: int,
) -> int:
    raw = str(
        os.getenv(
            name,
            str(default),
        )
        or ""
    ).strip()

    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"{name} must be an integer."
        ) from exc

    if value <= 0:
        raise RuntimeError(
            f"{name} must be greater than zero."
        )

    return value


def _env_float(
    name: str,
    default: float,
) -> float:
    raw = str(
        os.getenv(
            name,
            str(default),
        )
        or ""
    ).strip()

    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"{name} must be numeric."
        ) from exc

    if value <= 0:
        raise RuntimeError(
            f"{name} must be greater than zero."
        )

    return value


async def configure_connection(
    conn: AsyncConnection,
) -> None:
    """
    Configure every PostgreSQL connection created by the pool.

    Registering pgvector here allows numpy/Python vectors to be passed
    directly to VECTOR columns by tool embedding and retrieval code.
    """

    await register_vector_async(
        conn
    )


class Database:
    """
    Central async PostgreSQL/pgvector connection pool.

    Windows event-loop compatibility remains the responsibility of
    async_runtime.run_async(); this module intentionally does not create
    or replace event loops.
    """

    def __init__(
        self,
        database_url: str | None = None,
    ) -> None:

        resolved_url = str(
            database_url
            or os.getenv(
                "DATABASE_URL",
                "",
            )
            or ""
        ).strip()

        if not resolved_url:
            raise RuntimeError(
                "DATABASE_URL is missing from the root .env file: "
                f"{ENV_FILE}"
            )

        min_size = _env_int(
            "DATABASE_POOL_MIN_SIZE",
            DEFAULT_POOL_MIN_SIZE,
        )

        max_size = _env_int(
            "DATABASE_POOL_MAX_SIZE",
            DEFAULT_POOL_MAX_SIZE,
        )

        if max_size < min_size:
            raise RuntimeError(
                "DATABASE_POOL_MAX_SIZE cannot be smaller than "
                "DATABASE_POOL_MIN_SIZE."
            )

        timeout_seconds = _env_float(
            "DATABASE_POOL_TIMEOUT_SECONDS",
            DEFAULT_POOL_TIMEOUT_SECONDS,
        )

        self.database_url = resolved_url

        self.pool = AsyncConnectionPool(
            conninfo=self.database_url,
            min_size=min_size,
            max_size=max_size,
            timeout=timeout_seconds,
            open=False,
            configure=configure_connection,
        )

        self._opened = False

    async def open(
        self,
    ) -> None:
        """
        Open the connection pool and wait until its initial connections
        are usable.

        A failure here means PostgreSQL itself, the host port mapping,
        credentials, or pgvector registration must be fixed before the
        chatbot starts.
        """

        if self._opened:
            return

        await self.pool.open()

        try:
            await self.pool.wait()
        except Exception:
            # Do not leave a half-open pool behind when initialization fails.
            await self.pool.close()
            raise

        self._opened = True

    async def close(
        self,
    ) -> None:
        if not self._opened:
            return

        await self.pool.close()
        self._opened = False

    async def test_connection(
        self,
    ) -> dict[str, Any]:
        """
        Verify the pieces required by the chatbot:

        - PostgreSQL connection
        - pgvector extension
        - mcp_tools table
        - required embedding columns
        - VECTOR(256) storage contract
        """

        if not self._opened:
            raise RuntimeError(
                "Database pool is not open. Call await db.open() first."
            )

        async with (
            self.pool.connection()
            as conn
        ):
            async with conn.cursor() as cur:

                await cur.execute(
                    """
                    SELECT
                        current_database(),
                        current_user
                    """
                )

                row = await cur.fetchone()

                if row is None:
                    raise RuntimeError(
                        "Database test returned no result."
                    )

                database_name = str(
                    row[0]
                )

                user_name = str(
                    row[1]
                )

                # ------------------------------------------------
                # PGVECTOR EXTENSION
                # ------------------------------------------------

                await cur.execute(
                    """
                    SELECT extversion
                    FROM pg_extension
                    WHERE extname = 'vector'
                    """
                )

                vector_row = await cur.fetchone()

                if not vector_row:
                    raise RuntimeError(
                        "PostgreSQL extension 'vector' is not installed "
                        "in the current database."
                    )

                vector_version = str(
                    vector_row[0]
                )

                # ------------------------------------------------
                # MCP_TOOLS TABLE
                # ------------------------------------------------

                await cur.execute(
                    """
                    SELECT to_regclass('public.mcp_tools')
                    """
                )

                table_row = await cur.fetchone()

                if (
                    not table_row
                    or table_row[0] is None
                ):
                    raise RuntimeError(
                        "Required table public.mcp_tools does not exist."
                    )

                await cur.execute(
                    """
                    SELECT
                        column_name,
                        data_type,
                        udt_name
                    FROM information_schema.columns
                    WHERE
                        table_schema = 'public'
                        AND table_name = 'mcp_tools'
                    """
                )

                column_rows = await cur.fetchall()

                available_columns = {
                    str(column_name)
                    for (
                        column_name,
                        _data_type,
                        _udt_name,
                    ) in column_rows
                }

                required_columns = {
                    "id",
                    "server_name",
                    "tool_name",
                    "description",
                    "input_schema",
                    "searchable_text",
                    "content_hash",
                    "is_read_only",
                    "embedding",
                    "embedding_model",
                    "embedding_content_hash",
                    "updated_at",
                }

                missing_columns = sorted(
                    required_columns
                    - available_columns
                )

                if missing_columns:
                    raise RuntimeError(
                        "mcp_tools is missing required columns: "
                        f"{missing_columns}"
                    )

                # ------------------------------------------------
                # VECTOR DIMENSION CONTRACT
                # ------------------------------------------------
                #
                # pg_attribute/format_type returns values such as:
                #     vector(256)
                #
                # This verifies the same dimension used by settings.py,
                # embeddings.py, embed_tools.py, and pg_retriever.py.
                # ------------------------------------------------

                await cur.execute(
                    """
                    SELECT
                        format_type(
                            a.atttypid,
                            a.atttypmod
                        )
                    FROM pg_attribute AS a
                    JOIN pg_class AS c
                        ON c.oid = a.attrelid
                    JOIN pg_namespace AS n
                        ON n.oid = c.relnamespace
                    WHERE
                        n.nspname = 'public'
                        AND c.relname = 'mcp_tools'
                        AND a.attname = 'embedding'
                        AND a.attnum > 0
                        AND NOT a.attisdropped
                    """
                )

                embedding_type_row = (
                    await cur.fetchone()
                )

                if not embedding_type_row:
                    raise RuntimeError(
                        "Could not determine mcp_tools.embedding type."
                    )

                embedding_type = str(
                    embedding_type_row[0]
                ).strip()

                expected_type = (
                    f"vector({EXPECTED_VECTOR_DIMENSIONS})"
                )

                if (
                    embedding_type.casefold()
                    != expected_type.casefold()
                ):
                    raise RuntimeError(
                        "mcp_tools.embedding dimension mismatch. "
                        f"Expected {expected_type}, "
                        f"found {embedding_type}."
                    )

                # ------------------------------------------------
                # TOOL / EMBEDDING COUNTS
                # ------------------------------------------------

                await cur.execute(
                    """
                    SELECT
                        COUNT(*) AS total_tools,
                        COUNT(embedding)
                            AS tools_with_embeddings
                    FROM mcp_tools
                    """
                )

                count_row = (
                    await cur.fetchone()
                )

                total_tools = (
                    int(count_row[0])
                    if count_row
                    else 0
                )

                tools_with_embeddings = (
                    int(count_row[1])
                    if count_row
                    else 0
                )

        return {
            "database":
                database_name,
            "user":
                user_name,
            "pgvector_version":
                vector_version,
            "embedding_column":
                embedding_type,
            "mcp_tool_count":
                total_tools,
            "embedded_tool_count":
                tools_with_embeddings,
            "tools_needing_embeddings":
                max(
                    0,
                    total_tools
                    - tools_with_embeddings,
                ),
        }
