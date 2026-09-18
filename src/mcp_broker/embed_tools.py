#this script keeps your MCP tool vectors updated so the retriever can semantically match a user’s question to the correct tool from servers
from __future__ import annotations

from typing import Any

import numpy as np

from mcp_broker.async_runtime import run_async
from mcp_broker.db import Database
from mcp_broker.embeddings import (
    EmbeddingProvider,
    HashEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)
from mcp_broker.settings import get_settings


DEFAULT_EMBEDDING_MODE = "openai_compatible"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONS = 256
DEFAULT_BATCH_SIZE = 64


# ============================================================
# SETTINGS HELPERS
# ============================================================

def _read_secret(
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


def _setting_text(
    settings: Any,
    *names: str,
    default: str = "",
) -> str:
    for name in names:
        value = getattr(
            settings,
            name,
            None,
        )

        if value is None:
            continue

        text = _read_secret(
            value
        )

        if text:
            return text

    return default


def _setting_int(
    settings: Any,
    *names: str,
    default: int,
) -> int:
    for name in names:
        value = getattr(
            settings,
            name,
            None,
        )

        if value in (
            None,
            "",
        ):
            continue

        try:
            number = int(value)
        except (
            TypeError,
            ValueError,
        ):
            continue

        if number > 0:
            return number

    return int(default)


# ============================================================
# EMBEDDING PROVIDER
# ============================================================

def build_embedding_provider(
) -> tuple[
    EmbeddingProvider,
    str,
    int,
]:
    """
    Build the embedding provider from the same settings used by
    pg_retriever.py.

    Critical rule:
        stored tool embeddings
        and
        temporary question embeddings

    MUST use the same:
        provider
        model
        dimensions
    """

    settings = get_settings()

    mode = _setting_text(
        settings,
        "embedding_mode",
        default=DEFAULT_EMBEDDING_MODE,
    ).casefold().replace(
        "-",
        "_",
    )

    dimensions = _setting_int(
        settings,
        "embedding_dimensions",
        default=DEFAULT_EMBEDDING_DIMENSIONS,
    )

    if mode in {
        "hash",
        "local_hash",
        "local",
    }:
        provider = HashEmbeddingProvider(
            dimensions=dimensions,
        )

        model_name = (
            f"local-hash-{dimensions}"
        )

        return (
            provider,
            model_name,
            dimensions,
        )

    if mode in {
        "openai",
        "openai_compatible",
        "semantic",
    }:
        base_url = _setting_text(
            settings,
            "embedding_base_url",
            "llm_base_url",
        )

        api_key = _setting_text(
            settings,
            "embedding_api_key",
            "llm_api_key",
        )

        model_name = _setting_text(
            settings,
            "embedding_model",
            default=DEFAULT_EMBEDDING_MODEL,
        )

        batch_size = _setting_int(
            settings,
            "embedding_batch_size",
            default=DEFAULT_BATCH_SIZE,
        )

        if not base_url:
            raise RuntimeError(
                "Embedding base URL is missing. "
                "Set EMBEDDING_BASE_URL or LLM_BASE_URL."
            )

        if not api_key:
            raise RuntimeError(
                "Embedding API key is missing. "
                "Set EMBEDDING_API_KEY or LLM_API_KEY."
            )

        provider = (
            OpenAICompatibleEmbeddingProvider(
                base_url=base_url,
                api_key=api_key,
                model=model_name,
                dimensions=dimensions,
                batch_size=batch_size,
            )
        )

        return (
            provider,
            model_name,
            dimensions,
        )

    raise RuntimeError(
        "Unsupported EMBEDDING_MODE: "
        f"{mode!r}. "
        "Use 'openai_compatible' or 'hash'."
    )


async def _close_provider(
    provider: EmbeddingProvider,
) -> None:
    close_method = getattr(
        provider,
        "close",
        None,
    )

    if close_method is None:
        return

    result = close_method()

    if hasattr(
        result,
        "__await__",
    ):
        await result


# ============================================================
# MAIN
# ============================================================

async def main() -> None:
    db = Database()
    db_opened = False

    provider: (
        EmbeddingProvider
        | None
    ) = None

    try:
        # ----------------------------------------------------
        # 1. BUILD THE SAME EMBEDDING CONFIG AS QUERY SEARCH
        # ----------------------------------------------------

        (
            provider,
            embedding_model,
            embedding_dimensions,
        ) = build_embedding_provider()

        print(
            "Embedding model:",
            embedding_model,
        )

        print(
            "Embedding dimensions:",
            embedding_dimensions,
        )

        # ----------------------------------------------------
        # 2. DATABASE
        # ----------------------------------------------------

        await db.open()
        db_opened = True

        print(
            "Database connected."
        )

        async with (
            db.pool.connection()
            as conn
        ):
            # ------------------------------------------------
            # 3. FIND ONLY STALE/MISSING TOOL EMBEDDINGS
            # ------------------------------------------------

            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT
                        id,
                        server_name,
                        tool_name,
                        searchable_text,
                        content_hash
                    FROM mcp_tools
                    WHERE
                        embedding IS NULL
                        OR embedding_model
                           IS DISTINCT FROM %s
                        OR embedding_content_hash
                           IS DISTINCT FROM content_hash
                    ORDER BY
                        server_name,
                        tool_name,
                        id
                    """,
                    (
                        embedding_model,
                    ),
                )

                rows = (
                    await cur.fetchall()
                )

            print(
                "Tools requiring embeddings:",
                len(rows),
            )

            if not rows:
                print(
                    "All tool embeddings are already up to date."
                )
                return

            # ------------------------------------------------
            # 4. VALIDATE SEARCHABLE TEXT
            # ------------------------------------------------

            invalid_rows = [
                (
                    row[1],
                    row[2],
                )
                for row in rows
                if not str(
                    row[3]
                    or ""
                ).strip()
            ]

            if invalid_rows:
                preview = invalid_rows[:10]

                raise RuntimeError(
                    "Some tools have empty searchable_text. "
                    "Run sync_tools.py after fixing registry/tool_store. "
                    f"Examples: {preview}"
                )

            texts = [
                str(
                    row[3]
                ).strip()
                for row in rows
            ]

            # ------------------------------------------------
            # 5. GENERATE TOOL EMBEDDINGS
            # ------------------------------------------------

            vectors = await provider.embed(
                texts
            )

            if len(vectors) != len(rows):
                raise RuntimeError(
                    "Embedding count does not match tool count. "
                    f"tools={len(rows)}, vectors={len(vectors)}"
                )

            # ------------------------------------------------
            # 6. VALIDATE ALL VECTORS BEFORE WRITING ANYTHING
            # ------------------------------------------------

            prepared_updates: list[
                tuple[
                    np.ndarray,
                    str,
                    str,
                    Any,
                ]
            ] = []

            for row, vector in zip(
                rows,
                vectors,
            ):
                (
                    tool_id,
                    server_name,
                    tool_name,
                    _,
                    content_hash,
                ) = row

                vector_array = np.asarray(
                    vector,
                    dtype=np.float32,
                )

                if vector_array.ndim != 1:
                    raise RuntimeError(
                        "Embedding must be one-dimensional for "
                        f"{server_name}/{tool_name}."
                    )

                if (
                    vector_array.shape[0]
                    != embedding_dimensions
                ):
                    raise RuntimeError(
                        "Wrong embedding dimension for "
                        f"{server_name}/{tool_name}: "
                        f"expected={embedding_dimensions}, "
                        f"actual={vector_array.shape[0]}"
                    )

                if not np.isfinite(
                    vector_array
                ).all():
                    raise RuntimeError(
                        "Embedding contains non-finite values for "
                        f"{server_name}/{tool_name}."
                    )

                prepared_updates.append(
                    (
                        vector_array,
                        embedding_model,
                        content_hash,
                        tool_id,
                    )
                )

            # ------------------------------------------------
            # 7. STORE EMBEDDINGS
            # ------------------------------------------------

            async with conn.cursor() as cur:
                for row, update in zip(
                    rows,
                    prepared_updates,
                ):
                    (
                        _tool_id,
                        server_name,
                        tool_name,
                        _searchable_text,
                        _content_hash,
                    ) = row

                    await cur.execute(
                        """
                        UPDATE mcp_tools
                        SET
                            embedding = %s,
                            embedding_model = %s,
                            embedding_content_hash = %s,
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        update,
                    )

                    print(
                        "Embedded:",
                        f"{server_name}/{tool_name}",
                    )

            await conn.commit()

        # ----------------------------------------------------
        # 8. SUMMARY
        # ----------------------------------------------------

        print()
        print(
            "========== TOOL EMBEDDINGS =========="
        )

        print(
            "Embeddings generated:",
            len(rows),
        )

        print(
            "Model:",
            embedding_model,
        )

        print(
            "Dimensions:",
            embedding_dimensions,
        )

        print(
            "TOOL EMBEDDING SYNC OK"
        )

    finally:
        if provider is not None:
            await _close_provider(
                provider
            )

        if db_opened:
            await db.close()


if __name__ == "__main__":
    run_async(
        main()
    )
