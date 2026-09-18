from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mcp_broker.db import Database
from mcp_broker.embeddings import (
    EmbeddingProvider,
    HashEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)
from mcp_broker.settings import get_settings


DEFAULT_EMBEDDING_DIMENSIONS = 256
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_MODE = "openai_compatible"


@dataclass
class PgToolSearchResult:
    server_name: str
    tool_name: str
    description: str
    input_schema: dict
    searchable_text: str
    similarity: float
    is_read_only: bool


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
            result = int(value)
        except (
            TypeError,
            ValueError,
        ):
            continue

        if result > 0:
            return result

    return int(default)


def _build_default_embedding_provider(
) -> tuple[
    EmbeddingProvider,
    str,
    int,
]:
    """
    Build the SAME embedding configuration used for query search.

    embed_tools.py must use the same:
        mode
        model
        dimensions

    This prevents the previous failure mode where tools were stored with
    one embedding model but user questions were embedded with local hashes.
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


class PgVectorToolRetriever:
    """
    Semantic search against MCP tool embeddings stored in PostgreSQL/pgvector.

    Critical guarantees:
    - query and tool vectors use the same configured model
    - dimensions are checked before SQL execution
    - old embeddings from a different model are never mixed into search
    - read-only filtering is optional because final authorization is enforced
      against the live MCP ToolDescriptor catalog by PgHybridToolRetriever
    """

    def __init__(
        self,
        db: Database,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_model: str | None = None,
        embedding_dimensions: int | None = None,
        server_name: str | None = None,
    ) -> None:

        self.db = db

        if embedding_provider is None:
            (
                provider,
                configured_model,
                configured_dimensions,
            ) = _build_default_embedding_provider()

            self.embedding_provider = provider
            self.embedding_model = (
                str(
                    embedding_model
                    or configured_model
                ).strip()
            )
            self.embedding_dimensions = int(
                embedding_dimensions
                or configured_dimensions
            )

        else:
            self.embedding_provider = (
                embedding_provider
            )

            provider_dimensions = getattr(
                embedding_provider,
                "dimensions",
                None,
            )

            dimensions = (
                embedding_dimensions
                or provider_dimensions
                or DEFAULT_EMBEDDING_DIMENSIONS
            )

            self.embedding_dimensions = int(
                dimensions
            )

            if embedding_model:
                self.embedding_model = str(
                    embedding_model
                ).strip()

            elif isinstance(
                embedding_provider,
                HashEmbeddingProvider,
            ):
                self.embedding_model = (
                    "local-hash-"
                    f"{self.embedding_dimensions}"
                )

            else:
                model = getattr(
                    embedding_provider,
                    "model",
                    None,
                )

                if not model:
                    raise ValueError(
                        "embedding_model is required when the "
                        "provider does not expose a model name."
                    )

                self.embedding_model = str(
                    model
                ).strip()

        if self.embedding_dimensions <= 0:
            raise ValueError(
                "Embedding dimensions must be greater than zero."
            )

        if not self.embedding_model:
            raise ValueError(
                "Embedding model name is required."
            )

        self.server_name = (
            str(server_name).strip()
            if server_name
            else None
        )

    async def close(
        self,
    ) -> None:
        """
        Close the embedding provider when it owns an async HTTP client.
        """

        close_method = getattr(
            self.embedding_provider,
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

    async def retrieve(
        self,
        query: str,
        limit: int = 10,
        read_only_only: bool = True,
    ) -> list[
        PgToolSearchResult
    ]:

        query = str(
            query or ""
        ).strip()

        if not query:
            return []

        try:
            limit = int(limit)
        except (
            TypeError,
            ValueError,
        ):
            limit = 10

        limit = max(
            1,
            min(
                limit,
                100,
            ),
        )

        # ----------------------------------------------------
        # 1. CREATE TEMPORARY USER-QUESTION EMBEDDING
        # ----------------------------------------------------

        vectors = (
            await self.embedding_provider.embed(
                [query]
            )
        )

        if not vectors:
            raise RuntimeError(
                "Question embedding was not generated."
            )

        if len(vectors) != 1:
            raise RuntimeError(
                "Expected exactly one question embedding, "
                f"received {len(vectors)}."
            )

        query_vector = np.asarray(
            vectors[0],
            dtype=np.float32,
        )

        if query_vector.ndim != 1:
            raise RuntimeError(
                "Question embedding must be one-dimensional."
            )

        if (
            query_vector.shape[0]
            != self.embedding_dimensions
        ):
            raise RuntimeError(
                "Question embedding dimension mismatch. "
                f"Expected {self.embedding_dimensions}, "
                f"received {query_vector.shape[0]}."
            )

        if not np.isfinite(
            query_vector
        ).all():
            raise RuntimeError(
                "Question embedding contains non-finite values."
            )

        # ----------------------------------------------------
        # 2. SEARCH ONLY MATCHING STORED EMBEDDINGS
        # ----------------------------------------------------

        where_parts = [
            "embedding IS NOT NULL",
            "embedding_model = %s",
        ]

        parameters: list[
            Any
        ] = [
            self.embedding_model,
        ]

        if read_only_only:
            where_parts.append(
                "is_read_only = TRUE"
            )

        if self.server_name:
            where_parts.append(
                "server_name = %s"
            )
            parameters.append(
                self.server_name
            )

        where_sql = (
            "\nAND ".join(
                where_parts
            )
        )

        sql = f"""
            SELECT
                server_name,
                tool_name,
                description,
                input_schema,
                searchable_text,
                (
                    1 - (
                        embedding <=> %s
                    )
                ) AS similarity,
                is_read_only

            FROM mcp_tools

            WHERE
                {where_sql}

            ORDER BY
                embedding <=> %s

            LIMIT %s
        """

        # Vector appears before WHERE placeholders in the SQL.
        sql_parameters: list[
            Any
        ] = [
            query_vector,
            *parameters,
            query_vector,
            limit,
        ]

        async with (
            self.db.pool.connection()
            as conn
        ):
            async with conn.cursor() as cur:

                await cur.execute(
                    sql,
                    tuple(
                        sql_parameters
                    ),
                )

                rows = (
                    await cur.fetchall()
                )

        # ----------------------------------------------------
        # 3. CLEAR DIAGNOSTIC FOR STALE/MISSING EMBEDDINGS
        # ----------------------------------------------------

        if not rows:
            async with (
                self.db.pool.connection()
                as conn
            ):
                async with conn.cursor() as cur:

                    diagnostic_where = [
                        "embedding IS NOT NULL",
                    ]

                    diagnostic_params: list[
                        Any
                    ] = []

                    if self.server_name:
                        diagnostic_where.append(
                            "server_name = %s"
                        )
                        diagnostic_params.append(
                            self.server_name
                        )

                    await cur.execute(
                        f"""
                        SELECT DISTINCT
                            embedding_model
                        FROM mcp_tools
                        WHERE
                            {' AND '.join(diagnostic_where)}
                        ORDER BY embedding_model
                        """,
                        tuple(
                            diagnostic_params
                        ),
                    )

                    model_rows = (
                        await cur.fetchall()
                    )

            available_models = [
                str(row[0])
                for row in model_rows
                if row
                and row[0] is not None
            ]

            if (
                available_models
                and self.embedding_model
                not in available_models
            ):
                raise RuntimeError(
                    "No tool embeddings exist for the configured "
                    f"embedding model {self.embedding_model!r}. "
                    "Stored models are: "
                    f"{available_models}. "
                    "Run tool sync and embedding refresh."
                )

            return []

        # ----------------------------------------------------
        # 4. CONVERT DB ROWS
        # ----------------------------------------------------

        results: list[
            PgToolSearchResult
        ] = []

        for row in rows:

            (
                server_name,
                tool_name,
                description,
                input_schema,
                searchable_text,
                similarity,
                is_read_only,
            ) = row

            try:
                similarity_value = float(
                    similarity
                )
            except (
                TypeError,
                ValueError,
            ):
                similarity_value = (
                    0.0
                )

            if not np.isfinite(
                similarity_value
            ):
                similarity_value = (
                    0.0
                )

            results.append(
                PgToolSearchResult(
                    server_name=str(
                        server_name
                        or ""
                    ),
                    tool_name=str(
                        tool_name
                    ),
                    description=str(
                        description
                        or ""
                    ),
                    input_schema=(
                        input_schema
                        if isinstance(
                            input_schema,
                            dict,
                        )
                        else {}
                    ),
                    searchable_text=str(
                        searchable_text
                        or ""
                    ),
                    similarity=(
                        similarity_value
                    ),
                    is_read_only=bool(
                        is_read_only
                    ),
                )
            )

        return results
