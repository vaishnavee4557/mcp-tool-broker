from __future__ import annotations

import asyncio

from mcp_broker.db import Database
from mcp_broker.pg_retriever import (
    PgVectorToolRetriever,
)


async def main() -> None:

    db = Database()

    try:
        await db.open()

        retriever = PgVectorToolRetriever(
            db
        )

        question = (
            "show number of machines"
        )

        print()
        print(
            "Question:",
            question,
        )

        results = await retriever.retrieve(
            query=question,
            limit=8,
        )

        print()
        print(
            "========== PGVECTOR RESULTS =========="
        )

        for index, result in enumerate(
            results,
            start=1,
        ):

            print(
                f"{index}. "
                f"{result.tool_name}"
            )

            print(
                "   Similarity:",
                f"{result.similarity:.4f}",
            )

        print()
        print(
            "PGVECTOR SEARCH OK"
        )

    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(
        main()
    )