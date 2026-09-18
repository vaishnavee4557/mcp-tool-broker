import asyncio

from mcp_broker.db import Database


async def main() -> None:

    db = Database()

    try:
        await db.open()

        result = await db.test_connection()

        print()
        print("========== DATABASE TEST ==========")

        print(
            "Database:",
            result["database"],
        )

        print(
            "User:",
            result["user"],
        )

        print(
            "pgvector:",
            result["pgvector_version"],
        )

        print(
            "Stored MCP tools:",
            result["mcp_tool_count"],
        )

        print()
        print("DATABASE CONNECTION OK")

    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())