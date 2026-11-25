from __future__ import annotations

import uuid 
from typing import Any

from mcp_broker.db import Database
from mcp_broker.models import ChatTurn


MAX_CONVERSATION_TITLE_LENGTH = 80


class ConversationStore:

    def __init__(self, db: Database) -> None:
        self.db = db

    @staticmethod
    def _build_title(
        content: str,
        *,
        max_length: int = MAX_CONVERSATION_TITLE_LENGTH,
    ) -> str:
        """
        Build a readable title from the first user message.

        No extra LLM call is used. Whitespace is normalized and long
        messages are truncated with an ellipsis.
        """
        normalized = " ".join(
            str(content or "").strip().split()
        )

        if not normalized:
            return "Untitled conversation"

        max_length = max(
            4,
            int(max_length),
        )

        if len(normalized) <= max_length:
            return normalized

        return (
            normalized[
                : max_length - 3
            ].rstrip()
            + "..."
        )

    def _pool(self) -> Any:
        pool = getattr(self.db, "pool", None)

        if pool is None:
            raise RuntimeError(
                "Database does not expose an active 'pool'. "
                "Open the Database before using ConversationStore."
            )

        return pool

    @staticmethod
    def _normalize_conversation_id(
        conversation_id: str,
    ) -> str:
        try:
            parsed = uuid.UUID(
                str(conversation_id).strip()
            )
        except (
            ValueError,
            TypeError,
            AttributeError,
        ) as exc:
            raise ValueError(
                "Invalid conversation_id. Expected a valid UUID."
            ) from exc

        return str(parsed)
    @staticmethod
    def _normalize_user_id(
        user_id: str,
    ) -> str:
        normalized = str(
            user_id or ""
        ).strip()

        if not normalized:
            raise ValueError(
                "user_id cannot be empty."
            )

        return normalized
    

    async def ensure_schema(self) -> None:
        pool = self._pool()

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                # conversations table
                await cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversations (
                        id UUID PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        title TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )

                # Existing installations may have been created before
                # user_id was introduced.
                await cur.execute(
                    """
                    ALTER TABLE conversations
                    ADD COLUMN IF NOT EXISTS user_id TEXT
                    """
                )

                # Keep pre-existing rows valid during migration.
                await cur.execute(
                    """
                    UPDATE conversations
                    SET user_id = 'legacy-user'
                    WHERE user_id IS NULL
                       OR BTRIM(user_id) = ''
                    """
                )

                await cur.execute(
                    """
                    ALTER TABLE conversations
                    ALTER COLUMN user_id SET NOT NULL
                    """
                )
                 # ------------------------------------------------
                # MESSAGES TABLE
                # ------------------------------------------------

                await cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_messages (
                        id BIGSERIAL PRIMARY KEY,
                        conversation_id UUID NOT NULL
                            REFERENCES conversations(id)
                            ON DELETE CASCADE,
                        role TEXT NOT NULL
                            CHECK (role IN ('user', 'assistant')),
                        content TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                # ------------------------------------------------
                # INDEXES
                # ------------------------------------------------

                await cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                        idx_conversation_messages_conversation_id_id
                    ON conversation_messages (
                        conversation_id,
                        id DESC
                    )
                    """
                )

                await cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                        idx_conversations_user_updated
                    ON conversations (
                        user_id,
                        updated_at DESC
                    )
                    """
                )

            await conn.commit()

    async def create_conversation(
        self,
        user_id :str,
        title: str | None = None,
    ) -> str:
        user_id = self._normalize_user_id(
            user_id
        )
        conversation_id = str(uuid.uuid4())

        clean_title = (
            str(title).strip()
            if title is not None
            else None
        )

        if clean_title == "":
            clean_title = None

        pool = self._pool()

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO conversations (
                        id,
                        user_id,
                        title
                    )
                    VALUES (%s, %s, %s)
                    """,
                    (
                        conversation_id,
                        user_id,
                        clean_title,
                    
                    ),
                )
            await conn.commit()

        return conversation_id

    async def exists(
        self,
        user_id: str,
        conversation_id: str,
    ) -> bool:
        user_id = self._normalize_user_id(
            user_id
        )
        conversation_id = self._normalize_conversation_id(
            conversation_id
        )

        pool = self._pool()

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT 1
                    FROM conversations
                    WHERE id = %s
                    AND user_id = %s
                    LIMIT 1
                    """,
                    (conversation_id, user_id,),
                )

                row = await cur.fetchone()

        return row is not None

    async def add_message(
        self,
        user_id: str,
        conversation_id: str,
        role: str,
        content: str,
    ) -> None:
        user_id = self._normalize_user_id(
            user_id
        )
        conversation_id = self._normalize_conversation_id(
            conversation_id
        )

        role = str(role or "").strip().lower()

        if role not in {
            "user",
            "assistant",
        }:
            raise ValueError(
                "role must be either 'user' or 'assistant'."
            )

        content = str(
            content if content is not None else ""
        )

        if not content.strip():
            raise ValueError(
                "Conversation message content cannot be empty."
            )

        pool = self._pool()

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT 1
                    FROM conversations
                    WHERE id = %s
                    AND user_id = %s
                    LIMIT 1
                    """,
                    (conversation_id, user_id,),
                )

                if await cur.fetchone() is None:
                    raise ValueError(
                        "Conversation not found: "
                        f"{conversation_id}"
                    )

                await cur.execute(
                    """
                    INSERT INTO conversation_messages (
                        conversation_id,
                        role,
                        content
                    )
                    VALUES (%s, %s, %s)
                    """,
                    (
                        conversation_id,
                        role,
                        content,
                    ),
                )

                if role == "user":
                    generated_title = self._build_title(
                        content
                    )

                    await cur.execute(
                        """
                        UPDATE conversations
                        SET
                            title = COALESCE(
                                NULLIF(BTRIM(title), ''),
                                %s
                            ),
                            updated_at = NOW()
                        WHERE id = %s
                        AND user_id = %s
                        """,
                        (
                            generated_title,
                            conversation_id,
                            user_id,
                        ),
                    )
                else:
                    await cur.execute(
                        """
                        UPDATE conversations
                        SET updated_at = NOW()
                        WHERE id = %s
                        AND user_id = %s
                        """,
                        (conversation_id,
                         user_id),
                    )
            await conn.commit()

    async def list_recent_conversations(
        self,
        user_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Return the most recently updated conversations.

        Intended for CLI/UI history pickers. It does not load message
        bodies and does not touch pgvector.
        """
        user_id = self._normalize_user_id(
            user_id
        )
        limit = int(limit)

        if limit <= 0:
            return []

        pool = self._pool()

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT
                        id,
                        COALESCE(
                            NULLIF(BTRIM(title), ''),
                            'Untitled conversation'
                        ) AS title,
                        created_at,
                        updated_at
                    FROM conversations
                    WHERE user_id = %s
                    ORDER BY
                        updated_at DESC,
                        created_at DESC
                    LIMIT %s
                    """,
                    (user_id, limit),
                )

                rows = await cur.fetchall()

        return [
            {
                "id": str(row[0]),
                "title": str(row[1]),
                "created_at": row[2],
                "updated_at": row[3],
            }
            for row in rows
        ]

    async def get_all_messages(
        self,
        user_id: str,
        conversation_id: str,
    ) -> list[dict[str, Any]]:
        """
        Return the full persisted conversation in chronological order.

        This method is intended for display/history views only.
        It does NOT change the bounded LLM runtime history.
        """
        user_id = self._normalize_user_id(
            user_id
        )
        conversation_id = self._normalize_conversation_id(
            conversation_id
        )

        pool = self._pool()

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT
                        m.id,
                        m.role,
                        m.content,
                        m.created_at
                    FROM conversation_messages AS m
                    JOIN conversations AS c 
                    ON c.id = m.conversation_id
                    WHERE m.conversation_id = %s AND c.user_id = %s
                    ORDER BY m.id ASC
                    """,
                    (conversation_id, user_id),
                )

                rows = await cur.fetchall()

        return [
            {
                "id": int(row[0]),
                "role": str(row[1]),
                "content": str(row[2]),
                "created_at": row[3],
            }
            for row in rows
        ]

    async def get_recent_turns(
        self,
        user_id: str,
        conversation_id: str,
        limit: int = 10,
    ) -> list[ChatTurn]:
        user_id = self._normalize_user_id(
            user_id
        )
        conversation_id = self._normalize_conversation_id(
            conversation_id
        )

        limit = int(limit)

        if limit <= 0:
            return []

        pool = self._pool()

        fetch_limit = (limit * 2) + 1

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT
                        role,
                        content
                    FROM conversation_messages AS m
                    JOIN conversations AS c ON c.id = m.conversation_id
                    WHERE m.conversation_id = %s AND c.user_id = %s
                    ORDER BY m.id DESC
                    LIMIT %s
                    """,
                    (
                        conversation_id,
                        user_id,
                        fetch_limit,
                    ),
                )

                rows = await cur.fetchall()

        if not rows:
            return []

        chronological = list(reversed(rows))

        completed_pairs: list[
            tuple[
                tuple[Any, Any],
                tuple[Any, Any],
            ]
        ] = []

        index = 0

        while index < len(chronological) - 1:
            first = chronological[index]
            second = chronological[index + 1]

            first_role = str(
                first[0]
            ).strip().lower()

            second_role = str(
                second[0]
            ).strip().lower()

            if (
                first_role == "user"
                and second_role == "assistant"
            ):
                completed_pairs.append(
                    (
                        first,
                        second,
                    )
                )
                index += 2
                continue

            index += 1

        completed_pairs = completed_pairs[-limit:]

        history: list[ChatTurn] = []

        for user_row, assistant_row in completed_pairs:
            history.append(
                ChatTurn(
                    role="user",
                    content=str(user_row[1]),
                )
            )
            
            history.append(
                ChatTurn(
                    role="assistant",
                    content=str(assistant_row[1]),
                )
            )

        return history
    async def delete_conversation(
        self,
        user_id: str,
        conversation_id: str,
    ) -> bool:
        user_id = self._normalize_user_id(
            user_id
        )
        conversation_id = self._normalize_conversation_id(
            conversation_id
        )

        pool = self._pool()

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    DELETE FROM conversations
                    WHERE id = %s AND user_id = %s
                    """,
                    (conversation_id, user_id),
                )
                deleted = cur.rowcount > 0 
            await conn.commit()
        return deleted
