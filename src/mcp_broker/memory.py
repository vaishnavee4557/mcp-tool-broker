from __future__ import annotations

import asyncio
import time
from collections import defaultdict

from mcp_broker.models import ChatTurn


class InMemoryConversationStore:
    """Single-process reference implementation. Replace with Redis in multi-replica deployments."""

    def __init__(self, *, ttl_seconds: int = 3600, max_turns: int = 30) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_turns = max_turns
        
        self._turns: dict[str, list[ChatTurn]] = defaultdict(list)
        self._updated_at: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def get(self, session_id: str) -> list[ChatTurn]:
        async with self._lock:
            self._evict_expired()
            return list(self._turns.get(session_id, []))

    async def append(self, session_id: str, *turns: ChatTurn) -> None:
        async with self._lock:
            self._evict_expired()
            values = self._turns[session_id]
            values.extend(turns)
            self._turns[session_id] = values[-self.max_turns :]
            self._updated_at[session_id] = time.monotonic()
    # to delete expired sessions from the store
    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [
            session_id
            for session_id, updated in self._updated_at.items()
            if now - updated > self.ttl_seconds
        ]
        for session_id in expired:
            self._turns.pop(session_id, None)
            self._updated_at.pop(session_id, None)
    