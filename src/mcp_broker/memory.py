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
    def _make_key(self, user_id: str , session_id:str,) -> tuple[str ,str]:
        return (user_id, session_id)
    
    async def get(self,user_id: str, session_id: str) -> list[ChatTurn]:
        async with self._lock:
            self._evict_expired()
            key = self._make_key(user_id, session_id)
            return list(self._turns.get(key, [],))

    async def append(self, user_id: str, session_id: str, *turns: ChatTurn) -> None:
        async with self._lock:
            self._evict_expired()
            key = self._make_key(user_id, session_id)
            values = self._turns[key]
            values.extend(turns)
            self._turns[key] = values[-self.max_turns :]
            self._updated_at[key] = time.monotonic()
    async def delete(self , user_id: str, session_id: str) -> bool:
        async with self._lock:
            self._evict_expired()
            key = self._make_key(user_id, session_id)
            existed = key in self._turns
            self._turns.pop(key, None)
            self._updated_at.pop(key, None)
            return existed
    async def list_sessions(self, user_id: str) -> list[str]:
        async with self._lock:
            self._evict_expired()
            sessions = [
                session_id
                for (stored_user_id , session_id) in self._turns.keys()
                if stored_user_id == user_id
            
            return sessions 
            ]
    # to delete expired sessions from the store
    def _evict_expired(self) -> None:

        now = time.monotonic()

        expired = [
            key
            for key, updated
            in self._updated_at.items()
            if (
                now - updated
                > self.ttl_seconds
            )
        ]

        for key in expired:
            self._turns.pop(
                key,
                None,
            )

            self._updated_at.pop(
                key,
                None,
            )