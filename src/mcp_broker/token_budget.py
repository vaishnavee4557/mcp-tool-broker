from __future__ import annotations

import json

from mcp_broker.models import ChatTurn, RetrievalCandidate, ToolDescriptor


def estimate_tokens(value: object) -> int:
    """Fast conservative estimate for budgeting before provider-side token counting."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return max(1, (len(text) + 3) // 4)


class TokenBudgeter:
    def __init__(
        self,
        *,
        max_tool_schema_tokens: int,
        max_history_tokens: int,
        max_history_turns: int,
    ) -> None:
        self.max_tool_schema_tokens = max_tool_schema_tokens
        self.max_history_tokens = max_history_tokens
        self.max_history_turns = max_history_turns

    def select_tools(self, candidates: list[RetrievalCandidate]) -> tuple[list[ToolDescriptor], int]:
        selected: list[ToolDescriptor] = []
        used = 0
        for candidate in candidates:
            cost = estimate_tokens(candidate.tool.as_llm_tool())
            if selected and used + cost > self.max_tool_schema_tokens:
                continue
            if not selected and cost > self.max_tool_schema_tokens:
                # Always keep the top tool; an oversized schema should be fixed at source.
                selected.append(candidate.tool)
                used += cost
                break
            selected.append(candidate.tool)
            used += cost
        return selected, used

    def select_history(self, history: list[ChatTurn], query: str) -> tuple[list[ChatTurn], int]:
        if not history or self.max_history_turns == 0 or self.max_history_tokens == 0:
            return [], 0

        query_words = set(query.lower().split())
        latest = history[-self.max_history_turns :]
        ranked = sorted(
            enumerate(latest),
            key=lambda item: (
                bool(query_words.intersection(item[1].content.lower().split())),
                item[0],
            ),
            reverse=True,
        )

        chosen: list[tuple[int, ChatTurn]] = []
        used = 0
        for index, turn in ranked:
            cost = estimate_tokens(turn.model_dump())
            if used + cost <= self.max_history_tokens:
                chosen.append((index, turn))
                used += cost

        chosen.sort(key=lambda item: item[0])
        return [turn for _, turn in chosen], used
