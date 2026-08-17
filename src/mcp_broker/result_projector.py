from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


class ResultProjector:
    """Bounds tool output before it is placed back into model context."""

    def __init__(self, *, max_rows: int, max_chars: int, max_depth: int = 8) -> None:
        self.max_rows = max_rows
        self.max_chars = max_chars
        self.max_depth = max_depth

    def project(self, value: Any) -> Any:
        normalized = self._trim(value, depth=0)
        encoded = json.dumps(normalized, ensure_ascii=False, default=str)
        if len(encoded) <= self.max_chars:
            return normalized
        return {
            "truncated": True,
            "reason": "tool result exceeded context budget",
            "preview": encoded[: self.max_chars],
            "original_characters": len(encoded),
        }

    def _trim(self, value: Any, *, depth: int) -> Any:
        if depth >= self.max_depth:
            return "<max-depth-reached>"
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if hasattr(value, "model_dump"):
            return self._trim(value.model_dump(mode="json"), depth=depth + 1)
        if isinstance(value, Mapping):
            return {
                str(key): self._trim(child, depth=depth + 1)
                for key, child in value.items()
                if child is not None
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            trimmed = [self._trim(item, depth=depth + 1) for item in value[: self.max_rows]]
            if len(value) > self.max_rows:
                trimmed.append(
                    {
                        "truncated": True,
                        "remaining_items": len(value) - self.max_rows,
                    }
                )
            return trimmed
        return str(value)
