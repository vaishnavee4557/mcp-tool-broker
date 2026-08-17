from __future__ import annotations

import asyncio
import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

import httpx


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


class EmbeddingProvider(Protocol):
    async def embed(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        ...


class HashEmbeddingProvider:
    def __init__(
        self,
        dimensions: int = 256,
    ) -> None:
        self.dimensions = dimensions

    async def embed(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        return await asyncio.to_thread(
            self._embed_sync,
            texts,
        )

    def _embed_sync(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        return [
            self._one(text)
            for text in texts
        ]

    def _one(
        self,
        text: str,
    ) -> list[float]:
        vector = [0.0] * self.dimensions

        for token in _TOKEN_RE.findall(
            text.lower()
        ):
            digest = hashlib.blake2b(
                token.encode(),
                digest_size=8,
            ).digest()

            value = int.from_bytes(
                digest,
                "big",
            )

            index = value % self.dimensions

            sign = (
                1.0
                if (value >> 1) % 2 == 0
                else -1.0
            )

            vector[index] += sign

        norm = (
            math.sqrt(
                sum(
                    value * value
                    for value in vector
                )
            )
            or 1.0
        )

        return [
            value / norm
            for value in vector
        ]


class OpenAICompatibleEmbeddingProvider:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = "text-embedding-3-small",
        timeout_seconds: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError(
                "An embedding API key is required"
            )

        self.model = model

        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout_seconds,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def embed(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        print(
            self.model,
        )

        response = await self.client.post(
            "/embeddings",
            json={
                "model": self.model,
                "input": list(texts),
            },
        )

        if response.status_code >= 400:
            print(
                "\nEmbedding status:",
                response.status_code,
            )
            print(
                "Embedding response:",
                response.text,
            )

        response.raise_for_status()

        response_json = response.json()

        data = response_json.get(
            "data",
            [],
        )

        if not data:
            raise RuntimeError(
                "Embedding API returned no embedding data"
            )

        ordered = sorted(
            data,
            key=lambda item: item["index"],
        )

        return [
            item["embedding"]
            for item in ordered
        ]