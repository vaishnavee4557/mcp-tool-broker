from __future__ import annotations

import asyncio
import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

import aiohttp


# ============================================================
# COMMON
# ============================================================

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


class EmbeddingProvider(Protocol):
    """
    Common interface used for both:
    - persisted MCP tool embeddings
    - temporary user-query embeddings

    Tool vectors and query vectors MUST use the same provider,
    model, and dimensions.
    """

    dimensions: int

    async def embed(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        ...

    async def close(
        self,
    ) -> None:
        ...


def _validate_texts(
    texts: Sequence[str],
) -> list[str]:
    normalized = [
        str(text or "").strip()
        for text in texts
    ]

    if not normalized:
        return []

    if any(not text for text in normalized):
        raise ValueError(
            "Embedding input contains an empty text value."
        )

    return normalized


def _clean_secret(
    value: object,
    *,
    name: str,
) -> str:
    """
    Normalize a credential at the transport boundary.

    API keys must never contain ASCII control characters. Windows shell /
    copied environment values can occasionally carry hidden characters
    such as 0x16. Those characters are not part of a valid OpenAI API key,
    so remove them once here before the value becomes an HTTP header.

    Printable characters are preserved exactly.
    """

    if value is None:
        return ""

    if hasattr(value, "get_secret_value"):
        value = value.get_secret_value()

    text = str(value).strip()

    cleaned = "".join(
        char
        for char in text
        if ord(char) >= 32
        and ord(char) != 127
    ).strip()

    if not cleaned:
        raise ValueError(
            f"{name} is missing."
        )

    return cleaned


def _validate_header_value(
    name: str,
    value: str,
) -> str:
    """
    Final defense before placing a value in an HTTP header.
    """

    text = str(value)

    bad = [
        f"0x{ord(char):02x}"
        for char in text
        if ord(char) < 32
        or ord(char) == 127
    ]

    if bad:
        raise ValueError(
            f"{name} still contains forbidden HTTP control "
            f"characters after normalization: {bad}"
        )

    return text


def _validate_vectors(
    vectors: list[list[float]],
    *,
    expected_count: int,
    expected_dimensions: int,
) -> list[list[float]]:
    """
    Protect pgvector from malformed or dimension-mismatched vectors.
    """

    if len(vectors) != expected_count:
        raise RuntimeError(
            "Embedding API returned an unexpected number of vectors: "
            f"expected={expected_count}, actual={len(vectors)}"
        )

    validated: list[list[float]] = []

    for index, vector in enumerate(vectors):
        if not isinstance(vector, list):
            raise RuntimeError(
                f"Embedding #{index} is not a list."
            )

        if len(vector) != expected_dimensions:
            raise RuntimeError(
                "Embedding dimension mismatch: "
                f"expected={expected_dimensions}, "
                f"actual={len(vector)}, "
                f"index={index}"
            )

        converted: list[float] = []

        for value in vector:
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Embedding #{index} contains a non-numeric value."
                ) from exc

            if not math.isfinite(number):
                raise RuntimeError(
                    f"Embedding #{index} contains a non-finite value."
                )

            converted.append(number)

        validated.append(converted)

    return validated


# ============================================================
# LOCAL HASH EMBEDDING
# ============================================================

class HashEmbeddingProvider:
    """
    Deterministic development/offline fallback.

    Do not mix these vectors with OpenAI-generated vectors in the same
    retrieval index.
    """

    def __init__(
        self,
        dimensions: int = 256,
    ) -> None:
        dimensions = int(dimensions)

        if dimensions <= 0:
            raise ValueError(
                "Embedding dimensions must be greater than zero."
            )

        self.dimensions = dimensions

    async def embed(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        normalized = _validate_texts(texts)

        if not normalized:
            return []

        return await asyncio.to_thread(
            self._embed_sync,
            normalized,
        )

    async def close(
        self,
    ) -> None:
        return None

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
            text.casefold()
        ):
            digest = hashlib.blake2b(
                token.encode("utf-8"),
                digest_size=8,
            ).digest()

            value = int.from_bytes(
                digest,
                "big",
            )

            index = value % self.dimensions

            sign = (
                1.0
                if ((value >> 1) % 2 == 0)
                else -1.0
            )

            vector[index] += sign

        norm = math.sqrt(
            sum(
                value * value
                for value in vector
            )
        )

        if norm <= 0.0:
            return vector

        return [
            value / norm
            for value in vector
        ]


# ============================================================
# OPENAI-COMPATIBLE EMBEDDINGS
# ============================================================

class OpenAICompatibleEmbeddingProvider:
    """
    Production semantic embedding provider.

    Uses aiohttp directly with a minimal, controlled request:
        POST {base_url}/embeddings

    This avoids the transport/header issue observed on the current Windows
    environment and keeps the existing project interface unchanged.

    Important:
    - credentials are normalized exactly once at this boundary
    - hidden ASCII control characters are removed from the API key
    - only required HTTP headers are sent
    - JSON is serialized by aiohttp's json= parameter
    - VECTOR(256) is enforced by response validation
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int = 256,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        batch_size: int = 64,
    ) -> None:
        base_url = str(
            base_url or ""
        ).strip().rstrip("/")

        api_key = _clean_secret(
            api_key,
            name="Embedding API key",
        )

        model = str(
            model or ""
        ).strip()

        dimensions = int(dimensions)
        max_retries = int(max_retries)
        batch_size = int(batch_size)

        if not base_url:
            raise ValueError(
                "Embedding base URL is required."
            )

        if not model:
            raise ValueError(
                "Embedding model is required."
            )

        if dimensions <= 0:
            raise ValueError(
                "Embedding dimensions must be greater than zero."
            )

        if max_retries <= 0:
            raise ValueError(
                "max_retries must be greater than zero."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if base_url.endswith("/embeddings"):
            base_url = base_url[
                :-len("/embeddings")
            ].rstrip("/")

        self.base_url = base_url
        self.endpoint = f"{base_url}/embeddings"

        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.timeout_seconds = float(
            timeout_seconds
        )

        self._session: aiohttp.ClientSession | None = None

    async def _get_session(
        self,
    ) -> aiohttp.ClientSession:
        if (
            self._session is None
            or self._session.closed
        ):
            timeout = aiohttp.ClientTimeout(
                total=self.timeout_seconds
            )

            authorization = _validate_header_value(
                "Authorization",
                f"Bearer {self.api_key}",
            )

            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "Authorization": authorization,
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                trust_env=False,
            )

        return self._session

    async def close(
        self,
    ) -> None:
        if (
            self._session is not None
            and not self._session.closed
        ):
            await self._session.close()

        self._session = None

    async def embed(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        normalized = _validate_texts(texts)

        if not normalized:
            return []

        all_vectors: list[list[float]] = []

        for start in range(
            0,
            len(normalized),
            self.batch_size,
        ):
            batch = normalized[
                start:
                start + self.batch_size
            ]

            batch_vectors = (
                await self._embed_batch(batch)
            )

            all_vectors.extend(
                batch_vectors
            )

        return _validate_vectors(
            all_vectors,
            expected_count=len(normalized),
            expected_dimensions=self.dimensions,
        )

    async def _embed_batch(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        payload = {
            "model": self.model,
            "input": list(texts),
            "dimensions": self.dimensions,
        }

        last_error: Exception | None = None

        for attempt in range(
            1,
            self.max_retries + 1,
        ):
            try:
                session = await self._get_session()

                async with session.post(
                    self.endpoint,
                    json=payload,
                ) as response:
                    body_text = await response.text()

                    if 200 <= response.status < 300:
                        try:
                            response_json = (
                                await response.json()
                            )
                        except Exception as exc:
                            raise RuntimeError(
                                "Embedding API returned a non-JSON "
                                "success response."
                            ) from exc

                        if not isinstance(
                            response_json,
                            dict,
                        ):
                            raise RuntimeError(
                                "Embedding API response must be "
                                "a JSON object."
                            )

                        data = response_json.get(
                            "data",
                            [],
                        )

                        if (
                            not isinstance(data, list)
                            or not data
                        ):
                            raise RuntimeError(
                                "Embedding API returned no "
                                "embedding data."
                            )

                        try:
                            ordered = sorted(
                                data,
                                key=lambda item: int(
                                    item["index"]
                                ),
                            )

                            vectors = [
                                list(
                                    item["embedding"]
                                )
                                for item in ordered
                            ]

                        except (
                            KeyError,
                            TypeError,
                            ValueError,
                        ) as exc:
                            raise RuntimeError(
                                "Embedding API returned malformed data."
                            ) from exc

                        return _validate_vectors(
                            vectors,
                            expected_count=len(texts),
                            expected_dimensions=self.dimensions,
                        )

                    request_id = (
                        response.headers.get(
                            "x-request-id"
                        )
                        or ""
                    )

                    detail = (
                        "Embedding API returned "
                        f"HTTP {response.status}: "
                        f"{body_text[:4000] if body_text else '<empty response body>'}"
                    )

                    if request_id:
                        detail += (
                            f" | request_id={request_id}"
                        )

                    error = RuntimeError(
                        detail
                    )

                    # These errors will not improve by retrying the exact
                    # same request.
                    if response.status in {
                        400,
                        401,
                        403,
                        404,
                        405,
                        422,
                    }:
                        raise error

                    last_error = error

            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
            ) as exc:
                last_error = RuntimeError(
                    "Could not connect to embedding API: "
                    f"{type(exc).__name__}: {exc}"
                )

            if attempt < self.max_retries:
                await asyncio.sleep(
                    0.75
                    * (
                        2 ** (attempt - 1)
                    )
                )

        raise RuntimeError(
            "Embedding request failed after all retries: "
            f"{last_error}"
        ) from last_error
