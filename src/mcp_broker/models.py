from __future__ import annotations

import hashlib
import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


class Operation(StrEnum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class ToolPolicy(BaseModel):
    model_config = ConfigDict(extra="ignore")

    domain: str = "general"
    operation: Operation = Operation.READ
    required_scopes: set[str] = Field(default_factory=set)
    required_roles: set[str] = Field(default_factory=set)
    keywords: list[str] = Field(default_factory=list)
    summary: str | None = None
    enabled: bool = True

    # Explicit override for read-only POST/report/search tools.
    read_only: bool | None = None


class ToolDescriptor(BaseModel):
    """
    Normalized live MCP/OpenAPI tool.

    searchable_text is writable because registry.py enriches it before
    persistence/embedding.
    """

    model_config = ConfigDict(
        extra="ignore",
        validate_assignment=True,
    )

    server_name: str
    name: str
    description: str = ""

    input_schema: dict[str, Any] = Field(default_factory=dict)

    # Optional response metadata when provided by the MCP/OpenAPI adapter.
    output_schema: dict[str, Any] | None = None
    response_schema: dict[str, Any] | None = None

    # MCP annotations such as readOnlyHint.
    annotations: dict[str, Any] | None = None

    policy: ToolPolicy = Field(default_factory=ToolPolicy)

    base_url: str | None = None
    http_method: str | None = None
    api_path: str | None = None

    # Populated/enriched by registry.py.
    searchable_text: str = ""

    @computed_field
    @property
    def registry_id(self) -> str:
        return f"{self.server_name}.{self.name}"

    @computed_field
    @property
    def llm_name(self) -> str:
        """
        Produce a stable tool alias no longer than 64 characters.

        Long names receive a hash suffix to avoid collisions caused by
        simple truncation.
        """

        raw = f"{self.server_name}__{self.name}"

        clean = re.sub(
            r"[^a-zA-Z0-9_-]",
            "_",
            raw,
        )

        if len(clean) <= 64:
            return clean

        suffix = hashlib.sha1(
            clean.encode("utf-8")
        ).hexdigest()[:10]

        prefix_length = 64 - len(suffix) - 1

        return (
            clean[:prefix_length]
            + "_"
            + suffix
        )

    @property
    def effective_read_only(self) -> bool | None:
        """
        Return an explicit read-only hint when available.

        None means the caller should use its conservative fallback.
        """

        if self.policy.read_only is not None:
            return self.policy.read_only

        if isinstance(self.annotations, dict):
            for key in (
                "readOnlyHint",
                "read_only",
                "is_read_only",
            ):
                value = self.annotations.get(key)

                if isinstance(value, bool):
                    return value

        if self.http_method:
            method = self.http_method.strip().upper()

            if method in {
                "GET",
                "HEAD",
                "OPTIONS",
            }:
                return True

        return None

    def as_llm_tool(self) -> dict[str, Any]:
        """
        Tool schema exposed to the LLM.

        Only the live input schema controls which parameters are valid.
        Optional site/zone/line/cell fields therefore remain optional.
        """

        parameters = (
            self.input_schema
            or {
                "type": "object",
                "properties": {},
            }
        )

        return {
            "type": "function",
            "function": {
                "name": self.llm_name,
                "description": (
                    self.policy.summary
                    or self.description
                    or self.name
                ),
                "parameters": parameters,
            },
        }


class UserContext(BaseModel):
    user_id: str
    tenant_id: str = "default"
    roles: set[str] = Field(default_factory=set)
    scopes: set[str] = Field(default_factory=set)
    allow_write: bool = False


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=20_000,
    )
    session_id: str = Field(
        min_length=1,
        max_length=200,
    )


class CandidateInfo(BaseModel):
    tool: str
    domain: str
    score: float


class TokenStats(BaseModel):
    estimated_history_tokens: int = 0
    estimated_tool_schema_tokens: int = 0
    estimated_tool_result_tokens: int = 0


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    selected_tool: str | None = None
    selected_server: str | None = None
    candidates: list[CandidateInfo] = Field(default_factory=list)
    token_estimates: TokenStats = Field(default_factory=TokenStats)
    latency_ms: float


class RetrievalCandidate(BaseModel):
    tool: ToolDescriptor
    score: float
    lexical_score: float
    embedding_score: float
    domain_score: float


class ToolDecision(BaseModel):
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    direct_answer: str | None = None


class ToolExecution(BaseModel):
    descriptor: ToolDescriptor
    arguments: dict[str, Any]
    projected_result: Any