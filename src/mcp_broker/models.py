from __future__ import annotations

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


class ToolDescriptor(BaseModel):
    server_name: str
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    policy: ToolPolicy = Field(default_factory=ToolPolicy)
    base_url: str | None = None
    http_method: str | None = None
    api_path: str | None = None
    @computed_field
    @property
    def registry_id(self) -> str:
        return f"{self.server_name}.{self.name}"

    @computed_field
    @property
    def llm_name(self) -> str:
        raw = f"{self.server_name}__{self.name}"
        clean = re.sub(r"[^a-zA-Z0-9_-]", "_", raw)
        return clean[:64]

    @computed_field
    @property
    def searchable_text(self) -> str:
        summary = self.policy.summary or self.description
        return " ".join(
            [
                self.name.replace("_", " "),
                summary,
                self.policy.domain.replace("_", " "),
                " ".join(self.policy.keywords),
            ]
        ).strip()

    def as_llm_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.llm_name,
                "description": self.policy.summary or self.description or self.name,
                "parameters": self.input_schema or {"type": "object", "properties": {}},
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
    query: str = Field(min_length=1, max_length=20_000)
    session_id: str = Field(min_length=1, max_length=200)


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
