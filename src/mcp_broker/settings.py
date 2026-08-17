from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# .env is located beside settings.py:
# src/mcp_broker/.env
ENV_FILE = Path(__file__).resolve().parents[1] / "mcp_broker.env"
#ENV_FILE = Path(__file__).resolve().parent / ".env"
#ENV_FILE = PROJECT_ROOT / "src" / "mcp_broker.env"

class Settings(BaseSettings):
    # Swagger / Factigent configuration
    openapi_url: str = "http://151.185.44.114:30020/openapi.json"

    factigent_api_base_url: str = "http://151.185.44.114:30020"

    # Application configuration
    app_env: Literal[
        "development",
        "test",
        "production",
    ] = "development"

    log_level: str = "INFO"

    # MCP configuration
    mcp_servers_json: str = (
        '{"factigent_api":"http://127.0.0.1:8001/mcp"}'
    )

    tool_policies_path: Path = Path(
        "src/mcp_broker/tool_policies.json"
    )

    # LLM configuration
    llm_mode: Literal[
        "mock",
        "openai_compatible",
    ] = "mock"

    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "replace-with-your-tool-calling-model"

    # Your CLI expects this exact field name.
    llm_max_output_tokens: int = Field(
        default=500,
        ge=64,
        le=4096,
    )

    # Embedding configuration

    embedding_mode: Literal[
    "hash",
    "openai_compatible",
           ] = "openai_compatible"

    embedding_base_url: str = "https://api.openai.com/v1"

    embedding_api_key: str = ""

    embedding_model: str = "text-embedding-3-small"

    embedding_dimensions: int = Field(
    default=1536,
    ge=64,
    le=4096,
)
    

    # Retrieval and token limits
    max_tool_candidates: int = Field(
        default=8,
        ge=1,
        le=30,
    )

    max_tool_schema_tokens: int = Field(
        default=2600,
        ge=500,
    )

    max_history_turns: int = Field(
        default=6,
        ge=0,
        le=30,
    )

    max_history_tokens: int = Field(
        default=1600,
        ge=0,
    )

    max_tool_result_chars: int = Field(
        default=12_000,
        ge=1000,
    )

    max_tool_result_rows: int = Field(
        default=20,
        ge=1,
        le=500,
    )

    max_final_output_tokens: int = Field(
        default=500,
        ge=64,
        le=4096,
    )

    admin_token: str = "change-me"

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator(
        "llm_base_url",
        "embedding_base_url",
    )
    @classmethod
    def strip_trailing_slash(
        cls,
        value: str,
    ) -> str:
        return value.rstrip("/")

    @property
    def mcp_servers(self) -> dict[str, str]:
        try:
            value = json.loads(
                self.mcp_servers_json
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                "MCP_SERVERS_JSON must contain valid JSON"
            ) from exc

        if not isinstance(value, dict) or not value:
            raise ValueError(
                "MCP_SERVERS_JSON must be a "
                "non-empty JSON object"
            )

        result: dict[str, str] = {}

        for name, url in value.items():
            if (
                not isinstance(name, str)
                or not isinstance(url, str)
            ):
                raise ValueError(
                    "MCP server names and URLs "
                    "must be strings"
                )

            result[name] = url.rstrip("/")

        return result


@lru_cache(maxsize=1)
def get_settings() -> Settings:
 return Settings()