from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

# Current PostgreSQL schema stores VECTOR(256).
# Change this only together with an intentional DB vector migration.
DATABASE_VECTOR_DIMENSIONS = 256


class Settings(BaseSettings):
    """
    Central application configuration.

    API/tool behavior must come from configuration + live MCP/OpenAPI
    metadata, not from individual user questions.
    """

    # ========================================================
    # FACTIGENT / OPENAPI SOURCES
    # ========================================================

    # Existing predictive-maintenance backend.
    openapi_url: str = (
        "http://13.203.193.234:30020/openapi.json"
    )
    factigent_api_base_url: str = (
        "http://13.203.193.234:30020"
    )

    # Additional Factigent platform backend.
    # The OpenAPI loader/MCP server can use these to expose Context,
    # Tags, Reports, and future operations discovered from its spec.
    platform_openapi_url: str = (
        "http://151.185.44.114:30100/openapi.json"
    )
    platform_api_base_url: str = (
        "http://151.185.44.114:30100"
    )

    # Optional generic multi-source configuration for future API sources.
    # Keeping API endpoints in configuration means new operations within a
    # configured OpenAPI spec do not require chatbot retrieval/LLM code edits.
    openapi_sources_json: str = (
        "{"
        "\"predictive_maintenance\":{"
        "\"openapi_url\":\"http://13.203.193.234:30020/openapi.json\","
        "\"api_base_url\":\"http://13.203.193.234:30020\""
        "},"
        "\"factigent_platform\":{"
        "\"openapi_url\":\"http://151.185.44.114:30100/openapi.json\","
        "\"api_base_url\":\"http://151.185.44.114:30100\""
        "}"
        "}"
    )

    # ========================================================
    # APPLICATION
    # ========================================================

    app_env: Literal[
        "development",
        "test",
        "production",
    ] = "development"

    log_level: str = "INFO"

    # ========================================================
    # MCP
    # ========================================================

    # Generic multi-server configuration.
    mcp_servers_json: str = (
        '{"factigent_api":"http://127.0.0.1:8001/mcp"}'
    )

    # Compatibility fields used by some existing modules.
    # For multi-server code, prefer the mcp_servers property below.
    mcp_server_name: str = "factigent_api"
    mcp_server_url: str = (
        "http://127.0.0.1:8001/mcp"
    )

    # Correct project-level policy location:
    # C:\...\mcp_tool_broker\config\tool_policies.json
    tool_policies_path: Path = (
        PROJECT_ROOT
        / "config"
        / "tool_policies.json"
    )

    # ========================================================
    # LLM
    # ========================================================

    llm_mode: Literal[
        "mock",
        "openai_compatible",
    ] = "openai_compatible"

    llm_base_url: str = (
        "https://api.openai.com/v1"
    )

    llm_api_key: str = ""

    llm_model: str = "gpt-5"

    llm_max_output_tokens: int = Field(
        default=2000,
        ge=64,
        le=8000,
    )

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    embedding_mode: Literal[
        "hash",
        "openai_compatible",
    ] = "openai_compatible"

    embedding_base_url: str = (
        "https://api.openai.com/v1"
    )

    # May be omitted in .env; the LLM key is reused below.
    embedding_api_key: str = ""

    embedding_model: str = (
        "text-embedding-3-small"
    )

    embedding_dimensions: int = Field(
        default=DATABASE_VECTOR_DIMENSIONS,
        ge=1,
        le=4096,
    )

    embedding_batch_size: int = Field(
        default=64,
        ge=1,
        le=256,
    )

    # ========================================================
    # RETRIEVAL / CONTEXT LIMITS
    # ========================================================

    max_tool_candidates: int = Field(
        default=12,
        ge=1,
        le=50,
    )

    max_tool_schema_tokens: int = Field(
        default=2600,
        ge=500,
    )

    max_history_turns: int = Field(
        default=10,
        ge=0,
        le=30,
    )

    max_history_tokens: int = Field(
        default=2000,
        ge=0,
    )

    max_tool_result_chars: int = Field(
        default=50_000,
        ge=1000,
    )

    max_tool_result_rows: int = Field(
        default=300,
        ge=1,
        le=1000,
    )

    max_final_output_tokens: int = Field(
        default=2000,
        ge=64,
        le=8000,
    )

    admin_token: str = "change-me"

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    @field_validator(
        "openapi_url",
        "factigent_api_base_url",
        "platform_openapi_url",
        "platform_api_base_url",
        "llm_base_url",
        "embedding_base_url",
        "mcp_server_url",
        mode="before",
    )
    @classmethod
    def normalize_url(
        cls,
        value: object,
    ) -> str:
        return str(
            value or ""
        ).strip().rstrip("/")

    @field_validator(
        "llm_api_key",
        "llm_model",
        "embedding_api_key",
        "embedding_model",
        "mcp_server_name",
        mode="before",
    )
    @classmethod
    def strip_string_value(
        cls,
        value: object,
    ) -> str:
        return str(
            value or ""
        ).strip()

    # ========================================================
    # CROSS-FIELD VALIDATION
    # ========================================================

    @model_validator(
        mode="after"
    )
    def validate_embedding_configuration(
        self,
    ) -> "Settings":

        # Reuse the same credential when a separate embedding key
        # is not configured.
        if (
            not self.embedding_api_key
            and self.llm_api_key
        ):
            self.embedding_api_key = (
                self.llm_api_key
            )

        # The current DB column is VECTOR(256). Allowing settings to silently
        # request another dimension would break insert/search operations.
        if (
            self.embedding_dimensions
            != DATABASE_VECTOR_DIMENSIONS
        ):
            raise ValueError(
                "EMBEDDING_DIMENSIONS must be "
                f"{DATABASE_VECTOR_DIMENSIONS} because the current "
                "mcp_tools.embedding column is VECTOR(256). "
                "Perform an explicit DB vector migration before changing it."
            )

        if (
            self.embedding_mode
            == "openai_compatible"
            and not self.embedding_model
        ):
            raise ValueError(
                "EMBEDDING_MODEL is required when "
                "EMBEDDING_MODE=openai_compatible."
            )

        return self

    # ========================================================
    # PARSED CONFIGURATION
    # ========================================================

    @staticmethod
    def _parse_json_object(
        raw_value: str,
        *,
        setting_name: str,
    ) -> dict[str, Any]:
        try:
            value = json.loads(
                raw_value
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{setting_name} must contain valid JSON."
            ) from exc

        if (
            not isinstance(
                value,
                dict,
            )
            or not value
        ):
            raise ValueError(
                f"{setting_name} must be a non-empty JSON object."
            )

        return value

    @property
    def mcp_servers(
        self,
    ) -> dict[str, str]:
        value = self._parse_json_object(
            self.mcp_servers_json,
            setting_name="MCP_SERVERS_JSON",
        )

        result: dict[
            str,
            str,
        ] = {}

        for name, url in value.items():

            if (
                not isinstance(
                    name,
                    str,
                )
                or not isinstance(
                    url,
                    str,
                )
            ):
                raise ValueError(
                    "MCP server names and URLs must be strings."
                )

            clean_name = name.strip()
            clean_url = (
                url.strip()
                .rstrip("/")
            )

            if (
                not clean_name
                or not clean_url
            ):
                raise ValueError(
                    "MCP server names and URLs cannot be empty."
                )

            result[
                clean_name
            ] = clean_url

        return result

    @property
    def openapi_sources(
        self,
    ) -> dict[
        str,
        dict[str, str],
    ]:
        """
        Parsed API-source configuration for the OpenAPI loader/MCP layer.

        Each source has:
            openapi_url
            api_base_url
        """

        value = self._parse_json_object(
            self.openapi_sources_json,
            setting_name="OPENAPI_SOURCES_JSON",
        )

        result: dict[
            str,
            dict[str, str],
        ] = {}

        for source_name, source in (
            value.items()
        ):
            if (
                not isinstance(
                    source_name,
                    str,
                )
                or not source_name.strip()
            ):
                raise ValueError(
                    "OpenAPI source names must be non-empty strings."
                )

            if not isinstance(
                source,
                dict,
            ):
                raise ValueError(
                    "Each OpenAPI source must be a JSON object."
                )

            openapi_url = str(
                source.get(
                    "openapi_url",
                    "",
                )
                or ""
            ).strip().rstrip("/")

            api_base_url = str(
                source.get(
                    "api_base_url",
                    "",
                )
                or ""
            ).strip().rstrip("/")

            if (
                not openapi_url
                or not api_base_url
            ):
                raise ValueError(
                    "Each OpenAPI source requires both "
                    "openapi_url and api_base_url."
                )

            result[
                source_name.strip()
            ] = {
                "openapi_url":
                    openapi_url,
                "api_base_url":
                    api_base_url,
            }

        return result


@lru_cache(
    maxsize=1
)
def get_settings(
) -> Settings:
    return Settings()
