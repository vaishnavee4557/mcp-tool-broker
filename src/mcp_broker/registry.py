from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mcp_broker.mcp_gateway import MCPGateway
from mcp_broker.models import (
    ToolDescriptor,
    ToolPolicy,
)

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Live MCP tool registry.

    Responsibilities:
    - discover tools from every configured MCP server
    - attach policy metadata
    - build rich generic searchable metadata for retrieval/embeddings
    - keep stable lookup maps for registry_id and llm_name

    Important:
    This class contains no question-specific, machine-specific,
    sensor-specific, site/zone/line/cell-specific, or endpoint-specific
    routing logic.
    """

    def __init__(
        self,
        gateway: MCPGateway,
        policy_path: str | Path,
    ) -> None:
        self.gateway = gateway
        self.policy_path = Path(policy_path)

        self._tools_by_registry_id: dict[
            str,
            ToolDescriptor,
        ] = {}

        self._tools_by_llm_name: dict[
            str,
            ToolDescriptor,
        ] = {}

        self._tools_by_name: dict[
            str,
            ToolDescriptor,
        ] = {}

    @property
    def tools(
        self,
    ) -> list[ToolDescriptor]:
        return list(
            self._tools_by_registry_id.values()
        )

    def get_by_llm_name(
        self,
        name: str,
    ) -> ToolDescriptor | None:
        return self._tools_by_llm_name.get(
            str(name or "").strip()
        )

    def get_by_name(
        self,
        name: str,
    ) -> ToolDescriptor | None:
        return self._tools_by_name.get(
            str(name or "").strip()
        )

    def get_by_registry_id(
        self,
        registry_id: str,
    ) -> ToolDescriptor | None:
        return self._tools_by_registry_id.get(
            str(registry_id or "").strip()
        )

    async def refresh(
        self,
    ) -> list[ToolDescriptor]:
        """
        Discover the current live MCP catalog.

        The live catalog is the source of truth.
        PostgreSQL/pgvector is only a searchable copy of this metadata.
        """
        policies = self._load_policies()

        discovered: list[
            ToolDescriptor
        ] = []

        for server_name in self.gateway.servers:
            logger.info(
                "discovering_tools server=%s",
                server_name,
            )

            server_tools = await self.gateway.list_tools(
                server_name
            )

            for tool in server_tools:
                # Ensure server_name is available when the gateway/model
                # permits assignment.
                current_server_name = str(
                    getattr(
                        tool,
                        "server_name",
                        "",
                    )
                    or ""
                ).strip()

                if (
                    not current_server_name
                    and hasattr(
                        tool,
                        "server_name",
                    )
                ):
                    try:
                        tool.server_name = server_name
                    except Exception:
                        pass

                tool.policy = self._policy_for(
                    tool=tool,
                    policies=policies,
                )

                self._enrich_searchable_text(
                    tool
                )

                discovered.append(
                    tool
                )

        self._validate_unique_tools(
            discovered
        )

        self._tools_by_registry_id = {
            tool.registry_id: tool
            for tool in discovered
        }

        self._tools_by_llm_name = {
            tool.llm_name: tool
            for tool in discovered
            if str(
                getattr(
                    tool,
                    "llm_name",
                    "",
                )
                or ""
            ).strip()
        }

        self._tools_by_name = {
            tool.name: tool
            for tool in discovered
            if str(
                getattr(
                    tool,
                    "name",
                    "",
                )
                or ""
            ).strip()
        }

        logger.info(
            "tool_registry_refreshed count=%s",
            len(discovered),
        )

        return discovered

    # ========================================================
    # POLICY LOADING
    # ========================================================

    def _load_policies(
        self,
    ) -> dict[str, Any]:
        """
        Load policy JSON.

        Missing policy files are allowed so development does not crash,
        but the warning includes the exact resolved path.
        """
        path = self.policy_path

        if not path.is_absolute():
            path = (
                Path.cwd()
                / path
            )

        path = path.resolve()

        self.policy_path = path

        if not path.exists():
            logger.warning(
                "tool_policy_file_missing path=%s",
                path,
            )

            return {
                "defaults": {},
                "tools": {},
            }

        try:
            policy_data = json.loads(
                path.read_text(
                    encoding="utf-8",
                )
            )

        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Invalid JSON in tool policy file: "
                f"{path}"
            ) from exc

        if not isinstance(
            policy_data,
            dict,
        ):
            raise RuntimeError(
                "Tool policy file must contain "
                "a JSON object."
            )

        defaults = policy_data.get(
            "defaults",
            {},
        )

        tools = policy_data.get(
            "tools",
            {},
        )

        if not isinstance(
            defaults,
            dict,
        ):
            raise RuntimeError(
                "'defaults' in tool policy file "
                "must be a JSON object."
            )

        if not isinstance(
            tools,
            dict,
        ):
            raise RuntimeError(
                "'tools' in tool policy file "
                "must be a JSON object."
            )

        return policy_data

    @staticmethod
    def _policy_for(
        tool: ToolDescriptor,
        policies: dict[str, Any],
    ) -> ToolPolicy:
        """
        Resolve policy in this precedence:

            global defaults
            -> server defaults
            -> tool override

        Tool overrides may be keyed by:
            registry_id
            tool.name
            tool.llm_name

        Supporting all three prevents policy loss when generated MCP
        identifiers change format while the underlying API/tool stays the same.
        """
        defaults = policies.get(
            "defaults",
            {},
        )

        if not isinstance(
            defaults,
            dict,
        ):
            defaults = {}

        tool_overrides = policies.get(
            "tools",
            {},
        )

        if not isinstance(
            tool_overrides,
            dict,
        ):
            tool_overrides = {}

        server_name = str(
            getattr(
                tool,
                "server_name",
                "",
            )
            or ""
        ).strip()

        # Some policy files store ToolPolicy fields directly in defaults.
        # Others store them under defaults.<server_name>.
        policy_fields = set(
            ToolPolicy.model_fields.keys()
        )

        global_defaults = {
            key: value
            for key, value in defaults.items()
            if key in policy_fields
        }

        raw_server_defaults = defaults.get(
            server_name,
            {},
        )

        server_defaults = (
            raw_server_defaults
            if isinstance(
                raw_server_defaults,
                dict,
            )
            else {}
        )

        override: dict[
            str,
            Any,
        ] = {}

        lookup_keys = [
            str(
                getattr(
                    tool,
                    "registry_id",
                    "",
                )
                or ""
            ).strip(),
            str(
                getattr(
                    tool,
                    "name",
                    "",
                )
                or ""
            ).strip(),
            str(
                getattr(
                    tool,
                    "llm_name",
                    "",
                )
                or ""
            ).strip(),
        ]

        for key in lookup_keys:
            if not key:
                continue

            candidate = tool_overrides.get(
                key
            )

            if isinstance(
                candidate,
                dict,
            ):
                override = candidate
                break

        merged_policy = {
            **global_defaults,
            **server_defaults,
            **override,
        }

        return ToolPolicy.model_validate(
            merged_policy
        )

    # ========================================================
    # SEARCHABLE METADATA
    # ========================================================

    @classmethod
    def _enrich_searchable_text(
        cls,
        tool: ToolDescriptor,
    ) -> None:
        """
        Build retrieval/embedding text from live tool metadata.

        This is the important bridge between MCP/OpenAPI and retrieval.

        It includes:
        - registry/tool/LLM names
        - description
        - complete input schema field names
        - schema descriptions/types/enums
        - required-field names
        - policy domain/keywords

        No query-specific synonyms or API-specific aliases are added.
        """
        parts: list[str] = []

        for attr in (
            "registry_id",
            "name",
            "llm_name",
            "description",
        ):
            value = getattr(
                tool,
                attr,
                None,
            )

            if value is not None:
                text = str(
                    value
                ).strip()

                if text:
                    parts.append(
                        text
                    )

        existing_searchable_text = str(
            getattr(
                tool,
                "searchable_text",
                "",
            )
            or ""
        ).strip()

        if existing_searchable_text:
            parts.append(
                existing_searchable_text
            )

        input_schema = getattr(
            tool,
            "input_schema",
            None,
        )

        cls._collect_schema_metadata(
            input_schema,
            parts,
        )

        policy = getattr(
            tool,
            "policy",
            None,
        )

        if policy is not None:
            domain = str(
                getattr(
                    policy,
                    "domain",
                    "",
                )
                or ""
            ).strip()

            if domain:
                parts.append(
                    domain
                )

            keywords = getattr(
                policy,
                "keywords",
                [],
            )

            if isinstance(
                keywords,
                (
                    list,
                    tuple,
                    set,
                ),
            ):
                parts.extend(
                    str(item).strip()
                    for item in keywords
                    if str(
                        item
                    ).strip()
                )

        # Stable de-duplication while preserving original order.
        unique_parts = list(
            dict.fromkeys(
                part
                for part in parts
                if part
            )
        )

        searchable_text = " ".join(
            unique_parts
        )

        # ToolDescriptor in this project exposes searchable_text.
        # Keep this guarded so registry discovery does not fail if a future
        # descriptor version makes it optional/frozen.
        try:
            tool.searchable_text = searchable_text
        except Exception:
            logger.debug(
                "tool_searchable_text_not_assignable tool=%s",
                getattr(
                    tool,
                    "name",
                    "<unknown>",
                ),
            )

    @classmethod
    def _collect_schema_metadata(
        cls,
        value: Any,
        parts: list[str],
        *,
        depth: int = 0,
    ) -> None:
        """
        Recursively flatten useful JSON-schema/OpenAPI metadata.

        This is generic and therefore works for future Context, Tag, Report,
        machine, hierarchy, and other APIs without code changes.
        """
        if depth > 12:
            return

        if isinstance(
            value,
            dict,
        ):
            for field in (
                "title",
                "description",
                "type",
                "format",
            ):
                item = value.get(
                    field
                )

                if (
                    isinstance(
                        item,
                        str,
                    )
                    and item.strip()
                ):
                    parts.append(
                        item.strip()
                    )

            required = value.get(
                "required"
            )

            if isinstance(
                required,
                list,
            ):
                for item in required:
                    if isinstance(
                        item,
                        str,
                    ):
                        parts.append(
                            item
                        )

            enum_values = value.get(
                "enum"
            )

            if isinstance(
                enum_values,
                list,
            ):
                for item in enum_values:
                    if isinstance(
                        item,
                        (
                            str,
                            int,
                            float,
                            bool,
                        ),
                    ):
                        parts.append(
                            str(item)
                        )

            properties = value.get(
                "properties"
            )

            if isinstance(
                properties,
                dict,
            ):
                for (
                    property_name,
                    property_schema,
                ) in properties.items():
                    parts.append(
                        str(
                            property_name
                        )
                    )

                    cls._collect_schema_metadata(
                        property_schema,
                        parts,
                        depth=depth + 1,
                    )

            items = value.get(
                "items"
            )

            if items is not None:
                cls._collect_schema_metadata(
                    items,
                    parts,
                    depth=depth + 1,
                )

            additional_properties = value.get(
                "additionalProperties"
            )

            if isinstance(
                additional_properties,
                (
                    dict,
                    list,
                ),
            ):
                cls._collect_schema_metadata(
                    additional_properties,
                    parts,
                    depth=depth + 1,
                )

            for key in (
                "allOf",
                "anyOf",
                "oneOf",
                "prefixItems",
            ):
                children = value.get(
                    key
                )

                if isinstance(
                    children,
                    list,
                ):
                    for child in children:
                        cls._collect_schema_metadata(
                            child,
                            parts,
                            depth=depth + 1,
                        )

        elif isinstance(
            value,
            list,
        ):
            for child in value:
                cls._collect_schema_metadata(
                    child,
                    parts,
                    depth=depth + 1,
                )

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _validate_unique_tools(
        tools: list[
            ToolDescriptor
        ],
    ) -> None:
        registry_ids = [
            str(
                getattr(
                    tool,
                    "registry_id",
                    "",
                )
                or ""
            ).strip()
            for tool in tools
        ]

        if any(
            not item
            for item in registry_ids
        ):
            raise RuntimeError(
                "A discovered MCP tool is missing registry_id."
            )

        if (
            len(registry_ids)
            != len(
                set(
                    registry_ids
                )
            )
        ):
            raise RuntimeError(
                "Duplicate registry tool IDs were discovered."
            )

        llm_names = [
            str(
                getattr(
                    tool,
                    "llm_name",
                    "",
                )
                or ""
            ).strip()
            for tool in tools
        ]

        non_empty_llm_names = [
            name
            for name in llm_names
            if name
        ]

        if (
            len(
                non_empty_llm_names
            )
            != len(
                set(
                    non_empty_llm_names
                )
            )
        ):
            raise RuntimeError(
                "Tool aliases collide after sanitization."
            )

# from __future__ import annotations

# import json
# import logging
# from pathlib import Path
# from typing import Any

# from mcp_broker.mcp_gateway import MCPGateway
# from mcp_broker.models import ToolDescriptor, ToolPolicy

# logger = logging.getLogger(__name__)


# class ToolRegistry:
#     def __init__(self, gateway: MCPGateway, policy_path: Path) -> None:
#         self.gateway = gateway
#         self.policy_path = policy_path
#         self._tools_by_registry_id: dict[str, ToolDescriptor] = {}
#         self._tools_by_llm_name: dict[str, ToolDescriptor] = {}

#     @property
#     def tools(self) -> list[ToolDescriptor]:
#         return list(self._tools_by_registry_id.values())

#     def get_by_llm_name(self, name: str) -> ToolDescriptor | None:
#         return self._tools_by_llm_name.get(name)

#     async def refresh(self) -> list[ToolDescriptor]:
#         policies = self._load_policies()
#         discovered: list[ToolDescriptor] = []
#         for server_name in self.gateway.servers:
#             server_tools = await self.gateway.list_tools(server_name)
#             for tool in server_tools:
#                 tool.policy = self._policy_for(tool, policies)
#                 discovered.append(tool)

#         registry_ids = [tool.registry_id for tool in discovered]
#         if len(registry_ids) != len(set(registry_ids)):
#             raise RuntimeError("Duplicate registry tool IDs were discovered")
#         llm_names = [tool.llm_name for tool in discovered]
#         if len(llm_names) != len(set(llm_names)):
#             raise RuntimeError("Tool aliases collide after sanitization")

#         self._tools_by_registry_id = {tool.registry_id: tool for tool in discovered}
#         self._tools_by_llm_name = {tool.llm_name: tool for tool in discovered}
#         logger.info("tool_registry_refreshed count=%s", len(discovered))
#         return discovered

#     def _load_policies(self) -> dict[str, Any]:
#         if not self.policy_path.exists():
#             logger.warning("tool_policy_file_missing path=%s", self.policy_path)
#             return {"defaults": {}, "tools": {}}
#         return json.loads(self.policy_path.read_text(encoding="utf-8"))

#     @staticmethod
#     def _policy_for(tool: ToolDescriptor, policies: dict[str, Any]) -> ToolPolicy:
#         server_defaults = policies.get("defaults", {}).get(tool.server_name, {})
#         override = policies.get("tools", {}).get(tool.registry_id, {})
#         merged = {**server_defaults, **override}
#         return ToolPolicy.model_validate(merged)
