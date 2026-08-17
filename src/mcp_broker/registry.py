from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mcp_broker.mcp_gateway import MCPGateway
from mcp_broker.models import ToolDescriptor, ToolPolicy

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(
        self,
        gateway: MCPGateway,
        policy_path: str | Path,
    ) -> None:
        self.gateway = gateway

        # Always convert the supplied value into a Path object.
        self.policy_path = Path(policy_path)

        self._tools_by_registry_id: dict[str, ToolDescriptor] = {}
        self._tools_by_llm_name: dict[str, ToolDescriptor] = {}

    @property
    def tools(self) -> list[ToolDescriptor]:
        return list(self._tools_by_registry_id.values())

    def get_by_llm_name(
        self,
        name: str,
    ) -> ToolDescriptor | None:
        return self._tools_by_llm_name.get(name)

    async def refresh(self) -> list[ToolDescriptor]:
        policies = self._load_policies()
        discovered: list[ToolDescriptor] = []

        for server_name in self.gateway.servers:
            logger.info(
                "discovering_tools server=%s",
                server_name,
            )

            server_tools = await self.gateway.list_tools(server_name)

            for tool in server_tools:
                tool.policy = self._policy_for(
                    tool=tool,
                    policies=policies,
                )
                discovered.append(tool)

        registry_ids = [
            tool.registry_id
            for tool in discovered
        ]

        if len(registry_ids) != len(set(registry_ids)):
            raise RuntimeError(
                "Duplicate registry tool IDs were discovered"
            )

        llm_names = [
            tool.llm_name
            for tool in discovered
        ]

        if len(llm_names) != len(set(llm_names)):
            raise RuntimeError(
                "Tool aliases collide after sanitization"
            )

        self._tools_by_registry_id = {
            tool.registry_id: tool
            for tool in discovered
        }

        self._tools_by_llm_name = {
            tool.llm_name: tool
            for tool in discovered
        }

        logger.info(
            "tool_registry_refreshed count=%s",
            len(discovered),
        )

        return discovered

    def _load_policies(self) -> dict[str, Any]:
        if not self.policy_path.exists():
            logger.warning(
                "tool_policy_file_missing path=%s",
                self.policy_path,
            )

            return {
                "defaults": {},
                "tools": {},
            }

        try:
            policy_data = json.loads(
                self.policy_path.read_text(
                    encoding="utf-8",
                )
            )
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid JSON in policy file: "
                f"{self.policy_path}"
            ) from exc

        if not isinstance(policy_data, dict):
            raise RuntimeError(
                "Tool policy file must contain a JSON object"
            )

        return policy_data

    @staticmethod
    def _policy_for(
        tool: ToolDescriptor,
        policies: dict[str, Any],
    ) -> ToolPolicy:
        server_defaults = (
            policies
            .get("defaults", {})
            .get(tool.server_name, {})
        )

        tool_override = (
            policies
            .get("tools", {})
            .get(tool.registry_id, {})
        )

        merged_policy = {
            **server_defaults,
            **tool_override,
        }

        return ToolPolicy.model_validate(
            merged_policy
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
