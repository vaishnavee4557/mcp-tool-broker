from __future__ import annotations

from mcp_broker.models import Operation, ToolDescriptor, UserContext


class AccessPolicy:
    """Applies authorization before a tool becomes visible to the LLM."""

    @staticmethod
    def is_allowed(user: UserContext, tool: ToolDescriptor) -> bool:
        policy = tool.policy
        if not policy.enabled:
            return False
        if policy.operation in {Operation.WRITE, Operation.ADMIN} and not user.allow_write:
            return False
        if not policy.required_scopes.issubset(user.scopes):
            return False
        if policy.required_roles and not policy.required_roles.intersection(user.roles):
            return False
        return True
