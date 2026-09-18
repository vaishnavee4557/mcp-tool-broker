from __future__ import annotations

from mcp_broker.models import (
    Operation,
    ToolDescriptor,
    UserContext,
)


class AccessPolicy:
    """
    Authorization boundary applied before a tool can reach the LLM.

    This class does not decide tool relevance and does not force
    hierarchy values such as site, zone, line, or cell.
    """

    @staticmethod
    def is_allowed(
        user: UserContext,
        tool: ToolDescriptor,
    ) -> bool:
        if user is None or tool is None:
            return False

        policy = getattr(
            tool,
            "policy",
            None,
        )

        if policy is None:
            return False

        # Disabled tools are never visible.
        if not bool(
            getattr(
                policy,
                "enabled",
                True,
            )
        ):
            return False

        # WRITE / ADMIN tools require explicit permission.
        operation = getattr(
            policy,
            "operation",
            Operation.READ,
        )

        if (
            operation
            in {
                Operation.WRITE,
                Operation.ADMIN,
            }
            and not bool(
                getattr(
                    user,
                    "allow_write",
                    False,
                )
            )
        ):
            return False

        # Scopes use all-of semantics.
        required_scopes = set(
            getattr(
                policy,
                "required_scopes",
                set(),
            )
            or set()
        )

        user_scopes = set(
            getattr(
                user,
                "scopes",
                set(),
            )
            or set()
        )

        if not required_scopes.issubset(
            user_scopes
        ):
            return False

        # Roles use any-of semantics, matching the original behavior.
        required_roles = set(
            getattr(
                policy,
                "required_roles",
                set(),
            )
            or set()
        )

        user_roles = set(
            getattr(
                user,
                "roles",
                set(),
            )
            or set()
        )

        if (
            required_roles
            and required_roles.isdisjoint(
                user_roles
            )
        ):
            return False

        return True
