from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any


USER_ROLES_HEADER = "X-USER-ROLES"
USER_SCOPE_HEADER = "X-USER-SCOPE-CONTEXT"

_PROTECTED_HEADERS = {
    USER_ROLES_HEADER,
    USER_SCOPE_HEADER,
}


def _normalize_name(value: str) -> str:
    """Match header/schema names across hyphen/snake/case variants."""

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value or "").casefold(),
    )


def _compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class TrustedRequestContext:
    """
    Trusted per-user API request context.

    The LLM may choose normal API arguments such as site/zone/line, but it
    must not choose authorization headers. Those values come from this
    trusted context and are injected only when the selected live MCP tool
    declares matching input fields.

    For the current CLI, values are read from environment variables. In a
    future HTTP/UI layer, the same object can be created from authenticated
    user/session data without changing retrieval or tool-selection code.
    """

    headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_values(
        cls,
        *,
        user_roles: str = "",
        user_scope_context: Any = None,
    ) -> "TrustedRequestContext":
        headers: dict[str, str] = {}

        clean_roles = str(user_roles or "").strip()
        if clean_roles:
            headers[USER_ROLES_HEADER] = clean_roles

        if user_scope_context not in (None, ""):
            if isinstance(user_scope_context, str):
                raw_scope = user_scope_context.strip()
                if raw_scope:
                    try:
                        parsed_scope = json.loads(raw_scope)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            "FACTIGENT_USER_SCOPE_CONTEXT must contain valid JSON."
                        ) from exc
                else:
                    parsed_scope = None
            else:
                parsed_scope = user_scope_context

            if parsed_scope is not None:
                if not isinstance(parsed_scope, (list, dict)):
                    raise ValueError(
                        "FACTIGENT_USER_SCOPE_CONTEXT must be a JSON array or object."
                    )

                headers[USER_SCOPE_HEADER] = _compact_json(parsed_scope)

        return cls(headers=headers)

    @classmethod
    def from_environment(cls) -> "TrustedRequestContext":
        """
        CLI adapter only.

        Production can build TrustedRequestContext from authenticated request
        data instead of environment variables.
        """

        return cls.from_values(
            user_roles=os.getenv(
                "FACTIGENT_USER_ROLES",
                "",
            ),
            user_scope_context=os.getenv(
                "FACTIGENT_USER_SCOPE_CONTEXT",
                "",
            ),
        )

    @property
    def configured_header_names(self) -> list[str]:
        return sorted(self.headers)

    def apply_to_tool(
        self,
        tool: Any,
        arguments: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], list[str]]:
        """
        Merge trusted headers into one selected tool call.

        Security rules:
        - site/zone/line/cell/machine/date arguments from the LLM are preserved.
        - protected user-role/scope values supplied by the LLM are removed.
        - trusted values override only matching fields declared by the live
          tool schema.
        - no unknown fields are added to a tool call.
        """

        result = dict(arguments or {})

        input_schema = getattr(
            tool,
            "input_schema",
            {},
        ) or {}

        properties = input_schema.get(
            "properties",
            {},
        )

        if not isinstance(properties, dict):
            return result, []

        normalized_properties: dict[str, list[str]] = {}
        for property_name in properties:
            normalized_properties.setdefault(
                _normalize_name(str(property_name)),
                [],
            ).append(str(property_name))

        # Never trust authorization/scope values proposed by the LLM.
        for protected_header in _PROTECTED_HEADERS:
            matches = normalized_properties.get(
                _normalize_name(protected_header),
                [],
            )
            if len(matches) == 1:
                result.pop(matches[0], None)

        injected_fields: list[str] = []

        for header_name, header_value in self.headers.items():
            matches = normalized_properties.get(
                _normalize_name(header_name),
                [],
            )

            # Do not guess when a malformed schema exposes ambiguous aliases.
            if len(matches) != 1:
                continue

            schema_field = matches[0]
            result[schema_field] = header_value
            injected_fields.append(schema_field)
            
            
            

        return result, injected_fields 
    