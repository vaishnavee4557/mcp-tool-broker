from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from mcp_broker.models import UserContext
from mcp_broker.request_context import TrustedRequestContext 

@dataclass(frozen=True)
class BrokerChatResult:
    conversation_id: str
    answer: str
    selected_tool: str | None = None
    selected_server: str | None = None


class BrokerService:
    def __init__(
        self,
        *,
        settings: Any,
        conversation_store: Any,
        guardrails: Any,
        gateway: Any,
        retriever: Any,
        llm: Any,
        max_tool_steps: int,
        retrieval_limit: int,
        llm_shortlist_limit: int,
        max_history_turns: int,
        normalize_query_fn: Callable[[str], str],
        build_retrieval_query_fn: Callable[..., str],
        is_read_only_tool_fn: Callable[[Any], bool],
        build_candidate_map_fn: Callable[[list[Any]], dict[str, Any]],
        validate_arguments_fn: Callable[
            [Any, dict[str, Any]],
            tuple[dict[str, Any], list[str]],
        ],
        apply_required_schema_defaults_fn: Callable[
            [Any, dict[str, Any]],
            dict[str, Any],
        ],
        canonicalize_schema_values_fn: Callable[
            [Any, dict[str, Any]],
            dict[str, Any],
        ],
        resolve_required_from_context_fn: Callable[
            ...,
            tuple[
                dict[str, Any],
                list[str],
                list[str],
            ],
        ],
        tool_call_signature_fn: Callable[..., str],
        safe_call_fn: Callable[..., Awaitable[dict[str, Any]]],
        update_validated_context_fn: Callable[
            [
                dict[str, list[Any]],
                dict[str, Any],
            ],
            None,
        ],
        compact_value_fn: Callable[[Any], Any],
    ) -> None:

        self.settings = settings
        self.conversation_store = conversation_store
        self.guardrails = guardrails
        self.gateway = gateway
        self.retriever = retriever
        self.llm = llm

        self.max_tool_steps = max(
            1,
            int(max_tool_steps),
        )

        self.retrieval_limit = max(
            1,
            int(retrieval_limit),
        )

        self.llm_shortlist_limit = max(
            1,
            int(llm_shortlist_limit),
        )

        self.max_history_turns = max(
            1,
            int(max_history_turns),
        )

        self.normalize_query = (
            normalize_query_fn
        )

        self.build_retrieval_query = (
            build_retrieval_query_fn
        )

        self.is_read_only_tool = (
            is_read_only_tool_fn
        )

        self.build_candidate_map = (
            build_candidate_map_fn
        )

        self.validate_arguments = (
            validate_arguments_fn
        )

        self.apply_required_schema_defaults = (
            apply_required_schema_defaults_fn
        )

        self.canonicalize_schema_values = (
            canonicalize_schema_values_fn
        )

        self.resolve_required_from_context = (
            resolve_required_from_context_fn
        )

        self.tool_call_signature = (
            tool_call_signature_fn
        )

        self.safe_call = (
            safe_call_fn
        )

        self.update_validated_context = (
            update_validated_context_fn
        )

        self.compact_value = (
            compact_value_fn
        )

        # Structured context learned from successful MCP results.
        #
        # Key:
        #   (user_id, conversation_id)
        #
        # Value example:
        #   {
        #       "machine": ["MACH-1"],
        #       "zone": ["Zone-1"]
        #   }
        self._conversation_contexts: dict[
            tuple[str, str],
            dict[str, list[Any]],
        ] = {}

    # ========================================================
    # STRUCTURED CONVERSATION CONTEXT
    # ========================================================

    def _context_for(
        self,
        *,
        user_id: str,
        conversation_id: str,
    ) -> dict[str, list[Any]]:

        key = (
            str(user_id).strip(),
            str(conversation_id).strip(),
        )

        return (
            self._conversation_contexts.setdefault(
                key,
                {},
            )
        )

    def forget_conversation_context(
        self,
        *,
        user_id: str,
        conversation_id: str,
    ) -> None:

        key = (
            str(user_id).strip(),
            str(conversation_id).strip(),
        )

        self._conversation_contexts.pop(
            key,
            None,
        )

    # ========================================================
    # MAIN CHAT WORKFLOW
    # ========================================================

    async def chat(
        self,
        *,
        user_id: str,
        query: str,
        conversation_id: str | None = None,
        roles: set[str] | None = None,
        trusted_request_context: (
            TrustedRequestContext | None
        ) = None,
    ) -> BrokerChatResult:

        user_id = str(
            user_id or ""
        ).strip()

        raw_query = str(
            query or ""
        ).strip()

        if not user_id:
            raise ValueError(
                "user_id cannot be empty."
            )

        # ====================================================
        # 1. INPUT GUARDRAILS
        # ====================================================

        input_guard = (
            self.guardrails.validate_input(
                raw_query
            )
        )

        if not input_guard.allowed:

            return BrokerChatResult(
                conversation_id=(
                    conversation_id
                    or ""
                ),
                answer=input_guard.message,
            )

        # ====================================================
        # 2. NORMALIZE QUERY
        # ====================================================

        normalized_query = (
            self.normalize_query(
                input_guard.safe_text
                or raw_query
            )
        )

        # ====================================================
        # 3. CONVERSATION
        # ====================================================

        if conversation_id:

            exists = (
                await self.conversation_store.exists(
                    user_id=user_id,
                    conversation_id=(
                        conversation_id
                    ),
                )
            )

            if not exists:
                raise LookupError(
                    "Conversation not found."
                )

        else:

            conversation_id = (
                await self.conversation_store.create_conversation(
                    user_id=user_id
                )
            )

        assert conversation_id is not None

        # ====================================================
        # 4. LOAD PREVIOUS HISTORY
        # Important: Load history BEFORE saving the current user message.
        # ====================================================

        history = (
            await self.conversation_store.get_recent_turns(
                user_id=user_id,
                conversation_id=conversation_id,
                limit=self.max_history_turns,
            )
        )

        # ====================================================
        # 5. SAVE USER MESSAGE
        # ====================================================

        await self.conversation_store.add_message(
            user_id=user_id,
            conversation_id=conversation_id,
            role="user",
            content=normalized_query,
        )

        # ====================================================
        # 6. GET STRUCTURED CONTEXT
        # ====================================================

        conversation_context = (
            self._context_for(
                user_id=user_id,
                conversation_id = (conversation_id),
            )
        )

        # ====================================================
        # 7. TRUSTED REQUEST CONTEXT
        # ====================================================

        if trusted_request_context is None:

            trusted_request_context = (
                TrustedRequestContext.from_values()
            )

        # ====================================================
        # 8. USER CONTEXT
        # ====================================================

        user = UserContext(
            user_id=user_id,
            roles=set(
                roles or set()
            ),
        )

        # ====================================================
        # 9. MULTI-STEP MCP STATE
        # ====================================================

        tool_context: list[
            dict[str, Any]
        ] = []

        executed_signatures: set[
            str
        ] = set()

        final_answer: str | None = None

        last_tool_reference: (
            str | None
        ) = None

        selected_tool_name: (
            str | None
        ) = None

        selected_server_name: (
            str | None
        ) = None

        # ====================================================
        # 10. MULTI-STEP TOOL LOOP
        # ====================================================

        for step in range(
            1,
            self.max_tool_steps + 1,
        ):

            # -----------------------------------------------
            # BUILD RETRIEVAL QUERY
            # -----------------------------------------------

            retrieval_query = (
                self.build_retrieval_query(
                    original_query=(
                        normalized_query
                    ),
                    tool_context=(
                        tool_context
                    ),
                )
            )

            # -----------------------------------------------
            # RETRIEVE TOOLS
            # -----------------------------------------------

            candidates = (
                await self.retriever.retrieve(
                    query=retrieval_query,
                    user=user,
                    limit=(
                        self.retrieval_limit
                    ),
                )
            )

            # -----------------------------------------------
            # READ-ONLY FILTER + SHORTLIST
            # -----------------------------------------------

            candidates = [
                candidate
                for candidate
                in candidates
                if self.is_read_only_tool(
                    candidate.tool
                )
            ][
                : self.llm_shortlist_limit
            ]

            if not candidates:
                break

            candidate_tools = [
                candidate.tool
                for candidate
                in candidates
            ]

            # -----------------------------------------------
            # LLM CALL #1:
            # SELECT TOOL + ARGUMENTS
            # -----------------------------------------------

            decision = (
                await self.llm.choose_tool(
                    query=normalized_query,
                    history=history,
                    tools=candidate_tools,
                    tool_context=(
                        tool_context
                    ),
                )
            )

            # -----------------------------------------------
            # NO TOOL SELECTED
            # -----------------------------------------------

            if not decision.tool_name:

                final_answer = (
                    "I could not find an appropriate "
                    "authorized API for this request."
                )

                break

            # -----------------------------------------------
            # VERIFY TOOL IS IN AUTHORIZED SHORTLIST
            # -----------------------------------------------

            candidate_map = (
                self.build_candidate_map(
                    candidate_tools
                )
            )

            selected_tool = (
                candidate_map.get(
                    decision.tool_name
                )
            )

            if selected_tool is None:

                tool_context.append(
                    {
                        "success": False,
                        "stage": (
                            "tool_selection"
                        ),
                        "requested_tool": (
                            decision.tool_name
                        ),
                        "error": (
                            "The tool selection was outside "
                            "the authorized retriever shortlist."
                        ),
                    }
                )

                continue

            selected_tool_name = (
                selected_tool.name
            )

            selected_server_name = (
                selected_tool.server_name
            )

            # -----------------------------------------------
            # TRUSTED ROLE / SCOPE INJECTION
            # -----------------------------------------------

            (
                trusted_arguments,
                _injected_fields,
            ) = (
                trusted_request_context.apply_to_tool(
                    selected_tool,
                    decision.arguments or {},
                )
            )

            # -----------------------------------------------
            # REMOVE UNKNOWN ARGUMENTS
            # -----------------------------------------------

            (
                arguments,
                _removed_fields,
            ) = self.validate_arguments(
                selected_tool,
                trusted_arguments,
            )

            # -----------------------------------------------
            # REQUIRED SCHEMA DEFAULTS
            # -----------------------------------------------

            arguments = (
                self.apply_required_schema_defaults(
                    selected_tool,
                    arguments,
                )
            )

            # -----------------------------------------------
            # ENUM CANONICALIZATION
            # -----------------------------------------------

            arguments = (
                self.canonicalize_schema_values(
                    selected_tool,
                    arguments,
                )
            )

            # -----------------------------------------------
            # RESOLVE REQUIRED VALUES FROM VALIDATED CONTEXT
            # -----------------------------------------------

            (
                arguments,
                missing_fields,
                _auto_resolved_fields,
            ) = (
                self.resolve_required_from_context(
                    selected_tool,
                    arguments,
                    conversation_context,
                )
            )

            # -----------------------------------------------
            # REQUIRED VALUES STILL UNKNOWN
            # -----------------------------------------------

            if missing_fields:

                tool_context.append(
                    {
                        "success": False,
                        "stage": (
                            "argument_validation"
                        ),
                        "server": (
                            selected_tool.server_name
                        ),
                        "tool": (
                            selected_tool.name
                        ),
                        "arguments": (
                            arguments
                        ),
                        "missing_required_fields": (
                            missing_fields
                        ),
                        "message": (
                            "Required API fields are not known yet. "
                            "Choose an authorized read-only discovery "
                            "tool to resolve them. Do not guess values."
                        ),
                    }
                )

                continue

            # -----------------------------------------------
            # DUPLICATE CALL SIGNATURE
            # -----------------------------------------------

            try:

                signature = (
                    self.tool_call_signature(
                        selected_tool.server_name,
                        selected_tool.name,
                        arguments,
                    )
                )

            except TypeError:

                # Compatibility with older helper version.
                signature = (
                    self.tool_call_signature(
                        selected_tool.name,
                        arguments,
                    )
                )

            # -----------------------------------------------
            # DUPLICATE CALL PROTECTION
            # -----------------------------------------------

            if (
                signature
                in executed_signatures
            ):

                tool_context.append(
                    {
                        "success": False,
                        "stage": (
                            "duplicate_call_guard"
                        ),
                        "server": (
                            selected_tool.server_name
                        ),
                        "tool": (
                            selected_tool.name
                        ),
                        "arguments": (
                            arguments
                        ),
                        "message": (
                            "The exact same MCP tool call has already "
                            "been executed in this request."
                        ),
                    }
                )

                continue

            executed_signatures.add(
                signature
            )

            # -----------------------------------------------
            # MCP CALL
            # -----------------------------------------------

            raw_result = (
                await self.safe_call(
                    gateway=self.gateway,
                    tool=selected_tool,
                    arguments=arguments,
                )
            )

            last_tool_reference = (
                getattr(
                    selected_tool,
                    "registry_id",
                    None,
                )
                or selected_tool.name
            )

            # -----------------------------------------------
            # UPDATE VALIDATED STRUCTURED CONTEXT
            # -----------------------------------------------

            self.update_validated_context(
                conversation_context,
                raw_result,
            )

            # -----------------------------------------------
            # COMPACT RESULT
            # -----------------------------------------------

            tool_context.append(
                self.compact_value(
                    raw_result
                )
            )

            # -----------------------------------------------
            # DOES THE LLM NEED ANOTHER TOOL?
            # -----------------------------------------------

            if (
                step
                < self.max_tool_steps
            ):

                try:

                    needs_more_tools = (
                        await self.llm.needs_more_tools(
                            query=(
                                normalized_query
                            ),
                            history=history,
                            tool_context=(
                                tool_context
                            ),
                        )
                    )

                except Exception:

                    # Safe behavior:
                    # avoid unnecessary extra API calls.
                    needs_more_tools = False

                if not needs_more_tools:
                    break

        # ====================================================
        # 11. FINAL GROUNDED ANSWER
        # ====================================================

        if final_answer is None:

            if not tool_context:

                final_answer = (
                    "I could not retrieve enough authorized "
                    "Factigent API data to answer this request."
                )

            else:

                final_answer = (
                    await self.llm.answer(
                        query=normalized_query,
                        history=history,
                        tool_name=(
                            last_tool_reference
                            or "factigent_workflow"
                        ),
                        tool_result={
                            "tool_calls": (
                                tool_context
                            )
                        },
                        max_output_tokens=int(
                            getattr(
                                self.settings,
                                "llm_max_output_tokens",
                                2000,
                            )
                            or 2000
                        ),
                    )
                )

        if final_answer is None:
            final_answer = (
                "I could not generate a response."
            )

        # ====================================================
        # 12. OUTPUT GUARDRAILS
        # ====================================================

        output_guard = (
            self.guardrails.validate_output(
                final_answer
            )
        )

        if output_guard.allowed:

            final_answer = (
                output_guard.safe_text
                or final_answer
            )

        else:

            final_answer = (
                output_guard.message
                or final_answer
            )

        # ====================================================
        # 13. SAVE ASSISTANT RESPONSE
        # ====================================================

        await self.conversation_store.add_message(
            user_id=user_id,
            conversation_id=conversation_id,
            role="assistant",
            content=final_answer,
        )

        # ====================================================
        # 14. RETURN RESULT
        # ====================================================

        return BrokerChatResult(
            conversation_id=(
                conversation_id
            ),
            answer=final_answer,
            selected_tool=(
                selected_tool_name
            ),
            selected_server=(
                selected_server_name
            ),
        )

