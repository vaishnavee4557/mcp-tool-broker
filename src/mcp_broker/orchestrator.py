from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from mcp_broker.llm import ChatModel
from mcp_broker.mcp_gateway import MCPGateway
from mcp_broker.memory import InMemoryConversationStore
from mcp_broker.models import (
    CandidateInfo,
    ChatRequest,
    ChatResponse,
    ChatTurn,
    TokenStats,
    UserContext,
)
from mcp_broker.registry import ToolRegistry
from mcp_broker.result_projector import ResultProjector
from mcp_broker.retrieval import HybridToolRetriever
from mcp_broker.token_budget import (
    TokenBudgeter,
    estimate_tokens,
)


logger = logging.getLogger(__name__)


class ToolBrokerOrchestrator:
    """
    Handles the complete chatbot request flow.

    Flow:

        User request
            ↓
        Load user's conversation history
            ↓
        Select allowed conversation history
            ↓
        Retrieve relevant MCP tools
            ↓
        Apply token budget
            ↓
        Ask LLM which tool to use
            ↓
        Validate tool arguments
            ↓
        Call MCP tool
            ↓
        Project / limit MCP response
            ↓
        Generate final LLM answer
            ↓
        Save conversation for this user
            ↓
        Return response

    Conversation history is isolated using:

        user_id + session_id

    Therefore, one user's conversation cannot be loaded
    through another user's session.
    """

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        retriever: HybridToolRetriever,
        gateway: MCPGateway,
        model: ChatModel,
        memory: InMemoryConversationStore,
        budgeter: TokenBudgeter,
        projector: ResultProjector,
        max_candidates: int,
        max_output_tokens: int,
    ) -> None:

        # MCP tool registry.
        self.registry = registry

        # Finds the most relevant tools for a user question.
        self.retriever = retriever

        # Actually communicates with the MCP server.
        self.gateway = gateway

        # LLM used for tool selection and final answer generation.
        self.model = model

        # Stores conversation history.
        self.memory = memory

        # Prevents too much conversation/tool information
        # from being sent to the LLM.
        self.budgeter = budgeter

        # Reduces / sanitizes MCP tool output before
        # giving it to the LLM.
        self.projector = projector

        # Maximum number of tool candidates exposed to the LLM.
        self.max_candidates = max(
            1,
            max_candidates,
        )

        self.max_output_tokens = max_output_tokens

    async def handle(
        self,
        request: ChatRequest,
        user: UserContext,
    ) -> ChatResponse:
        """
        Main entry point for every chat request.

        request contains:
            session_id
            query

        user contains:
            user_id
            roles / permissions

        Both user_id and session_id are used to isolate
        conversation history.
        """

        started = time.perf_counter()

        # Unique identifier for this individual API request.
        request_id = uuid.uuid4().hex

        # ====================================================
        # 1. LOAD USER-SPECIFIC CONVERSATION HISTORY
        # ====================================================

        full_history = await self.memory.get(
            user.user_id,
            request.session_id,
        )

        # We don't necessarily send the entire conversation
        # history to the LLM.
        #
        # TokenBudgeter selects only the portion that fits
        # within the configured token budget.
        history, history_tokens = (
            self.budgeter.select_history(
                full_history,
                request.query,
            )
        )

        # ====================================================
        # 2. RETRIEVE RELEVANT MCP TOOLS
        # ====================================================

        candidates = await self.retriever.retrieve(
            request.query,
            user,
            limit=self.max_candidates,
        )

        # ====================================================
        # 3. APPLY TOOL TOKEN BUDGET
        # ====================================================

        selected_tools, schema_tokens = (
            self.budgeter.select_tools(
                candidates
            )
        )

        logger.info(
            (
                "retrieval_completed "
                "request_id=%s "
                "user_id=%s "
                "session_id=%s "
                "candidates=%s "
                "selected=%s"
            ),
            request_id,
            user.user_id,
            request.session_id,
            len(candidates),
            len(selected_tools),
        )

        # ====================================================
        # 4. NO TOOL AVAILABLE
        # ====================================================

        if not selected_tools:

            answer = (
                "No authorized tool is available "
                "for this request."
            )

            await self._save(
                user.user_id,
                request.session_id,
                request.query,
                answer,
            )

            return self._response(
                request_id=request_id,
                answer=answer,
                candidates=candidates,
                history_tokens=history_tokens,
                schema_tokens=schema_tokens,
                result_tokens=0,
                started=started,
            )

        # ====================================================
        # 5. ASK LLM TO SELECT A TOOL
        # ====================================================

        decision = await self.model.choose_tool(
            query=request.query,
            history=history,
            tools=selected_tools,
        )


        if not decision.tool_name:
            answer = (
                " I can only answer questions related to the tools available factigent apis and supported machines data "
            )
            await self._save(
                user.user_id,
                request.session_id,
                request.query,
                answer,
            )
            return self._response(
                request_id = request_id,
                answer = answer,
                candidates = candidates,
                history_tokens = history_tokens,
                schema_tokens = schema_tokens,
                result_tokens = 0,
                started = started,
            )
        # ====================================================
        # 7. FIND SELECTED TOOL
        # ====================================================

        descriptor = (
            self.registry.get_by_llm_name(
                decision.tool_name
            )
        )

        # Only tools that were exposed to the LLM
        # are allowed to be executed.
        visible_names = {
            tool.llm_name
            for tool in selected_tools
        }

        if (
            descriptor is None
            or descriptor.llm_name
            not in visible_names
        ):
            raise RuntimeError(
                (
                    "Model attempted to call a tool "
                    "outside the authorized candidate set"
                )
            )

        # ====================================================
        # 8. VALIDATE TOOL ARGUMENTS
        # ====================================================

        arguments = decision.arguments or {}

        self._validate_arguments(
            descriptor.input_schema,
            arguments,
        )

        # ====================================================
        # 9. CALL MCP TOOL
        # ====================================================

        raw_result = await self.gateway.call_tool(
            descriptor,
            arguments,
        )

        # ====================================================
        # 10. REDUCE / PROJECT MCP RESULT
        # ====================================================

        projected_result = (
            self.projector.project(
                raw_result
            )
        )

        result_tokens = estimate_tokens(
            projected_result
        )

        # ====================================================
        # 11. GENERATE FINAL ANSWER
        # ====================================================

        answer = await self.model.answer(
            query=request.query,
            history=history,
            tool_name=descriptor.llm_name,
            tool_result=projected_result,
            max_output_tokens=(
                self.max_output_tokens
            ),
        )

        # ====================================================
        # 12. SAVE USER-SPECIFIC CONVERSATION
        # ====================================================

        await self._save(
            user.user_id,
            request.session_id,
            request.query,
            answer,
        )

        logger.info(
            (
                "request_completed "
                "request_id=%s "
                "user_id=%s "
                "session_id=%s "
                "tool=%s "
                "candidates=%s"
            ),
            request_id,
            user.user_id,
            request.session_id,
            descriptor.registry_id,
            len(selected_tools),
        )

        # ====================================================
        # 13. RETURN API RESPONSE
        # ====================================================

        return self._response(
            request_id=request_id,
            answer=answer,
            candidates=candidates,
            history_tokens=history_tokens,
            schema_tokens=schema_tokens,
            result_tokens=result_tokens,
            started=started,
            selected_tool=descriptor.name,
            selected_server=(
                descriptor.server_name
            ),
        )

    # ========================================================
    # TOOL ARGUMENT VALIDATION
    # ========================================================

    @staticmethod
    def _validate_arguments(
        schema: dict[str, Any],
        arguments: dict[str, Any],
    ) -> None:
        """
        Checks that arguments generated by the LLM match
        the selected tool's JSON schema.

        Example:

        Tool requires:
            machine_id: string

        LLM sends:
            machine_id: "MAC01"

        -> valid

        LLM sends:
            random_field: 123

        -> validation may fail depending on schema.
        """

        try:
            Draft202012Validator(
                schema
                or {
                    "type": "object"
                }
            ).validate(
                arguments
            )

        except ValidationError as exc:
            raise ValueError(
                (
                    "Invalid tool arguments: "
                    f"{exc.message}"
                )
            ) from exc

    # ========================================================
    # SAVE CONVERSATION
    # ========================================================

    async def _save(
        self,
        user_id: str,
        session_id: str,
        query: str,
        answer: str,
    ) -> None:
        """
        Save one user + assistant conversation turn.

        IMPORTANT:

        Memory key is now:

            user_id + session_id

        instead of only:

            session_id

        This is what isolates conversations between users.
        """

        await self.memory.append(
            user_id,
            session_id,
            ChatTurn(
                role="user",
                content=query,
            ),
            ChatTurn(
                role="assistant",
                content=answer,
            ),
        )

    # ========================================================
    # BUILD RESPONSE
    # ========================================================

    @staticmethod
    def _response(
        *,
        request_id: str,
        answer: str,
        candidates: list[Any],
        history_tokens: int,
        schema_tokens: int,
        result_tokens: int,
        started: float,
        selected_tool: str | None = None,
        selected_server: str | None = None,
    ) -> ChatResponse:
        """
        Convert the internal orchestration result into the
        final ChatResponse returned by the API.
        """

        return ChatResponse(
            request_id=request_id,
            answer=answer,
            selected_tool=selected_tool,
            selected_server=selected_server,
            candidates=[
                CandidateInfo(
                    tool=item.tool.registry_id,
                    domain=(
                        item.tool.policy.domain
                    ),
                    score=round(
                        item.score,
                        4,
                    ),
                )
                for item in candidates
            ],
            token_estimates=TokenStats(
                estimated_history_tokens=(
                    history_tokens
                ),
                estimated_tool_schema_tokens=(
                    schema_tokens
                ),
                estimated_tool_result_tokens=(
                    result_tokens
                ),
            ),
            latency_ms=round(
                (
                    time.perf_counter()
                    - started
                )
                * 1000,
                2,
            ),
        )