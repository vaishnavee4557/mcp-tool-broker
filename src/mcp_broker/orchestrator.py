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
from mcp_broker.token_budget import TokenBudgeter, estimate_tokens

logger = logging.getLogger(__name__)


class ToolBrokerOrchestrator:
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
        self.registry = registry
        self.retriever = retriever
        self.gateway = gateway
        self.models = model
        self.memory = memory
        self.budgeter = budgeter
        self.projector = projector
        self.max_candidates = max_candidates
        self.max_output_tokens = max_output_tokens

    async def handle(self, request: ChatRequest, user: UserContext) -> ChatResponse:
        started = time.perf_counter()
        request_id = uuid.uuid4().hex

        full_history = await self.memory.get(request.session_id)
        history, history_tokens = self.budgeter.select_history(full_history, request.query)

        candidates = await self.retriever.retrieve(
            request.query,
            user,
            limit=self.max_candidates,
        )
        selected_tools, schema_tokens = self.budgeter.select_tools(candidates)

        if not selected_tools:
            answer = "No authorized tool is available for this request."
            await self._save(request.session_id, request.query, answer)
            return self._response(
                request_id=request_id,
                answer=answer,
                candidates=candidates,
                history_tokens=history_tokens,
                schema_tokens=schema_tokens,
                result_tokens=0,
                started=started,
            )

        decision = await self.model.choose_tool(
            query=request.query,
            history=history,
            tools=selected_tools,
        )

        if decision.direct_answer and not decision.tool_name:
            await self._save(request.session_id, request.query, decision.direct_answer)
            return self._response(
                request_id=request_id,
                answer=decision.direct_answer,
                candidates=candidates,
                history_tokens=history_tokens,
                schema_tokens=schema_tokens,
                result_tokens=0,
                started=started,
            )

        descriptor = self.registry.get_by_llm_name(decision.tool_name or "")
        visible_names = {tool.llm_name for tool in selected_tools}
        if descriptor is None or descriptor.llm_name not in visible_names:
            raise RuntimeError("Model attempted to call a tool outside the authorized candidate set")

        self._validate_arguments(descriptor.input_schema, decision.arguments)
        raw_result = await self.gateway.call_tool(descriptor, decision.arguments)
        projected_result = self.projector.project(raw_result)
        result_tokens = estimate_tokens(projected_result)

        answer = await self.model.answer(
            query=request.query,
            history=history,
            tool_name=descriptor.llm_name,
            tool_result=projected_result,
            max_output_tokens=self.max_output_tokens,
        )
        await self._save(request.session_id, request.query, answer)

        logger.info(
            "request_completed tool=%s candidates=%s",
            descriptor.registry_id,
            len(selected_tools),
        )
        return self._response(
            request_id=request_id,
            answer=answer,
            candidates=candidates,
            history_tokens=history_tokens,
            schema_tokens=schema_tokens,
            result_tokens=result_tokens,
            started=started,
            selected_tool=descriptor.name,
            selected_server=descriptor.server_name,
        )

    @staticmethod
    def _validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
        try:
            Draft202012Validator(schema or {"type": "object"}).validate(arguments)
        except ValidationError as exc:
            raise ValueError(f"Invalid tool arguments: {exc.message}") from exc

    async def _save(self, session_id: str, query: str, answer: str) -> None:
        await self.memory.append(
            session_id,
            ChatTurn(role="user", content=query),
            ChatTurn(role="assistant", content=answer),
        )

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
        return ChatResponse(
            request_id=request_id,
            answer=answer,
            selected_tool=selected_tool,
            selected_server=selected_server,
            candidates=[
                CandidateInfo(
                    tool=item.tool.registry_id,
                    domain=item.tool.policy.domain,
                    score=round(item.score, 4),
                )
                for item in candidates
            ],
            token_estimates=TokenStats(
                estimated_history_tokens=history_tokens,
                estimated_tool_schema_tokens=schema_tokens,
                estimated_tool_result_tokens=result_tokens,
            ),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
