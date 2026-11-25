from __future__ import annotations

from contextlib import asynccontextmanager
from inspect import isawaitable
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from mcp_broker.conversation_store import ConversationStore
from mcp_broker.db import Database
from mcp_broker.guardrails_service import FactigentGuardrails
from mcp_broker.llm import OpenAICompatibleChatModel
from mcp_broker.mcp_gateway import MCPGateway
from mcp_broker.registry import ToolRegistry
from mcp_broker.settings import get_settings

from mcp_broker.cli import (
    build_retriever,
    is_read_only_tool,
    read_secret,
    resolve_mcp_servers,
    resolve_policy_path,
)

class ChatRequest(BaseModel):
    query: str = Field(min_length=1,max_length=10_000,)
    conversation_id: str | None = None

class ChatResponse(BaseModel):
    conversation_id: str
    answer: str

class HealthResponse(BaseModel):
    status: str

class AppState:
    """
    Stores long-lived objects that should be created only once
    when FastAPI starts.
    These objects are reused for every request.
    """

    settings: Any
    db: Database
    conversation_store: ConversationStore
    guardrails: FactigentGuardrails
    gateway: MCPGateway
    registry: ToolRegistry
    retriever: Any
    llm: OpenAICompatibleChatModel


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):

    print(
        "\n=========================================="
    )
    print(
        "Starting Factigent FastAPI service..."
    )
    print(
        "=========================================="
    )
    settings = get_settings()
    state.settings = settings
    state.db = Database()
    await state.db.open()
    print(
        "PostgreSQL/pgvector connected."
    )
    state.conversation_store = ConversationStore(
        state.db
    )
    await state.conversation_store.ensure_schema()

    print(
        "Conversation storage ready."
    )

    print(
        "Loading Guardrails AI..."
    )

    state.guardrails = FactigentGuardrails()

    print(
        "Guardrails AI ready."
    )

    mcp_servers = resolve_mcp_servers(
        settings
    )

    if not mcp_servers:
        raise RuntimeError(
            "No MCP servers are configured."
        )

    print(
        "\nConfigured MCP servers:"
    )

    for (
        server_name,
        server_url,
    ) in mcp_servers.items():

        print(
            f"  {server_name}: {server_url}"
        )

    state.gateway = MCPGateway(
        servers=mcp_servers
    )

    # ========================================================
    # 5. TOOL REGISTRY
    # ========================================================

    state.registry = ToolRegistry(
        gateway=state.gateway,
        policy_path=resolve_policy_path(
            settings
        ),
    )

    tools = await state.registry.refresh()

    if not tools:
        raise RuntimeError(
            "No MCP tools were discovered."
        )

    print(
        f"\nLoaded {len(tools)} MCP tools."
    )

    # ========================================================
    # 6. READ-ONLY TOOL FILTER
    # ========================================================

    read_only_tools = [
        tool
        for tool in tools
        if is_read_only_tool(
            tool
        )
    ]

    if not read_only_tools:
        raise RuntimeError(
            "No read-only MCP tools are available."
        )

    print(
        "Read-only tools available:",
        len(read_only_tools),
    )

    # ========================================================
    # 7. RETRIEVER
    # ========================================================

    state.retriever = await build_retriever(
        state.db,
        read_only_tools,
    )

    print(
        "PGVector hybrid retriever ready."
    )

    # ========================================================
    # 8. LLM
    # ========================================================

    api_key = read_secret(
        settings.llm_api_key
    )

    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY is missing from configuration."
        )

    state.llm = OpenAICompatibleChatModel(
        base_url=settings.llm_base_url,
        api_key=api_key,
        model=settings.llm_model,
        timeout_seconds=60,
        max_retries=3,
    )

    print(
        "LLM client ready."
    )

    print(
        "\n=========================================="
    )
    print(
        "Factigent FastAPI service ready."
    )
    print(
        "=========================================="
    )

    try:
        yield

    finally:

        print(
            "\nShutting down Factigent FastAPI service..."
        )

        # ====================================================
        # CLOSE LLM
        # ====================================================

        llm = getattr(
            state,
            "llm",
            None,
        )

        if llm is not None:

            close_method = getattr(
                llm,
                "close",
                None,
            )

            if callable(
                close_method
            ):

                result = close_method()

                if isawaitable(result):
                    await result

        # ====================================================
        # CLOSE DATABASE
        # ====================================================

        db = getattr(
            state,
            "db",
            None,
        )

        if db is not None:
            await db.close()

        print(
            "Factigent FastAPI service stopped."
        )


# ============================================================
# FASTAPI APPLICATION
# ============================================================


app = FastAPI(
    title="Factigent MCP Tool ",
    description=(
        "Factigent chatbot backend service."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# ROOT
# ============================================================


@app.get("/")
async def root() -> dict[str, str]:

    return {
        "service": "Factigent MCP Tool Broker",
        "status": "running",
    }

@app.get(
    "/health",
    response_model=HealthResponse,
)
async def health() -> HealthResponse:

    return HealthResponse(
        status="ok",
    )

@app.post(
    "/v1/chat",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
    x_user_id: str = Header(
        ...,
        alias="X-User-Id",
    ),
) -> ChatResponse:

    # ========================================================
    # 1. USER ID VALIDATION
    # ========================================================

    user_id = str(
        x_user_id or ""
    ).strip()

    if not user_id:

        raise HTTPException(
            status_code=400,
            detail=(
                "X-User-Id cannot be empty."
            ),
        )
    input_guard = (
        state.guardrails.validate_input(
            request.query
        )
    )

    if not input_guard.allowed:

        raise HTTPException(
            status_code=400,
            detail=input_guard.message,
        )

    query = (
        input_guard.safe_text
        or request.query
    )

    query = query.strip()


    conversation_id = (
        request.conversation_id
    )

    if conversation_id:

        exists = (
            await state.conversation_store.exists(
                user_id=user_id,
                conversation_id=conversation_id,
            )
        )

        if not exists:

            raise HTTPException(
                status_code=404,
                detail=(
                    "Conversation not found."
                ),
            )

    else:

        conversation_id = (
            await state.conversation_store.create_conversation(
                user_id=user_id,
            )
        )


    await state.conversation_store.add_message(
        user_id=user_id,
        conversation_id=conversation_id,
        role="user",
        content=query,
    )

    answer = (
        "FastAPI is connected successfully. "
        "The broker workflow will be connected next."
    )

    # ========================================================
    # 6. OUTPUT GUARDRAILS
    # ========================================================

    output_guard = (
        state.guardrails.validate_output(
            answer
        )
    )

    if output_guard.allowed:

        answer = (
            output_guard.safe_text
            or answer
        )

    else:

        answer = (
            output_guard.message
        )


    await state.conversation_store.add_message(
       user_id = user_id, conversation_id=conversation_id,role = "assistant", content = answer )

    # ========================================================
    # 8. RETURN JSON RESPONSE
    # ========================================================

    return ChatResponse(
        conversation_id=conversation_id,answer = answer )
    