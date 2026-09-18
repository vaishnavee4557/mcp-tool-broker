"""Token-efficient MCP tool broker."""

__all__ = ["__version__"]
__version__ = "0.1.0"
"""retrieval.py    → selects top 2–3 candidates
token_budget.py → checks how many candidates fit
llm.py          → chooses one final tool
mcp_gateway.py  → executes that tool
orchestrator.py → controls the complete process

Swagger APIs
    ↓
Automatically create ToolDescriptor for each useful API
    ↓
Create searchable_text and embeddings
    ↓
Retriever selects top 2–3 relevant tools
             |
llm.py
    ↓
LLM chooses one final tool and creates arguments
    ↓
Passes selected tool to mcp_gateway.py
    ↓
MCP client calls that tool on the MCP server
    ↓
MCP server executes the Python tool function
    ↓
Tool calls Factigent API|

# Mock model: used without an actual LLM API 
User Question
    ↓
cli.py
raw_query = input(...)
    ↓
normalize_query(raw_query)
    ↓
build_retrieval_query(...)
    ↓
retriever.retrieve(...)
    ↓
pgvector + BM25
    ↓
Top relevant MCP tools
    ↓
LLM choose_tool()
    ↓
MCP API call
    ↓
LLM final answer

#     """
#     Used for local testing.

#     It selects the first retrieved tool and creates basic arguments
#     from the user's question.
#     """
# class MockChatModel:
#     async def choose_tool(
#         self,
#         query: str,
#         history: list[ChatTurn],
#         tools: list[ToolDescriptor],
#     ) -> ToolDecision:

#         if not tools:
#             return ToolDecision(
#                 direct_answer="No suitable authorized tool was found."
#             )

#         selected_tool = tools[0]

#         arguments = self._extract_arguments(
#             query=query,
#             tool=selected_tool,
#         )

#         return ToolDecision(
#             tool_name=selected_tool.llm_name,
#             arguments=arguments,
#         )

#     async def answer(
#         self,
#         query: str,
#         history: list[ChatTurn],
#         tool_name: str | None,
#         tool_result: Any | None,
#         max_output_tokens: int,
#     ) -> str:

#         if not tool_name:
#             return "No suitable tool was available."

#         readable_result = json.dumps(
#             tool_result,
#             ensure_ascii=False,
#             default=str,
#         )

#         return f"Tool {tool_name} returned: {readable_result}"

#     def _extract_arguments(
#         self,
#         query: str,
#         tool: ToolDescriptor,
#     ) -> dict[str, Any]:
#         """
#         Create simple tool arguments from the user's question.

#         This is only for testing. A real LLM understands arguments
#         more intelligently.
#         """

#         arguments: dict[str, Any] = {}

#         lowered_query = query.lower()

#         properties = tool.input_schema.get(
#             "properties",
#             {},
#         )

#         # Extract machine ID such as MAC1 or MAC2.
#         if "machine_id" in properties:
#             for word in query.replace(",", " ").split():
#                 cleaned_word = word.strip("?.").upper()

#                 if cleaned_word.startswith("MAC"):
#                     arguments["machine_id"] = cleaned_word
#                     break

#         if "limit" in properties:
#             arguments["limit"] = 5

#         if "hours" in properties:
#             arguments["hours"] = 24

#         if "summary" in properties:
#             arguments["summary"] = query[:500]

#         if "parameters" in properties:
#             supported_parameters = [
#                 "temperature",
#                 "vibration",
#                 "voltage",
#                 "humidity",
#                 "power",
#             ]

#             selected_parameters = [
#                 parameter
#                 for parameter in supported_parameters
#                 if parameter in lowered_query
#             ]

#             if selected_parameters:
#                 arguments["parameters"] = selected_parameters

#         return arguments """

# For the MCP server, Terminal 1:

# cd C:\Users\vaish\Downloads\mcp_tool_broker

# $env:PYTHONPATH = (Resolve-Path .\src).Path

# C:\Users\vaish\Downloads\mcp_tool_broker.venv\Scripts\python.exe -m mcp_broker.mcp_server

# cd C:\Users\vaish\Downloads\mcp_tool_broker

# $env:PYTHONPATH = (Resolve-Path .\src).Path

# C:\Users\vaish\Downloads\mcp_tool_broker.venv\Scripts\python.exe -m mcp_broker.cli
"""








"""
# cd C:\Users\vaish\Downloads\mcp_tool_broker
# .\.venv\Scripts\Activate.ps1
# $env:PYTHONPATH = (Resolve-Path .\src).Path
# python -m mcp_broker.mcp_server
# for cli 
# cd C:\Users\vaish\Downloads\mcp_tool_broker
# .\.venv\Scripts\Activate.ps
# $env:PYTHONPATH = (Resolve-Path .\src).Path
# python -m mcp_broker.cli
#  for pg vector db
# Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
# docker start factigent_pgvector
# docker ps --filter "name=factigent_pgvector"
# docker info
#   Test-NetConnection 127.0.0.1 -Port 5433     
#cd C:\Users\vaish\Downloads\mcp_tool_broker
#.\.venv\Scripts\Activate.ps1
#$env:PYTHONPATH=(Resolve-Path .\src).Path
#$env:OPENAPI_SOURCE_NAME="ums"
#$env:MCP_HOST="127.0.0.1"
#$env:MCP_PORT="8003"
#python -m mcp_broker.mcp_server     
    # cd C:\Users\vaish\Downloads\mcp_tool_broker
# .\.venv\Scripts\Activate.ps1
# $env:PYTHONPATH=(Resolve-Path .\src).Path
# $env:OPENAPI_SOURCE_NAME="play"
# $env:MCP_HOST="127.0.0.1"
# $env:MCP_PORT="8002"
# python -m mcp_broker.mcp_server
    # TO SEE PGVECTORE 
    
    # cd C:\Users\vaish\Downloads\mcp_tool_broker
#docker ps --format "table {{.Names}}\t{{.Status}}"
    # docker exec -it factigent_pgvector psql -U factigent_app -d factigent_chatbot
#     SELECT
#     COUNT(*) AS total_tools,
#     COUNT(embedding) AS embedded_tools,
#     COUNT(*) - COUNT(embedding) AS tools_without_embedding
# FROM mcp_tools;
# SELECT
#     id,
#     server_name,
#     tool_name,
#     embedding_model,
#     left(searchable_text, 180) AS embedded_text,
#     left(embedding::text, 180) AS embedding_preview
# FROM public.mcp_tools
# WHERE embedding IS NOT NULL
# ORDER BY server_name, tool_name
# LIMIT 3;