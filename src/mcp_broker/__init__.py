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

