import os

from collections.abc import AsyncIterable
from typing import Any, Literal

import httpx

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from contextlib import asynccontextmanager
import json
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
memory = MemorySaver()


server_params = StdioServerParameters(command="npx",
                                                args=['-y','@notionhq/notion-mcp-server'],
                                                env={"OPENAPI_MCP_HEADERS":json.dumps({
                                                  "Authorization": f"Bearer {NOTION_API_KEY}",
                                                  "Notion-Version": "2022-06-28"
                                                })
                                                  
                                                })
@asynccontextmanager
async def get_retriever_mcp_agent():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            yield tools


            
class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


class NotionADKAgent:
    """NotionADKAgent - an assistant to search in Notion workspace"""

    SYSTEM_INSTRUCTION = (
    """
    You are a Notion Information Retrieval agent. You help users search for and retrieve information from their Notion workspace.

    Your goal is to efficiently find and present the most relevant information from the user's Notion workspace, and present it clearly.

    ## Rules

    1.  Use the available Notion tools to search for pages and blocks based on user queries. For these, provide clear, formatted responses with titles, summaries, and links.
    2.  When a user asks you to query or get information from a database (e.g., "count entries in 'Sermon Notes'"), you MUST follow this specific two-step process:
        a.  **Find the Database by Name**: First, use the `notion.search` tool to locate the database by its name. The tool will return information that includes a URL.
        b.  **Extract and Use the ID**: The `id` you need for the `notion.queryDatabase` tool is contained within the URL returned in the previous step. You must parse this URL, extract the ID, and then use it to perform the second query.
    3.  **CRITICAL - Do Not Ask for IDs unncessarily**: You are explicitly forbidden from asking the user for a database or page ID if you have already found the item via search. Your job is to extract the URL you were provided with.
    4.  If a search yields no results, clearly state that and suggest alternative search terms.
    5.  When presenting database query results, format them in a structured way that shows the key properties.
    6.  Always provide the source (page title and URL) when presenting retrieved information.
    """
    )

    FORMAT_INSTRUCTION = (
        'Set response status to input_required if the user needs to provide more information to complete the request.'
        'Set response status to error if there is an error while processing the request.'
        'Set response status to completed if the request is complete.'
    )

    def __init__(self, tools: list):  # <-- Accept tools as an argument
        self.tools = tools           # <-- Assign the tools
        model_source = os.getenv('model_source', 'google')
        if model_source == 'google':
            self.model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
        else:
            self.model = ChatOpenAI(
                model=os.getenv('TOOL_LLM_NAME'),
                openai_api_key=os.getenv('API_KEY', 'EMPTY'),
                openai_api_base=os.getenv('TOOL_LLM_URL'),
                temperature=0,
            )

        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=(self.FORMAT_INSTRUCTION, ResponseFormat),
        )

    async def stream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        inputs = {'messages': [('user', query)]}
        config = {'configurable': {'thread_id': context_id}}

        for item in self.graph.stream(inputs, config, stream_mode='values'):
            message = item['messages'][-1]
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Looking up the notion workspace...',
                }
            elif isinstance(message, ToolMessage):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Processing the notion workspace..',
                }

        yield self.get_agent_response(config)

    def get_agent_response(self, config):
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get('structured_response')
        if structured_response and isinstance(
            structured_response, ResponseFormat
        ):
            if structured_response.status == 'input_required':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'error':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'completed':
                return {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': structured_response.message,
                }

        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': (
                'We are unable to process your request at the moment. '
                'Please try again.'
            ),
        }

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

# NEW aync factory function
async def create_notion_agent() -> NotionADKAgent:
    """
    Asynchronously fetches tools and initializes the NotionADKAgent.
    """
    async with get_retriever_mcp_agent() as tools:
        agent = NotionADKAgent(tools=tools)
        return agent