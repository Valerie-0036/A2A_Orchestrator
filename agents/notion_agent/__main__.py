import asyncio
import logging
import os
import sys

import httpx
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryPushNotifier, InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from dotenv import load_dotenv

# Assuming .agent and agents.notion_agent.agent_executor are correctly handled in your project structure
from .agent import NotionADKAgent
from agents.notion_agent.agent_executor import create_executor

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""

host = "localhost"
port = 10003

def main(): # <--- This is now a synchronous function
    """Starts the Notion Agent server."""
    try:
        if os.getenv('model_source', 'google') == 'google':
            if not os.getenv('GOOGLE_API_KEY'):
                raise MissingAPIKeyError(
                    'GOOGLE_API_KEY environment variable not set.'
                )
        else:
            if not os.getenv('TOOL_LLM_URL'):
                raise MissingAPIKeyError(
                    'TOOL_LLM_URL environment variable not set.'
                )
            if not os.getenv('TOOL_LLM_NAME'):
                raise MissingAPIKeyError(
                    'TOOL_LLM_NAME environment not variable not set.'
                )

        capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
        skill = AgentSkill(
            id="notion_search",
            name="Search Notion workspace",
            description="Searches and retrieves information from Notion pages, databases, and blocks.",
            tags=["notion","search","retrieval","knowledge","workspace"],
            examples=[
                "Search for 'project plan'",
                "Find pages about Q3 goals",
                "Retrieve information from the meeting notes database",
            ],
        )

        agent_card = AgentCard(
            name="Notion Search Agent",
            description="Provides information retrieval services from Notion workspace using MCP.",
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=NotionADKAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=NotionADKAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        # --8<-- [start:DefaultRequestHandler]
        httpx_client = httpx.AsyncClient()

        # Execute create_executor() in its own event loop if it's async
        # We need to run this *before* uvicorn starts its loop
        agent_executor_instance = asyncio.run(create_executor()) # <--- Run this async call

        request_handler = DefaultRequestHandler(
            agent_executor=agent_executor_instance, # <--- Use the awaited instance
            task_store=InMemoryTaskStore(),
            push_notifier=InMemoryPushNotifier(httpx_client),
        )
        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )

        logger.info(f"Starting Notion Search Agent on http://{host}:{port}/")
        uvicorn.run(server.build(), host=host, port=port) # <--- Uvicorn starts the loop here
        # --8<-- [end:DefaultRequestHandler]

    except MissingAPIKeyError as e:
        logger.error(f'Error: {e}')
        sys.exit(1)
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main() # <--- Call the synchronous main function