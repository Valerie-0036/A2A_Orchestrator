# File: agents/host_agent/agent.py

import asyncio
import uuid
from typing import Any, AsyncIterable, Dict, List

import httpx
import nest_asyncio
from a2a.client import A2ACardResolver, A2AClient
# --- MODIFIED: Cleaned up imports and added the correct response types ---
from a2a.types import (
    AgentCard, SendStreamingMessageRequest, Task, TextPart,
    JSONRPCErrorResponse, SendStreamingMessageSuccessResponse,
    TaskArtifactUpdateEvent, TaskStatusUpdateEvent
)
from dotenv import load_dotenv
from google.adk import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from .remote_agent_connection import RemoteAgentConnections

# Load environment variables and apply nest_asyncio for environments like notebooks
load_dotenv()
nest_asyncio.apply()


class HostAgent:
    """The Host agent, powered by Google ADK, designed to orchestrate child agents."""

    def __init__(self):
        # This will hold connections to child agents discovered at startup
        self.remote_agent_connections: Dict[str, RemoteAgentConnections] = {}
        self.cards: Dict[str, AgentCard] = {}
        # This string is dynamically built to be part of the LLM's system prompt
        self.agents_for_prompt: str = "No child agents found."

        # Standard Google ADK setup
        self._agent = self._create_agent()
        self._user_id = "host_agent_user"
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    async def _async_init_components(self, remote_agent_addresses: List[str]):
        """Connects to child agents to get their capabilities (AgentCard)."""
        async with httpx.AsyncClient(timeout=30) as client:
            for address in remote_agent_addresses:
                try:
                    card_resolver = A2ACardResolver(client, address)
                    card = await card_resolver.get_agent_card()
                    # This assumes RemoteAgentConnections correctly initializes an A2AClient
                    remote_connection = RemoteAgentConnections(
                        agent_card=card, agent_url=address
                    )
                    self.remote_agent_connections[card.name] = remote_connection
                    self.cards[card.name] = card
                    print(f"Successfully connected to child agent: {card.name}")
                except Exception as e:
                    print(f"ERROR: Failed to initialize connection for {address}: {e}")

        # Build a descriptive list of agents for the LLM prompt
        agent_info = [
            f"- {card.name}: {card.description}" for card in self.cards.values()
        ]
        if agent_info:
            self.agents_for_prompt = "\n".join(agent_info)

    @classmethod
    async def create(cls, remote_agent_addresses: List[str]):
        """Asynchronous factory method to create and initialize the agent."""
        instance = cls()
        await instance._async_init_components(remote_agent_addresses)
        return instance

    def _root_instruction(self, context: ReadonlyContext) -> str:
        """System prompt for the LLM, instructing it on how to act as an orchestrator."""
        return (
            "You are a helpful and friendly orchestrator agent. Your job is to understand the user's "
            "request and delegate it to the appropriate child agent using the `delegate_task` tool. "
            "First, you can use `list_agents` to see who is available. "
            "Then, use `delegate_task` to assign the work. Respond to the user with the result from the child agent."
            "\n\nAvailable child agents:\n"
            f"{self.agents_for_prompt}"
        )

    def _list_agents(self) -> List[str]:
        """Tool function: returns a list of available child agents."""
        return list(self.remote_agent_connections.keys())

    async def _delegate_task(
        self, agent_name: str, message: str, tool_context: ToolContext
    ) -> str:
        """
        Tool function (CONSUMER): forwards the `message` to a specific child agent
        and returns its final text response.
        """
        if agent_name not in self.remote_agent_connections:
            return f"Error: Unknown agent '{agent_name}'. Please use list_agents() to see available agents."

        print(f"Delegating task to {agent_name}: '{message}'")

        # This variable will hold the final answer. It is the STATE we need to maintain.
        final_response_text = ""

        try:
            # Get the stream of updates from the producer function.
            response_stream = self._stream_message_to_remote(
                agent_name, message, tool_context
            )

            # Asynchronously loop through every update yielded by the stream.
            async for update in response_stream:
                # The update is a dictionary. Check if it has a 'content' key with a value.
                content_chunk = update.get("content")
                if content_chunk is not None:
                    # We found new text from the child agent. SAVE IT.
                    # This will be overwritten by later content updates, ensuring we
                    # always have the most recent valid content.
                    final_response_text = content_chunk

            # After the loop is finished, final_response_text holds the last valid content received.
            print(f"DEBUG: Final content from child agent: '{final_response_text}'")
            return final_response_text

        except Exception as e:
            print(f"ERROR: Failed to delegate task to {agent_name}: {e}", exc_info=True)
            return f"An error occurred while communicating with {agent_name}."

    def _create_agent(self) -> Agent:
        """Defines the agent for the Google ADK framework, including its tools."""
        return Agent(
            model="gemini-2.0-flash",
            name="HostAgentCore",
            instruction=self._root_instruction,
            description="Delegates user queries to child A2A agents.",
            tools=[self._delegate_task, self._list_agents],
        )

    async def stream(self, query: str, session_id: str) -> AsyncIterable[Dict[str, Any]]:
        """The main entry point for the ADK agent. This is the method our A2A adapter will call."""
        session = await self._runner.session_service.get_session(
            app_name=self._agent.name, user_id=self._user_id, session_id=session_id
        )
        if session is None:
            session = await self._runner.session_service.create_session(
                app_name=self._agent.name, user_id=self._user_id, session_id=session_id
            )

        content = types.Content(role="user", parts=[types.Part.from_text(text=query)])

        async for event in self._runner.run_async(
            user_id=self._user_id, session_id=session.id, new_message=content
        ):
            if event.is_final_response():
                response_text = "\n".join(
                    [p.text for p in event.content.parts if p.text]
                )
                yield {"is_task_complete": True, "content": response_text}


    async def _stream_message_to_remote(
        self, agent_name: str, task: str, tool_context: ToolContext
    ) -> AsyncIterable[Dict[str, Any]]:
        """
        Helper (PRODUCER): streams a message to a remote agent and yields
        simplified update dictionaries. THIS FUNCTION SHOULD BE STATELESS.
        """
        connection = self.remote_agent_connections[agent_name]
        a2a_client = connection.agent_client

        task_id = str(uuid.uuid4())
        message_id = str(uuid.uuid4())
        context_id = tool_context.state.get("context_id", str(uuid.uuid4()))
        message_payload = {
            "role": "user",
            "parts": [{"kind": "text", "text": task}],
            "messageId": message_id,
            "taskId": task_id,
            "contextId": context_id,
        }
        stream_request = SendStreamingMessageRequest(
            id=task_id, params={"message": message_payload}
        )

        async for response_update in a2a_client.send_message_streaming(stream_request):
            print(f"DEBUG: Received update from child agent: {response_update.model_dump_json(indent=2)}")
            
            # --- MODIFIED: Check for the correct streaming response type ---
            if not isinstance(response_update.root, SendStreamingMessageSuccessResponse):
                if isinstance(response_update.root, JSONRPCErrorResponse):
                    print(f"ERROR received from child agent: {response_update.root.error.message}")
                continue

            task_update = response_update.root.result
            
            # For each update, create a fresh dictionary to yield.
            # Default to having no new content.
            yield_dict = {"content": None}
            
            artifacts_to_check = None

            # Check if the update is an artifact update.
            if isinstance(task_update, TaskArtifactUpdateEvent):
                artifacts_to_check = [task_update.artifact]
            # Also check the initial task object which might contain artifacts.
            elif isinstance(task_update, Task) and task_update.artifacts:
                 artifacts_to_check = task_update.artifacts

            # If the update included an artifact, extract the text content.
            if artifacts_to_check:
                for artifact in artifacts_to_check:
                    for part in artifact.parts:
                        if isinstance(part.root, TextPart):
                            # Put the found text into our dictionary for this specific yield.
                            yield_dict["content"] = part.root.text
            
            # Yield the processed dictionary to the consumer (_delegate_task).
            # It will either have content or content will be None.
            yield yield_dict