# File: agents/host_agent/entry.py

import asyncio
import logging
from uuid import uuid4

import click
from server.server import A2AServer
from a2a.types import (AgentCard, AgentCapabilities, AgentSkill, Artifact,
                       Message, SendStreamingMessageRequest, Task, TaskState,
                       TaskStatus,Part, TextPart, TaskState, TaskStatus,
                       TaskStatusUpdateEvent, TaskArtifactUpdateEvent, Artifact,
                      TaskArtifactUpdateEvent, TaskStatusUpdateEvent)
from a2a.server.tasks import TaskStore, TaskManager,  InMemoryTaskStore
from collections.abc import AsyncIterable

# Make sure these paths are correct for your project structure
from .agent import HostAgent
from utilities.discovery import DiscoveryClient # Assuming you have this helper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class A2AHostAgentAdapter:
    """
    Adapter that uses a TaskManager helper to bridge the Google ADK HostAgent
    with the A2A Server.
    """
    def __init__(self, host_agent: HostAgent, task_store: TaskStore):
        self._agent = host_agent
        self._task_store = task_store

    # THIS IS THE METHOD THE SERVER IS LOOKING FOR
    # Make sure the name is exactly 'message_stream'

    async def message_stream(self, request: SendStreamingMessageRequest) -> AsyncIterable[Task]:
        # ... (code to initialize task_id, context_id, etc. is fine) ...
        user_message = request.params.message
        task_id = str(request.id)
        context_id = user_message.contextId or str(uuid4())

        task_manager_helper = TaskManager(
            task_id=task_id,
            context_id=context_id,
            task_store=self._task_store,
            initial_message=user_message,
        )

        initial_event = TaskStatusUpdateEvent(
            taskId=task_id,
            contextId=context_id,
            status=TaskStatus(state=TaskState.working, description="Host agent received task."),
            final=False
        )
        initial_task = await task_manager_helper.save_task_event(initial_event)
        yield initial_task

        # --- START OF THE FIX ---

        # Ensure the message has parts before trying to access them
        if not user_message.parts:
            # Handle cases where the message is empty
            logger.warning(f"Received message for task {task_id} with no parts.")
            # You might want to yield a "failed" task event here
            return # Stop processing this request

        # Get the root of the first part
        first_part = user_message.parts[0].root

        # Check if the part is a TextPart before accessing its text attribute
        if isinstance(first_part, TextPart):
            query = first_part.text
        else:
            # Handle cases where the first part is not text (e.g., a file or data)
            logger.error(f"The first part of the message for task {task_id} is not text. Cannot process.")
            # You should yield a proper error event to the client
            error_status = TaskStatus(
                state=TaskState.failed,
                message="This agent only accepts text input."
            )
            error_event = TaskStatusUpdateEvent(
                taskId=task_id,
                contextId=context_id,
                status=error_status,
                final=True
            )
            error_task = await task_manager_helper.save_task_event(error_event)
            yield error_task
            return # Stop processing

        # --- END OF THE FIX ---

        session_id = str(uuid4())

        async for event in self._agent.stream(query=query, session_id=session_id):
            is_complete = event.get("is_task_complete", False)
            content = event.get("content", event.get("updates", "..."))

            # When creating the response artifact, you also need to use the correct models
            response_artifact = Artifact(
                artifactId=str(uuid4()),
                parts=[Part(root=TextPart(text=content))] # Wrap the TextPart in a Part
            )

            artifact_event = TaskArtifactUpdateEvent(
                taskId=task_id,
                contextId=context_id,
                artifact=response_artifact
            )
            updated_task = await task_manager_helper.save_task_event(artifact_event)

            if is_complete:
                final_status_event = TaskStatusUpdateEvent(
                    taskId=task_id,
                    contextId=context_id,
                    status=TaskStatus(state=TaskState.completed, description="Task complete."),
                    final=True
                )
                updated_task = await task_manager_helper.save_task_event(final_status_event)

            yield updated_task

@click.command()
@click.option("--host", default="127.0.0.1", help="Host to bind the server to.")
@click.option("--port", default=10000, help="Port for the HostAgent server.")
@click.option("--registry", default="utilities/agent_registry.json", help="Path to JSON file of child-agent URLs.")
def main(host: str, port: int, registry: str):
    """Main function to start the server, managed by click."""
    asyncio.run(start_server(host, port, registry))

async def start_server(host: str, port: int, registry: str):
    """Asynchronously initializes and starts the A2A server."""
    try:
        discovery = DiscoveryClient(registry_file=registry)
        agent_urls = discovery.list_agent_urls()
        logger.info(f"Found {len(agent_urls)} child agents to orchestrate: {agent_urls}")
    except Exception as e:
        logger.error(f"Could not read agent registry at '{registry}': {e}")
        return

    # 2. Initialize the core ADK-based agent
    orchestrator = await HostAgent.create(remote_agent_addresses=agent_urls)
    logger.info("HostAgent initialized successfully.")

    orchestrator_card = AgentCard(
        name="HostAgent",
        description="An orchestrator that intelligently delegates tasks to a team of specialized child agents.",
        url=f"http://{host}:{port}/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[AgentSkill(
            id="orchestrate",
            name="General Orchestration",
            description="Can answer questions and perform tasks by routing requests to other agents.",
            examples=["What is the time?", "Tell me a joke.", "Who can you talk to?"],
            tags=["routing", "orchestration", "general"] # <-- THE FIX IS HERE
        )]
    )

    # Note: A2AHostAgentAdapter was expecting task_store in the previous version.
    # Make sure you're using the corrected version from the last answer.
    task_store = InMemoryTaskStore()
    task_manager_adapter = A2AHostAgentAdapter(
        host_agent=orchestrator,
        task_store=task_store
    )
    
    server = A2AServer(
        host=host,
        port=port,
        agent_card=orchestrator_card,
        task_manager=task_manager_adapter
    )
    
    logger.info(f"Starting A2A HostAgent server at http://{host}:{port}")
    await server.start()

if __name__ == "__main__":
    main()