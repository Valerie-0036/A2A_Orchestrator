from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.requests import Request
from typing import Protocol, AsyncIterable

from a2a.types import (
    AgentCard,
    GetTaskRequest,
    InternalError,
    JSONRPCResponse,
    SendStreamingMessageRequest,
    Task,
)

from a2a.server.tasks import task_manager  

# 🛠️ General utilities
import json                                              # Used for printing the request payloads (for debugging)
import logging                                           # Used to log errors and info messages
logger = logging.getLogger(__name__)                     # Setup logger for this file

# 🕒 datetime import for serialization
from datetime import datetime

# 📦 Encoder to help convert complex data like datetime into JSON
from fastapi.encoders import jsonable_encoder


# -----------------------------------------------------------------------------
# 🔧 Serializer for datetime
# -----------------------------------------------------------------------------
class A2ATaskManager(Protocol):
    """Defines the interface a task manager must implement to be used by the A2AServer."""

    async def message_stream(self, request: SendStreamingMessageRequest) -> AsyncIterable[Task]:
        """Processes a streaming message request and yields task updates."""
        ...

    # We can also define other methods the server might need to call.
    # If your server doesn't handle 'task/get', you can remove this.
    async def get_task(self, request: GetTaskRequest) -> JSONRPCResponse:
        """Retrieves the state of a specific task."""
        ...


# -----------------------------------------------------------------------------
# 🌐 The Generic A2A Server
# -----------------------------------------------------------------------------
class A2AServer:
    """A generic A2A server that handles web requests and delegates to a task manager."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5000,
        agent_card: AgentCard = None,
        task_manager: A2ATaskManager = None,  # ✨ Use the Protocol as the type hint
    ):
        self.host = host
        self.port = port
        self.agent_card = agent_card
        self.task_manager = task_manager
        self.app = Starlette(debug=True)
        self.app.add_route("/", self._handle_request, methods=["POST"])
        self.app.add_route("/.well-known/agent.json", self._get_agent_card, methods=["GET"])

    async def start(self):
        """Starts the Uvicorn web server."""
        if not self.agent_card or not self.task_manager:
            raise ValueError("An AgentCard and a compliant TaskManager are required.")
        import uvicorn

        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

    def _get_agent_card(self, request: Request) -> JSONResponse:
        """Serves the agent's capability card."""
        return JSONResponse(self.agent_card.model_dump(exclude_none=True))

    async def _handle_request(self, request: Request):
        """
        The main request handler. It validates the JSON-RPC request and calls the
        appropriate method on the provided task_manager.
        """
        body = None
        try:
            body = await request.json()
            method = body.get("method")
            logger.debug("Received A2A request with method: %s", method)

            if method == "message/stream":
                # Validate the request body against the Pydantic model
                json_rpc = SendStreamingMessageRequest.model_validate(body)

                # This is the event stream that will be sent to the client
                async def event_generator():
                    # This line calls your A2AHostAgentAdapter.message_stream()
                    async for task_update in self.task_manager.message_stream(json_rpc):
                        response_payload = JSONRPCResponse(id=json_rpc.id, result=task_update)
                        # Use jsonable_encoder to safely convert Pydantic models (including datetimes) to JSON
                        json_str = json.dumps(jsonable_encoder(response_payload.model_dump(exclude_none=True)))
                        yield f"data: {json_str}\n\n"

                return StreamingResponse(event_generator(), media_type="text/event-stream")

            # This part handles non-streaming task retrieval if needed
            elif method == "task/get" and hasattr(self.task_manager, "get_task"):
                json_rpc = GetTaskRequest.model_validate(body)
                # The adapter would need to implement get_task for this to work
                result = await self.task_manager.get_task(json_rpc)
                return JSONResponse(content=jsonable_encoder(result.model_dump(exclude_none=True)))

            else:
                raise ValueError(f"Unsupported A2A method: '{method}'")

        except Exception as e:
            logger.error(f"Error handling request: {e}", exc_info=True)
            req_id = body.get("id") if body else None
            error_response = JSONRPCResponse(id=req_id, error=InternalError(message=str(e)))
            return JSONResponse(
                content=jsonable_encoder(error_response.model_dump(exclude_none=True)),
                status_code=400,
            )