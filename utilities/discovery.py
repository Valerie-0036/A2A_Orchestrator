# utilities/discovery.py
# =============================================================================
# 🎯 Purpose:
# A shared utility module for discovering Agent-to-Agent (A2A) servers.
# It reads a registry of agent base URLs (from a JSON file) and fetches
# each agent's metadata (AgentCard) from the standard discovery endpoint.
# This allows any client or agent to dynamically learn about available agents.
# =============================================================================

import os
import json
import logging
from typing import List

import httpx
from a2a.types import AgentSkill, AgentCapabilities, AgentCard

logger = logging.getLogger(__name__)


class DiscoveryClient:
    """
    🔍 Discovers A2A agents by reading a registry file of URLs and querying
    each one's /.well-known/agent.json endpoint to retrieve an AgentCard.

    Attributes:
        registry_file (str): Path to the JSON file listing base URLs (strings).
        base_urls (List[str]): Loaded list of agent base URLs.
    """

    def __init__(self, registry_file: str = None):
        """
        Initialize the DiscoveryClient.

        Args:
            registry_file (str, optional): Path to the registry JSON. If None,
                defaults to 'agent_registry.json' in this utilities folder.
        """
        if registry_file:
            self.registry_file = registry_file
        else:
            self.registry_file = os.path.join(
                os.path.dirname(__file__),
                "agent_registry.json"
            )
        self.base_urls = self._load_registry()

    def _load_registry(self) -> List[str]:
        """
        Load and parse the registry JSON file into a list of URLs.

        Returns:
            List[str]: The list of agent base URLs, or empty list on error.
        """
        try:
            with open(self.registry_file, "r") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Registry file must contain a JSON list of URLs.")
            return data
        except FileNotFoundError:
            logger.warning(f"Registry file not found: {self.registry_file}")
            return []
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing registry file: {e}")
            return []

    async def list_agent_cards(self) -> List[AgentCard]:
        """
        Asynchronously fetch the discovery endpoint from each registered URL
        and parse the returned JSON into AgentCard objects.

        Returns:
            List[AgentCard]: Successfully retrieved agent cards.
        """
        cards: List[AgentCard] = []
        async with httpx.AsyncClient() as client:
            for base in self.base_urls:
                url = base.rstrip("/") + "/.well-known/agent.json"
                try:
                    response = await client.get(url, timeout=5.0)
                    response.raise_for_status()
                    card = AgentCard.model_validate(response.json())
                    cards.append(card)
                except Exception as e:
                    logger.warning(f"Failed to discover agent at {url}: {e}")
        return cards

    # <<< NEW METHOD ADDED HERE >>>
    def list_agent_urls(self) -> List[str]:
        """
        Returns the list of base URLs loaded from the registry file.
        This is a synchronous method as the URLs are loaded in the constructor.

        Returns:
            List[str]: The list of agent base URLs.
        """
        return self.base_urls