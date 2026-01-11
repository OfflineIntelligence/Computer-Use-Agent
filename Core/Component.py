# Entity/Core/Component.py

import sys
import os

from typing import Dict, Optional, List, Union, Any
from Entity.Core.Multimodal_llm import MultiLLM


class BaseComponent:
    def __init__(self, engine_parameters: Dict, platform: str):
        """
        Base component for Computer Use Agent functionality.
        
        Args:
            engine_parameters: Configuration for Rust server connection
            platform: Operating system platform ('windows', 'linux', 'macos')
        """
        self.engine_parameters = engine_parameters
        self.platform = platform.lower()
        self._agents = {}  # Optional: cache agents by system prompt

    def _create_agent(self, system_prompt: Optional[str] = None, 
                     engine_parameters: Optional[Dict] = None) -> MultiLLM:
        """
        Create a new MultiLLM agent instance optimized for Rust server.
        
        Args:
            system_prompt: Optional custom system prompt
            engine_parameters: Optional override parameters
            
        Returns:
            MultiLLM: Agent ready for conversation
        """
        final_engine_parameters = (engine_parameters or self.engine_parameters).copy()
        
        # Ensure base_url is set (critical for Rust server)
        if "base_url" not in final_engine_parameters:
            final_engine_parameters["base_url"] = "http://127.0.0.1:8000"
        
        # Create agent with Rust server configuration
        agent = MultiLLM(engine_parameters=final_engine_parameters)
        
        # Set system prompt if provided
        if system_prompt:
            agent.add_system_prompt(system_prompt)
            
        return agent

    def create_platform_agent(self, component_type: str) -> MultiLLM:
        """
        Create a pre-configured agent for specific platform components.
        
        Args:
            component_type: Type of component ('file_system', 'process', 'network', 'gui')
            
        Returns:
            MultiLLM: Pre-configured agent for the component
        """
        system_prompts = {
            'file_system': f"You are a {self.platform} file system expert. Help with file operations, directory management, and path handling.",
            'process': f"You are a {self.platform} process management expert. Help with process control, task management, and system monitoring.",
            'network': f"You are a {self.platform} networking expert. Help with network configuration, connectivity, and troubleshooting.",
            'gui': f"You are a {self.platform} GUI automation expert. Help with window management, UI interaction, and automation tasks.",
            'system_info': f"You are a {self.platform} system information expert. Help with hardware details, system status, and performance monitoring."
        }
        
        prompt = system_prompts.get(component_type, f"You are a {self.platform} system expert.")
        return self._create_agent(system_prompt=prompt)

    def batch_create_agents(self, system_prompts: List[str]) -> List[MultiLLM]:
        """
        Create multiple agents with different system prompts.
        
        Args:
            system_prompts: List of system prompts for each agent
            
        Returns:
            List[MultiLLM]: Multiple specialized agents
        """
        return [self._create_agent(prompt) for prompt in system_prompts]

    def get_engine_config(self) -> Dict:
        """
        Get the current engine configuration.
        
        Returns:
            Dict: Current Rust server configuration
        """
        return self.engine_parameters.copy()

    def update_engine_config(self, new_parameters: Dict):
        """
        Update engine configuration parameters.
        
        Args:
            new_parameters: New parameters to merge
        """
        self.engine_parameters.update(new_parameters)

    def test_connection(self) -> bool:
        """
        Test connection to Rust server.
        
        Returns:
            bool: True if connection successful
        """
        try:
            agent = self._create_agent(system_prompt="Test connection")
            response = agent.get_response("Hello, are you connected?")
            return bool(response and len(response.strip()) > 0)
        except Exception:
            return False
