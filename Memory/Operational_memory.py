# Entity/Memory/Operational_memory.py

import sys
import os
import inspect
import textwrap
import asyncio
import hashlib
from collections import OrderedDict
from typing import List, Type, AsyncGenerator, Dict, Any


# LRU Cache implementation for efficient memory management
class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


# In-memory caches for prompts and trajectories with size limits
_prompt_cache = LRUCache(50)  # Limit to 50 prompt entries
_trajectory_cache = LRUCache(20)  # Limit to 20 trajectory entries

# Max actions displayed based on GPU or device capacity
MAX_ACTIONS_DISPLAYED = 10


class OP_MEMORY:
    @staticmethod
    def _hash_key(*args) -> str:
        """Generate a consistent hash key from input args for caching."""
        m = hashlib.sha256()
        combined = "|".join(map(str, args))
        m.update(combined.encode('utf-8'))
        return m.hexdigest()

    # ------------------------
    # Section Generators
    # ------------------------
    @staticmethod
    def _generate_agent_methods_section(agent_class: Type, skipped_actions: List[str] = None,
                                        max_actions: int = MAX_ACTIONS_DISPLAYED) -> str:
        skipped_actions = skipped_actions or []
        action_methods = []
        for attr_name in dir(agent_class):
            if attr_name.startswith('_') or attr_name in skipped_actions:
                continue
            attr = getattr(agent_class, attr_name)
            if callable(attr) and hasattr(attr, "is_agent_action"):
                try:
                    signature = inspect.signature(attr)
                    docstring = (attr.__doc__ or "").strip()
                    method_desc = f"    def {attr_name}{signature}:"
                    if docstring:
                        method_desc += f"\n        '''{docstring}'''"
                    action_methods.append(method_desc)
                except (ValueError, TypeError):
                    continue
        return "\n".join(action_methods[:max_actions])

    @staticmethod
    def _generate_rules_section() -> str:
        return textwrap.dedent(
            """
            CRITICAL RULES FOR ANALYSIS:
            1. ALWAYS analyze the ACTUAL screenshot provided, not assumptions.
            2. If the screenshot shows a different state than expected, ADAPT your plan.
            3. NEVER continue with previous actions if the screen state has changed.
            4. Verify each action was successful before proceeding.

            YOUR RESPONSE MUST FOLLOW THIS EXACT STRUCTURE:

            PREVIOUS ACTION VERIFICATION:
            [Describe what the screenshot shows about the previous action's success or failure]

            SCREENSHOT ANALYSIS:
            [Detailed analysis of CURRENT screen state, applications visible, UI elements]

            NEXT ACTION:
            [Exactly one action to perform based on current screen state]

            GROUNDED ACTION:
            ```python
            agent.method_name(arguments_if_needed)
            ```
            """
        )

    @staticmethod
    def _generate_examples_section() -> str:
        return textwrap.dedent(
            """
            CRITICAL NOTES FOR GROUNDED ACTION:
            1. Only perform one action at a time.
            2. Do not put anything other than python code in the code block.
            3. You must use only the available methods provided above.
            4. Only return one code block with a single function call.
            5. NEVER specify coordinates or UI element descriptions in method parameters.
            6. The system will automatically detect UI elements and set coordinates for you.
            7. Examples of CORRECT usage:
               ```python
               agent.click()
               agent.type("hello world")
               agent.open("notepad")
               agent.hotkey(["ctrl", "s"])
               ```
            8. Examples of INCORRECT usage:
               ```python
               agent.click("menu button")  # NO
               agent.click(100, 200)       # NO
               agent.custom_method()       # NO
               ```
            9. Return with `agent.done()` when the task is fully complete.
            10. Return with `agent.fail()` only if the task is impossible after thorough trying.
            11. Use `agent.wait(seconds)` for delays between actions.
            12. My computer's password is 'osworld-public-evaluation' for sudo rights.
            13. Prefer hotkeys (`agent.hotkey()`) over clicking when possible.
            """
        )

    # ------------------------
    # Main Task Prompt Constructor
    # ------------------------
    @staticmethod
    def task_construct(agent_class: Type, skipped_actions: List[str] = None, task_desc: str = "TASK_DESCRIPTION",
                       max_actions: int = MAX_ACTIONS_DISPLAYED) -> str:
        """Construct a cached, hardware-aware task prompt for the agent."""
        cache_key = OP_MEMORY._hash_key(agent_class.__name__, tuple(skipped_actions or []), task_desc)

        # Use LRU cache get method
        cached_result = _prompt_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        op_memory = textwrap.dedent(
            f"""
            You are an expert in graphical user interfaces and Python code. 
            You are responsible for executing the task: `{task_desc}`.
            You are working in CURRENT_OS.

            You are provided with:
            1. A screenshot of the current time step.
            2. The history of your previous interactions with the UI.
            3. Access to the following class and methods to interact with the UI:

            class Agent:
            """
        )

        # Compose sections
        op_memory += OP_MEMORY._generate_agent_methods_section(agent_class, skipped_actions, max_actions)
        op_memory += "\n" + OP_MEMORY._generate_rules_section()
        op_memory += "\n" + OP_MEMORY._generate_examples_section()

        op_memory_final = op_memory.strip()

        # Use LRU cache put method
        _prompt_cache.put(cache_key, op_memory_final)
        return op_memory_final

    # ------------------------
    # Async Streaming Version
    # ------------------------
    @staticmethod
    async def stream_task_construct(agent_class: Type, skipped_actions: List[str] = None,
                                    task_desc: str = "TASK_DESCRIPTION", max_actions: int = MAX_ACTIONS_DISPLAYED) -> \
    AsyncGenerator[str, None]:
        """Stream the task prompt in sections for async generation."""
        yield textwrap.dedent(
            f"""
            You are an expert in graphical user interfaces and Python code. 
            You are responsible for executing the task: `{task_desc}`.
            You are working in CURRENT_OS.

            You are provided with:
            1. A screenshot of the current time step.
            2. The history of your previous interactions with the UI.
            3. Access to the following class and methods to interact with the UI:

            class Agent:
            """
        )
        await asyncio.sleep(0)

        yield OP_MEMORY._generate_agent_methods_section(agent_class, skipped_actions, max_actions)
        await asyncio.sleep(0)

        yield OP_MEMORY._generate_rules_section()
        await asyncio.sleep(0)

        yield OP_MEMORY._generate_examples_section()

    # ------------------------
    # Reflection / Trajectory
    # ------------------------
    REFLECTION_ON_TRAJECTORY = textwrap.dedent("""
    You are an expert computer use agent designed to reflect on the trajectory of a task.

    TASK: Analyze the provided trajectory and determine if the agent is making progress.

    CURRENT TRAJECTORY: A sequence of screenshots, reasoning, and actions.

    YOUR ROLE: Provide constructive feedback on the trajectory without suggesting specific future actions.
    """)

    # ------------------------
    # Coordinate / Vision Prompt
    # ------------------------
    PROMPT = textwrap.dedent("""
    You are an expert in graphical user interfaces. Your task is to process a text phrase and identify relevant UI elements.

    INPUT:
    - Phrase: Text description of what to find
    - Screenshot: Current screen state
    - Text elements: Available UI text with identifiers

    YOUR PROCESS:
    1. Analyze the phrase and screenshot to understand the context.
    2. Identify the most relevant UI element that matches the phrase.
    3. Return ONLY a structured JSON:

    {
        "x": int,
        "y": int,
        "element_id": str,
        "confidence": float
    }

    RULES:
    1. Respond ONLY with JSON.
    2. If multiple elements match, choose the most contextually appropriate.
    3. If no element matches, return {"x":0,"y":0,"element_id":"none","confidence":0.0}
    """)

    # ------------------------
    # System Prompt Getter
    # ------------------------
    @staticmethod
    def get_system_prompt(prompt_type: str = "default") -> str:
        prompts = {
            "default": "You are a helpful assistant specialized in computer operating tasks.",
            "coordinate_detection": OP_MEMORY.PROMPT,
            "reflection": OP_MEMORY.REFLECTION_ON_TRAJECTORY,
            "vision": "You are a computer vision expert. Analyze screenshots and describe UI elements.",
            "planning": "You are a task planning expert. Break down computer tasks into executable steps.",
        }
        return prompts.get(prompt_type, prompts["default"])

    # ------------------------
    # Action Help
    # ------------------------
    @staticmethod
    def get_action_help(agent_class: Type) -> str:
        help_text = ["Available Actions:"]
        for attr_name in dir(agent_class):
            attr = getattr(agent_class, attr_name)
            if callable(attr) and hasattr(attr, "is_agent_action"):
                docstring = (attr.__doc__ or "No documentation").split('\n')[0]
                help_text.append(f"- {attr_name}: {docstring}")
        return "\n".join(help_text)

    # ------------------------
    # Cache Management
    # ------------------------
    @staticmethod
    def clear_caches():
        """Clear all LRU caches - useful when agent methods change"""
        global _prompt_cache, _trajectory_cache
        _prompt_cache = LRUCache(50)
        _trajectory_cache = LRUCache(20)
        print("Operational memory caches cleared")

    @staticmethod
    def get_cache_stats() -> Dict[str, Any]:
        """Get current cache statistics for monitoring"""
        return {
            "prompt_cache_size": len(_prompt_cache.cache),
            "prompt_cache_capacity": _prompt_cache.capacity,
            "trajectory_cache_size": len(_trajectory_cache.cache),
            "trajectory_cache_capacity": _trajectory_cache.capacity
        }