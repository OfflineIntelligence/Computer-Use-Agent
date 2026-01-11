# Entity/Utils/Common_utilities.py

import sys
import os
import re
import time
import logging
from typing import Optional, Tuple, Any, Dict, List
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger("Common_utils")


@dataclass
class AgentContext:
    """Unified context object to replace redundant state tracking across components"""
    instruction: str
    current_step: int
    total_steps: int
    visual_summary: str  # Instead of full screenshots
    last_actions: List[str]  # Last 2 actions only
    reflection_summary: str  # Compressed reflections

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for easy serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentContext':
        """Create context from dictionary"""
        return cls(**data)

    def compress_actions(self, max_actions: int = 2) -> None:
        """Keep only the most recent actions"""
        if len(self.last_actions) > max_actions:
            self.last_actions = self.last_actions[-max_actions:]

    def update_step(self, current: int, total: int) -> None:
        """Update step progress"""
        self.current_step = current
        self.total_steps = total

    def add_action(self, action: str) -> None:
        """Add action and compress the list"""
        self.last_actions.append(action)
        self.compress_actions()


def llm_call(
        agent,
        temperature: float = 0.0,
        thinking: bool = False,
        max_retries: int = 3,
        **kwargs
) -> str:
    attempt = 0
    while attempt < max_retries:
        try:
            response = agent.get_response(
                temperature=temperature,
                thinking=thinking,
                **kwargs
            )

            if response is not None and str(response).strip():
                return str(response).strip()
            else:
                raise ValueError("Agent returned empty or None response")

        except Exception as e:
            attempt += 1
            logger.warning(f"LLM call attempt {attempt} failed: {e}")
            time.sleep(1.0 * attempt)

    logger.error(f"LLM call failed after {max_retries} attempts.")
    return ""


def split_thinking_response(full_response: str) -> Tuple[str, str]:
    if not full_response:
        return "", ""

    full_response = full_response.strip()
    thoughts, answer = "", full_response

    try:
        thoughts_match = re.search(r"<thoughts>(.*?)</thoughts>", full_response, re.DOTALL | re.IGNORECASE)
        if thoughts_match:
            thoughts = thoughts_match.group(1).strip()

        answer_match = re.search(r"<answer>(.*?)</answer>", full_response, re.DOTALL | re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).strip()
        elif thoughts:
            answer = re.sub(r"<thoughts>.*?</thoughts>", "", full_response, flags=re.DOTALL | re.IGNORECASE).strip()

    except Exception as e:
        logger.warning(f"XML parsing failed: {e}")
        thoughts, answer = "", full_response

    if not thoughts and "\n\n" in answer:
        parts = answer.split("\n\n", 1)
        reasoning_indicators = ["think", "reason", "analyze", "consider", "because", "therefore", "thus"]
        first_part_lower = parts[0].lower()

        if any(indicator in first_part_lower for indicator in reasoning_indicators):
            thoughts = parts[0].strip()
            answer = parts[1].strip() if len(parts) > 1 else ""

    if thoughts:
        thoughts = re.sub(r"</?thoughts>", "", thoughts, flags=re.IGNORECASE).strip()
    if answer:
        answer = re.sub(r"</?answer>", "", answer, flags=re.IGNORECASE).strip()

    return answer, thoughts


def string_parse(input_string: str) -> Optional[str]:
    if not input_string:
        return None

    input_string = input_string.strip()

    if input_string in ["WAIT", "DONE", "FAIL"]:
        return input_string

    extraction_patterns = [
        (r"<answer>(.*?)</answer>", re.DOTALL),
        (r"<action>(.*?)</action>", re.DOTALL),
        (r"```(?:\w+\s+)?(.*?)```", re.DOTALL),
        (r"'''(.*?)'''", re.DOTALL),
        (r"`(.*?)`", re.DOTALL),
        (r"\"\"\"(.*?)\"\"\"", re.DOTALL),
        (r'agent\.\w+\([^)]*\)', re.DOTALL),
    ]

    for pattern, flags in extraction_patterns:
        matches = re.findall(pattern, input_string, flags)
        if matches:
            code = matches[0].strip()
            if code:
                return code

    code_indicators = ["agent.", "pyautogui.", "import ", "def ", "class "]
    if any(indicator in input_string for indicator in code_indicators):
        return input_string

    return "FAIL"


def sanitize_code(code: str) -> str:
    if not code or "\n" not in code:
        return code

    quote_patterns = [
        (r'(")([^"]*?\\n[^"]*?)(")', '"""'),
        (r"(')([^']*?\\n[^']*?)(')", "'''"),
    ]

    for pattern, replacement in quote_patterns:
        matches = re.findall(pattern, code, re.DOTALL)
        if matches:
            for match in matches:
                old_quoted = f"{match[0]}{match[1]}{match[2]}"
                new_quoted = f"{replacement}{match[1]}{replacement}"
                code = code.replace(old_quoted, new_quoted, 1)
            break

    return code


def extract_first_agent_function(code_string: str) -> Optional[str]:
    if not code_string:
        return None

    # Look for agent.method() patterns
    patterns = [
        r'agent\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)',
        r'self\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)',
        r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*#\s*agent'
    ]

    for pattern in patterns:
        matches = re.findall(pattern, code_string, re.DOTALL)
        if matches:
            return matches[0]

    return None


def validate_coordinates(response: str) -> Optional[Tuple[int, int]]:
    # Add more robust coordinate validation
    if not response:
        return None

    # Try to extract coordinates from various formats
    coordinate_patterns = [
        r'\((\d+),\s*(\d+)\)',  # (123, 456)
        r'\[(\d+),\s*(\d+)\]',  # [123, 456]
        r'\{(\d+),\s*(\d+)\}',  # {123, 456}
        r'x[:=]\s*(\d+)\s*[,;]\s*y[:=]\s*(\d+)',  # x:123, y:456
        r'(\d+)\s*,\s*(\d+)',  # 123, 456
        r'(\d+)\s+(\d+)',  # 123 456
    ]

    for pattern in coordinate_patterns:
        matches = re.findall(pattern, response)
        if matches:
            try:
                x = int(matches[0][0])
                y = int(matches[0][1])
                # Validate reasonable screen coordinates
                if 0 <= x <= 5000 and 0 <= y <= 5000:
                    return (x, y)
            except (ValueError, IndexError):
                continue

    return None


def normalize_coordinates(coords: Any, screen_width: int = 1920, screen_height: int = 1080) -> Optional[
    Tuple[int, int]]:
    """
    Normalize coordinates from various formats to (x, y) tuple
    """
    if not coords:
        return None

    try:
        # Handle tuple/list
        if isinstance(coords, (tuple, list)) and len(coords) >= 2:
            x, y = int(coords[0]), int(coords[1])
            if 0 <= x <= screen_width and 0 <= y <= screen_height:
                return (x, y)

        # Handle string
        elif isinstance(coords, str):
            return validate_coordinates(coords)

        # Handle dictionary
        elif isinstance(coords, dict):
            x = coords.get('x', coords.get('X', 0))
            y = coords.get('y', coords.get('Y', 0))
            if x != 0 or y != 0:  # Only return if non-zero
                return (int(x), int(y))

    except (ValueError, TypeError):
        pass

    return None


def create_initial_context(instruction: str, total_steps: int = 1) -> AgentContext:
    """Create initial AgentContext for a new command"""
    return AgentContext(
        instruction=instruction,
        current_step=0,
        total_steps=total_steps,
        visual_summary="",
        last_actions=[],
        reflection_summary=""
    )


def compress_reflections(reflections: List[str]) -> str:
    """Compress reflection list into summary string"""
    if not reflections:
        return ""

    # Keep only the most recent 2 reflections
    recent_reflections = reflections[-2:] if len(reflections) > 2 else reflections

    # Create a concise summary
    if len(recent_reflections) == 1:
        return recent_reflections[0]
    else:
        return f"Recent reflections: {', '.join(recent_reflections)}"


# Planning Data Classes
@dataclass
class PlanStep:
    """Represents a single step in an execution plan"""
    step_id: int
    action_type: str
    description: str
    target: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    required_elements: Optional[List[str]] = None
    max_retries: int = 3
    retry_count: int = 0
    status: str = "pending"  # pending, executing, completed, failed
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlanStep':
        return cls(**data)

@dataclass
class ExecutionPlan:
    """Represents a complete execution plan for a task"""
    plan_id: str
    original_command: str
    steps: List[PlanStep]
    status: str = "active"  # active, completed, failed, replaced
    current_step_index: int = 0
    completed_steps: List[int] = None
    failed_steps: List[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def __post_init__(self):
        if self.completed_steps is None:
            self.completed_steps = []
        if self.failed_steps is None:
            self.failed_steps = []
        
    def get_current_step(self) -> Optional[PlanStep]:
        """Get the current step to execute"""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None
    
    def mark_step_completed(self, step_id: int):
        """Mark a step as completed"""
        if step_id not in self.completed_steps:
            self.completed_steps.append(step_id)
        # Move to next step
        if self.current_step_index < len(self.steps) - 1:
            self.current_step_index += 1
        else:
            self.status = "completed"
            
    def mark_step_failed(self, step_id: int):
        """Mark a step as failed"""
        if step_id not in self.failed_steps:
            self.failed_steps.append(step_id)
        # Increment retry count for the step
        if 0 <= step_id < len(self.steps):
            self.steps[step_id].retry_count += 1
            
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Convert steps to dictionaries
        data['steps'] = [step.to_dict() for step in self.steps]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionPlan':
        # Convert step dictionaries back to PlanStep objects
        steps = [PlanStep.from_dict(step_data) for step_data in data['steps']]
        data['steps'] = steps
        return cls(**data)