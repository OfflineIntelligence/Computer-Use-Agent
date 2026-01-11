# Entity/Agent_hub/Action_board.py

import sys
import os
import logging
import json
import uuid
import re
import time
import threading
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import asdict

from Entity.Core.Core_engine import RustServer
from Entity.Agent_hub.Execution_console import ACTIO
from Entity.Core.Component import BaseComponent
from Entity.Memory.Operational_memory import OP_MEMORY
from Entity.Utils.Common_utilities import (
    llm_call,
    ExecutionPlan,
    PlanStep
)

logger = logging.getLogger("Audio.agent")


class AB(BaseComponent):
    """
    Pure Planning Component - Generates and manages execution plans
    Does NOT execute actions - only plans and adjusts strategies
    """

    def __init__(
            self,
            engine_parameters: Dict,
            action_agent: ACTIO,
            platform: str = "windows",
            trajectory_maxlen: int = 8,
            reflection: bool = True,
    ):
        super().__init__(engine_parameters, platform)
        self.action_agent = action_agent
        self.trajectory_maxlen = trajectory_maxlen
        self.reflection = reflection
        self.temperature = engine_parameters.get("temperature", 0.1)  # Lower temp for planning
        self.rust_engine = RustServer(**engine_parameters)
        self.command_state_lock = threading.RLock()

        # Planning-specific state
        self.active_plans: Dict[str, ExecutionPlan] = {}
        self.plan_history: List[ExecutionPlan] = []
        self.planning_agent = None

        self.reset()

    def reset(self):
        """Reset planning state and initialize planning agent"""
        planning_prompt = self._get_planning_system_prompt()
        self.planning_agent = self._create_agent(planning_prompt)
        self.active_plans = {}
        self.plan_history = []
        logger.info("Action Board reset - ready for planning")

    def _get_planning_system_prompt(self) -> str:
        """Get the specialized planning system prompt"""
        return f"""
You are an expert computer task planner for {self.platform} systems. Your role is to break down user commands into executable, step-by-step plans.

PLANNING PRINCIPLES:
1. Break complex tasks into simple, atomic steps that can be executed independently
2. Each step should be clear, specific, and executable by an automation system
3. Consider the platform: {self.platform} - use appropriate applications and shortcuts
4. Include necessary waits between steps for UI responsiveness
5. Plan for verification where needed
6. Use standard applications and system commands

AVAILABLE ACTION TYPES:
- open_app: Open an application (target: application name)
- click_element: Click on a UI element (target: element description/text)
- type_text: Type text into a field (parameters: {{"text": "content"}})
- press_key: Press keyboard key(s) (target: key name like "enter", "tab")
- navigate: Navigate to URL in browser (target: URL)
- wait: Wait for specified time (parameters: {{"seconds": number}})
- hotkey: Use keyboard shortcut (target: key combination like "ctrl+s")
- scroll: Scroll the screen (parameters: {{"clicks": number, "direction": "up/down"}})
- done: Mark task as completed

RESPONSE FORMAT:
Return ONLY a JSON array of step objects. Example:
[
  {{"step_id": 1, "action_type": "open_app", "description": "Open web browser", "target": "chrome"}},
  {{"step_id": 2, "action_type": "navigate", "description": "Go to website", "target": "https://example.com"}},
  {{"step_id": 3, "action_type": "click_element", "description": "Click search box", "target": "search input"}},
  {{"step_id": 4, "action_type": "type_text", "description": "Type search query", "parameters": {{"text": "query"}}}},
  {{"step_id": 5, "action_type": "press_key", "description": "Press enter to search", "target": "enter"}}
]

IMPORTANT:
- Be specific in descriptions but concise in execution
- Include necessary waits for applications to load
- Use common application names that work on {self.platform}
- Consider alternative approaches for reliability
"""

    def generate_plan(self, instruction: str) -> ExecutionPlan:
        """
        Generate a complete execution plan from user instruction

        Args:
            instruction: User command like "Open notepad and type hello world"

        Returns:
            ExecutionPlan: Structured plan with steps to execute
        """
        plan_id = str(uuid.uuid4())[:8]

        logger.info(f"Generating plan for: '{instruction}'")

        try:
            # Use LLM to generate step-by-step plan
            user_prompt = f"""
TASK: {instruction}

PLATFORM: {self.platform}

Please break this task down into executable steps. Consider:
1. What applications need to be opened?
2. What UI elements need interaction?
3. What text needs to be typed?
4. What navigation is required?
5. What verification might be needed?

Return ONLY the JSON array of steps.
"""

            self.planning_agent.reset()
            self.planning_agent.add_message(user_prompt, role="user")

            response = llm_call(self.planning_agent, temperature=self.temperature, max_retries=2)

            # Parse the response into structured plan steps
            steps = self._parse_plan_response(response, instruction)

            # Create the execution plan
            plan = ExecutionPlan(
                plan_id=plan_id,
                original_command=instruction,
                steps=steps
            )

            self.active_plans[plan_id] = plan
            logger.info(f"Generated plan {plan_id} with {len(steps)} steps")

            return plan

        except Exception as e:
            logger.error(f"Plan generation failed for '{instruction}': {e}")
            # Return a minimal fallback plan
            return self._create_fallback_plan(plan_id, instruction)

    def _parse_plan_response(self, response: str, original_command: str) -> List[PlanStep]:
        """
        Parse LLM response into structured plan steps

        Args:
            response: LLM response containing step definitions
            original_command: Original user command for context

        Returns:
            List[PlanStep]: Parsed plan steps
        """
        try:
            # Clean the response and extract JSON
            cleaned_response = self._clean_llm_response(response)

            # Try to parse as JSON array
            steps_data = json.loads(cleaned_response)

            if not isinstance(steps_data, list):
                raise ValueError("Expected JSON array")

            steps = []
            for step_data in steps_data:
                step = PlanStep(
                    step_id=step_data.get("step_id", len(steps) + 1),
                    action_type=step_data.get("action_type", "wait"),
                    description=step_data.get("description", "Unnamed step"),
                    target=step_data.get("target"),
                    parameters=step_data.get("parameters", {}),
                    required_elements=step_data.get("required_elements", []),
                    max_retries=step_data.get("max_retries", 3)
                )

                # Validate step has required fields
                if not step.action_type or not step.description:
                    logger.warning(f"Skipping invalid step: {step_data}")
                    continue

                steps.append(step)

            logger.debug(f"Successfully parsed {len(steps)} plan steps")
            return steps

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse plan response: {e}. Using fallback planning.")
            return self._create_fallback_steps(original_command)

    def _clean_llm_response(self, response: str) -> str:
        """Clean and extract JSON from LLM response"""
        # Remove markdown code blocks
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)

        # Extract JSON array pattern
        json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
        if json_match:
            return json_match.group()

        # If no array found, try to find any JSON structure
        brace_match = re.search(r'\{.*\}', response, re.DOTALL)
        if brace_match:
            potential_json = brace_match.group()
            try:
                data = json.loads(potential_json)
                if isinstance(data, list):
                    return potential_json
                elif "steps" in data and isinstance(data["steps"], list):
                    return json.dumps(data["steps"])
            except json.JSONDecodeError:
                pass

        # Return original response as last resort
        return response.strip()

    def _create_fallback_steps(self, instruction: str) -> List[PlanStep]:
        """Create basic fallback steps when parsing fails"""
        steps = []

        # Simple keyword-based planning
        instruction_lower = instruction.lower()

        if any(word in instruction_lower for word in ["open", "start", "launch"]):
            # Extract application name
            app_match = re.search(r'(?:open|start|launch)\s+([^\s]+)', instruction_lower)
            app_name = app_match.group(1) if app_match else "application"
            steps.append(PlanStep(1, "open_app", f"Open {app_name}", app_name))

        if any(word in instruction_lower for word in ["type", "write", "enter"]):
            # Extract text to type
            text_match = re.search(r'(?:type|write|enter)\s+["\']?([^"\']+)["\']?', instruction_lower)
            text_content = text_match.group(1) if text_match else "text"
            steps.append(PlanStep(len(steps) + 1, "type_text", f"Type {text_content}",
                                  parameters={"text": text_content}))

        if any(word in instruction_lower for word in ["click", "press", "select"]):
            # Extract element to click
            element_match = re.search(r'(?:click|press|select)\s+([^\s]+)', instruction_lower)
            element_name = element_match.group(1) if element_match else "button"
            steps.append(PlanStep(len(steps) + 1, "click_element", f"Click {element_name}", element_name))

        if not steps:
            steps.append(PlanStep(1, "open_app", f"Open application for: {instruction}", instruction))

        steps.append(PlanStep(len(steps) + 1, "done", "Task execution completed"))

        return steps

    def _create_fallback_plan(self, plan_id: str, instruction: str) -> ExecutionPlan:
        """Create a fallback plan when generation fails completely"""
        steps = self._create_fallback_steps(instruction)
        return ExecutionPlan(plan_id, instruction, steps)

    def get_next_step(self, plan: ExecutionPlan, previous_result: Dict = None) -> Optional[PlanStep]:
        """
        Get the next step to execute, considering previous execution results

        Args:
            plan: The execution plan
            previous_result: Result from the previously executed step

        Returns:
            Optional[PlanStep]: Next step to execute, or None if plan is complete
        """
        if plan.status != "active":
            logger.info(f"Plan {plan.plan_id} is {plan.status}, no next step")
            return None

        current_step = plan.get_current_step()
        if not current_step:
            logger.info(f"Plan {plan.plan_id} has no more steps")
            return None

        # Handle previous step result if provided
        if previous_result and not previous_result.get("success", True):
            failed_step_id = plan.current_step_index - 1
            plan.mark_step_failed(failed_step_id)

            failed_step = plan.steps[failed_step_id] if failed_step_id < len(plan.steps) else None

            if failed_step and failed_step.retry_count >= failed_step.max_retries:
                logger.warning(f"Step {failed_step_id} exceeded max retries, skipping to next step")
                plan.current_step_index += 1
                return self.get_next_step(plan, previous_result)

        logger.debug(f"Plan {plan.plan_id} - Next step: {current_step.step_id}")
        return current_step

    def update_plan_from_result(self, plan: ExecutionPlan, step_id: int, result: Dict):
        """
        Update plan based on step execution result

        Args:
            plan: The execution plan to update
            step_id: ID of the step that was executed
            result: Execution result dictionary
        """
        if result.get("success", False):
            plan.mark_step_completed(step_id)
            logger.info(f"Plan {plan.plan_id} - Step {step_id} completed successfully")
        else:
            plan.mark_step_failed(step_id)
            error_msg = result.get('error', 'Unknown error')
            logger.warning(f"Plan {plan.plan_id} - Step {step_id} failed: {error_msg}")

    def replan_on_failure(self, plan: ExecutionPlan, failed_step: PlanStep, failure_reason: str) -> ExecutionPlan:
        """
        Generate a new plan when execution fails critically

        Args:
            plan: The original plan that failed
            failed_step: The step that failed
            failure_reason: Reason for failure

        Returns:
            ExecutionPlan: New adjusted plan
        """
        logger.info(f"Replanning due to critical failure in step {failed_step.step_id}: {failure_reason}")

        # Create replanning instruction with failure context
        replan_instruction = (
            f"{plan.original_command} - Previous approach failed at step {failed_step.step_id}: "
            f"{failed_step.description}. Failure reason: {failure_reason}. "
            f"Please suggest an alternative approach."
        )

        # Generate new plan
        new_plan = self.generate_plan(replan_instruction)

        # Archive the old plan
        plan.status = "replaced"
        self.plan_history.append(plan)

        logger.info(f"Generated new plan {new_plan.plan_id} to replace failed plan {plan.plan_id}")
        return new_plan

    def get_plan_status(self, plan_id: str) -> Dict[str, Any]:
        """
        Get current status of a plan

        Args:
            plan_id: ID of the plan to check

        Returns:
            Dict: Plan status information
        """
        if plan_id in self.active_plans:
            plan = self.active_plans[plan_id]
            current_step = plan.get_current_step()

            return {
                "plan_id": plan.plan_id,
                "original_command": plan.original_command,
                "status": plan.status,
                "current_step": plan.current_step_index,
                "current_step_description": current_step.description if current_step else None,
                "total_steps": len(plan.steps),
                "completed_steps": len(plan.completed_steps),
                "failed_steps": len(plan.failed_steps),
                "progress": f"{len(plan.completed_steps)}/{len(plan.steps)}"
            }

        return {"error": f"Plan {plan_id} not found"}

    def get_all_plans_status(self) -> Dict[str, Any]:
        """Get status of all active plans"""
        return {
            "active_plans": len(self.active_plans),
            "historical_plans": len(self.plan_history),
            "plans": {pid: self.get_plan_status(pid) for pid in self.active_plans.keys()}
        }

    # Legacy interface for backward compatibility
    def action_generator(self, instruction: str) -> Tuple[Dict, List]:
        """
        Legacy interface - generates a plan but returns empty actions
        Maintains compatibility with existing code during transition

        Args:
            instruction: User command

        Returns:
            Tuple[Dict, List]: Plan information and empty actions list
        """
        plan = self.generate_plan(instruction)
        current_step = plan.get_current_step()

        executor_info = {
            "plan": plan.to_dict(),
            "current_step": current_step.to_dict() if current_step else None,
            "plan_id": plan.plan_id,
            "instruction": instruction,
            "status": "plan_generated",
            "total_steps": len(plan.steps),
            "message": "Plan generated - use execute_plan_step for execution"
        }

        # Return empty actions list - execution will be handled separately
        return executor_info, []

    def del_mssgs(self):
        """Clean up planning agent message history"""
        if self.planning_agent and hasattr(self.planning_agent, 'messages'):
            # Keep only system prompt and last exchange
            if len(self.planning_agent.messages) > 3:
                self.planning_agent.messages = [
                    self.planning_agent.messages[0],  # System prompt
                    self.planning_agent.messages[-2] if len(self.planning_agent.messages) >= 2 else None,
                    self.planning_agent.messages[-1] if len(self.planning_agent.messages) >= 1 else None
                ]
                # Remove None values
                self.planning_agent.messages = [msg for msg in self.planning_agent.messages if msg is not None]

    def cleanup_old_plans(self, max_active_plans: int = 10):
        """Clean up old plans to prevent memory buildup"""
        if len(self.active_plans) > max_active_plans:
            # Remove oldest completed plans
            completed_plans = [pid for pid, plan in self.active_plans.items()
                               if plan.status in ["completed", "failed"]]

            for pid in completed_plans[:len(self.active_plans) - max_active_plans]:
                del self.active_plans[pid]
                logger.debug(f"Cleaned up old plan: {pid}")