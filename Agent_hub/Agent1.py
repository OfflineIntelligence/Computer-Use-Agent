# Entity/Agent_hub/Agent1.py









import sys
import os
import logging
import platform as _platform
from typing import Dict, List, Tuple, Optional
import threading
import time

######### PROJECT ROOT
######################################################

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

######### MODULES-IMPORT
######################################################

from Entity.Agent_hub.Execution_console import ACTIO
from Entity.Agent_hub.Action_board import AB
from Entity.Utils.Config import Config



#########
######################################################

logger = logging.getLogger("_Aud.io")



######################################################
#########

class AudioAgent:
    def __init__(self, engine_parameters: Dict, action_agent: ACTIO, platform: str = _platform.system().lower()):
        self.engine_parameters = engine_parameters
        self.action_agent = action_agent
        self.platform = platform
    def reset(self) -> None:
        pass
    def predict(self, instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
        return {}, []


class Agent(AudioAgent):

    def __init__(
            self,
            platform: str = None,
            trajectory_maxlen: int = 16,
            enable_reflection: bool = True,
    ):
        agent_config = Config.get_agent_config()
        rust_config = Config.get_rust_config()
        platform = platform or agent_config["platform"]

        engine_parameters = {
            "base_url": rust_config["base_url"],
            "temperature": 0.1,
            "model": rust_config["model"],
            "coordinate_width": agent_config["coordinate_width"],
            "coordinate_height": agent_config["coordinate_height"],
            "max_tokens": rust_config["max_tokens"],
            "stream": False,
            "timeout": 120,
        }

        action_agent = ACTIO(
            platform=platform,
            engine_parameters=engine_parameters,
        )

        super().__init__(engine_parameters=engine_parameters, action_agent=action_agent, platform=platform)
        self.max_task_iterations = agent_config.get("max_task_iterations", 11)
        self.trajectory_maxlen = trajectory_maxlen
        self.enable_reflection = enable_reflection
        self.reset()



    def reset(self) -> None:
        self.executor = AB(
            engine_parameters=self.engine_parameters,
            action_agent=self.action_agent,
            platform=self.platform,
            trajectory_maxlen=self.trajectory_maxlen,
            reflection=self.enable_reflection,
        )

    def predict(self, instruction: str, observation: Dict = None) -> Tuple[Dict, List[str]]:
        """Stateless across user intents, stateful within a single multi-step task (up to 11+ steps)."""
        MAX_ITERATIONS = self.max_task_iterations  # Allow complex tasks (e.g., booking, multi-app workflows)
        MIN_ITERATIONS = 2   # Ensure at least 2 attempts before stagnation checks
        all_actions = []
        combined_info = {}

        # 🔥 CRITICAL: Full reset BEFORE this task starts → no carryover from prior commands
        self.reset()

        # Initialize fresh command state
        self.executor.command_state = {
            "active_command": instruction,
            "current_step": 0,
            "total_steps": 0,
            "step_results": [],
            "command_start_time": time.time()
        }

        # Clear stale visual memory
        self.action_agent.last_visual_feedback = None

        for iteration in range(MAX_ITERATIONS):
            logger.info(f"Iteration {iteration + 1}/{MAX_ITERATIONS} for: '{instruction}'")

            try:
                # ActionBoard generates plan and provides current step
                executor_info, actions = self.executor.action_generator(instruction=instruction)
                
                # Extract current step information for execution
                current_step_info = executor_info.get("current_step")
                if current_step_info:
                    # Execute the planned step using Execution Console
                    step_meta, step_result = self.action_agent.execute_planned_step(
                        step_description=current_step_info.get("description", ""),
                        action_type=current_step_info.get("action_type", "wait"),
                        target=current_step_info.get("target"),
                        parameters=current_step_info.get("parameters", {})
                    )
                    
                    # Update plan based on execution result
                    plan_id = executor_info.get("plan_id")
                    step_id = current_step_info.get("step_id")
                    if plan_id and step_id is not None:
                        self.executor.update_plan_from_result(
                            self.executor.active_plans[plan_id], 
                            step_id, 
                            step_meta
                        )
                    
                    # Add execution result to actions
                    all_actions.append(step_result)
                    combined_info.update(step_meta)
                else:
                    # Fallback to legacy action handling
                    all_actions.extend(actions)
                    combined_info.update(executor_info)

                # Log progress (optional; disable in prod for latency)
                current_step = self.executor.command_state.get("current_step", 0)
                total_steps = self.executor.command_state.get("total_steps", 0)
                if total_steps > 0:
                    logger.debug(f"Plan progress: {current_step}/{total_steps}")

                # 🛑 Smart early stopping (respects multi-step needs)
                if self._should_stop_iteration(executor_info, actions, iteration):
                    logger.info(f"Task completed or failed after {iteration + 1} iterations")
                    break

            except Exception as e:
                logger.error(f"Error in iteration {iteration + 1}: {e}", exc_info=True)
                all_actions.append("wait")
                combined_info["error"] = str(e)
                # Allow recovery for first few errors; stop after repeated failures
                if iteration >= 3:
                    logger.warning("Stopping due to repeated errors")
                    break

            # Brief pause for UI to stabilize (keep minimal)
            time.sleep(0.25)

            # 🔍 Stagnation detection (only after MIN_ITERATIONS)
            if (iteration >= MIN_ITERATIONS and
                len(all_actions) >= 2 and
                all("wait" in str(a).lower() for a in all_actions[-2:])):
                logger.info("Stopping: No progress detected (repeated 'wait' actions)")
                break

        # 🔥 Final cleanup: purge LLM message history to ensure next task starts fresh
        self.executor.del_mssgs()

        return combined_info, all_actions



    def _should_stop_iteration(self, executor_info: Dict, actions: List, iteration: int) -> bool:
        """Enhanced iteration stopping criteria with configurable max iterations."""

        # Stop if task is explicitly marked as done
        if any("done" in str(action).lower() for action in actions):
            logger.info("Stopping: Task marked as DONE")
            return True

        # Stop if task failed
        if any("fail" in str(action).lower() for action in actions):
            logger.info("Stopping: Task marked as FAIL")
            return True

        # Stop if no meaningful action after multiple iterations
        if (len(actions) == 1 and
                "wait" in str(actions[0]).lower() and
                iteration >= 2):  # Allow 2 wait iterations before stopping
            logger.info("Stopping: No meaningful actions after multiple iterations")
            return True

        # Check command state from Action Board
        if not self.executor.command_state.get("active_command"):
            logger.info("Stopping: Action Board indicates command completed")
            return True

        # 🔥 STOP if maximum iterations reached (configurable via self.max_task_iterations)
        if iteration >= self.max_task_iterations - 1:
            logger.info(f"Stopping: Maximum iterations ({self.max_task_iterations}) reached")
            return True

        # Stop if visual feedback indicates completion
        visual_feedback = executor_info.get("visual_feedback", {})
        if (visual_feedback.get("action_effective") and
                not visual_feedback.get("significant_change") and
                iteration > 0):
            logger.info("Stopping: Action effective but no further changes needed")
            return True

        return False



    #################################################################################
    #################################################################################   CHAT INTERFACE //////////////
    #################################################################################



    def chat_interface(self):
        """Interactive chat interface for commanding the system"""
        print("\n" + "=" * 60)
        print("🤖 COMPUTER USE AGENT - CHAT INTERFACE")
        print("=" * 60)
        print("Commands:")
        print("  - Type any computer task")
        print("  - 'screenshot' - Capture and analyze current screen")
        print("  - 'status' - Show agent status")
        print("  - 'reset' - Reset agent state")
        print("  - 'quit' or 'exit' - Exit the interface")
        print("=" * 60)

        while True:
            try:
                user_input = input("\n🎯 Command: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break

                elif user_input.lower() == 'screenshot':
                    print("📸 Capturing screenshot and analyzing...")
                    try:
                        screenshot = self.action_agent.capture_screenshot()
                        analysis = self.action_agent._describe_screenshot_elements(screenshot)
                        print(f"📊 Screen Analysis:\n{analysis}")
                    except Exception as e:
                        print(f"❌ Screenshot failed: {e}")

                elif user_input.lower() == 'status':
                    print("📊 Agent Status:")
                    print(f"  Platform: {self.platform}")
                    print(f"  Model: {self.engine_parameters.get('model', 'Unknown')}")
                    print(f"  Server: {self.engine_parameters.get('base_url', 'Unknown')}")
                    print(f"  Safety Mode: {getattr(self.action_agent, 'safety_mode', 'Unknown')}")

                    # Check backend connection
                    try:
                        from Entity.Core.Core_engine import RustServer
                        engine = RustServer(**self.engine_parameters)
                        # NOTE: Using a simple test message instead of a full generate call
                        # to check connectivity without full model inference.
                        # Assuming RustServer has a simpler connectivity check or a light endpoint.
                        # For now, let's assume successful initialization implies basic connectivity.
                        print("  ✅ Backend: Assumed Connected (Initialization successful)")
                    except Exception as e:
                        print(f"  ❌ Backend: Connection failed - {e}")

                elif user_input.lower() == 'reset':
                    print("🔄 Resetting agent...")
                    self.reset()
                    print("✅ Agent reset complete")

                elif user_input:
                    # Process computer task
                    print(f"🚀 Executing: '{user_input}'")
                    print("⏳ Thinking...")

                    def execute_task():
                        try:
                            # The predict method now handles the full iterative loop
                            info, actions = self.predict(user_input)

                            print(f"\n✅ Execution Complete!")
                            print(f"📋 Final Plan: {info.get('executor_plan', 'No final plan generated')}")
                            print(f"💭 Last Thoughts: {info.get('plan_thoughts', 'No thoughts')}")
                            print(f"⚡ Last Action: {info.get('plan_code', 'No action')}")

                            # Show visual feedback if available
                            visual_feedback = info.get('visual_feedback', {})
                            if visual_feedback:
                                print(f"📊 Last Visual Feedback: {visual_feedback}")

                            # Show all actions taken
                            if actions:
                                print(f"🎯 Total Actions Taken: {len(actions)}")
                                print(f"  Details: {actions}")

                        except Exception as e:
                            print(f"❌ Execution failed: {e}")
                            logger.error(f"Task execution error: {e}", exc_info=True)

                    # Run in a thread to prevent blocking
                    task_thread = threading.Thread(target=execute_task)
                    task_thread.daemon = True
                    task_thread.start()

                    # Simple progress indicator
                    for i in range(5):  # Increased progress steps to reflect potential iterative execution
                        if not task_thread.is_alive():
                            break
                        print(".", end='', flush=True)
                        time.sleep(1)
                    print()

                else:
                    print("❌ Please enter a command")

            except KeyboardInterrupt:
                print("\n\n🛑 Interrupted by user")
                break
            except Exception as e:
                print(f"❌ Interface error: {e}")
                logger.error(f"Chat interface error: {e}", exc_info=True)


def quick_test():
    """Quick test function"""
    print("🧪 Running quick test...")
    try:
        agent = Agent()
        print("✅ Agent created successfully!")
        print(f"   Platform: {agent.platform}")
        print(f"   Model: {agent.engine_parameters.get('model')}")
        print(f"   Server: {agent.engine_parameters.get('base_url')}")
        return agent
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return None


if __name__ == "__main__":
    print("🤖 Agent1.py - Computer Use Agent")
    print("=" * 50)

    # Run quick test
    agent = quick_test()

    if agent:
        # Start chat interface
        agent.chat_interface()
    else:
        print("❌ Cannot start chat interface due to initialization errors")