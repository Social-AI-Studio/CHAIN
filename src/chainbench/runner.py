"""
Main runner for the CHAIN benchmark (chainbench).

This module coordinates the execution of benchmark tasks, bringing together
environments, agents, tasks, and evaluation components.
"""

import json
import os
import time
from typing import List, Optional, Dict, Tuple, Callable
from chainbench.agents import VLMAgent, ActionCandidate, StepSelectionManager, create_reward_agent, BaseRewardAgent
from chainbench.core import Config, ENVIRONMENT_REGISTRY, TASK_REGISTRY, AGENT_REGISTRY
from chainbench.core.base import Observation, State, TaskResult, Action, EvaluationResult
from chainbench.environment.base_env import PhysicsEnvironment
try:
    import pybullet as p
except ImportError:
    from chainbench.environment.base_env import p
from chainbench.evaluation import BenchmarkEvaluator
from chainbench.tasks import PhysicsTask
from chainbench.utils.logger import ExperimentLogger
from chainbench.utils.display import ProgressDisplay, StatusDisplay, LiveLogger

class BenchmarkRunner:
    """Main runner for chainbench benchmark execution."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = ExperimentLogger(
            log_dir=config.runner.log_dir,
            experiment_name=config.runner.experiment_name
        )

        # Initialize components
        self.environment: Optional[PhysicsEnvironment] = None
        self.task: Optional[PhysicsTask] = None
        self.agent: Optional[VLMAgent] = None
        self.evaluator: Optional[BenchmarkEvaluator] = None
        
        # Step selection components
        self.reward_agent: Optional[BaseRewardAgent] = None
        self.step_selection_manager: Optional[StepSelectionManager] = None
        
        # Execution state
        # Each entry: {"response": str, "actions": List[Action], "observations": List[Observation]}
        self.interaction_history: List[Dict] = []
        self.start_time = None
        
        # Display utilities
        self.live_logger = LiveLogger(verbose=True)
        self.progress_display = None
        
    def setup(self) -> None:
        """Setup all components for benchmark execution."""
        
        # Ensure log directory exists (deferred from RunnerConfig to avoid side-effects on import)
        os.makedirs(self.config.runner.log_dir, exist_ok=True)
        
        # Create task first (required for environment configuration)
        self.task = self._create_task()
        
        # Create environment based on task configuration
        self.environment = self._create_environment()
        
        # Create agent
        self.agent = self._create_agent()
        
        # Create evaluator
        self.evaluator = self._create_evaluator()
        
        # Setup step selection if enabled
        self._setup_step_selection()
        
        self.live_logger.log_info("Benchmark setup complete")
    
    def _setup_step_selection(self) -> None:
        """Setup step selection components if enabled."""
        if self.config.step_selection and self.config.step_selection.enabled:
            mode = "greedy" if self.config.step_selection.top_k == 1 else f"beam search (width={self.config.step_selection.top_k})"
            self.live_logger.log_info(
                f"Initializing step selection: {mode}, "
                f"method='{self.config.step_selection.selection_method}'"
            )
            
            # Create reward agent for action scoring/comparison
            self.reward_agent = create_reward_agent(self.config.step_selection)
            
            # Create step selection manager
            self.step_selection_manager = StepSelectionManager(
                self.config.step_selection,
                self.reward_agent
            )
            
            self.live_logger.log_info("Step selection components initialized")

    def _create_task(self) -> PhysicsTask:
        """Create task based on configuration."""
        task_cls = TASK_REGISTRY.get(self.config.task.type)
        if task_cls is None:
            raise RuntimeError(f"Unknown or unsupported task type: {self.config.task.type}")
        return task_cls(self.config.task)
        
    def _create_environment(self) -> PhysicsEnvironment:
        """Create physics environment based on task type."""
        # Create environment based on task type
        env_cls = ENVIRONMENT_REGISTRY.get(self.config.environment.type)
        if env_cls is None:
            raise RuntimeError(f"Unknown or unsupported environment type: {self.config.environment.type}")
        return env_cls(self.config.environment)
            
    def _create_agent(self) -> VLMAgent:
        """Create VLM agent based on configuration."""
        agent_cls = AGENT_REGISTRY.get(self.config.agent.type)
        if agent_cls is None:
            raise RuntimeError(f"Unknown or unsupported agent type: {self.config.agent.type}")
        return agent_cls(self.config.agent)
            
    def _create_evaluator(self) -> BenchmarkEvaluator:
        """Create evaluator with judge if configured."""
        return BenchmarkEvaluator(self.config.judgement)
        
    def run_single_task(self) -> TaskResult:
        """Run a single task instance."""
        self._validate_components()
        StatusDisplay.print_header(f"Starting Task: {self.task.task_id}")
        
        self.interaction_history = []
        self.start_time = time.time()
        
        try:
            # Initialize task execution
            observation = self._initialize_task_execution()
            
            # Main interaction loop
            total_steps = self._execute_interaction_loop(observation)
            
            # Create and return task result
            return self._create_task_result(total_steps)
            
        except Exception as e:
            return self._handle_task_failure(e)
            
        finally:
            self._cleanup_environment()
            
    def _validate_components(self) -> None:
        """Validate that all required components are initialized."""
        if not all([self.environment, self.task, self.agent]):
            raise RuntimeError("Components not properly initialized. Call setup() first.")
    
    def _initialize_task_execution(self) -> Observation:
        """Initialize environment and set up task prompts."""
        # Reset environment
        self.live_logger.log_action("Initializing environment")
        observation = self.task.configure_environment(self.environment)
        # Record initial reset action
        self.interaction_history.append({
            "response": "",
            "actions": [Action(action_type="reset", parameters={})],
            "observations": [observation]
        })
        self.live_logger.log_result("Environment ready")
        
        # Setup task prompts
        self.live_logger.log_action("Preparing task prompts")
        self.agent.set_system_prompt(self.task.get_system_prompt())
        self.live_logger.log_result("Prompts ready")
        
        # Log initial state
        self.live_logger.log_action("Logging initial state")
        self.logger.log_step(0, {
            "step_type": "initial",
            "observation": observation.to_dict(),
            "system_prompt": self.agent.system_prompt,
            "user_prompt": self.task.get_user_prompt(),
            "tools": self.environment.get_tool_schemas(),
            "image": observation.image, 
        })
        self.live_logger.log_result("Initial state logged")
        
        return observation
    
    def _execute_interaction_loop(self, observation: Observation) -> int:
        """Execute the main agent-environment interaction loop."""
        max_steps = self.environment.config.max_steps
        StatusDisplay.print_section(f"Starting Interaction Loop (max {max_steps} steps)")
        
        # Check if step selection is enabled
        use_step_selection = (
            self.config.step_selection and 
            self.config.step_selection.enabled and 
            self.step_selection_manager is not None
        )
        
        if use_step_selection:
            mode = "greedy" if self.config.step_selection.top_k == 1 else f"beam search (width={self.config.step_selection.top_k})"
            self.live_logger.log_info(
                f"Step selection enabled: {mode}, "
                f"method={self.config.step_selection.selection_method}, "
                f"num_candidates={self.config.step_selection.num_candidates}, "
                f"beam_width(top_k)={self.config.step_selection.top_k}"
            )
        
        self.progress_display = ProgressDisplay(max_steps)
        
        for step in range(1, max_steps + 1):
            self.progress_display.update(step, f"Step {step}/{max_steps}: Processing agent response")
            
            # Check token limit before processing
            if self._is_token_limit_exceeded():
                return self._handle_token_limit_exceeded(step)
            
            # Build prompt
            prompt_history = self._build_prompt_history()
            
            # Get current image for beam search
            current_image = observation.image if observation else None
            
            # Get agent response - use step selection if enabled
            if use_step_selection and current_image is not None:
                response, tool_calls = self._select_best_actions_with_step_selection(
                    step, prompt_history, current_image
                )
            else:
                response, tool_calls = self._get_agent_response(step, prompt_history)
            
            self.live_logger.log_info(f"Step {step}:")
            self.live_logger.log_info(f"Prompt history: \n{prompt_history}")
            self.live_logger.log_info(f"Agent response: \n{response}")
            self.live_logger.log_info(f"Tool calls: \n{tool_calls}")

            # Check if agent wants to finish
            if self._should_finish_task(tool_calls):
                break
                
            # Execute tools or log response
            if tool_calls:
                self._execute_tool_calls(step, tool_calls, response)
                # Update observation from the last executed action
                if self.interaction_history:
                    last_step = self.interaction_history[-1]
                    observations = last_step.get("observations", [])
                    if observations:
                        observation = observations[-1]
                
                # Check if task is automatically completed based on environment state
                if self._is_task_auto_completed(observation):
                    self.live_logger.log_result("Task automatically completed! All objectives achieved.", success=True)
                    break
                        
                # Check if task is solved (for tasks that support this)
                if self._is_task_solved(observation):
                    self.live_logger.log_info("Task solved! Ending task execution.")
                    break
            else:
                # Record text-only response in interaction history
                self.interaction_history.append({
                    "response": response,
                    "actions": [],
                    "observations": []
                })
                
                self.logger.log_step(step, {
                    "step_type": "response",
                    "agent_response": response,
                    "observation": observation.to_dict()
                })
            
            self.live_logger.log_info(f"Step {step}, Token count: {self.agent.get_token_count()}")
            
            # Log step selection statistics if enabled
            if use_step_selection:
                best_path, best_score = self.step_selection_manager.get_best_path()
                self.live_logger.log_info(
                    f"Step selection: best path length={len(best_path)}, cumulative score={best_score:.3f}"
                )
        
        # Finish progress display
        # Check if task was completed (either by breaking early or by auto-completion)
        task_completed = step < max_steps
        
        # Additional check: verify if the last observation shows completion
        if self.interaction_history:
            last_step = self.interaction_history[-1]
            observations = last_step.get("observations", [])
            if observations:
                last_obs = observations[-1]
                if self._is_task_auto_completed(last_obs):
                    task_completed = True
        
        self.progress_display.finish(success=task_completed)
        
        if not task_completed:
            self.live_logger.log_info("Step limit reached")
        else:
            self.live_logger.log_info("Task completed successfully")
            
        return step
    
    def _build_prompt_history(self) -> str:
        """Build prompt history from interaction history."""
        user_prompt = self.task.get_user_prompt()
        
        # Add object mapping information so the agent knows which object_id corresponds to which color/object
        object_mapping = self.get_object_mapping()
        
        prompt_history = ""
        for idx, step_data in enumerate(self.interaction_history):
            if idx == 0:  # Skip initial reset step
                continue
                
            prompt_history += f"\n{'='*60}\nStep {idx}:\n"
            
            # Add agent response if present
            response = step_data.get("response", "")
            if response:
                prompt_history += f"\n[Agent Response]:\n{response}\n"
            
            # Add actions if present
            actions = step_data.get("actions", [])
            if actions:
                prompt_history += f"\n[Actions]:\n"
                for i, action in enumerate(actions, 1):
                    # Handle both Action objects and dicts
                    if hasattr(action, 'action_type'):
                        action_type = action.action_type
                        parameters = action.parameters
                    else:
                        action_type = action.get("action_type", "unknown")
                        parameters = action.get("parameters", {})
                    prompt_history += f"  {i}. {action_type}({parameters})\n"
            
            # Add observations (tool results)
            observations = step_data.get("observations", [])
            if observations:
                prompt_history += f"\n[Results]:\n"
                for i, obs in enumerate(observations, 1):
                    # Handle both Observation objects and dicts
                    if hasattr(obs, 'description'):
                        description = obs.description
                    else:
                        description = obs.get("description", "")
                    if description:
                        prompt_history += f"  {i}. {description}\n"
        
        # Include object mapping at the beginning so agent can reference it
        full_prompt = f"{user_prompt}\n\n{object_mapping}\n"
        
        if prompt_history:
            full_prompt += f"\nInteraction History:{prompt_history}\n"
            
        full_prompt += "\nNow, what's your next action?"
        
        return full_prompt
    
    def get_object_mapping(self) -> str:
        """
        Returns a string describing the complete object information including position, color, and properties.
        This information is updated dynamically at each step.
        
        Returns:
            str: Human-readable description of all objects with their properties for the LLM to understand.
        """
        # Allow environment-specific summaries (non-PyBullet environments).
        if hasattr(self.environment, "describe_objects"):
            try:
                return self.environment.describe_objects()  # type: ignore[attr-defined]
            except Exception:
                pass

        lines = ["🧩 OBJECT MAPPING (Complete object information - updated this step):"]
        lines.append("=" * 80)
        
        # First, show container information
        container_info = None
        for obj_info in self.environment.objects:
            if obj_info.properties.get('is_container', False):
                container_info = obj_info
                pos = obj_info.position
                
                # Get container AABB size
                try:
                    aabb_min, aabb_max = p.getAABB(obj_info.object_id)
                    width = aabb_max[0] - aabb_min[0]
                    depth = aabb_max[1] - aabb_min[1]
                    height = aabb_max[2] - aabb_min[2]
                    container_size_str = f"({width:.3f}, {depth:.3f}, {height:.3f})"
                    
                    # Also show internal volume bounds for reference
                    lines.append(f"📦 Container:")
                    lines.append(f"   - object_id: {obj_info.object_id}")
                    lines.append(f"   - name: {obj_info.name}")
                    lines.append(f"   - position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
                    lines.append(f"   - size: {container_size_str}")
                    lines.append(f"   - internal bounds: x[{aabb_min[0]:.3f}, {aabb_max[0]:.3f}], y[{aabb_min[1]:.3f}, {aabb_max[1]:.3f}], z[{aabb_min[2]:.3f}, {aabb_max[2]:.3f}]")
                except Exception as e:
                    lines.append(f"📦 Container:")
                    lines.append(f"   - object_id: {obj_info.object_id}")
                    lines.append(f"   - name: {obj_info.name}")
                    lines.append(f"   - position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
                    lines.append(f"   - size: unknown")
                
                lines.append("")
                break
        
        # Then show all movable objects
        non_container_count = 0
        
        for obj_info in self.environment.objects:
            # Skip container objects
            if obj_info.properties.get('is_container', False):
                continue
                
            non_container_count += 1
            obj_id = obj_info.object_id
            pos = obj_info.position
            
            # Get visual shape data to retrieve color
            color_str = "unknown"
            try:
                visual_shapes = p.getVisualShapeData(obj_id)
                
                if visual_shapes:
                    # Get color from the first visual shape (most objects have one main shape)
                    # visual_shapes is a list of tuples: (objectUniqueId, linkIndex, visualGeometryType, 
                    #                                      dimensions, filename, meshScale, rgbaColor, ...)
                    rgba_color = visual_shapes[0][7]  # Index 7 is rgbaColor
                    
                    # Format color as RGB values (0-255 scale for readability)
                    r = int(rgba_color[0] * 255)
                    g = int(rgba_color[1] * 255)
                    b = int(rgba_color[2] * 255)
                    
                    color_str = f"RGB=({r}, {g}, {b})"
                    
            except Exception as e:
                color_str = f"error ({str(e)})"
            
            # Get AABB (bounding box) to calculate object size
            try:
                aabb_min, aabb_max = p.getAABB(obj_id)
                # Calculate actual dimensions
                width = aabb_max[0] - aabb_min[0]   # x dimension
                depth = aabb_max[1] - aabb_min[1]   # y dimension
                height = aabb_max[2] - aabb_min[2]  # z dimension
                size_str = f"({width:.3f}, {depth:.3f}, {height:.3f})"
            except Exception as e:
                size_str = "unknown"
            
            # Format object information - ID, color, position, and size
            lines.append(f"🧩 Object #{non_container_count} (object_id: {obj_id}):")
            #lines.append(f"   - name: {obj_info.name}")
            lines.append(f"   - color: {color_str}")
            lines.append(f"   - position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            lines.append(f"   - size: {size_str}")  # AABB size: (width, depth, height) in meters
            lines.append("")
        
        lines.append("=" * 80)
        lines.append(f"Total movable objects: {non_container_count}")
        lines.append("\n💡 IMPORTANT:")
        lines.append("   - Use object_id (integer) to interact with objects in tool calls")
        lines.append("   - Position format: (x, y, z) in meters, where z is height")
        lines.append("   - Size format: (width, depth, height) in meters - the actual dimensions of the object")
        lines.append("   - Positions and sizes are updated after each action - always check current values!")
        lines.append("\n📏 SIZE USAGE:")
        lines.append("   - Use object size to calculate if it fits inside the container")
        lines.append("   - Compare object size with container bounds to plan placement")
        lines.append("   - Consider object height when stacking on top of another object")
        
        return "\n".join(lines)
    
    def _get_agent_response(self, step: int, prompt_history: str) -> tuple:
        """Get response from agent."""
        self.live_logger.log_step_start(step, "Getting agent response")
        
        # Get recent observations with images from interaction_history
        recent_obs = []
        for step_data in self.interaction_history[-self.config.runner.history_length:]:
            observations = step_data.get("observations", [])
            recent_obs.extend(observations)
        
        response, tool_calls = self.agent.process_observation(
            recent_obs, prompt_history, self.environment.get_tool_schemas()
        )
        
        # Log agent's intention
        if tool_calls:
            self.live_logger.log_info(f"Agent wants to execute {len(tool_calls)} tool(s): {tool_calls}")
        else:
            self.live_logger.log_info("Agent provided text response only")
            
        return response, tool_calls
    
    def _generate_candidate_actions(self, step: int, prompt_history: str, num_candidates: int) -> List[ActionCandidate]:
        """
        Generate multiple candidate actions from the agent.
        
        Args:
            step: Current step number
            prompt_history: The prompt history string
            num_candidates: Number of candidate actions to generate
            
        Returns:
            List of ActionCandidate objects
        """
        candidates = []
        
        # Get recent observations
        recent_obs = []
        for step_data in self.interaction_history[-self.config.runner.history_length:]:
            observations = step_data.get("observations", [])
            recent_obs.extend(observations)
        
        # Store original temperature
        original_temperature = self.agent.config.temperature
        
        # Use higher temperature for diversity if configured
        if self.config.step_selection:
            self.agent.config.temperature = self.config.step_selection.candidate_temperature
        
        self.live_logger.log_info(f"Generating {num_candidates} candidate actions...")
        
        for i in range(num_candidates):
            try:
                response, tool_calls = self.agent.process_observation(
                    recent_obs, prompt_history, self.environment.get_tool_schemas()
                )
                
                # Convert tool calls to ActionCandidate objects
                for tool_call in tool_calls:
                    try:
                        tool_name = tool_call["function"]["name"]
                        arguments = json.loads(tool_call["function"]["arguments"])
                        
                        # Skip finish action in candidates
                        if tool_name == "finish":
                            continue
                        
                        candidate = ActionCandidate(
                            action_type=tool_name,
                            parameters=arguments,
                            response=response,
                            tool_call=tool_call
                        )
                        candidates.append(candidate)
                        
                    except (KeyError, json.JSONDecodeError) as e:
                        self.live_logger.log_warning(f"Failed to parse tool call: {e}")
                        continue
                        
            except Exception as e:
                self.live_logger.log_warning(f"Failed to generate candidate {i+1}: {e}")
                continue
        
        # Restore original temperature
        self.agent.config.temperature = original_temperature
        
        self.live_logger.log_info(f"Generated {len(candidates)} valid candidate actions")
        return candidates
    
    def _select_best_actions_with_step_selection(
        self,
        step: int,
        prompt_history: str,
        current_image
    ) -> Tuple[str, List[Dict]]:
        """
        Use step selection (greedy/beam search) to choose the best action(s).
        
        Args:
            step: Current step number
            prompt_history: The prompt history string
            current_image: Current observation image
            
        Returns:
            Tuple of (response, tool_calls) for the best action(s)
        """
        # Get task description and object mapping
        task_description = self.task.get_user_prompt()
        object_mapping = self.get_object_mapping()
        
        # Build interaction history string (simplified)
        interaction_str = self._build_interaction_summary()
        
        # Initialize step selection manager if first step
        if step == 1:
            self.step_selection_manager.initialize()
        
        # Define candidate generator function
        def generate_candidates() -> List[ActionCandidate]:
            return self._generate_candidate_actions(
                step, prompt_history, self.config.step_selection.num_candidates
            )
        
        # Use step selection manager to expand and select
        best_actions = self.step_selection_manager.expand_beams(
            generate_candidates,
            current_image,
            task_description,
            object_mapping,
            interaction_str
        )
        
        if not best_actions:
            # Fallback to regular agent response if no candidates
            self.live_logger.log_warning("No valid candidates from step selection, falling back to regular response")
            return self._get_agent_response(step, prompt_history)
        
        # Convert best action to response and tool_calls format
        best_action = best_actions[0]
        
        self.live_logger.log_info(
            f"Step selection chose action: {best_action.action_type} "
            f"(score: {best_action.score:.3f})"
        )
        
        return best_action.response, [best_action.tool_call]
    
    def _build_interaction_summary(self) -> str:
        """Build a summary string of the interaction history."""
        summary_lines = []
        for idx, step_data in enumerate(self.interaction_history):
            if idx == 0:  # Skip initial reset
                continue
            
            actions = step_data.get("actions", [])
            for action in actions:
                if hasattr(action, 'action_type'):
                    summary_lines.append(f"Step {idx}: {action.action_type}({action.parameters})")
                else:
                    summary_lines.append(f"Step {idx}: {action.get('action_type', 'unknown')}")
        
        return "\n".join(summary_lines) if summary_lines else "No previous actions"
    
    def _should_finish_task(self, tool_calls: list) -> bool:
        """Check if agent wants to finish the task."""
        if not tool_calls:
            return False
            
        tool_names = [tool_call.get("function", {}).get("name", "") for tool_call in tool_calls]
        if "finish" in tool_names:
            self.live_logger.log_info("Agent called 'finish' tool. Task complete.")
            return True
            
        return False
    
    def _is_task_auto_completed(self, observation: Observation) -> bool:
        """
        Check if the task is automatically completed based on environment state.
        
        For stacking game: check if puzzle is complete (is_complete=True in metadata).
        """
        if not observation or not observation.state:
            return False
        
        # Check metadata for completion status
        metadata = observation.state.metadata
        if metadata and metadata.get("is_complete", False):
            return True
            
        return False
    
    def _is_task_solved(self, observation: Observation) -> bool:
        """Check if task is solved based on observation state."""
        if not observation or not observation.state:
            return False
        
        metadata = observation.state.metadata or {}
        is_solved = metadata.get("is_solved", False)
        
        # Handle both boolean and string representations
        if isinstance(is_solved, bool):
            return is_solved
        elif isinstance(is_solved, str):
            return is_solved.lower() in ("true", "1", "yes")
        
        return False
    
    def _is_task_auto_completed(self, observation: Observation) -> bool:
        """
        Check if the task is automatically completed based on environment state.
        
        For stacking game: check if puzzle is complete (is_complete=True in metadata).
        """
        if not observation or not observation.state:
            return False
        
        # Check metadata for completion status
        metadata = observation.state.metadata
        if metadata and metadata.get("is_complete", False):
            return True
            
        return False
    
    def _execute_tool_calls(self, step: int, tool_calls: list, response: str) -> None:
        """Execute all tool calls from the agent."""
        step_actions = []
        step_observations = []
        
        for i, tool_call in enumerate(tool_calls, 1):
            try:
                action, observation = self._execute_single_tool(step, i, len(tool_calls), tool_call, response)
                step_actions.append(action)
                step_observations.append(observation)
            except Exception as e:
                self._log_tool_error(step, tool_call, response, e)
        
        # Record the complete step in interaction history (with actual objects)
        self.interaction_history.append({
            "response": response,
            "actions": step_actions,
            "observations": step_observations
        })
    
    def _execute_single_tool(self, step: int, tool_index: int, total_tools: int, 
                           tool_call: dict, response: str) -> Tuple[Action, Observation]:
        """Execute a single tool call and return the action and observation."""
        tool_name = tool_call["function"]["name"]

        # safe load json
        arg_str = tool_call["function"]["arguments"]
        if arg_str == '':
            arguments = {}
        else:
            arguments = json.loads(arg_str)
        
        self.live_logger.log_action(f"Executing tool {tool_index}/{total_tools}: {tool_name}")
        
        # Create and execute action
        action = Action(action_type=tool_name, parameters=arguments)
        observation = self.environment.step(action)
        
        # Log results
        self.live_logger.log_result(f"Tool completed: {observation.state.metadata['tool_result']}")
        
        self.logger.log_step(step, {
            "step_type": "action",
            "agent_response": response,
            "tool_call": tool_call,
            "action": action.to_dict(),
            "observation": observation.to_dict(),
            "tool_result": observation.state.metadata['tool_result'],
            "image": observation.image,
        })
        
        return action, observation
    
    def _log_tool_error(self, step: int, tool_call: dict, response: str, error: Exception) -> None:
        """Log tool execution error."""
        tool_name = tool_call.get('function', {}).get('name', 'unknown')
        error_msg = f"Error executing tool {tool_name}: {error}"
        
        self.live_logger.log_error(f"Tool failed: {error_msg}")
        
        self.logger.log_step(step, {
            "step_type": "tool_error",
            "error": error_msg,
            "tool_call": tool_call,
            "agent_response": response
        })
    
    def _is_token_limit_exceeded(self) -> bool:
        """Check if token count exceeds the configured limit."""
        return self.agent.get_token_count() > self.config.agent.max_content_size
    
    def _handle_token_limit_exceeded(self, step: int) -> int:
        """Handle case when token limit is exceeded."""
        token_count = self.agent.get_token_count()
        max_tokens = self.config.agent.max_content_size
        
        self.live_logger.log_warning(f"Token count exceeded: {token_count}/{max_tokens}")
        self.progress_display.finish(success=False)
        
        raise RuntimeError(f"Token count exceeded max content size: {token_count}, Max: {max_tokens}")
    
    def _create_task_result(self, total_steps: int) -> TaskResult:
        """Create the final task result."""
        execution_time = time.time() - self.start_time
        
        # Display completion results
        StatusDisplay.print_section("Task Execution Completed")
        results = {
            "Execution Time": f"{execution_time:.2f}s",
            "Total Steps": total_steps,
            "Total Tokens": self.agent.get_token_count()
        }
        StatusDisplay.print_results(results, "Execution Summary")
        
        task_result = TaskResult(
            task_id=self.task.task_id,
            task_type=self.config.task.type,
            total_steps=total_steps,
            execution_time=execution_time,
            trajectory=self.interaction_history,
            success=False,  # Success determined during evaluation
            metadata={
                "difficulty": self.config.task.difficulty.value,
                "optimal_steps": self.task.optimal_steps,
                "total_tokens": self.agent.get_token_count(),
                "total_tokens_in": self.agent.get_total_tokens_in(),
                "total_tokens_out": self.agent.get_total_tokens_out(),
                "agent_model": self.config.agent.model_name,
            }
        )
        
        # Save logs
        self.live_logger.log_action("Saving experiment logs")
        self.logger.save_logs()
        self.live_logger.log_result("Logs saved successfully")
        
        return task_result
    
    def _handle_task_failure(self, error: Exception) -> TaskResult:
        """Handle task execution failure."""
        execution_time = time.time() - self.start_time if self.start_time else 0
        
        # Cleanup progress display
        if self.progress_display:
            self.progress_display.finish(success=False)
        
        # Display error information
        StatusDisplay.print_section("Task Execution Failed")
        error_info = {
            "Error": str(error),
            "Execution Time": f"{execution_time:.2f}s",
            "Steps Completed": len(self.interaction_history)
        }
        StatusDisplay.print_results(error_info, "Error Summary")
        self.live_logger.log_error(f"Task execution failed: {error}")
        
        # Create error result
        error_result = TaskResult(
            task_id=self.task.task_id,
            task_type=self.config.task.type,
            total_steps=len(self.interaction_history),
            execution_time=execution_time,
            trajectory=self.interaction_history,
            success=False,
            error_message=str(error),
            metadata={
                "difficulty": self.config.task.difficulty.value,
                "total_tokens": self.agent.get_token_count() if self.agent else 0,
                "total_tokens_in": self.agent.get_total_tokens_in() if self.agent else 0,
                "total_tokens_out": self.agent.get_total_tokens_out() if self.agent else 0,
                "agent_model": self.config.agent.model_name if self.agent else "unknown",
            }
        )
        
        # Save error logs
        self._save_error_logs()
        
        return error_result
    
    def _save_error_logs(self) -> None:
        """Attempt to save error logs."""
        try:
            self.live_logger.log_action("Saving error logs")
            self.logger.save_logs()
            self.live_logger.log_result("Error logs saved")
        except Exception:
            self.live_logger.log_error("Could not save error logs")
    
    def _cleanup_environment(self) -> None:
        """Clean up environment resources."""
        if self.environment:
            self.environment.close()
                
    def run_multiple_tasks(self, num_runs: int = 1) -> List[TaskResult]:
        """Run multiple instances of the same task."""
        results = []
        
        if num_runs > 1:
            StatusDisplay.print_header(f"Running Multiple Tasks ({num_runs} instances)")
            # Create progress display for multiple runs
            multi_run_progress = ProgressDisplay(num_runs)
        
        for i in range(num_runs):
            if num_runs > 1:
                multi_run_progress.update(i, f"Running task instance {i+1}/{num_runs}")
                self.live_logger.log_step_start(i+1, f"Running task instance {i+1}/{num_runs}")
            else:
                self.live_logger.log_info(f"Running single task instance")
            
            # Reinitialize for each run
            self.setup()
            
            result = self.run_single_task()
            results.append(result)
                
            if num_runs > 1:
                self.live_logger.log_step_end(i+1, f"Task instance {i+1} completed", result.success)
        
        if num_runs > 1:
            multi_run_progress.finish(success=True)
            
        return results
        
    def evaluate(self, task_results: List[TaskResult]) -> EvaluationResult:
        """
        Evaluate one or more task results using task-specific success criteria,
        and return comprehensive metrics.
        """
        StatusDisplay.print_section("Starting Task Evaluation")
        self.live_logger.log_action("Getting task-specific success criteria")
        self.live_logger.log_result("Success criteria defined")

        try:
            evaluation_result = self.task.evaluate_tasks(self.evaluator, task_results)
        except Exception as e:
            self.live_logger.log_warning(f"Evaluator failed during task evaluation: {e}")
            evaluation_result = None
        return evaluation_result
        
    def run_benchmark(self, num_runs: int = 1) -> EvaluationResult:
        """Run complete benchmark with evaluation."""
        StatusDisplay.print_header(f"chainbench Benchmark - {num_runs} Run{'s' if num_runs > 1 else ''}")
        
        # Display configuration
        config_info = {
            "Agent Model": self.config.agent.model_name,
            "Task Type": self.config.task.type,
            "Task Name": self.config.task.name,
            "Difficulty": self.config.task.difficulty.value,
            "Number of Runs": num_runs
        }
        StatusDisplay.print_config(config_info, "Benchmark Configuration")
        
        # Run tasks
        task_results = self.run_multiple_tasks(num_runs)
        
        # Evaluate results with task-specific criteria
        evaluation_result = self.evaluate(task_results)
        
        # Export results to Excel
        self.evaluator.export_results_to_excel(
            evaluation_result,
            os.path.join(self.logger.run_dir, self.config.runner.results_excel_path),
            self.config.agent.model_name
        )
        
        # Export detailed report
        self.evaluator.export_detailed_report(
            evaluation_result,
            os.path.join(self.logger.run_dir, "detailed_reports"),
            self.config.agent.model_name
        )
        
        # Print summary
        self._print_benchmark_summary(evaluation_result)

        return evaluation_result
        
    def _print_benchmark_summary(self, evaluation_result) -> None:
        """Print benchmark summary to console."""
        StatusDisplay.print_header("BENCHMARK SUMMARY")
        
        # Basic information
        basic_info = {
            "Model": self.config.agent.model_name,
            "Task": f"{self.config.task.name} ({self.config.task.difficulty.value})",
            "Total Tasks": len(evaluation_result.task_results),
            "Successful": sum(1 for r in evaluation_result.task_results if r.success),
            "Accuracy": evaluation_result.accuracy
        }
        StatusDisplay.print_results(basic_info, "Basic Results")
        
        # Advanced metrics
        advanced_metrics = {}
        if evaluation_result.pass_at_k:
            for k, rate in evaluation_result.pass_at_k.items():
                advanced_metrics[f"Pass@{k}"] = rate
                
        if evaluation_result.distance_to_optimal != float('inf'):
            advanced_metrics["Distance to Optimal"] = evaluation_result.distance_to_optimal
            
        if evaluation_result.token_efficiency != float('inf'):
            advanced_metrics["Token Efficiency"] = f"{evaluation_result.token_efficiency:.0f} tokens/success"
        
        if advanced_metrics:
            StatusDisplay.print_results(advanced_metrics, "Advanced Metrics")
        
        StatusDisplay.print_section("Output Files")
        self.live_logger.log_info(f"Results saved to: {self.config.runner.results_excel_path}")
        StatusDisplay.print_separator("=", 60)
