"""
Luban Lock disassembly task implementation (Unity-backed).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from chainbench.core import register_task, register_task_config, TaskConfig
from chainbench.core.base import Observation, TaskResult
from chainbench.tasks.base_task import PhysicsTask

from chainbench.environment.luban_env import LubanEnvironment

@register_task_config("luban_disassembly")
@dataclass
class LubanTaskConfig(TaskConfig):
    urdf_root: str = "assets/pybullet/phobos_models/luban-3-piece"
    level_index: int = 0
    target_displacement_threshold: float = 0.08
    ruled_evaluation: bool = True

@register_task("luban_disassembly")
class LubanDisassemblyTask(PhysicsTask):
    """
    Task: Disassemble a Luban Lock.
    Loads pieces using RuntimeAssembly logic (Fixed Constraints).
    """

    def __init__(self, config: LubanTaskConfig):
        super().__init__(config)
        self.target_piece_id: Optional[int] = None
        self.initial_positions: Dict[int, Tuple[float, float, float]] = {}

    def _calculate_optimal_steps(self) -> int:
        return 1

    def configure_environment(self, environment) -> Observation:  # type: ignore[override]
        """Configure Unity-backed environment (no PyBullet setup)."""
        if not isinstance(environment, LubanEnvironment):
            raise ValueError("LubanTask requires LubanEnvironment")
        self.environment = environment
        observation = self.environment.reset()
        self._cache_initial_state(observation)
        # Cache observation for _get_initial_instruction if called before environment is fully set up
        self._cached_initial_obs = observation
        return observation

    def _configure_environment(self) -> None:
        """No-op for Unity-backed environment."""
        return

    def _cache_initial_state(self, observation: Observation) -> None:
        self.initial_positions = {}
        for obj in observation.state.objects:
            self.initial_positions[obj.object_id] = obj.position
        if observation.state.objects:
            self.target_piece_id = observation.state.objects[0].object_id

    def _evaluate_success(self, task_results: List[TaskResult]) -> List[TaskResult]:
        if not task_results:
            return []

        for task_result in task_results:
            last_obs = self._extract_last_observation(task_result)
            solved_raw = False
            if last_obs is not None:
                if hasattr(last_obs, "state"):
                    solved_raw = last_obs.state.metadata.get("is_solved")
                elif isinstance(last_obs, dict):
                    meta = last_obs.get("state", {}).get("metadata", {})
                    solved_raw = meta.get("is_solved")

            if isinstance(solved_raw, bool):
                is_solved = solved_raw
            elif isinstance(solved_raw, str):
                is_solved = solved_raw.strip().lower() in ("true", "1", "yes")
            else:
                is_solved = bool(solved_raw)

            task_result.success = is_solved
            task_result.metadata["is_solved"] = is_solved
            task_result.metadata["feedback"] = f"Solved={solved_raw}"
        return task_results

    def _extract_last_observation(self, task_result: TaskResult):
        if not task_result.trajectory:
            return None
        last = task_result.trajectory[-1]
        if isinstance(last, dict):
            observations = last.get("observations", [])
            return observations[-1] if observations else None
        try:
            _, obs = last
            return obs
        except Exception:
            return None

    def _get_initial_system_prompt(self) -> str:
        # Get piece counts from environment state
        num_pieces = 0
        num_blocked = 0
        
        # Method 1: Try to get from state metadata (preferred - already calculated in _get_current_state)
        if self.environment and hasattr(self.environment, 'current_state') and self.environment.current_state:
            metadata = self.environment.current_state.metadata or {}
            num_pieces = metadata.get("num_pieces", 0)
            num_blocked = metadata.get("num_blocked", 0)
        
        # Method 2: Fallback - calculate from objects if metadata not available
        if num_pieces == 0 and self.environment and hasattr(self.environment, 'objects') and self.environment.objects:
            num_pieces = len(self.environment.objects)
            num_finished = sum(
                1 for obj in self.environment.objects 
                if obj.properties.get("is_finished", False)
            )
            num_blocked = num_pieces - num_finished
        
        # Method 3: Last resort - use cached initial observation if available
        # (This handles the case where _get_initial_instruction is called before environment is fully set up)
        if num_pieces == 0 and hasattr(self, '_cached_initial_obs'):
            obs = self._cached_initial_obs
            if obs and hasattr(obs, 'state') and obs.state:
                metadata = obs.state.metadata or {}
                num_pieces = metadata.get("num_pieces", 0)
                num_blocked = metadata.get("num_blocked", 0)
                if num_pieces == 0 and obs.state.objects:
                    num_pieces = len(obs.state.objects)
                    num_finished = sum(
                        1 for obj in obs.state.objects
                        if obj.properties.get("is_finished", False)
                    )
                    num_blocked = num_pieces - num_finished
        
        system_prompt_template = """You are an expert mechanical puzzle solver for a “Luban Lock” (interlocking burr puzzle). Your job is to find the KEY piece (it is currently unblocked) and slide it completely out.

CONTEXT
- Total pieces: {NUM_PIECES}
- Pieces still not removed: {NUM_BLOCKED} (these are still inside the lock; not necessarily “immovable”)
- Pieces are mechanically interlocked; a piece can only translate (slide) along its allowed axis/rails.

GOAL
- Identify the KEY piece (must be unblocked in the current state) and extract it by sliding it out along the correct direction.
- A piece is considered “unlocked/solved” if it is moved away from the center (its original position) by a distance greater than 200."""
        return system_prompt_template.format(NUM_PIECES=num_pieces, NUM_BLOCKED=num_blocked)

    def _get_initial_instruction(self) -> str:
        """
        Get initial instruction with dynamic piece counts from environment state.
        Retrieves NUM_PIECES and NUM_BLOCKED from the environment's state metadata
        or calculates them from the objects list.
        """
        prompt = f"""
You will receive shotscreens of the current state of the puzzle. The images are taken from 3 different viewpoints and the coordinates are given in each image.

CRITICAL RULES (NO RANDOM TRIES)
- Do NOT propose “try moving X in some direction” unless you can justify the direction from the images and the axis reference.
- Every move must be a constrained translation along the piece’s allowed rail/axis. No rotation, no diagonal motion.
- If the movement direction of a piece is blocked (you can directly see the piece is blocked in image), you should not propose to move in that direction.

HOW TO REASON FROM IMAGES
1) Align coordinate frames:
   - Identify +X/-X, +Y/-Y, +Z/-Z directions using the axis markers shown in each screenshot.
   - Cross-check between the 3 viewpoints to avoid sign confusion.

2) Infer each piece’s allowed sliding axis:
   - From grooves/rails and the piece geometry, decide whether the piece can only move along X, Y, or Z (and which sign is outward).
   - If the axis is ambiguous from one view, resolve it using the other views. If still ambiguous, state what visual evidence is missing.

3) Feasibility test BEFORE suggesting a move:
   - A move is feasible only if there is visible clearance along that axis (no immediate collision).
   - Prefer moves that increase “distance from the center” (moving outward), because the goal is extraction (>200 from original center).
"""
        return prompt
