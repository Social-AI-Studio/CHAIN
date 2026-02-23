"""
Stacking-game task for the CHAIN benchmark.

This task asks the agent to pack all polycube pieces into the target box using
the stacking_game environment tools.
"""

from dataclasses import dataclass
from typing import List, Optional

from chainbench.core import TaskConfig, register_task, register_task_config
from chainbench.core.base import Observation, TaskResult
from chainbench.tasks.base_task import PhysicsTask


@register_task_config("stacking_game")
@dataclass
class StackingGameTaskConfig(TaskConfig):
    """Configuration for the stacking_game task."""

    puzzle_size: str = "2x2x2"
    puzzle_id: str = "puzzle_001"
    ruled_evaluation: bool = True
    allow_random_puzzle: bool = False
    init_seed: Optional[int] = None


@register_task("stacking_game")
class StackingGameTask(PhysicsTask):
    """Task wrapper for the stacking_game environment."""

    def __init__(self, config: StackingGameTaskConfig):
        super().__init__(config)

    def _calculate_optimal_steps(self) -> int:
        import json
        import os
        import warnings
        
        # Parse puzzle_id format: "puzzle_mid_001" -> difficulty="mid", number="001"
        # or handle legacy format if puzzle_id doesn't start with "puzzle_"
        puzzle_id = self.config.puzzle_id
        if puzzle_id.startswith("puzzle_"):
            puzzle_id = puzzle_id[7:]  # Remove "puzzle_" prefix
        
        # Split by underscore: "mid_001" -> ["mid", "001"]
        parts = puzzle_id.split("_")
        if len(parts) >= 2:
            difficulty = parts[0]  # e.g., "easy", "mid", "hard"
            number = parts[1]      # e.g., "001", "002"
        else:
            # Fallback: assume the entire puzzle_id is the number
            difficulty = "easy"
            number = puzzle_id
        
        # Build correct path: assets/stacking_game/puzzles_full_v9/{size}/{difficulty}/{number}/{size}_{difficulty}_{number}.json
        size = self.config.puzzle_size
        puzzle_dir = os.environ.get("PUZZLE_DIR", "assets/stacking_game/puzzles_full_v9")
        config_path = os.path.join(
            puzzle_dir,
            size,
            difficulty,
            number,
            f"{size}_{difficulty}_{number}.json"
        )

        if not os.path.exists(config_path):
            warnings.warn(
                f"Puzzle file not found at {config_path}. "
                "Falling back to built-in demo optimal_steps=2."
            )
            return 2

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        piece_num = len(config.get("pieces", []))
        return max(piece_num, 1)


    def configure_environment(self, environment) -> Observation:  # type: ignore[override]
        """Configure environment with the puzzle specified in the task config."""
        self.environment = environment
        # Pass puzzle selection hints to the environment before reset.
        if hasattr(environment, "current_size"):
            environment.current_size = self.config.puzzle_size
        if hasattr(environment, "current_puzzle_id"):
            environment.current_puzzle_id = self.config.puzzle_id
        if hasattr(environment, "_task_seed"):
            environment._task_seed = self.config.init_seed
        observation = environment.reset()
        return observation

    def _configure_environment(self) -> None:
        """Unused because configure_environment is fully overridden."""
        return None

    def _evaluate_success(self, task_results: List[TaskResult]) -> List[TaskResult]:
        """Rule-based success: puzzle cells must be fully filled."""
        for task_result in task_results:
            is_complete = False
            if task_result.trajectory:
                # Traverse backwards to find the latest observation with metadata.
                for step in reversed(task_result.trajectory):
                    observations = step.get("observations", []) if isinstance(step, dict) else []
                    if observations:
                        last_obs = observations[-1]
                        try:
                            meta = last_obs.state.metadata if hasattr(last_obs, "state") else {}
                            is_complete = bool(meta.get("is_complete"))
                        except Exception:
                            is_complete = False
                        break
            task_result.success = is_complete
            task_result.metadata["is_complete"] = is_complete
        return task_results

    def _get_initial_system_prompt(self) -> str:
        """Provide system instructions tailored for stacking_game."""
        # return (
        #     "You are a 3D stacking game player. "
        # )
        return ""

    def _get_initial_instruction(self) -> str:
        """User-facing task instruction."""
        cfg = self.config

        instruction = f"""
You are solving a 3D packing puzzle.

**Goal:** Pack every piece into the **{cfg.puzzle_size}** box for puzzle **`{cfg.puzzle_id}`**. You must fill **all {len(self.environment.game_state.spec.box)} cells** with **no collisions** and **no out-of-bounds** placements.

**Critical rules:**

1. **First, carefully read the list of available tools** (their names, required arguments, and what state they return).
2. **Use the current image / board progress** as the ground-truth state of what is already placed and what remains.
3. Think in 3D and plan explicitly: build a mental model of the box as **(x, y, z)** layers, check **cross-sections** layer-by-layer, and reason about **connectivity/voids** (avoid creating unreachable 1-cell cavities). Prefer a strategy like **anchor corners/edges → fill constrained regions → resolve remaining gaps**, and consider symmetry/rotations to match piece geometry.
4. Work in a safe order: **inspect pieces → plan a feasible sequence → place pieces step-by-step**, verifying after each placement (collision-free, in-bounds, and consistent with your 3D plan).
5. If a placement fails (collision / invalid) or you realize it blocks the remaining space, **backtrack**: use **`pickup`** to remove the previously placed piece (or the piece causing the conflict), then try a different orientation/order and re-place. Use `pickup` especially when a recent move seems to create trapped voids or misaligns with your layer-by-layer plan.
6. Only call **`finish`** when the box is fully filled and valid.

**Output format constraints:**

* Do **not** output multiple actions.
* You **MUST** end your response with **exactly one tool call** wrapped as:

  * `<action> XXX </action>`
* `XXX` must be the **exact tool invocation content** (use the tool’s required schema/arguments).
* **ALWAYS** call a tool to advance the game state. Do not just describe your plan without taking action.

**Now do the next step.**

Example format for placing a piece (e.g., placing piece "3" at specific coordinates):
<action>{{"tool":"place","piece_id":"3","cells":[[1,1,2],[2,1,2]]}}</action>        
        """
        # return (
        #     f"Pack every piece into the {cfg.puzzle_size} box for puzzle '{cfg.puzzle_id}'. "
        #     "You must fill all cells without collisions. "
        #     "Inspect pieces, plan a feasible order, place them, and call finish once solved."
        # )
        return instruction
