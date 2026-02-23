"""
Stacking-game environment integration for CHAIN benchmark.

This environment wraps the discrete polycube stacking game from
`assets/stacking_game` so an LLM agent can interact with it via tools.
"""

from __future__ import annotations

import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

from chainbench.core import (
    EnvironmentConfig,
    BaseEnvironment,
    register_environment,
    register_environment_config,
)
from chainbench.core.base import Action, Observation, ObjectInfo, State

# --- Optional stacking_game imports (loaded lazily to avoid hard dependency) ---

STACKING_GAME_ROOT = Path("assets/stacking_game")
if STACKING_GAME_ROOT.exists() and str(STACKING_GAME_ROOT) not in sys.path:
    sys.path.append(str(STACKING_GAME_ROOT))

STACKING_IMPORT_ERROR: Optional[Exception] = None
try:
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib
        matplotlib.use("Agg")
    except Exception:
        pass
    from game_core import GameState, LevelSpec, PieceDef, Vec3  # type: ignore
    from initialization import initialize_pieces_on_ground  # type: ignore
    from loader import (  # type: ignore
        create_game_state,
        load_puzzle_by_name,
        preprocess_piece,
    )
    from placement import (  # type: ignore
        pickup_piece,
        place_piece_by_cells,
    )
    from visualizer_3d import visualize_state_3d  # type: ignore
except Exception as exc:  # pragma: no cover - fallback when dependency missing
    STACKING_IMPORT_ERROR = exc


def _ensure_stacking_available() -> None:
    """Raise a helpful error if stacking_game is not importable."""
    if STACKING_IMPORT_ERROR is not None:
        raise RuntimeError(
            "stacking_game modules are unavailable. Please ensure the "
            "'assets/stacking_game' directory exists and dependencies "
            "like matplotlib are installed."
        ) from STACKING_IMPORT_ERROR


def _vec_center(cells: List[Vec3]) -> Tuple[float, float, float]:
    """Compute centroid of a voxel set (1-based coordinates)."""
    if not cells:
        return (0.0, 0.0, 0.0)
    xs = [c.x for c in cells]
    ys = [c.y for c in cells]
    zs = [c.z for c in cells]
    return (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))


def _to_vec3_list(coords: List[List[int]]) -> List[Vec3]:
    """Convert a list of [x, y, z] to Vec3 list."""
    return [Vec3(int(x), int(y), int(z)) for x, y, z in coords]


@register_environment_config("stacking_game")
@dataclass
class StackingGameConfig(EnvironmentConfig):
    """Configuration for the stacking_game environment."""

    puzzle_dir: str = "assets/stacking_game/puzzles_full_v9"
    default_size: str = "2x2x2"
    default_puzzle_id: str = "puzzle_001"
    randomize_on_reset: bool = True
    init_spacing: int = 2
    init_seed: Optional[int] = None
    render_unplaced: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if not Path(self.puzzle_dir).is_absolute():
            self.puzzle_dir = str(Path(self.puzzle_dir).resolve())


@register_environment("stacking_game")
class StackingGameEnvironment(BaseEnvironment):
    """LLM-friendly wrapper around the stacking_game logic."""

    def __init__(self, config: StackingGameConfig):
        _ensure_stacking_available()
        super().__init__(config)
        self.config: StackingGameConfig
        self.step_count: int = 0
        self.current_state: Optional[State] = None
        self.game_state: Optional[GameState] = None
        self.objects: List[ObjectInfo] = []
        self._object_id_map: Dict[str, int] = {}
        self._task_seed: Optional[int] = config.init_seed
        self._used_fallback: bool = False
        self._tool_handlers = {
            "state": self._tool_state,
            "place": self._tool_place,
            "pickup": self._tool_pickup,
            "get_piece_info": self._tool_get_piece_info,
        }
        self.current_size = config.default_size
        self.current_puzzle_id = config.default_puzzle_id

    # ------------------------------------------------------------------ #
    # BaseEnvironment API
    # ------------------------------------------------------------------ #
    def reset(self) -> Observation:
        """Reset environment to initial state."""
        self.step_count = 0
        self._load_game_state()
        self.current_state = self._get_current_state()
        return self._create_observation()

    def step(self, action: Action) -> Observation:
        """Execute an action (tool call) and return new observation."""
        self.step_count += 1
        tool_result = self.execute_tool_call(action.action_type, action.parameters)
        self.current_state = self._get_current_state(
            metadata={
                "tool_call": action.to_dict(),
                "tool_result": tool_result,
            }
        )
        return self._create_observation()

    def render(self, multi_view: bool = True) -> Union[Image.Image, Dict[str, Image.Image]]:
        """Render current game state to a PIL image."""
        if not self.game_state:
            return Image.new("RGB", (self.config.render_width, self.config.render_height), color="white")
        try:
            fig = visualize_state_3d(
                self.game_state,
                title=f"{self.current_size}/{self.current_puzzle_id}",
                show_unplaced=self.config.render_unplaced,
                figsize=(8, 6),
            )
            import matplotlib.pyplot as plt  # Local import to avoid hard dependency at module load
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
            plt.close(fig)
            return img
        except Exception as exc:  # pragma: no cover - visualization fallback
            return Image.new(
                "RGB",
                (self.config.render_width, self.config.render_height),
                color="red",
            )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return JSON schemas for tools exposed to the LLM agent."""
        def build_schema(name: str, desc: str, properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
            return {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }

        return [
            build_schema(
                "state",
                "Show current game state (same as CLI 'state'): box size, occupancy, and placed/unplaced pieces.",
                {},
                [],
            ),
            build_schema(
                "place",
                "Place a piece (CLI 'place <id>') using explicit cells. "
                "Provide all occupied cells in 1-based coordinates; length must match the piece voxel count. "
                "Example: [[1, 1, 1], [1, 2, 1], [2, 1, 1], [2, 1, 2]] (verify these cells form the same shape as the chosen piece; "
                "different valid cell layouts can represent rotations).",
                {
                    "piece_id": {"type": "string", "description": "Piece identifier to place."},
                    "cells": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 3,
                            "maxItems": 3,
                        },
                        "description": "Exact 1-based coordinates for every voxel of the piece.",
                    },
                },
                ["piece_id", "cells"],
            ),
            build_schema(
                "pickup",
                "Pick up a placed piece (CLI 'pickup <id>') so it can be repositioned. Use this to remove a previously placed block when it was placed incorrectly or needs adjusting.",
                {"piece_id": {"type": "string", "description": "Piece identifier to remove."}},
                ["piece_id"],
            ),
            # build_schema(
            #     "get_piece_info",
            #     "Inspect a piece (CLI 'piece <id>'): voxel count, local coords, and placement if present.",
            #     {"piece_id": {"type": "string", "description": "Piece identifier to query."}},
            #     ["piece_id"],
            # ),
        ]

    def execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch tool calls."""
        handler = self._tool_handlers.get(tool_name)
        if not handler:
            return {"status": "error", "message": f"Unknown tool '{tool_name}'"}
        try:
            return handler(**arguments)
        except Exception as exc:  # pragma: no cover - runtime safety
            return {"status": "error", "message": f"Tool '{tool_name}' failed: {exc}"}

    def close(self) -> None:
        """Cleanup (no-op for this environment)."""
        self.game_state = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _load_game_state(self, size: Optional[str] = None, puzzle_id: Optional[str] = None) -> None:
        """Load game state from disk or fallback demo level."""
        _ensure_stacking_available()
        self.current_size = size or self.current_size
        self.current_puzzle_id = puzzle_id or self.current_puzzle_id
        self._used_fallback = False
        spec = load_puzzle_by_name(self.config.puzzle_dir, self.current_size, self.current_puzzle_id)
        if not spec:
            spec = self._create_fallback_level()
            self._used_fallback = True
        self.game_state = create_game_state(spec)

        if self.config.randomize_on_reset:
            initialize_pieces_on_ground(
                self.game_state,
                spacing=self.config.init_spacing,
                seed=self._task_seed,
            )
        self._update_objects()

    def _create_fallback_level(self) -> LevelSpec:
        """Create a tiny built-in level if dataset assets are missing."""
        pieces = [
            PieceDef(id="0", local_voxels=[Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)]),
            PieceDef(id="1", local_voxels=[Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, 1, 0), Vec3(1, 1, 1)]),
        ]
        pieces = [preprocess_piece(p) for p in pieces]
        return LevelSpec(box=(2, 2, 2), pieces=pieces)

    def _get_current_state(self, metadata: Optional[Dict[str, Any]] = None) -> State:
        """Convert game_state to benchmark State."""
        if not self.game_state:
            return State(step=self.step_count, objects=[], time_stamp=0.0, metadata=metadata or {})

        objects = list(self.objects)
        meta = {
            "puzzle_size": self.current_size,
            "puzzle_id": self.current_puzzle_id,
            "placed_pieces": sorted(list(self.game_state.placed.keys())),
            "unplaced_pieces": sorted(list(self.game_state.unplaced)),
            "is_complete": self.game_state.is_complete(),
        }
        if metadata:
            meta.update(metadata)

        return State(
            step=self.step_count,
            objects=objects,
            time_stamp=self.step_count,
            metadata=meta,
        )

    def _create_observation(self) -> Observation:
        """Create Observation from current state."""
        image = self.render(multi_view=False)
        return Observation(image=image, state=self.current_state, description=self._get_state_description())

    def _get_state_description(self) -> str:
        """Textual description for the prompt history."""
        if not self.current_state or not self.game_state:
            return "Stacking game not initialized."

        desc_lines = [
            f"Puzzle: {self.current_size}/{self.current_puzzle_id}",
            f"Box cells: {self.game_state.spec.box[0]}x{self.game_state.spec.box[1]}x{self.game_state.spec.box[2]}",
            f"Placed: {len(self.game_state.placed)}, Unplaced: {len(self.game_state.unplaced)}",
        ]
        if self.current_state.metadata:
            tool_call = self.current_state.metadata.get("tool_call")
            tool_res = self.current_state.metadata.get("tool_result")
            if tool_call and tool_res:
                desc_lines.append(
                    f"Last tool: {tool_call.get('action_type')} with {tool_call.get('parameters')}, "
                    f"result: {tool_res.get('status')} - {tool_res.get('message')}"
                )
        if self.game_state.placed:
            placed_summary = ", ".join(
                [f"{pid}({len(pp.world_cells)} cells @ rot {pp.transform.rot})" for pid, pp in self.game_state.placed.items()]
            )
            desc_lines.append(f"Placed pieces: {placed_summary}")
        if self.game_state.unplaced:
            unplaced = ", ".join(sorted(self.game_state.unplaced))
            desc_lines.append(f"Unplaced pieces: {unplaced}")
        if self.game_state.is_complete():
            desc_lines.append("Puzzle complete.")
        return "\n".join(desc_lines)

    def _update_objects(self) -> None:
        """Refresh ObjectInfo list from game_state."""
        if not self.game_state:
            self.objects = []
            return

        objs: List[ObjectInfo] = []
        self._object_id_map = {}

        # Virtual box as container
        box_obj = ObjectInfo(
            object_id=0,
            name="box",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            object_type="container",
            properties={"is_container": True, "box_size": self.game_state.spec.box},
        )
        objs.append(box_obj)

        next_id = 1
        for pid, placed in self.game_state.placed.items():
            center = _vec_center(placed.world_cells)
            objs.append(
                ObjectInfo(
                    object_id=next_id,
                    name=f"piece_{pid}",
                    position=center,
                    orientation=(0.0, 0.0, 0.0, 1.0),
                    object_type="piece",
                    properties={
                        "piece_id": pid,
                        "rot": placed.transform.rot,
                        "placed": True,
                        "cells": [c.to_tuple() for c in placed.world_cells],
                    },
                )
            )
            self._object_id_map[pid] = next_id
            next_id += 1

        for pid in sorted(self.game_state.unplaced):
            init_pos = self.game_state.initial_placements.get(pid)
            center = _vec_center(init_pos.world_cells) if init_pos else (0.0, 0.0, 0.0)
            objs.append(
                ObjectInfo(
                    object_id=next_id,
                    name=f"piece_{pid}",
                    position=center,
                    orientation=(0.0, 0.0, 0.0, 1.0),
                    object_type="piece",
                    properties={
                        "piece_id": pid,
                        "placed": False,
                        "cells": [c.to_tuple() for c in init_pos.world_cells] if init_pos else [],
                    },
                )
            )
            self._object_id_map[pid] = next_id
            next_id += 1

        self.objects = objs

    # ------------------------------------------------------------------ #
    # Tool implementations
    # ------------------------------------------------------------------ #
    def _tool_state(self) -> Dict[str, Any]:
        if not self.game_state:
            return {"status": "error", "message": "No puzzle loaded"}
        A, B, C = self.game_state.spec.box
        placed = sorted(self.game_state.placed.keys())
        unplaced = sorted(self.game_state.unplaced)
        return {
            "status": "success",
            "message": "State retrieved",
            "state": {
                "box": [A, B, C],
                "occupied": len(self.game_state.occupied),
                "total_cells": A * B * C,
                "placed_pieces": placed,
                "unplaced_pieces": unplaced,
                "is_complete": self.game_state.is_complete(),
            },
        }

    def _tool_place(self, piece_id: str, cells: List[List[int]]) -> Dict[str, Any]:
        if not self.game_state:
            return {"status": "error", "message": "No puzzle loaded"}
        coords = _to_vec3_list(cells)
        result = place_piece_by_cells(self.game_state, piece_id, coords)
        if result.success:
            self._update_objects()
            return {
                "status": "success",
                "message": result.message,
                "transform": {
                    "rotation": result.transform.rot if result.transform else None,
                    "position": result.transform.t.to_tuple() if result.transform else None,
                },
            }
        return {"status": "error", "message": result.message, "error": result.error.value}

    def _tool_pickup(self, piece_id: str) -> Dict[str, Any]:
        if not self.game_state:
            return {"status": "error", "message": "No puzzle loaded"}
        result = pickup_piece(self.game_state, piece_id)
        if result.success:
            self._update_objects()
            return {"status": "success", "message": result.message}
        return {"status": "error", "message": result.message, "error": result.error.value}

    def _tool_get_piece_info(self, piece_id: str) -> Dict[str, Any]:
        if not self.game_state:
            return {"status": "error", "message": "No puzzle loaded"}
        piece = self.game_state.get_piece_def(piece_id)
        if not piece:
            return {"status": "error", "message": f"Piece {piece_id} not found"}
        placed = self.game_state.placed.get(piece_id)
        info = {
            "piece_id": piece_id,
            "voxel_count": len(piece.local_voxels),
            "voxels_local": [v.to_tuple() for v in piece.local_voxels],
            "placed": bool(placed),
        }
        if placed:
            info["world_cells"] = [c.to_tuple() for c in placed.world_cells]
            info["rotation"] = placed.transform.rot
            info["position"] = placed.transform.t.to_tuple()
        return {"status": "success", "message": "Piece info retrieved", "info": info}

    # ------------------------------------------------------------------ #
    # Extra helpers for prompt building
    # ------------------------------------------------------------------ #
    def describe_objects(self) -> str:
        """Return human-readable object summary for prompt history."""
        if not self.game_state:
            return "No puzzle loaded."

        lines = [
            f"Puzzle: {self.current_size}/{self.current_puzzle_id}",
            f"Box size: {self.game_state.spec.box[0]}x{self.game_state.spec.box[1]}x{self.game_state.spec.box[2]} (1-based grid)",
            f"Placed pieces: {len(self.game_state.placed)}, Unplaced: {len(self.game_state.unplaced)}",
            "Pieces:",
        ]
        for pid in sorted(self.game_state.unplaced):
            piece = self.game_state.get_piece_def(pid)
            lines.append(f" - Piece {pid}: {len(piece.local_voxels)} voxels (unplaced)")
        for pid, placed in self.game_state.placed.items():
            center = _vec_center(placed.world_cells)
            lines.append(
                f" - Piece {pid}: placed at center {tuple(round(c, 2) for c in center)}, "
                f"rotation {placed.transform.rot}, cells={len(placed.world_cells)}"
            )
        if self.game_state.is_complete():
            lines.append("All cells are filled. Puzzle complete.")
        return "\n".join(lines)
