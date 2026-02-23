"""
Luban Lock environment implementation backed by Unity.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image, ImageDraw

from chainbench.core import (
    BaseEnvironment,
    EnvironmentConfig,
    register_environment,
    register_environment_config,
)
from chainbench.core.base import Action, Observation, ObjectInfo, State


@register_environment_config("luban")
@dataclass
class LubanConfig(EnvironmentConfig):
    """Configuration for the Unity-backed Luban Lock environment."""

    host: str = "127.0.0.1"
    port: int = 9999
    env_id: int = 1
    level_index: int = 0
    exe_path: str = "assets/Luban/LuBanSuoEnv.exe"
    start_level_on_reset: bool = True
    startup_wait: float = 1.5
    connect_timeout: float = 2.0
    request_timeout: float = 10.0
    max_startup_retries: int = 30
    logic_unit_scale: float = 0.0001
    # Keep legacy fields for config compatibility (unused with Unity backend).
    move_unit_step: float = 0.01
    rotate_unit_step: float = 5.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.host, str) or not self.host:
            raise ValueError("host must be a non-empty string")
        if not isinstance(self.port, int) or self.port <= 0:
            raise ValueError("port must be a positive integer")
        if not isinstance(self.env_id, int) or not (0 <= self.env_id <= 8):
            raise ValueError("env_id must be within [0, 8]")
        if not isinstance(self.level_index, int) or not (0 <= self.level_index <= 31):
            raise ValueError("level_index must be within [0, 31]")
        if not os.path.isabs(self.exe_path):
            self.exe_path = str(Path(self.exe_path).resolve())


@register_environment("luban")
class LubanEnvironment(BaseEnvironment):
    """Unity-backed Luban Lock environment with socket control."""

    def __init__(self, config: LubanConfig):
        super().__init__(config)
        self.config: LubanConfig
        self.step_count: int = 0
        self.current_state: Optional[State] = None
        self.objects: List[ObjectInfo] = []
        self._last_state_data: Dict[str, Any] = {}
        self._process: Optional[subprocess.Popen] = None
        self._owns_process: bool = False
        self._server_ready: bool = False
        self._tool_handlers = {
            "get_state": self._tool_get_state,
            "move_piece": self._tool_move_piece,
            "rotate_piece": self._tool_rotate_piece,
        }

    # ------------------------------------------------------------------ #
    # BaseEnvironment API
    # ------------------------------------------------------------------ #
    def reset(self) -> Observation:
        """Reset environment to initial state."""
        self.step_count = 0
        self.objects = []
        self._last_state_data = {}
        self._ensure_ready()
        if self.config.start_level_on_reset:
            self._send_command("start_level", {"level_index": self.config.level_index})
            time.sleep(self.config.startup_wait)
        self._refresh_state()
        self.current_state = self._get_current_state()
        return self._create_observation()

    def step(self, action: Action) -> Observation:
        """Execute an action (tool call) and return new observation."""
        self.step_count += 1
        tool_result = self.execute_tool_call(action.action_type, action.parameters)
        self._refresh_state()
        self.current_state = self._get_current_state(
            metadata={
                "tool_call": action.to_dict(),
                "tool_result": tool_result,
            }
        )
        return self._create_observation()

    def render(self, multi_view: bool = True) -> Union[Image.Image, Dict[str, Image.Image]]:
        """Render current environment state by requesting Unity screenshots."""
        return self._capture_and_compose()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return JSON schemas for tools exposed to the agent."""
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
                "get_state",
                "Fetch current state (positions, rotations, step count). Does NOT trigger screenshots.",
                {},
                [],
            ),
            build_schema(
                "move_piece",
                "Move a piece along an axis by a distance in logic units (1 unit = 0.0001 Unity units).",
                {
                    "piece_id": {"type": "integer", "description": "Target piece ID."},
                    "axis": {"type": "string", "enum": ["x", "y", "z"], "description": "Move axis."},
                    "distance": {"type": "number", "description": "Move distance in logic units."}
                },
                ["piece_id", "axis", "distance"],
            ),
            build_schema(
                "rotate_piece",
                "Rotate a piece by 90-degree multiples around an axis.",
                {
                    "piece_id": {"type": "integer", "description": "Target piece ID."},
                    "axis": {"type": "string", "enum": ["x", "y", "z"], "description": "Rotation axis."},
                    "angle": {"type": "number", "description": "Rotation angle in degrees (multiples of 90)."},
                },
                ["piece_id", "axis", "angle"],
            ),
        ]

    def execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch tool calls."""
        handler = self._tool_handlers.get(tool_name)
        if not handler:
            return {"status": "error", "message": f"Unknown tool '{tool_name}'"}
        try:
            return handler(**arguments)
        except Exception as exc:
            return {"status": "error", "message": f"Tool '{tool_name}' failed: {exc}"}

    def close(self) -> None:
        """Clean up Unity process if started by this environment."""
        if self._process and self._owns_process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                pass
        self._process = None
        self._server_ready = False

    # ------------------------------------------------------------------ #
    # Unity communication helpers
    # ------------------------------------------------------------------ #
    def _ensure_ready(self) -> None:
        if self._server_ready:
            return
        if self._check_server_available():
            self._server_ready = True
            return
        # self._start_unity_process()
        self._wait_for_server()
        self._server_ready = True

    def _check_server_available(self) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.config.connect_timeout)
                sock.connect((self.config.host, self.config.port))
            return True
        except Exception:
            return False

    def _start_unity_process(self) -> None:
        exe_path = self.config.exe_path
        if not os.path.exists(exe_path):
            raise FileNotFoundError(f"LuBanSuoEnv.exe not found at: {exe_path}")
        if self._process and self._process.poll() is None:
            return
        self._process = subprocess.Popen(
            [exe_path],
            cwd=str(Path(exe_path).parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self._owns_process = True

    def _wait_for_server(self) -> None:
        for _ in range(self.config.max_startup_retries):
            if self._check_server_available():
                return
            time.sleep(0.2)
        raise RuntimeError("Unity server did not become ready in time.")

    def _send_command(self, command: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._ensure_ready()
        payload = {"command": command, "env_id": self.config.env_id, "params": params or {}}
        message = json.dumps(payload) + "\n"
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.config.request_timeout)
                sock.connect((self.config.host, self.config.port))
                sock.sendall(message.encode("utf-8"))
                response_data = b""
                while True:
                    chunk = sock.recv(1024 * 1024)
                    if not chunk:
                        break
                    response_data += chunk
                    if b"\n" in chunk:
                        break
            raw = response_data.split(b"\n")[0] if response_data else b""
            if not raw:
                return {"status": "error", "message": "Empty response from Unity"}
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:
            return {"status": "error", "message": f"Unity communication failed: {exc}"}

    # ------------------------------------------------------------------ #
    # State handling
    # ------------------------------------------------------------------ #
    def _refresh_state(self) -> None:
        result = self._send_command("get_state")
        if result.get("status") == "success":
            data = result.get("data", {}) or {}
            self._last_state_data = data
            self.objects = self._build_objects_from_state(data)

    def _build_objects_from_state(self, data: Dict[str, Any]) -> List[ObjectInfo]:
        objects: List[ObjectInfo] = []
        for piece in data.get("pieces", []) or []:
            pos = piece.get("pos", {}) or {}
            obj_id = int(piece.get("id", -1))
            objects.append(
                ObjectInfo(
                    object_id=obj_id,
                    name=f"piece_{obj_id}",
                    position=(float(pos.get("x", 0.0)), float(pos.get("y", 0.0)), float(pos.get("z", 0.0))),
                    orientation=(0.0, 0.0, 0.0, 1.0),
                    object_type="luban_piece",
                    properties={
                        "rot": piece.get("rot"),
                        "is_finished": piece.get("is_finished", False),
                    },
                )
            )
        return objects

    def _get_current_state(self, metadata: Optional[Dict[str, Any]] = None) -> State:
        # Calculate piece statistics
        num_pieces = len(self.objects)
        num_finished = sum(1 for obj in self.objects if obj.properties.get("is_finished", False))
        num_blocked = num_pieces - num_finished  # Pieces still inside the lock
        
        meta: Dict[str, Any] = {
            "puzzle_index": self._last_state_data.get("puzzle_index"),
            "step_count": self._last_state_data.get("step_count"),
            "is_solved": self._last_state_data.get("is_solved"),
            "num_pieces": num_pieces,
            "num_finished": num_finished,
            "num_blocked": num_blocked,
        }
        if metadata:
            meta.update(metadata)
        return State(
            step=self.step_count,
            objects=self.objects,
            time_stamp=time.time(),
            metadata=meta,
        )

    def _create_observation(self) -> Observation:
        image = self._capture_and_compose()
        return Observation(image=image, state=self.current_state, description=self._get_state_description())

    def _get_state_description(self) -> str:
        if not self.current_state:
            return "Luban environment not initialized."
        meta = self.current_state.metadata or {}
        lines = [
            f"Luban Lock (env_id={self.config.env_id})",
            f"Puzzle index: {meta.get('puzzle_index')}, step_count: {meta.get('step_count')}, solved: {meta.get('is_solved')}",
        ]
        tool_call = meta.get("tool_call")
        tool_result = meta.get("tool_result")
        if tool_call and tool_result:
            lines.append(
                f"Last tool: {tool_call.get('action_type')} {tool_call.get('parameters')} -> "
                f"{tool_result.get('status')} ({tool_result.get('message')})"
            )
        if self.objects:
            lines.append(f"Pieces ({len(self.objects)}):")
            for obj in self.objects:
                rot = obj.properties.get("rot")
                pos = obj.position
                lines.append(
                    f"- id={obj.object_id}, pos=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}), rot={rot}"
                )
        return "\n".join(lines)

    def _blank_image(self) -> Image.Image:
        return Image.new("RGB", (self.config.render_width, self.config.render_height), color="black")

    def _capture_and_compose(self) -> Image.Image:
        """Capture Unity screenshots and compose viewpoints 1/3/5."""
        result = self._send_command("capture")
        if result.get("status") != "success":
            return self._blank_image()

        data = result.get("data", {}) or {}
        path_str = data.get("screenshot_path", "")
        paths = [p for p in path_str.split("|") if p]
        if not paths:
            return self._blank_image()

        target_indices = [1, 3, 5]
        tiles: List[Image.Image] = []
        for idx in target_indices:
            if idx < len(paths) and os.path.exists(paths[idx]):
                try:
                    tiles.append(Image.open(paths[idx]).convert("RGB"))
                    continue
                except Exception:
                    pass
            tiles.append(self._blank_image())

        widths = [img.width for img in tiles]
        heights = [img.height for img in tiles]
        total_width = sum(widths)
        max_height = max(heights) if heights else self.config.render_height
        canvas = Image.new("RGB", (total_width, max_height), color="black")

        x_offset = 0
        for view_idx, img in enumerate(tiles):
            canvas.paste(img, (x_offset, 0))
            draw = ImageDraw.Draw(canvas)
            draw.text((x_offset + 8, 8), f"viewpoint {view_idx}", fill=(255, 255, 255))
            x_offset += img.width

        return canvas

    def _wait_until_stable(self) -> int:
        """Unity environment handles physics internally; no wait needed."""
        return 0

    def describe_objects(self) -> str:
        """Provide a summary for prompt history."""
        if not self.current_state:
            return "Luban environment not initialized."
        return self._get_state_description()

    # ------------------------------------------------------------------ #
    # Tool implementations
    # ------------------------------------------------------------------ #
    def _tool_get_state(self) -> Dict[str, Any]:
        return self._send_command("get_state")

    def _tool_move_piece(self, piece_id: int, axis: str, distance: float) -> Dict[str, Any]:
        return self._send_command(
            "move",
            {
                "piece_id": int(piece_id),
                "axis": axis,
                "distance": float(distance),
                "is_exploratory": False,
            },
        )

    def _tool_rotate_piece(self, piece_id: int, axis: str, angle: float) -> Dict[str, Any]:
        return self._send_command(
            "rotate",
            {
                "piece_id": int(piece_id),
                "axis": axis,
                "angle": float(angle),
            },
        )
