"""
Task implementations for the CHAIN benchmark.

This package contains:
- LubanDisassemblyTask: Disassemble interlocking Luban Lock puzzles
- StackingGameTask: Pack polycube pieces into a target 3D box
"""

# Normal imports to ensure proper task registration
from chainbench.tasks.base_task import PhysicsTask
from chainbench.tasks.luban_task import LubanDisassemblyTask
from chainbench.tasks.stacking_game_task import StackingGameTask

__all__ = [
    "PhysicsTask",
    "LubanDisassemblyTask",
    "StackingGameTask",
]
