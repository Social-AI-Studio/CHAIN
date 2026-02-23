"""
Environment implementations for the CHAIN benchmark.

This package contains simulation environment wrappers:
- LubanEnvironment: Unity-backed interlocking mechanical puzzle (Luban Lock)
- StackingGameEnvironment: Polycube packing under geometric constraints
"""

# Normal imports to ensure proper environment registration
from chainbench.environment.luban_env import LubanEnvironment, LubanConfig
from chainbench.environment.stacking_game_env import StackingGameEnvironment, StackingGameConfig

__all__ = [
    "LubanEnvironment",
    "LubanConfig",
    "StackingGameEnvironment",
    "StackingGameConfig",
]
