"""
3D Polycube Stacking Game Package
"""

from .game_core import (
    Vec3, Transform, PieceDef, PlacedPiece,
    LevelSpec, GameState, PlacementResult, ErrorCode
)

from .rotation import ROTATION_MATRICES, get_rotation_matrix

from .loader import (
    load_puzzle_from_json,
    load_puzzle_by_name,
    find_all_puzzles,
    create_game_state
)

from .placement import (
    place_piece_by_cells,
    place_piece_by_transform,
    move_piece_by_cells,
    pickup_piece
)

__version__ = "1.0.0"
__all__ = [
    # Core types
    'Vec3', 'Transform', 'PieceDef', 'PlacedPiece',
    'LevelSpec', 'GameState', 'PlacementResult', 'ErrorCode',
    # Rotation
    'ROTATION_MATRICES', 'get_rotation_matrix',
    # Loader
    'load_puzzle_from_json', 'load_puzzle_by_name',
    'find_all_puzzles', 'create_game_state',
    # Placement
    'place_piece_by_cells', 'place_piece_by_transform',
    'move_piece_by_cells', 'pickup_piece',
]
