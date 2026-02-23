"""
初始化工具 - 在地面外放置所有piece并随机旋转
"""

import random
import numpy as np
from typing import List, Optional
from game_core import Vec3, Transform, GameState, PlacedPiece
from rotation import ROTATION_MATRICES


def calculate_piece_bounds(piece_voxels: List[Vec3]) -> tuple:
    """
    计算piece的边界框

    Returns:
        (width, height, depth)
    """
    if not piece_voxels:
        return (0, 0, 0)

    min_x = min(v.x for v in piece_voxels)
    max_x = max(v.x for v in piece_voxels)
    min_y = min(v.y for v in piece_voxels)
    max_y = max(v.y for v in piece_voxels)
    min_z = min(v.z for v in piece_voxels)
    max_z = max(v.z for v in piece_voxels)

    return (max_x - min_x + 1, max_y - min_y + 1, max_z - min_z + 1)


def initialize_pieces_on_ground(state: GameState, spacing: int = 2, seed: Optional[int] = None):
    """
    在地面外初始化所有piece,每个piece随机旋转,独立放置在地上(不悬空)

    Args:
        state: 游戏状态
        spacing: piece之间的间距
        seed: 随机种子(用于可复现性)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    A, B, C = state.spec.box

    # 清除初始放置
    state.initial_placements.clear()

    # 计算放置位置(在盒子右侧)
    x_offset = A + spacing

    for piece in state.spec.pieces:
        if piece.id not in state.unplaced:
            continue  # 已经被放置的piece跳过

        # 随机选择一个旋转
        num_rotations = len(piece.rotation_signatures)
        if num_rotations > 0:
            rot_idx = random.randint(0, min(23, num_rotations - 1))
        else:
            rot_idx = 0

        # 应用旋转
        rot_matrix = ROTATION_MATRICES[rot_idx]
        rotated_voxels = []

        for v in piece.local_voxels:
            vec = np.array([v.x, v.y, v.z])
            rotated = rot_matrix @ vec
            rotated_voxels.append(Vec3(int(rotated[0]), int(rotated[1]), int(rotated[2])))

        # 规范化到min=0
        min_x = min(v.x for v in rotated_voxels)
        min_y = min(v.y for v in rotated_voxels)
        min_z = min(v.z for v in rotated_voxels)

        normalized_voxels = [
            Vec3(v.x - min_x, v.y - min_y, v.z - min_z)
            for v in rotated_voxels
        ]

        # 关键修改: 放置在地面上,z从1开始(确保最底层在地面)
        # position表示piece的最小点位置
        position = Vec3(x_offset, 1, 1)

        # 计算世界坐标
        world_cells = [
            Vec3(v.x + position.x, v.y + position.y, v.z + position.z)
            for v in normalized_voxels
        ]

        # 保存初始放置
        transform = Transform(rot=rot_idx, t=position)
        state.initial_placements[piece.id] = PlacedPiece(
            id=piece.id,
            transform=transform,
            world_cells=world_cells
        )

        # 计算piece的宽度和深度,更新offset
        max_x = max(v.x for v in normalized_voxels)
        max_y = max(v.y for v in normalized_voxels)

        piece_width = max_x + 1
        x_offset += piece_width + spacing


def reset_piece_to_initial(state: GameState, piece_id: str) -> bool:
    """
    将piece重置到初始位置

    Args:
        state: 游戏状态
        piece_id: piece ID

    Returns:
        是否成功
    """
    if piece_id not in state.initial_placements:
        return False

    # 如果piece当前在盒子里,需要先取出
    if piece_id in state.placed:
        from game_core import uncommit_placement
        uncommit_placement(state, piece_id)

    return True


def randomize_piece_rotation(state: GameState, piece_id: str, seed: Optional[int] = None):
    """
    随机化piece的旋转(在初始位置)

    Args:
        state: 游戏状态
        piece_id: piece ID
        seed: 随机种子
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    piece = state.get_piece_def(piece_id)
    if not piece or piece_id not in state.initial_placements:
        return

    # 随机选择旋转
    num_rotations = len(piece.rotation_signatures)
    if num_rotations > 0:
        rot_idx = random.randint(0, min(23, num_rotations - 1))
    else:
        rot_idx = 0

    # 更新初始放置的旋转
    current = state.initial_placements[piece_id]

    # 重新计算旋转后的体素
    rot_matrix = ROTATION_MATRICES[rot_idx]
    rotated_voxels = []

    for v in piece.local_voxels:
        vec = np.array([v.x, v.y, v.z])
        rotated = rot_matrix @ vec
        rotated_voxels.append(Vec3(int(rotated[0]), int(rotated[1]), int(rotated[2])))

    # 规范化
    min_x = min(v.x for v in rotated_voxels)
    min_y = min(v.y for v in rotated_voxels)
    min_z = min(v.z for v in rotated_voxels)

    normalized_voxels = [
        Vec3(v.x - min_x, v.y - min_y, v.z - min_z)
        for v in rotated_voxels
    ]

    # 保持位置不变,更新世界坐标
    position = current.transform.t
    world_cells = [
        Vec3(v.x + position.x, v.y + position.y, v.z + position.z)
        for v in normalized_voxels
    ]

    # 更新
    state.initial_placements[piece_id] = PlacedPiece(
        id=piece_id,
        transform=Transform(rot=rot_idx, t=position),
        world_cells=world_cells
    )



if __name__ == "__main__":
    print("Testing initialization...")

    from game_core import PieceDef, LevelSpec
    from loader import preprocess_piece, create_game_state

    # 创建测试pieces
    piece0 = PieceDef(id="0", local_voxels=[
        Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(1, 1, 0)
    ])
    piece1 = PieceDef(id="1", local_voxels=[
        Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0)
    ])

    piece0 = preprocess_piece(piece0)
    piece1 = preprocess_piece(piece1)

    spec = LevelSpec(box=(3, 3, 3), pieces=[piece0, piece1])
    state = create_game_state(spec)

    print(f"Box: {spec.box}")
    print(f"Pieces: {len(spec.pieces)}")

    # 初始化pieces在地面
    initialize_pieces_on_ground(state, seed=42)

    print("\nInitial placements:")
    for piece_id, placement in state.initial_placements.items():
        print(f"  Piece {piece_id}:")
        print(f"    Rotation: {placement.transform.rot}")
        print(f"    Position: {placement.transform.t.to_tuple()}")
        print(f"    Cells: {[c.to_tuple() for c in placement.world_cells]}")

    print("\n✓ Initialization test passed!")
