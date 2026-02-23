"""
形状匹配和放置判定逻辑
"""

from typing import List, Optional, Tuple
import numpy as np

from game_core import (
    Vec3, Transform, PieceDef, GameState, PlacementResult,
    ErrorCode, normalize_points, points_to_signature,
    within_box, has_support, commit_placement, uncommit_placement
)
from rotation import ROTATION_MATRICES


def infer_transform_from_target(
    piece: PieceDef,
    target_cells: List[Vec3],
    target_signature: str
) -> Optional[Transform]:
    """
    从目标格子推断变换(旋转+平移)

    Args:
        piece: piece定义
        target_cells: 目标格子 (1-based世界坐标)
        target_signature: 目标格子的规范化签名

    Returns:
        Transform对象或None
    """
    # 找到匹配的旋转索引
    rot_idx = -1
    for i, sig in enumerate(piece.rotation_signatures):
        if sig == target_signature:
            rot_idx = i
            break

    if rot_idx == -1:
        return None

    # 将目标格子转换为0-based
    target_0based = [Vec3(c.x - 1, c.y - 1, c.z - 1) for c in target_cells]

    # 规范化目标格子
    normalized_target = normalize_points(target_0based)

    # 旋转piece的本地体素
    rot_matrix = ROTATION_MATRICES[rot_idx]
    rotated_local = []
    for v in piece.local_voxels:
        vec = np.array([v.x, v.y, v.z])
        rotated = rot_matrix @ vec
        rotated_local.append(Vec3(int(rotated[0]), int(rotated[1]), int(rotated[2])))

    # 规范化旋转后的本地体素
    normalized_rotated = normalize_points(rotated_local)

    # 计算平移: 取第一个点计算偏移
    # target_0based[0] = normalized_rotated[0] + translation
    # 但我们需要找到原始坐标的对应关系

    # 重新计算: 不规范化,直接找对应
    # 旋转piece使其形状匹配,然后计算平移
    rotated_voxels = []
    for v in piece.local_voxels:
        vec = np.array([v.x, v.y, v.z])
        rotated = rot_matrix @ vec
        rotated_voxels.append(Vec3(int(rotated[0]), int(rotated[1]), int(rotated[2])))

    # 规范化旋转后的体素 (平移到min=0)
    min_x = min(v.x for v in rotated_voxels)
    min_y = min(v.y for v in rotated_voxels)
    min_z = min(v.z for v in rotated_voxels)

    normalized_rotated_voxels = [
        Vec3(v.x - min_x, v.y - min_y, v.z - min_z)
        for v in rotated_voxels
    ]

    # target也规范化到0-based
    target_normalized = normalize_points(target_0based)

    # 现在计算平移: target_cells[0] = normalized_rotated_voxels[0] + t (in 1-based)
    # 取第一个规范化后的点
    if not target_normalized or not normalized_rotated_voxels:
        return None

    # 找到target_cells中对应normalized_target[0]的原始坐标
    # 这需要建立映射关系
    # 简化: 直接用最小点计算

    # 目标最小点 (1-based)
    target_min_x = min(c.x for c in target_cells)
    target_min_y = min(c.y for c in target_cells)
    target_min_z = min(c.z for c in target_cells)

    # rotated_voxels的最小点需要对齐到target的最小点
    # 平移向量 = target_min - rotated_min (在0-based中)
    # 转换回1-based: t = target_min (1-based)

    translation = Vec3(target_min_x, target_min_y, target_min_z)

    return Transform(rot=rot_idx, t=translation)


def place_piece_by_cells(
    state: GameState,
    piece_id: str,
    target_cells: List[Vec3]
) -> PlacementResult:
    """
    按目标格子集合放置piece

    Args:
        state: 游戏状态
        piece_id: piece ID
        target_cells: 目标格子 (1-based世界坐标)

    Returns:
        PlacementResult
    """
    # 获取piece定义
    piece = state.get_piece_def(piece_id)
    if not piece:
        return PlacementResult(
            success=False,
            error=ErrorCode.PIECE_NOT_FOUND,
            message=f"Piece {piece_id} not found"
        )

    # 检查piece是否已经放置
    if piece_id in state.placed:
        return PlacementResult(
            success=False,
            error=ErrorCode.PIECE_ALREADY_PLACED,
            message=f"Piece {piece_id} is already placed"
        )

    # 1. 检查数量
    if len(target_cells) != len(piece.local_voxels):
        return PlacementResult(
            success=False,
            error=ErrorCode.WRONG_COUNT,
            message=f"Wrong count: expected {len(piece.local_voxels)}, got {len(target_cells)}"
        )

    # 2. 形状匹配
    # 将目标格子转换为0-based并规范化
    target_0based = [Vec3(c.x - 1, c.y - 1, c.z - 1) for c in target_cells]
    target_sig = points_to_signature(target_0based)

    if target_sig not in piece.rotation_signatures:
        return PlacementResult(
            success=False,
            error=ErrorCode.SHAPE_MISMATCH,
            message="Shape does not match any rotation of the piece"
        )

    # 推断变换
    transform = infer_transform_from_target(piece, target_cells, target_sig)
    if not transform:
        return PlacementResult(
            success=False,
            error=ErrorCode.SHAPE_MISMATCH,
            message="Failed to infer transform"
        )

    # 直接使用target_cells作为world_cells
    # 因为形状匹配已经确认它们是正确的
    world_cells_final = target_cells

    # 3. 边界检查
    if not within_box(world_cells_final, state.spec.box):
        return PlacementResult(
            success=False,
            error=ErrorCode.OUT_OF_BOUNDS,
            message="Piece exceeds box boundaries"
        )

    # 4. 碰撞检查
    for c in world_cells_final:
        if c.to_key() in state.occupied:
            return PlacementResult(
                success=False,
                error=ErrorCode.COLLISION,
                message=f"Collision at {c.to_tuple()}"
            )

    # 5. 支撑检查
    if not has_support(world_cells_final, state.by_cell, piece_id):
        return PlacementResult(
            success=False,
            error=ErrorCode.FLOATING,
            message="Piece is floating (no support)"
        )

    # 6. 提交放置
    commit_placement(state, piece_id, transform, world_cells_final)

    return PlacementResult(
        success=True,
        error=ErrorCode.OK,
        transform=transform,
        world_cells=world_cells_final,
        message="Piece placed successfully"
    )


def move_piece_by_cells(
    state: GameState,
    piece_id: str,
    target_cells: List[Vec3]
) -> PlacementResult:
    """
    移动piece到新位置

    Args:
        state: 游戏状态
        piece_id: piece ID
        target_cells: 目标格子 (1-based世界坐标)

    Returns:
        PlacementResult
    """
    # 检查piece是否已放置
    if piece_id not in state.placed:
        return PlacementResult(
            success=False,
            error=ErrorCode.PIECE_NOT_FOUND,
            message=f"Piece {piece_id} is not placed"
        )

    # 保存当前状态
    prev_placement = state.placed[piece_id]

    # 临时移除
    uncommit_placement(state, piece_id)

    # 尝试放置到新位置
    result = place_piece_by_cells(state, piece_id, target_cells)

    # 如果失败,恢复原状态
    if not result.success:
        commit_placement(
            state,
            piece_id,
            prev_placement.transform,
            prev_placement.world_cells
        )

    return result


def pickup_piece(state: GameState, piece_id: str) -> PlacementResult:
    """
    取出piece

    Args:
        state: 游戏状态
        piece_id: piece ID

    Returns:
        PlacementResult
    """
    if piece_id not in state.placed:
        return PlacementResult(
            success=False,
            error=ErrorCode.PIECE_NOT_FOUND,
            message=f"Piece {piece_id} is not placed"
        )

    uncommit_placement(state, piece_id)

    return PlacementResult(
        success=True,
        error=ErrorCode.OK,
        message=f"Piece {piece_id} picked up"
    )


def place_piece_by_transform(
    state: GameState,
    piece_id: str,
    rot: int,
    position: Vec3
) -> PlacementResult:
    """
    按变换(旋转+位置)直接放置piece

    Args:
        state: 游戏状态
        piece_id: piece ID
        rot: 旋转索引 (0-23)
        position: 位置 (1-based, 放置后的最小点位置)

    Returns:
        PlacementResult
    """
    # 获取piece定义
    piece = state.get_piece_def(piece_id)
    if not piece:
        return PlacementResult(
            success=False,
            error=ErrorCode.PIECE_NOT_FOUND,
            message=f"Piece {piece_id} not found"
        )

    # 检查piece是否已经放置
    if piece_id in state.placed:
        return PlacementResult(
            success=False,
            error=ErrorCode.PIECE_ALREADY_PLACED,
            message=f"Piece {piece_id} is already placed"
        )

    # 应用变换
    rot_matrix = ROTATION_MATRICES[rot]
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

    # 应用平移 (1-based)
    world_cells = [
        Vec3(v.x + position.x, v.y + position.y, v.z + position.z)
        for v in normalized_voxels
    ]

    transform = Transform(rot=rot, t=position)

    # 边界检查
    if not within_box(world_cells, state.spec.box):
        return PlacementResult(
            success=False,
            error=ErrorCode.OUT_OF_BOUNDS,
            message="Piece exceeds box boundaries"
        )

    # 碰撞检查
    for c in world_cells:
        if c.to_key() in state.occupied:
            return PlacementResult(
                success=False,
                error=ErrorCode.COLLISION,
                message=f"Collision at {c.to_tuple()}"
            )

    # 支撑检查
    if not has_support(world_cells, state.by_cell, piece_id):
        return PlacementResult(
            success=False,
            error=ErrorCode.FLOATING,
            message="Piece is floating (no support)"
        )

    # 提交放置
    commit_placement(state, piece_id, transform, world_cells)

    return PlacementResult(
        success=True,
        error=ErrorCode.OK,
        transform=transform,
        world_cells=world_cells,
        message="Piece placed successfully"
    )
