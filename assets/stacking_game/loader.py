"""
关卡加载器 - 从puzzles_full_v9读取JSON数据
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple
from game_core import Vec3, PieceDef, LevelSpec, GameState
from rotation import ROTATION_MATRICES
import numpy as np


def load_puzzle_from_json(json_path: str) -> Optional[LevelSpec]:
    """
    从JSON文件加载puzzle

    Args:
        json_path: puzzle的JSON文件路径

    Returns:
        LevelSpec对象或None
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # 解析box尺寸 (注意: JSON中是0-based, 我们要转换为实际尺寸)
        box = tuple(data['box'])  # [A, B, C]

        # 解析pieces
        pieces = []
        for i, piece_data in enumerate(data['pieces']):
            piece_id = str(i)  # 使用索引作为ID

            # 转换坐标列表为Vec3 (JSON中是0-based)
            local_voxels = [Vec3.from_list(coord) for coord in piece_data]

            # 创建PieceDef (会自动规范化到(0,0,0)起点)
            piece = PieceDef(id=piece_id, local_voxels=local_voxels)

            # 预处理旋转签名
            piece = preprocess_piece(piece)
            pieces.append(piece)

        return LevelSpec(box=box, pieces=pieces)

    except Exception as e:
        print(f"Error loading puzzle from {json_path}: {e}")
        return None


def preprocess_piece(piece: PieceDef) -> PieceDef:
    """
    预处理piece: 生成所有24个旋转的规范形签名

    Args:
        piece: 原始piece定义

    Returns:
        添加了rotation_signatures的piece
    """
    from game_core import normalize_points, points_to_signature

    seen_signatures = set()
    rotation_signatures = []

    # 对24个旋转
    for rot_idx in range(24):
        rot_matrix = ROTATION_MATRICES[rot_idx]

        # 旋转所有本地体素
        rotated_points = []
        for v in piece.local_voxels:
            vec = np.array([v.x, v.y, v.z])
            rotated = rot_matrix @ vec
            rotated_points.append(Vec3(int(rotated[0]), int(rotated[1]), int(rotated[2])))

        # 生成签名
        signature = points_to_signature(rotated_points)

        # 去重
        if signature not in seen_signatures:
            seen_signatures.add(signature)
            rotation_signatures.append(signature)

    piece.rotation_signatures = rotation_signatures
    return piece


def find_all_puzzles(base_dir: str) -> List[Tuple[str, str]]:
    """
    查找所有puzzle文件

    Args:
        base_dir: puzzles_full_v9目录路径

    Returns:
        [(puzzle_name, json_path), ...] 列表
    """
    puzzles = []
    base_path = Path(base_dir)

    # 遍历所有尺寸目录 (如2x2x2, 3x3x3等)
    for size_dir in base_path.iterdir():
        if not size_dir.is_dir() or size_dir.name.startswith('_'):
            continue

        # 遍历该尺寸下的所有puzzle
        for puzzle_dir in size_dir.iterdir():
            if not puzzle_dir.is_dir() or puzzle_dir.name.startswith('_'):
                continue

            # 查找JSON文件
            json_files = list(puzzle_dir.glob('*.json'))
            if json_files:
                json_path = str(json_files[0])
                puzzle_name = f"{size_dir.name}/{puzzle_dir.name}"
                puzzles.append((puzzle_name, json_path))

    return sorted(puzzles)


def load_puzzle_by_name(base_dir: str, size: str, puzzle_id: str) -> Optional[LevelSpec]:
    """
    按名称加载puzzle

    Args:
        base_dir: puzzles_full_v9目录路径
        size: 尺寸 (如 "2x2x2")
        puzzle_id: puzzle ID (如 "puzzle_001" 或 "puzzle_easy_001")

    Returns:
        LevelSpec对象或None
    """
    # 解析 puzzle_id: "puzzle_easy_001" -> difficulty="easy", number="001"
    # 或者 "puzzle_001" -> 直接尝试作为完整名称
    
    # 移除 "puzzle_" 前缀（如果有）
    if puzzle_id.startswith("puzzle_"):
        puzzle_id = puzzle_id[7:]  # Remove "puzzle_" prefix
    
    # 尝试解析为 difficulty_number 格式
    parts = puzzle_id.split("_")
    if len(parts) >= 2:
        difficulty = parts[0]  # e.g., "easy", "mid", "hard"
        number = parts[1]      # e.g., "001", "002"
        
        # 构建新的路径格式: {size}/{difficulty}/{number}/{size}_{difficulty}_{number}.json
        json_path = os.path.join(
            base_dir,
            size,
            difficulty,
            number,
            f"{size}_{difficulty}_{number}.json"
        )
        
        if os.path.exists(json_path):
            return load_puzzle_from_json(json_path)
    
    # 回退到旧格式（兼容性）
    old_json_path = os.path.join(base_dir, size, puzzle_id, f"{puzzle_id}_{size}.json")
    if os.path.exists(old_json_path):
        return load_puzzle_from_json(old_json_path)
    
    return None


def create_game_state(level_spec: LevelSpec) -> GameState:
    """
    从LevelSpec创建GameState

    Args:
        level_spec: 关卡规格

    Returns:
        初始化的游戏状态
    """
    return GameState(spec=level_spec)


def list_available_puzzles(base_dir: str):
    """
    列出所有可用的puzzle

    Args:
        base_dir: puzzles_full_v9目录路径
    """
    puzzles = find_all_puzzles(base_dir)
    print(f"Found {len(puzzles)} puzzles:")
    print()

    # 按尺寸分组
    by_size = {}
    for name, path in puzzles:
        size = name.split('/')[0]
        if size not in by_size:
            by_size[size] = []
        by_size[size].append(name)

    for size in sorted(by_size.keys()):
        print(f"{size}: {len(by_size[size])} puzzles")
        for name in by_size[size][:3]:  # 显示前3个
            print(f"  - {name}")
        if len(by_size[size]) > 3:
            print(f"  ... and {len(by_size[size]) - 3} more")
        print()


if __name__ == "__main__":
    # 测试加载
    base_dir = str(Path(__file__).resolve().parent / "puzzles_full_v9")

    print("=== Available Puzzles ===")
    list_available_puzzles(base_dir)

    print("\n=== Loading Sample Puzzle ===")
    spec = load_puzzle_by_name(base_dir, "2x2x2", "puzzle_001")

    if spec:
        print(f"Box size: {spec.box}")
        print(f"Number of pieces: {len(spec.pieces)}")
        for piece in spec.pieces:
            print(f"\nPiece {piece.id}:")
            print(f"  Voxels: {len(piece.local_voxels)}")
            print(f"  Unique rotations: {len(piece.rotation_signatures)}")
            print(f"  Coordinates: {[v.to_tuple() for v in piece.local_voxels]}")
