"""
3D Polycube Stacking Game - Core Logic
核心数据结构和游戏逻辑实现
"""

from typing import List, Set, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class ErrorCode(Enum):
    """错误码"""
    OK = "OK"
    WRONG_COUNT = "WrongCount"
    SHAPE_MISMATCH = "ShapeMismatch"
    OUT_OF_BOUNDS = "OutOfBounds"
    COLLISION = "Collision"
    FLOATING = "Floating"
    PIECE_NOT_FOUND = "PieceNotFound"
    PIECE_ALREADY_PLACED = "PieceAlreadyPlaced"


@dataclass
class Vec3:
    """3D向量 (1-based坐标对外，0-based内部使用)"""
    x: int
    y: int
    z: int

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def to_tuple(self) -> Tuple[int, int, int]:
        return (self.x, self.y, self.z)

    def to_key(self) -> str:
        """转换为字符串key用于集合/字典"""
        return f"{self.x},{self.y},{self.z}"

    @staticmethod
    def from_list(lst: List[int]) -> 'Vec3':
        return Vec3(lst[0], lst[1], lst[2])

    @staticmethod
    def from_key(key: str) -> 'Vec3':
        x, y, z = map(int, key.split(','))
        return Vec3(x, y, z)


@dataclass
class Transform:
    """刚体变换 (旋转 + 平移)"""
    rot: int  # 0-23 旋转索引
    t: Vec3   # 平移向量


@dataclass
class PieceDef:
    """方块定义"""
    id: str
    local_voxels: List[Vec3]  # 本地坐标 (0-based)
    rotation_signatures: List[str] = field(default_factory=list)  # 预处理的旋转签名

    def __post_init__(self):
        """确保local_voxels已规范化到(0,0,0)起点"""
        if not self.local_voxels:
            return
        # 规范化到最小点为(0,0,0)
        min_x = min(v.x for v in self.local_voxels)
        min_y = min(v.y for v in self.local_voxels)
        min_z = min(v.z for v in self.local_voxels)
        if min_x != 0 or min_y != 0 or min_z != 0:
            self.local_voxels = [
                Vec3(v.x - min_x, v.y - min_y, v.z - min_z)
                for v in self.local_voxels
            ]


@dataclass
class PlacedPiece:
    """已放置的方块"""
    id: str
    transform: Transform
    world_cells: List[Vec3]  # 世界坐标缓存


@dataclass
class LevelSpec:
    """关卡规格"""
    box: Tuple[int, int, int]  # (A, B, C) 盒子尺寸
    pieces: List[PieceDef]


@dataclass
class GameState:
    """游戏状态"""
    spec: LevelSpec
    occupied: Set[str] = field(default_factory=set)  # 已占用格子的key集合
    by_cell: Dict[str, str] = field(default_factory=dict)  # cell_key -> piece_id
    placed: Dict[str, PlacedPiece] = field(default_factory=dict)  # piece_id -> PlacedPiece
    unplaced: Set[str] = field(default_factory=set)  # 未放置的piece_id集合
    initial_placements: Dict[str, PlacedPiece] = field(default_factory=dict)  # 初始放置位置(在地面外)

    def __post_init__(self):
        """初始化时设置所有piece为未放置"""
        if not self.unplaced:
            self.unplaced = {p.id for p in self.spec.pieces}

    def is_complete(self) -> bool:
        """检查是否完成"""
        A, B, C = self.spec.box
        total_cells = A * B * C
        return len(self.occupied) == total_cells and len(self.unplaced) == 0

    def get_piece_def(self, piece_id: str) -> Optional[PieceDef]:
        """获取piece定义"""
        for p in self.spec.pieces:
            if p.id == piece_id:
                return p
        return None


@dataclass
class PlacementResult:
    """放置结果"""
    success: bool
    error: ErrorCode
    transform: Optional[Transform] = None
    world_cells: Optional[List[Vec3]] = None
    message: str = ""


def vec3_list_to_key_set(cells: List[Vec3]) -> Set[str]:
    """Vec3列表转换为key集合"""
    return {c.to_key() for c in cells}


def normalize_points(points: List[Vec3]) -> List[Vec3]:
    """
    规范化点集:将最小坐标平移到(0,0,0),然后按字典序排序
    """
    if not points:
        return []

    min_x = min(p.x for p in points)
    min_y = min(p.y for p in points)
    min_z = min(p.z for p in points)

    normalized = [
        Vec3(p.x - min_x, p.y - min_y, p.z - min_z)
        for p in points
    ]

    # 按字典序排序
    normalized.sort(key=lambda v: (v.x, v.y, v.z))
    return normalized


def points_to_signature(points: List[Vec3]) -> str:
    """
    点集转换为签名字符串
    """
    normalized = normalize_points(points)
    return ";".join(f"{p.x},{p.y},{p.z}" for p in normalized)


def apply_rotation_matrix(point: Vec3, rot_matrix: np.ndarray) -> Vec3:
    """应用旋转矩阵到点"""
    vec = np.array([point.x, point.y, point.z])
    rotated = rot_matrix @ vec
    return Vec3(int(rotated[0]), int(rotated[1]), int(rotated[2]))


def apply_transform(local_voxels: List[Vec3], rot_matrix: np.ndarray,
                   translation: Vec3, to_one_based: bool = True) -> List[Vec3]:
    """
    应用变换到本地坐标
    local_voxels: 0-based本地坐标
    translation: 平移向量
    to_one_based: 是否转换为1-based坐标
    """
    result = []
    for v in local_voxels:
        # 旋转
        rotated = apply_rotation_matrix(v, rot_matrix)
        # 平移
        if to_one_based:
            # translation是1-based的目标位置
            transformed = Vec3(
                rotated.x + translation.x,
                rotated.y + translation.y,
                rotated.z + translation.z
            )
        else:
            transformed = Vec3(
                rotated.x + translation.x,
                rotated.y + translation.y,
                rotated.z + translation.z
            )
        result.append(transformed)
    return result


def within_box(cells: List[Vec3], box: Tuple[int, int, int]) -> bool:
    """检查所有单元格是否在盒子范围内 (1-based坐标)"""
    A, B, C = box
    for c in cells:
        if not (1 <= c.x <= A and 1 <= c.y <= B and 1 <= c.z <= C):
            return False
    return True


def has_support(world_cells: List[Vec3], by_cell: Dict[str, str],
                current_piece_id: str) -> bool:
    """
    检查是否有支撑(非悬空)
    至少有一个体素满足:
    1. 在底部 (z=1)
    2. 或有一个六邻域格子被其他块占据
    """
    for c in world_cells:
        # 底部支撑
        if c.z == 1:
            return True

        # 检查六邻域
        neighbors = [
            Vec3(c.x + 1, c.y, c.z),
            Vec3(c.x - 1, c.y, c.z),
            Vec3(c.x, c.y + 1, c.z),
            Vec3(c.x, c.y - 1, c.z),
            Vec3(c.x, c.y, c.z + 1),
            Vec3(c.x, c.y, c.z - 1),
        ]

        for n in neighbors:
            n_key = n.to_key()
            if n_key in by_cell and by_cell[n_key] != current_piece_id:
                return True

    return False


def commit_placement(state: GameState, piece_id: str,
                    transform: Transform, world_cells: List[Vec3]):
    """提交放置操作"""
    # 添加到occupied和by_cell
    for c in world_cells:
        key = c.to_key()
        state.occupied.add(key)
        state.by_cell[key] = piece_id

    # 更新placed和unplaced
    state.placed[piece_id] = PlacedPiece(
        id=piece_id,
        transform=transform,
        world_cells=world_cells
    )
    state.unplaced.discard(piece_id)


def uncommit_placement(state: GameState, piece_id: str):
    """撤销放置操作"""
    if piece_id not in state.placed:
        return

    placed_piece = state.placed[piece_id]

    # 从occupied和by_cell移除
    for c in placed_piece.world_cells:
        key = c.to_key()
        state.occupied.discard(key)
        if key in state.by_cell and state.by_cell[key] == piece_id:
            del state.by_cell[key]

    # 更新placed和unplaced
    del state.placed[piece_id]
    state.unplaced.add(piece_id)
