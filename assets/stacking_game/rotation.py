"""
24个旋转矩阵生成和应用
立方体旋转群 (Rot24)
"""

import numpy as np
from typing import List


def generate_24_rotations() -> List[np.ndarray]:
    """
    生成24个正旋转矩阵 (SO(3)的离散子群)

    方法: 枚举6个面朝向 + 每个面4个旋转方向
    """
    rotations = []

    # 基础旋转矩阵
    # 绕X轴旋转90度
    Rx90 = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ], dtype=int)

    # 绕Y轴旋转90度
    Ry90 = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
    ], dtype=int)

    # 绕Z轴旋转90度
    Rz90 = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ], dtype=int)

    # 单位矩阵
    I = np.eye(3, dtype=int)

    # 通过组合生成24个旋转
    # 6个面朝向 (通过绕X和Y旋转实现)
    face_rotations = [
        I,                      # +Z朝上
        Rx90,                   # +Y朝上
        Rx90 @ Rx90,            # -Z朝上
        Rx90 @ Rx90 @ Rx90,     # -Y朝上
        Ry90,                   # +X朝上
        Ry90 @ Ry90 @ Ry90,     # -X朝上
    ]

    # 对每个面朝向,绕该面旋转0/90/180/270度
    for face_rot in face_rotations:
        for i in range(4):
            # 绕Z轴旋转i*90度,然后应用面朝向
            z_rot = np.linalg.matrix_power(Rz90, i)
            final_rot = face_rot @ z_rot
            rotations.append(final_rot)

    return rotations


# 全局预计算的24个旋转矩阵
ROTATION_MATRICES = generate_24_rotations()


def get_rotation_matrix(rot_index: int) -> np.ndarray:
    """
    获取旋转矩阵
    rot_index: 0-23
    """
    if not 0 <= rot_index < 24:
        raise ValueError(f"Rotation index must be 0-23, got {rot_index}")
    return ROTATION_MATRICES[rot_index]


def apply_rotation(point: np.ndarray, rot_index: int) -> np.ndarray:
    """应用旋转到点"""
    rot_matrix = get_rotation_matrix(rot_index)
    return rot_matrix @ point


def find_matching_rotation(source_points: List[np.ndarray],
                          target_signature: str,
                          all_signatures: List[str]) -> int:
    """
    找到使source_points规范化后等于target_signature的旋转索引

    Args:
        source_points: 源点列表 (0-based)
        target_signature: 目标签名
        all_signatures: 预计算的24个旋转的签名列表

    Returns:
        旋转索引 (0-23), 如果找不到返回-1
    """
    for i, sig in enumerate(all_signatures):
        if sig == target_signature:
            return i
    return -1


def test_rotations():
    """测试旋转矩阵的正确性"""
    print(f"Generated {len(ROTATION_MATRICES)} rotation matrices")

    # 测试每个矩阵都是正交的且行列式为1
    for i, R in enumerate(ROTATION_MATRICES):
        # 检查正交性: R @ R.T = I
        product = R @ R.T
        is_orthogonal = np.allclose(product, np.eye(3))

        # 检查行列式为1 (不是-1,那样是镜像)
        det = np.linalg.det(R)
        is_rotation = np.isclose(det, 1.0)

        if not (is_orthogonal and is_rotation):
            print(f"Warning: Rotation {i} is invalid!")
            print(f"  Orthogonal: {is_orthogonal}, Det=1: {is_rotation}")
        else:
            print(f"Rotation {i:2d}: Valid")

    # 测试唯一性
    unique_count = len(set(tuple(map(tuple, R)) for R in ROTATION_MATRICES))
    print(f"\nUnique rotations: {unique_count}/24")

    # 显示一些示例
    print("\nExample rotations:")
    for i in [0, 6, 12, 18]:
        print(f"\nRotation {i}:")
        print(ROTATION_MATRICES[i])


if __name__ == "__main__":
    test_rotations()
