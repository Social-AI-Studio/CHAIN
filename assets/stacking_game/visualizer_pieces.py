"""
单独绘制每个piece的可视化 - 类似原始pieces图片的风格
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import List, Tuple

from game_core import Vec3, GameState, PieceDef
from visualizer_3d import (
    get_piece_color, draw_voxel, create_cube_vertices, create_cube_faces,
    PIECE_COLORS
)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import to_rgba


def visualize_pieces_grid(state: GameState,
                          title: str = "Pieces",
                          cols: int = 3,
                          figsize: Tuple[int, int] = None) -> plt.Figure:
    """
    网格布局显示所有pieces，每个piece单独一个子图

    Args:
        state: 游戏状态
        title: 总标题
        cols: 每行显示几个piece
        figsize: 图表大小

    Returns:
        matplotlib Figure对象
    """
    # 获取所有未放置的pieces
    piece_ids = sorted(state.unplaced)
    num_pieces = len(piece_ids)

    if num_pieces == 0:
        print("No pieces to display")
        return None

    # 计算布局
    rows = (num_pieces + cols - 1) // cols

    # 自动计算figsize
    if figsize is None:
        figsize = (cols * 4, rows * 4)

    fig = plt.figure(figsize=figsize)

    # 为每个piece创建子图
    for idx, piece_id in enumerate(piece_ids):
        piece = state.get_piece_def(piece_id)
        if not piece:
            continue

        # 创建3D子图
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')

        # 获取该piece的初始放置信息（如果有）
        if piece_id in state.initial_placements:
            placement = state.initial_placements[piece_id]
            world_cells = placement.world_cells

            # 规范化到原点附近（用于单独显示）
            min_x = min(c.x for c in world_cells)
            min_y = min(c.y for c in world_cells)
            min_z = min(c.z for c in world_cells)

            # 转换为相对坐标（从1开始，更好看）
            display_cells = [
                Vec3(c.x - min_x + 1, c.y - min_y + 1, c.z - min_z + 1)
                for c in world_cells
            ]
        else:
            # 如果没有初始放置，使用本地坐标
            display_cells = [
                Vec3(v.x + 1, v.y + 1, v.z + 1)
                for v in piece.local_voxels
            ]

        # 获取颜色
        color = get_piece_color(piece_id)

        # 绘制piece的每个体素
        for cell in display_cells:
            draw_voxel(ax, cell, color, alpha=0.9, edge_color='black', linewidth=1.5)

        # 计算边界
        max_x = max(c.x for c in display_cells)
        max_y = max(c.y for c in display_cells)
        max_z = max(c.z for c in display_cells)

        max_size = max(max_x, max_y, max_z) + 1

        # 绘制地面平面
        ground_size = max_size
        ground_x = [0, ground_size]
        ground_y = [0, ground_size]
        xx, yy = np.meshgrid(ground_x, ground_y)
        zz = np.ones_like(xx) * 0.5
        ax.plot_surface(xx, yy, zz, alpha=0.1, color='lightgray')

        # 设置坐标轴
        ax.set_xlim([0, max_size])
        ax.set_ylim([0, max_size])
        ax.set_zlim([0, max_size])

        # 隐藏坐标轴刻度
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # 设置视角
        ax.view_init(elev=25, azim=45)

        # 标题：Piece ID 和体素数量
        num_voxels = len(display_cells)
        ax.set_title(f"Piece {piece_id} (|V|={num_voxels})",
                    fontsize=12, fontweight='bold')

        # 设置背景颜色
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # 设置轴线颜色为灰色
        ax.xaxis.pane.set_edgecolor('lightgray')
        ax.yaxis.pane.set_edgecolor('lightgray')
        ax.zaxis.pane.set_edgecolor('lightgray')

    # 设置总标题
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def save_pieces_visualization(state: GameState, filename: str,
                              title: str = "Puzzle Pieces",
                              dpi: int = 150):
    """
    保存pieces网格可视化

    Args:
        state: 游戏状态
        filename: 输出文件名
        title: 标题
        dpi: 分辨率
    """
    fig = visualize_pieces_grid(state, title=title)
    if fig:
        fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved pieces visualization to {filename}")
        plt.close(fig)


# 测试代码
if __name__ == "__main__":
    print("Testing pieces grid visualization...")

    import matplotlib
    matplotlib.use('Agg')

    from loader import load_puzzle_by_name, create_game_state
    from initialization import initialize_pieces_on_ground

    # 测试3x3x3 puzzle
    puzzles_dir = "/mnt/moonfs/wuyuhao-m2/wyh/VisualReasonBench/Stacking_scaling/puzzles_full_v9"
    spec = load_puzzle_by_name(puzzles_dir, "3x3x3", "puzzle_001")

    if spec:
        state = create_game_state(spec)
        initialize_pieces_on_ground(state, seed=42)

        # 生成可视化
        save_pieces_visualization(
            state,
            "/tmp/test_pieces_grid.png",
            title="3x3x3 Puzzle Pieces",
            dpi=200
        )

        print("✓ Test completed!")
