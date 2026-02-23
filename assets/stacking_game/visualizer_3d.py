"""
3D可视化模块 - 使用matplotlib绘制3D视图
"""

import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from typing import List, Dict, Optional, Tuple
from matplotlib.colors import to_rgba

from game_core import Vec3, GameState, PieceDef


# 预定义的颜色方案
PIECE_COLORS = [
    '#FF6B6B',  # 红色
    '#4ECDC4',  # 青色
    '#45B7D1',  # 蓝色
    '#96CEB4',  # 绿色
    '#FFEAA7',  # 黄色
    '#DFE6E9',  # 灰色
    '#FD79A8',  # 粉色
    '#A29BFE',  # 紫色
    '#74B9FF',  # 浅蓝
    '#55EFC4',  # 薄荷绿
    '#FDCB6E',  # 橙色
    '#E17055',  # 橙红
]


def get_piece_color(piece_id: str) -> str:
    """获取piece的颜色"""
    try:
        idx = int(piece_id)
        return PIECE_COLORS[idx % len(PIECE_COLORS)]
    except:
        return PIECE_COLORS[0]


def create_cube_vertices(pos: Vec3, size: float = 0.9) -> np.ndarray:
    """
    创建单位立方体的顶点

    Args:
        pos: 立方体中心位置 (1-based)
        size: 立方体大小 (0-1之间,留出间隙)

    Returns:
        8x3的顶点数组
    """
    # 将1-based坐标转换为实际坐标
    cx, cy, cz = pos.x - 0.5, pos.y - 0.5, pos.z - 0.5
    d = size / 2.0

    vertices = np.array([
        [cx - d, cy - d, cz - d],  # 0: 左下前
        [cx + d, cy - d, cz - d],  # 1: 右下前
        [cx + d, cy + d, cz - d],  # 2: 右上前
        [cx - d, cy + d, cz - d],  # 3: 左上前
        [cx - d, cy - d, cz + d],  # 4: 左下后
        [cx + d, cy - d, cz + d],  # 5: 右下后
        [cx + d, cy + d, cz + d],  # 6: 右上后
        [cx - d, cy + d, cz + d],  # 7: 左上后
    ])

    return vertices


def create_cube_faces(vertices: np.ndarray) -> List[np.ndarray]:
    """
    创建立方体的6个面

    Returns:
        6个面,每个面是4个顶点的数组
    """
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # 前面 (z min)
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # 后面 (z max)
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # 底面 (y min)
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # 顶面 (y max)
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # 左面 (x min)
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面 (x max)
    ]

    return faces


def draw_voxel(ax: Axes3D, pos: Vec3, color: str, alpha: float = 0.8,
               edge_color: str = 'black', linewidth: float = 0.8):
    """
    绘制一个体素(单位立方体)
    
    Args:
        ax: matplotlib 3D axes
        pos: 体素位置 (1-based)
        color: 颜色
        alpha: 透明度
        edge_color: 边缘颜色
        linewidth: 边缘线宽
    """
    vertices = create_cube_vertices(pos)
    faces = create_cube_faces(vertices)

    # 创建3D多边形集合
    face_collection = Poly3DCollection(
        faces,
        facecolors=to_rgba(color, alpha),
        edgecolors=edge_color,
        linewidths=linewidth
    )

    ax.add_collection3d(face_collection)


def draw_box_frame(ax: Axes3D, box_size: Tuple[int, int, int],
                   color: str = 'gray', linewidth: float = 2.0):
    """
    绘制盒子的线框

    Args:
        ax: matplotlib 3D axes
        box_size: (A, B, C) 盒子尺寸
        color: 线框颜色
        linewidth: 线宽
    """
    A, B, C = box_size

    # 定义盒子的8个顶点 (0-based坐标系统)
    vertices = np.array([
        [0, 0, 0],  # 0
        [A, 0, 0],  # 1
        [A, B, 0],  # 2
        [0, B, 0],  # 3
        [0, 0, C],  # 4
        [A, 0, C],  # 5
        [A, B, C],  # 6
        [0, B, C],  # 7
    ])

    # 定义12条边
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7],  # 竖边
    ]

    for edge in edges:
        points = vertices[edge]
        ax.plot3D(*points.T, color=color, linewidth=linewidth, alpha=0.3)


def visualize_state_3d(state: GameState,
                       title: str = "3D Polycube Puzzle",
                       show_unplaced: bool = True,
                       figsize: Tuple[int, int] = (14, 8),
                       keep_unplaced_initial_pose: bool = True,
                       normalize_unplaced_view: bool = False,
                       pad_unplaced_view: int = 1) -> plt.Figure:
    """
    可视化游戏状态(3D视图)

    Args:
        state: 游戏状态
        title: 图表标题
        show_unplaced: 是否显示未放置的piece
        figsize: 图表大小
        keep_unplaced_initial_pose: 未放置piece保持加载时姿态与位置
        normalize_unplaced_view: 右侧小视角是否归一化到局部坐标
        pad_unplaced_view: 右侧小视角边界留白

    Returns:
        matplotlib Figure对象
    """
    A, B, C = state.spec.box

    if show_unplaced and state.unplaced:
        # 左侧单视角，右侧为piece的网格小视角
        fig = plt.figure(figsize=figsize)
        piece_ids = sorted(state.initial_placements.keys())
        n_pieces = len(piece_ids)
        right_cols = 2
        right_rows = max(1, math.ceil(n_pieces / right_cols))

        # 计算所有piece的最大尺寸，用于统一缩放比例
        max_span = 1.0
        for pid in piece_ids:
            placement = state.initial_placements.get(pid)
            if placement and placement.world_cells:
                min_x = min(c.x for c in placement.world_cells)
                max_x = max(c.x for c in placement.world_cells)
                min_y = min(c.y for c in placement.world_cells)
                max_y = max(c.y for c in placement.world_cells)
                min_z = min(c.z for c in placement.world_cells)
                max_z = max(c.z for c in placement.world_cells)
                max_span = max(max_span, max_x - min_x + 1, max_y - min_y + 1, max_z - min_z + 1)
        
        # 稍微增加一点视口余量
        view_size = max_span + pad_unplaced_view * 2

        gs = fig.add_gridspec(
            nrows=1,
            ncols=2,
            width_ratios=[1.6, 1.0],
            wspace=0.18,  # 增加间距，自然分隔左右两部分
        )
        gs_right = gs[0, 1].subgridspec(right_rows, right_cols, wspace=0.05, hspace=0.2)

        # 左侧：单一视角盒子视图
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        _draw_box_view(ax1, state, title, elev=20, azim=45)

        # 右侧：每个piece独立小视角（已放置的标记）
        if piece_ids:
            for idx, pid in enumerate(piece_ids):
                row = idx // right_cols
                col = idx % right_cols
                ax = fig.add_subplot(gs_right[row, col], projection='3d')
                is_placed = pid not in state.unplaced
                _draw_single_unplaced_piece(
                    ax,
                    state,
                    pid,
                    is_placed=is_placed,
                    keep_initial_pose=keep_unplaced_initial_pose,
                    normalize_view=normalize_unplaced_view,
                    pad=pad_unplaced_view,
                    unified_view_size=view_size
                )
        else:
            ax = fig.add_subplot(gs_right[0, 0], projection='3d')
            ax.text(0.5, 0.5, 0.5, "No pieces", ha='center', va='center')
            ax.set_axis_off()

        # 右侧标题（移除突兀的分隔线，使用间距自然分隔）
        fig.text(0.73, 0.96, "Pieces", ha="center", va="center",
                 fontsize=10, style='italic', alpha=0.6, color='gray')

    else:
        # 只显示盒子
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        _draw_box_view(ax, state, title, elev=20, azim=45)

    plt.tight_layout()
    return fig


def _draw_box_view(ax: Axes3D, state: GameState, title: str,
                   elev: float = 20, azim: float = 45):
    """绘制盒子视图"""
    A, B, C = state.spec.box

    # 绘制盒子线框
    draw_box_frame(ax, (A, B, C))

    # 绘制已放置的pieces
    for piece_id, placed in state.placed.items():
        color = get_piece_color(piece_id)
        for cell in placed.world_cells:
            draw_voxel(ax, cell, color, alpha=0.85, linewidth=0.8)

    # 设置坐标轴
    ax.set_xlabel('X', labelpad=10)
    ax.set_ylabel('Y', labelpad=10)
    ax.set_zlabel('Z', labelpad=10)
    ax.set_xlim([0, A])
    ax.set_ylim([0, B])
    ax.set_zlim([0, C])
    
    # 禁用网格线，避免在标签位置（0.5, 1.5）绘制虚线
    ax.grid(False)
    
    # 设置刻度位置在单元格中心（0.5, 1.5, ...）用于标签显示
    tick_pos_x = [i + 0.5 for i in range(A)]
    tick_pos_y = [i + 0.5 for i in range(B)]
    tick_pos_z = [i + 0.5 for i in range(C)]
    ax.set_xticks(tick_pos_x)
    ax.set_yticks(tick_pos_y)
    ax.set_zticks(tick_pos_z)
    ax.set_xticklabels([str(i) for i in range(A)])
    ax.set_yticklabels([str(i) for i in range(B)])
    ax.set_zticklabels([str(i) for i in range(C)])
    # 隐藏刻度线本身，只显示标签
    ax.tick_params(axis='x', which='both', length=0, pad=2)
    ax.tick_params(axis='y', which='both', length=0, pad=2)
    ax.tick_params(axis='z', which='both', length=0, pad=2)
    try:
        ax.set_box_aspect((A, B, C))
    except Exception:
        pass

    # 设置视角
    ax.view_init(elev=elev, azim=azim)

    # 标题
    occupied = len(state.occupied)
    total = A * B * C
    status = "COMPLETE!" if state.is_complete() else f"{occupied}/{total} cells"
    ax.set_title(f"{title}\n{status}")


def _draw_unplaced_pieces(ax: Axes3D, state: GameState):
    """绘制未放置的pieces（使用initial_placements中的位置和旋转）"""
    ax.set_title("Unplaced Pieces")

    if not state.initial_placements:
        ax.text(0.5, 0.5, 0.5, "No pieces", ha='center', va='center')
        return

    # 找到所有初始放置的边界
    all_cells = []
    for piece_id in state.unplaced:
        if piece_id in state.initial_placements:
            all_cells.extend(state.initial_placements[piece_id].world_cells)

    if not all_cells:
        return

    # 计算边界
    min_x = min(c.x for c in all_cells)
    max_x = max(c.x for c in all_cells)
    min_y = min(c.y for c in all_cells)
    max_y = max(c.y for c in all_cells)
    min_z = min(c.z for c in all_cells)
    max_z = max(c.z for c in all_cells)

    # 绘制每个未放置的piece
    for piece_id in sorted(state.unplaced):
        if piece_id not in state.initial_placements:
            continue

        placement = state.initial_placements[piece_id]
        color = get_piece_color(piece_id)

        # 绘制piece的每个体素
        for cell in placement.world_cells:
            draw_voxel(ax, cell, color, alpha=0.7)

    # 绘制地面网格（帮助理解pieces在地上）
    ground_z = 0.5  # 稍微低于z=1
    ground_x = [max(0, min_x - 1), max_x + 1]
    ground_y = [max(0, min_y - 1), max_y + 1]

    from matplotlib.patches import Rectangle
    import matplotlib.patches as mpatches

    # 画一个地面矩形
    xx, yy = np.meshgrid(ground_x, ground_y)
    zz = np.ones_like(xx) * ground_z
    ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray')

    # 设置坐标轴
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    x0 = max(0, min_x - 1)
    y0 = max(0, min_y - 1)
    z0 = 0
    x1 = max_x + 1
    y1 = max_y + 1
    z1 = max_z + 1
    ax.set_xlim([x0, x1])
    ax.set_ylim([y0, y1])
    ax.set_zlim([z0, z1])
    ax.set_xticks(list(range(int(x0), int(x1) + 1)))
    ax.set_yticks(list(range(int(y0), int(y1) + 1)))
    ax.set_zticks(list(range(int(z0), int(z1) + 1)))

    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
    ax.view_init(elev=20, azim=45)


def _draw_single_unplaced_piece(ax: Axes3D, state: GameState, piece_id: str,
                                is_placed: bool = False,
                                keep_initial_pose: bool = True,
                                normalize_view: bool = False,
                                pad: int = 1,
                                unified_view_size: Optional[float] = None):
    """绘制单个piece的小视角（网格排布用）"""
    # keep_initial_pose: 保持初始加载姿态与位置
    # normalize_view: 归一化到局部坐标（与初始位置无关）
    # pad: 视角边界留白（单位为格）
    # unified_view_size: 统一的视图尺寸（用于保持所有piece比例一致）
    placement = state.initial_placements.get(piece_id)
    if not placement or not placement.world_cells:
        ax.text(0.5, 0.5, 0.5, "No data", ha='center', va='center')
        ax.set_axis_off()
        return

    color = get_piece_color(piece_id)
    min_x = min(c.x for c in placement.world_cells)
    max_x = max(c.x for c in placement.world_cells)
    min_y = min(c.y for c in placement.world_cells)
    max_y = max(c.y for c in placement.world_cells)
    min_z = min(c.z for c in placement.world_cells)
    max_z = max(c.z for c in placement.world_cells)

    if keep_initial_pose and not normalize_view:
        # 保持与加载时一致的朝向与位置
        draw_cells = placement.world_cells
        
        # 计算当前piece的中心
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        cz = (min_z + max_z) / 2
        
        if unified_view_size:
            # 使用统一的视图尺寸，居中显示
            d = unified_view_size / 2
            x0, x1 = cx - d, cx + d
            y0, y1 = cy - d, cy + d
            z0, z1 = cz - d, cz + d
        else:
            x0 = min_x - pad
            y0 = min_y - pad
            z0 = min_z - pad
            x1 = max_x + pad
            y1 = max_y + pad
            z1 = max_z + pad
    else:
        # 规范化到小视角坐标系，避免受全局摆放影响
        draw_cells = [
            Vec3(c.x - min_x + 1, c.y - min_y + 1, c.z - min_z + 1)
            for c in placement.world_cells
        ]
        
        if unified_view_size:
            # 规范化模式下如果也要求统一比例（虽然这里通常不需要）
            size_x = max_x - min_x + 1
            size_y = max_y - min_y + 1
            size_z = max_z - min_z + 1
            cx, cy, cz = size_x/2 + 0.5, size_y/2 + 0.5, size_z/2 + 0.5
            d = unified_view_size / 2
            x0, x1 = cx - d, cx + d
            y0, y1 = cy - d, cy + d
            z0, z1 = cz - d, cz + d
        else:
            size_x = max_x - min_x + 1
            size_y = max_y - min_y + 1
            size_z = max_z - min_z + 1
            x0 = 0
            y0 = 0
            z0 = 0
            x1 = size_x + pad
            y1 = size_y + pad
            z1 = size_z + pad

    for cell in draw_cells:
        draw_voxel(ax, cell, color, alpha=0.85, linewidth=0.8)

    ax.set_xlim([x0, x1])
    ax.set_ylim([y0, y1])
    ax.set_zlim([z0, z1])
    
    # 设置正确的空间比例，防止变形
    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))

    title = f"Piece {piece_id}"
    if is_placed:
        title = f"{title} (PLACED)"
        # 标记已放置
        ax.text2D(0.5, 0.5, "❌", transform=ax.transAxes,
                  ha='center', va='center', fontsize=28, alpha=0.7)
    ax.set_title(title)
    ax.view_init(elev=20, azim=45)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')


def visualize_piece_rotations(piece: PieceDef,
                              num_rotations: int = 8,
                              title: Optional[str] = None,
                              rotate_piece: bool = False) -> plt.Figure:
    """
    可视化piece的多个旋转

    Args:
        piece: piece定义
        num_rotations: 显示的旋转数量
        title: 图表标题
        rotate_piece: 是否对物体进行旋转

    Returns:
        matplotlib Figure对象
    """
    from rotation import ROTATION_MATRICES
    import numpy as np

    num_to_show = min(num_rotations, len(piece.rotation_signatures))

    # 计算子图布局
    cols = 4
    rows = (num_to_show + cols - 1) // cols

    fig = plt.figure(figsize=(3 * cols, 3 * rows))

    color = get_piece_color(piece.id)

    for i in range(num_to_show):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')

        # 应用旋转
        if rotate_piece:
            rot_matrix = ROTATION_MATRICES[i]
            rotated_voxels = []

            for v in piece.local_voxels:
                vec = np.array([v.x, v.y, v.z])
                rotated = rot_matrix @ vec
                rotated_voxels.append(Vec3(int(rotated[0]), int(rotated[1]), int(rotated[2])))
        else:
            rotated_voxels = list(piece.local_voxels)

        # 规范化到(1,1,1)起点
        min_x = min(v.x for v in rotated_voxels)
        min_y = min(v.y for v in rotated_voxels)
        min_z = min(v.z for v in rotated_voxels)

        for v in rotated_voxels:
            pos = Vec3(v.x - min_x + 1, v.y - min_y + 1, v.z - min_z + 1)
            draw_voxel(ax, pos, color, alpha=0.7)

        # 设置坐标轴
        max_coord = max(
            max(v.x - min_x for v in rotated_voxels),
            max(v.y - min_y for v in rotated_voxels),
            max(v.z - min_z for v in rotated_voxels)
        ) + 1

        ax.set_xlim([0, max_coord + 1])
        ax.set_ylim([0, max_coord + 1])
        ax.set_zlim([0, max_coord + 1])
        # 强制为正方体比例，因为坐标轴范围一致
        ax.set_box_aspect((1, 1, 1))
        
        ax.set_title(f"Rotation {i}")
        ax.view_init(elev=20, azim=45)

        # 隐藏坐标轴刻度
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    else:
        fig.suptitle(f"Piece {piece.id} - {len(piece.rotation_signatures)} Unique Rotations",
                    fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def animate_placement(state: GameState,
                     piece_id: str,
                     target_cells: List[Vec3],
                     num_frames: int = 20) -> List[plt.Figure]:
    """
    创建放置动画的帧序列

    Args:
        state: 游戏状态
        piece_id: 要放置的piece ID
        target_cells: 目标位置
        num_frames: 动画帧数

    Returns:
        图表列表
    """
    from placement import place_piece_by_cells

    # 先尝试放置(不提交)
    result = place_piece_by_cells(state, piece_id, target_cells)

    if not result.success:
        print(f"Cannot animate: {result.message}")
        return []

    # TODO: 实现动画逻辑
    # 这里可以添加piece从初始位置移动到目标位置的动画

    return []


def save_visualization(fig: plt.Figure, filename: str, dpi: int = 150):
    """
    保存可视化图表

    Args:
        fig: matplotlib Figure对象
        filename: 输出文件名
        dpi: 分辨率
    """
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"Saved visualization to {filename}")


def show_visualization(fig: plt.Figure):
    """显示可视化图表"""
    plt.show()


# 测试代码
if __name__ == "__main__":
    print("Testing 3D Visualizer...")

    # 创建一个简单的测试场景
    from game_core import PieceDef, LevelSpec, GameState
    from loader import preprocess_piece
    from placement import place_piece_by_transform

    # 创建piece
    piece0 = PieceDef(
        id="0",
        local_voxels=[Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(1, 1, 0)]
    )
    piece1 = PieceDef(
        id="1",
        local_voxels=[Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0)]
    )

    piece0 = preprocess_piece(piece0)
    piece1 = preprocess_piece(piece1)

    # 创建关卡
    spec = LevelSpec(box=(3, 3, 2), pieces=[piece0, piece1])
    state = GameState(spec=spec)

    # 放置一些pieces
    place_piece_by_transform(state, "0", rot=0, position=Vec3(1, 1, 1))

    # 可视化
    print("Creating visualization...")
    fig = visualize_state_3d(state, title="Test Scene", show_unplaced=True)

    print("Displaying... (close window to continue)")
    show_visualization(fig)

    # 显示piece的旋转
    print("\nShowing piece rotations...")
    fig2 = visualize_piece_rotations(piece1, num_rotations=8, rotate_piece=False)
    show_visualization(fig2)

    print("Done!")
