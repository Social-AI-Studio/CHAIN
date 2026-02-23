"""
3D交互式游戏 - 带可视化的puzzle游戏
"""

import matplotlib
# 设置交互式后端
try:
    matplotlib.use('TkAgg')  # 尝试使用TkAgg
except:
    try:
        matplotlib.use('Qt5Agg')  # 尝试Qt5
    except:
        matplotlib.use('TkAgg')  # 默认TkAgg

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import sys
import os

from game_core import Vec3, GameState
from loader import load_puzzle_by_name, create_game_state
from placement import place_piece_by_cells, pickup_piece, place_piece_by_transform
from visualizer_3d import visualize_state_3d, show_visualization
from initialization import initialize_pieces_on_ground, randomize_piece_rotation


class InteractiveGame3D:
    """带3D可视化的交互式游戏"""

    def __init__(self, puzzles_dir: str):
        self.puzzles_dir = puzzles_dir
        self.state: GameState = None
        self.puzzle_name = ""
        self.fig = None
        self.selected_piece = None

    def load_puzzle(self, size: str, puzzle_id: str, seed: int = None) -> bool:
        """加载关卡并初始化"""
        spec = load_puzzle_by_name(self.puzzles_dir, size, puzzle_id)
        if not spec:
            print(f"Failed to load puzzle {size}/{puzzle_id}")
            return False

        self.state = create_game_state(spec)
        self.puzzle_name = f"{size}/{puzzle_id}"

        # 初始化pieces在地面外
        initialize_pieces_on_ground(self.state, spacing=2, seed=seed)

        print(f"\n=== Loaded puzzle: {self.puzzle_name} ===")
        print(f"Box size: {spec.box[0]}x{spec.box[1]}x{spec.box[2]}")
        print(f"Pieces: {len(spec.pieces)}")
        print()

        return True

    def visualize(self):
        """显示3D可视化"""
        if not self.state:
            print("No puzzle loaded")
            return

        try:
            # 关闭之前的窗口
            if self.fig:
                plt.close(self.fig)

            # 创建新的可视化
            self.fig = visualize_state_3d(
                self.state,
                title=f"Puzzle: {self.puzzle_name}",
                show_unplaced=True
            )

            # 显示窗口
            plt.ion()  # 开启交互模式
            plt.show()
            plt.pause(0.1)

            print("✓ 3D窗口已打开")

        except Exception as e:
            print(f"⚠ 无法显示3D窗口: {e}")
            print("提示: 可能需要安装tkinter或配置显示环境")
            print("解决方案:")
            print("  1. 安装tkinter: sudo apt-get install python3-tk")
            print("  2. 或使用demo_3d.py生成图片文件")

    def show_status(self):
        """显示当前状态"""
        if not self.state:
            print("No puzzle loaded")
            return

        A, B, C = self.state.spec.box
        print("\n=== Status ===")
        print(f"Box: {A}x{B}x{C}")
        print(f"Occupied: {len(self.state.occupied)}/{A*B*C} cells")
        print(f"Placed: {len(self.state.placed)} pieces")
        print(f"Unplaced: {len(self.state.unplaced)} pieces")

        if self.state.is_complete():
            print("\n🎉 PUZZLE COMPLETE! 🎉")

    def place_piece_interactive(self, piece_id: str):
        """交互式放置piece"""
        if not self.state:
            print("No puzzle loaded")
            return

        piece = self.state.get_piece_def(piece_id)
        if not piece:
            print(f"Piece {piece_id} not found")
            return

        if piece_id in self.state.placed:
            print(f"Piece {piece_id} is already placed")
            return

        print(f"\n=== Placing piece {piece_id} ({len(piece.local_voxels)} cells) ===")
        print("Enter target cells (format: x,y,z), one per line")
        print("Or enter 'rot X Y Z R' to place by rotation (X,Y,Z=position, R=rotation 0-23)")
        print("Enter empty line to cancel:")

        mode = input("Mode [cells/rot]: ").strip().lower()

        if mode == "rot":
            # 按旋转放置
            try:
                print("Position (x y z):")
                pos_input = input().strip().split()
                x, y, z = map(int, pos_input)

                print(f"Rotation (0-{min(23, len(piece.rotation_signatures)-1)}):")
                rot = int(input().strip())

                result = place_piece_by_transform(
                    self.state, piece_id, rot, Vec3(x, y, z)
                )

                if result.success:
                    print(f"✓ {result.message}")
                    self.visualize()
                else:
                    print(f"✗ {result.error.value}: {result.message}")

            except (ValueError, KeyboardInterrupt) as e:
                print(f"\nCancelled: {e}")

        else:
            # 按格子放置
            target_cells = []
            try:
                while len(target_cells) < len(piece.local_voxels):
                    line = input(f"Cell {len(target_cells)+1}/{len(piece.local_voxels)}: ").strip()
                    if not line:
                        break

                    parts = line.split(',')
                    if len(parts) != 3:
                        print("Invalid format. Use: x,y,z")
                        continue

                    x, y, z = map(int, parts)
                    target_cells.append(Vec3(x, y, z))

            except (ValueError, KeyboardInterrupt):
                print("\nCancelled")
                return

            if len(target_cells) != len(piece.local_voxels):
                print(f"Wrong number of cells. Expected {len(piece.local_voxels)}, got {len(target_cells)}")
                return

            # 尝试放置
            result = place_piece_by_cells(self.state, piece_id, target_cells)

            if result.success:
                print(f"✓ {result.message}")
                self.visualize()
            else:
                print(f"✗ {result.error.value}: {result.message}")

    def pickup_piece_cmd(self, piece_id: str):
        """取出piece"""
        result = pickup_piece(self.state, piece_id)

        if result.success:
            print(f"✓ {result.message}")
            self.visualize()
        else:
            print(f"✗ {result.error.value}: {result.message}")

    def randomize_piece(self, piece_id: str):
        """随机化piece的旋转"""
        if not self.state:
            print("No puzzle loaded")
            return

        randomize_piece_rotation(self.state, piece_id)
        print(f"✓ Randomized piece {piece_id}")
        self.visualize()

    def run_interactive(self):
        """运行交互式游戏循环"""
        print("=" * 60)
        print("3D Interactive Polycube Stacking Game")
        print("=" * 60)
        print("\nCommands:")
        print("  load <size> <id> [seed] - Load a puzzle")
        print("  show                    - Show 3D visualization")
        print("  status                  - Show current status")
        print("  place <id>              - Place a piece")
        print("  pickup <id>             - Pick up a piece")
        print("  random <id>             - Randomize piece rotation")
        print("  help                    - Show this help")
        print("  quit                    - Exit")
        print()

        while True:
            try:
                cmd = input("\n> ").strip()

                if not cmd:
                    continue

                parts = cmd.split()
                command = parts[0].lower()

                if command == "help":
                    print("""
Available commands:
  load <size> <id> [seed] - Load a puzzle (e.g., load 2x2x2 puzzle_001)
  show                    - Show 3D visualization window
  status                  - Show current game status
  place <id>              - Place a piece interactively
  pickup <id>             - Pick up a placed piece
  random <id>             - Randomize piece rotation (in initial position)
  quit/exit               - Exit the game
                    """)

                elif command == "load":
                    if len(parts) < 3:
                        print("Usage: load <size> <puzzle_id> [seed]")
                        print("Example: load 2x2x2 puzzle_001 42")
                    else:
                        size = parts[1]
                        puzzle_id = parts[2]
                        seed = int(parts[3]) if len(parts) > 3 else None
                        if self.load_puzzle(size, puzzle_id, seed):
                            self.visualize()

                elif command == "show":
                    self.visualize()

                elif command == "status":
                    self.show_status()

                elif command == "place":
                    if len(parts) < 2:
                        print("Usage: place <piece_id>")
                    else:
                        self.place_piece_interactive(parts[1])

                elif command == "pickup":
                    if len(parts) < 2:
                        print("Usage: pickup <piece_id>")
                    else:
                        self.pickup_piece_cmd(parts[1])

                elif command == "random":
                    if len(parts) < 2:
                        print("Usage: random <piece_id>")
                    else:
                        self.randomize_piece(parts[1])

                elif command == "quit" or command == "exit":
                    print("Goodbye!")
                    break

                else:
                    print(f"Unknown command: {command}")
                    print("Type 'help' for available commands")

            except KeyboardInterrupt:
                print("\nUse 'quit' to exit")
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()


def main():
    """主函数"""
    puzzles_dir = "/mnt/moonfs/wuyuhao-m2/wyh/VisualReasonBench/Stacking_scaling/puzzles_full_v9"

    if len(sys.argv) > 1:
        puzzles_dir = sys.argv[1]

    game = InteractiveGame3D(puzzles_dir)
    game.run_interactive()


if __name__ == "__main__":
    main()
