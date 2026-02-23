"""
CLI交互界面 - 简单的命令行游戏界面
"""

import sys
from typing import List, Optional
from game_core import Vec3, GameState
from loader import load_puzzle_by_name, create_game_state, find_all_puzzles
from placement import (
    place_piece_by_cells,
    place_piece_by_transform,
    move_piece_by_cells,
    pickup_piece
)


class StackingGame:
    """堆叠游戏主类"""

    def __init__(self, puzzles_dir: str):
        self.puzzles_dir = puzzles_dir
        self.state: Optional[GameState] = None
        self.puzzle_name = ""

    def load_puzzle(self, size: str, puzzle_id: str) -> bool:
        """加载关卡"""
        spec = load_puzzle_by_name(self.puzzles_dir, size, puzzle_id)
        if not spec:
            print(f"Failed to load puzzle {size}/{puzzle_id}")
            return False

        self.state = create_game_state(spec)
        self.puzzle_name = f"{size}/{puzzle_id}"
        print(f"\n=== Loaded puzzle: {self.puzzle_name} ===")
        print(f"Box size: {spec.box[0]}x{spec.box[1]}x{spec.box[2]}")
        print(f"Pieces: {len(spec.pieces)}")
        print()
        return True

    def show_state(self):
        """显示当前状态"""
        if not self.state:
            print("No puzzle loaded")
            return

        print("\n=== Current State ===")
        A, B, C = self.state.spec.box
        print(f"Box: {A}x{B}x{C}")
        print(f"Occupied: {len(self.state.occupied)}/{A*B*C} cells")
        print(f"Placed pieces: {len(self.state.placed)}")
        print(f"Unplaced pieces: {len(self.state.unplaced)}")

        if self.state.placed:
            print("\nPlaced:")
            for piece_id, placed in self.state.placed.items():
                print(f"  Piece {piece_id}: {len(placed.world_cells)} cells")

        if self.state.unplaced:
            print("\nUnplaced:")
            for piece_id in sorted(self.state.unplaced):
                piece = self.state.get_piece_def(piece_id)
                if piece:
                    print(f"  Piece {piece_id}: {len(piece.local_voxels)} cells")

        if self.state.is_complete():
            print("\n🎉 PUZZLE COMPLETE! 🎉")

    def show_piece(self, piece_id: str):
        """显示piece信息"""
        if not self.state:
            print("No puzzle loaded")
            return

        piece = self.state.get_piece_def(piece_id)
        if not piece:
            print(f"Piece {piece_id} not found")
            return

        print(f"\n=== Piece {piece_id} ===")
        print(f"Voxels: {len(piece.local_voxels)}")
        print("Coordinates (local, 0-based):")
        for v in piece.local_voxels:
            print(f"  {v.to_tuple()}")

        print(f"Unique rotations: {len(piece.rotation_signatures)}")

        if piece_id in self.state.placed:
            placed = self.state.placed[piece_id]
            print("\nCurrently placed at:")
            for v in placed.world_cells:
                print(f"  {v.to_tuple()}")

    def visualize_2d(self):
        """简单的2D可视化(俯视图)"""
        if not self.state:
            print("No puzzle loaded")
            return

        A, B, C = self.state.spec.box
        print("\n=== Top View (Z-layers) ===")

        for z in range(C, 0, -1):
            print(f"\nLayer z={z}:")
            for y in range(B, 0, -1):
                row = ""
                for x in range(1, A + 1):
                    key = Vec3(x, y, z).to_key()
                    if key in self.state.by_cell:
                        piece_id = self.state.by_cell[key]
                        row += f"[{piece_id}]"
                    else:
                        row += " . "
                print(f"  {row}")

    def place_piece_interactive(self, piece_id: str):
        """交互式放置piece"""
        if not self.state:
            print("No puzzle loaded")
            return

        piece = self.state.get_piece_def(piece_id)
        if not piece:
            print(f"Piece {piece_id} not found")
            return

        print(f"\nPlacing piece {piece_id} ({len(piece.local_voxels)} cells)")
        print("Enter target cells (format: x,y,z), one per line")
        print("Enter empty line when done:")

        target_cells = []
        while len(target_cells) < len(piece.local_voxels):
            try:
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
            self.visualize_2d()
        else:
            print(f"✗ {result.error.value}: {result.message}")

    def run_cli(self):
        """运行CLI主循环"""
        print("=== 3D Polycube Stacking Game ===")
        print("Type 'help' for commands")

        while True:
            try:
                cmd = input("\n> ").strip().lower()

                if not cmd:
                    continue

                parts = cmd.split()
                command = parts[0]

                if command == "help":
                    self.show_help()

                elif command == "list":
                    self.list_puzzles()

                elif command == "load":
                    if len(parts) < 3:
                        print("Usage: load <size> <puzzle_id>")
                        print("Example: load 2x2x2 puzzle_001")
                    else:
                        self.load_puzzle(parts[1], parts[2])

                elif command == "state":
                    self.show_state()

                elif command == "view":
                    self.visualize_2d()

                elif command == "piece":
                    if len(parts) < 2:
                        print("Usage: piece <id>")
                    else:
                        self.show_piece(parts[1])

                elif command == "place":
                    if len(parts) < 2:
                        print("Usage: place <piece_id>")
                    else:
                        self.place_piece_interactive(parts[1])

                elif command == "pickup":
                    if len(parts) < 2:
                        print("Usage: pickup <piece_id>")
                    else:
                        result = pickup_piece(self.state, parts[1])
                        if result.success:
                            print(f"✓ {result.message}")
                            self.visualize_2d()
                        else:
                            print(f"✗ {result.error.value}: {result.message}")

                elif command == "quit" or command == "exit":
                    print("Goodbye!")
                    break

                else:
                    print(f"Unknown command: {command}")
                    print("Type 'help' for commands")

            except KeyboardInterrupt:
                print("\nUse 'quit' to exit")
            except Exception as e:
                print(f"Error: {e}")

    def show_help(self):
        """显示帮助信息"""
        print("""
Available commands:
  help                    - Show this help
  list                    - List available puzzles
  load <size> <id>        - Load a puzzle (e.g., load 2x2x2 puzzle_001)
  state                   - Show current game state
  view                    - Show 2D visualization
  piece <id>              - Show piece information
  place <id>              - Place a piece interactively
  pickup <id>             - Pick up a placed piece
  quit/exit               - Exit the game
        """)

    def list_puzzles(self):
        """列出可用的puzzle"""
        puzzles = find_all_puzzles(self.puzzles_dir)
        print(f"\nFound {len(puzzles)} puzzles:")

        # 按尺寸分组
        by_size = {}
        for name, _ in puzzles:
            size = name.split('/')[0]
            if size not in by_size:
                by_size[size] = []
            by_size[size].append(name)

        for size in sorted(by_size.keys()):
            print(f"\n{size}: ({len(by_size[size])} puzzles)")
            for name in by_size[size][:5]:  # 显示前5个
                puzzle_id = name.split('/')[1]
                print(f"  - {puzzle_id}")
            if len(by_size[size]) > 5:
                print(f"  ... and {len(by_size[size]) - 5} more")


def main():
    """主函数"""
    puzzles_dir = "/mnt/moonfs/wuyuhao-m2/wyh/VisualReasonBench/Stacking_scaling/puzzles_full_v9"

    # 检查命令行参数
    if len(sys.argv) > 1:
        puzzles_dir = sys.argv[1]

    game = StackingGame(puzzles_dir)
    game.run_cli()


if __name__ == "__main__":
    main()
