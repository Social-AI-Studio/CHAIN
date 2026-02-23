#!/usr/bin/env python3
"""Run stacking_game parallel tests via stacking_runner."""

from __future__ import annotations

import argparse

from chainbench import stacking_runner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="stacking_game parallel test wrapper.")
    parser.add_argument("--config", help="Path to a single stacking_game YAML config")
    parser.add_argument("--configs", nargs="+", help="Multiple stacking_game YAML configs")
    parser.add_argument("--k", type=int, default=1, help="K for KxK parallel testing")
    parser.add_argument("--workers", type=int, default=None, help="Max parallel workers")
    parser.add_argument("--runs-per-config", type=int, default=1, help="Runs per config")
    parser.add_argument("--price-in", type=float, default=0.1, help="USD per 1K input tokens")
    parser.add_argument("--price-out", type=float, default=0.1, help="USD per 1K output tokens")
    parser.add_argument("--seed-base", type=int, default=None, help="Base seed for deterministic runs")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pricing = stacking_runner.Pricing(
        input_per_1k=args.price_in,
        output_per_1k=args.price_out,
    )

    if args.config:
        stacking_runner.run_parallel_kxk(
            config_path=args.config,
            k=max(1, args.k),
            pricing=pricing,
            max_workers=args.workers,
            seed_base=args.seed_base,
        )
        return 0

    if args.configs:
        stacking_runner.run_parallel_difficulty_tasks(
            config_paths=args.configs,
            pricing=pricing,
            max_workers=args.workers,
            runs_per_config=max(1, args.runs_per_config),
            seed_base=args.seed_base,
        )
        return 0

    print("Please provide --config or --configs.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


# python examples/stacking_game_test.py --config eval_configs/stacking_game_222.yaml --k 1 --workers 1

# python examples/stacking_game_test.py --configs eval_configs/stacking_game_222.yaml eval_configs/stacking_game_223.yaml --runs-per-config 2 --workers 2

# python examples/stacking_game_test.py --configs eval_configs/stacking_game_222.yaml eval_configs/stacking_game_223.yaml eval_configs/stacking_game_234.yaml eval_configs/stacking_game_244.yaml eval_configs/stacking_game_333.yaml eval_configs/stacking_game_334.yaml eval_configs/stacking_game_344.yaml eval_configs/stacking_game_444.yaml  --runs-per-config 1 --workers 8
# python examples/stacking_game_test.py --configs eval_configs/stacking_game_222_seed.yaml eval_configs/stacking_game_223_seed.yaml eval_configs/stacking_game_234_seed.yaml eval_configs/stacking_game_244_seed.yaml eval_configs/stacking_game_333_seed.yaml eval_configs/stacking_game_334_seed.yaml eval_configs/stacking_game_344_seed.yaml eval_configs/stacking_game_444_seed.yaml  --runs-per-config 1 --workers 8