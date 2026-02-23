"""Stacking-game benchmark runner with parallel evaluation and custom metrics."""

from __future__ import annotations

import argparse
import os
from queue import Queue
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional

from chainbench import load_config, validate_config
from chainbench.core.base import TaskResult
from chainbench.evaluation.metrics import MetricsCalculator
from chainbench.runner import BenchmarkRunner

    
    
LEVEL_SIZE_MAPPING = {
    "0": "2x2x2_easy_001",
    "1": "2x2x2_mid_001",
    "2": "2x2x3_easy_001",
    "3": "2x2x3_easy_002",
    "4": "2x2x3_easy_003",
    "5": "2x2x3_mid_001",
    "6": "2x2x3_mid_002",
    "7": "2x2x3_mid_003",
    "8": "2x2x4_easy_001",
    "9": "2x2x4_easy_002",
    "10": "2x2x4_easy_003",
    "11": "2x2x4_mid_001",
    "12": "2x2x4_mid_002",
    "13": "2x3x3_easy_001",
    "14": "2x3x3_easy_002",
    "15": "2x3x3_mid_001",
    "16": "2x3x3_mid_002",
    "17": "2x3x3_mid_003",
    "18": "2x3x3_mid_004",
    "19": "2x3x3_hard_001",
    "20": "2x3x3_hard_002",
    "21": "2x3x3_hard_003",
    "22": "2x3x3_hard_004",
    "23": "2x3x4_easy_001",
    "24": "2x3x4_mid_001",
    "25": "2x3x4_mid_002",
    "26": "2x3x4_mid_003",
    "27": "2x3x4_hard_001",
    "28": "2x3x4_hard_002",
    "29": "2x3x4_hard_003",
    "30": "2x3x4_hard_004",
    "31": "2x3x4_hard_005",
    "32": "2x4x4_hard_001",
    "33": "2x4x4_hard_002",
    "34": "2x4x4_hard_003",
    "35": "2x4x4_hard_004",
    "36": "2x4x4_hard_005",
    "37": "2x4x4_hard_006",
    "38": "2x4x4_hard_007",
    "39": "2x4x4_hard_008",
    "40": "3x3x3_mid_001",
    "41": "3x3x3_mid_002",
    "42": "3x3x3_mid_003",
    "43": "3x3x3_mid_004",
    "44": "3x3x3_mid_005",
    "45": "3x3x3_hard_001",
    "46": "3x3x3_hard_002",
    "47": "3x3x3_hard_003",
    "48": "3x3x3_hard_004",
    "49": "3x3x3_hard_005",
    "50": "3x3x4_mid_001",
    "51": "3x3x4_mid_002",
    "52": "3x3x4_hard_001",
    "53": "3x3x4_hard_002",
    "54": "3x3x4_hard_003",
    "55": "3x3x4_hard_004",
    "56": "3x3x4_hard_005",
    "57": "3x4x4_hard_001",
    "58": "3x4x4_hard_002",
    "59": "3x4x4_hard_003",
    "60": "3x4x4_hard_004",
    "61": "3x4x4_hard_005",
    "62": "3x4x4_hard_006",
    "63": "3x4x4_hard_007",
    "64": "3x4x4_hard_008",
    "65": "3x4x4_hard_009",
    "66": "3x4x4_hard_010",
    "67": "4x4x4_hard_001",
    "68": "4x4x4_hard_002",
    "69": "4x4x4_hard_003",
    "70": "4x4x4_hard_004",
    "71": "4x4x4_hard_005",
    "72": "4x4x4_hard_006",
    "73": "4x4x4_hard_007",
    "74": "4x4x4_hard_008",
    "75": "4x4x4_hard_009",
    "76": "4x4x4_hard_010",
}

@dataclass
class Pricing:
    """Token pricing in USD per 1K tokens."""

    input_per_1k: float = 0.0
    output_per_1k: float = 0.0


@dataclass
class MetricSummary:
    accuracy: float
    pass_at_k: Dict[int, float]
    avg_at_k: Dict[int, float]
    avg_steps_solved: float
    distance_to_optimal: float
    tokens_per_solved: float
    tokens_per_step: float
    cost_total: float
    usd_per_solved: float
    total_tasks: int
    solved_tasks: int


def _clone_config_with_suffix(config_path: str, suffix: str):
    config = load_config(config_path)
    issues = validate_config(config)
    errors = [i for i in issues if i.startswith("ERROR")]
    if errors:
        raise RuntimeError(f"Config validation failed: {errors}")
    config.runner.experiment_name = f"{config.runner.experiment_name}_{suffix}"
    return config


def _parse_level_mapping(level_index: int) -> tuple[str, str, str]:
    level_key = str(level_index)
    mapped = LEVEL_SIZE_MAPPING.get(level_key)
    if not mapped:
        raise ValueError(f"Unknown level index {level_index}. Known levels: {sorted(LEVEL_SIZE_MAPPING.keys())}")
    size, puzzle_id = mapped.split("_", 1)
    if not puzzle_id.startswith("puzzle_"):
        puzzle_id = f"puzzle_{puzzle_id}"
    return size, puzzle_id, mapped


def _apply_seed_overrides(config, seed: Optional[int]) -> None:
    if seed is None:
        return
    if hasattr(config.task, "init_seed"):
        config.task.init_seed = seed
    if hasattr(config.environment, "init_seed"):
        config.environment.init_seed = seed


def _apply_stacking_overrides(
    config,
    level_index: int,
    env_id: int,
    seed: Optional[int],
) -> None:
    if hasattr(config.environment, "env_id"):
        config.environment.env_id = env_id
    size, puzzle_id, _ = _parse_level_mapping(level_index)
    if hasattr(config.task, "puzzle_size"):
        config.task.puzzle_size = size
    if hasattr(config.task, "puzzle_id"):
        config.task.puzzle_id = puzzle_id
    if hasattr(config.environment, "default_size"):
        config.environment.default_size = size
    if hasattr(config.environment, "default_puzzle_id"):
        config.environment.default_puzzle_id = puzzle_id
    _apply_seed_overrides(config, seed)


def _run_single_task(
    config_path: str,
    batch_id: int,
    run_id: int,
    level_index: Optional[int] = None,
    env_id: int = 1,
    seed: Optional[int] = None,
    level_log_dir: Optional[str] = None,
) -> TaskResult:
    level_tag = f"l{level_index}" if level_index is not None else "lna"
    suffix = f"b{batch_id}_r{run_id}_{level_tag}_e{env_id}_{int(time.time() * 1000)}"
    config = _clone_config_with_suffix(config_path, suffix)
    if level_index is not None:
        _apply_stacking_overrides(config, level_index=level_index, env_id=env_id, seed=seed)
    else:
        _apply_seed_overrides(config, seed)

    if level_log_dir and level_index is not None:
        _, _, level_name = _parse_level_mapping(level_index)
        level_base_name = f"level_{level_index}_{level_name}"
        config.runner.experiment_name = f"{level_base_name}_run_{run_id}"
        config.runner.log_dir = level_log_dir
    else:
        pass

    
    runner = BenchmarkRunner(config)
    runner.setup()
    result = runner.run_single_task()
    evaluation = runner.evaluate([result])
    evaluated = evaluation.task_results[0]
    evaluated.metadata["group_key"] = f"level_{level_index}" if level_index is not None else f"batch_{batch_id}"
    if level_index is not None:
        evaluated.metadata["level_index"] = level_index
    evaluated.metadata["env_id"] = env_id
    evaluated.metadata["config_path"] = config_path
    
    runner.evaluator.export_results_to_excel(
        evaluation,
        os.path.join(runner.logger.run_dir, config.runner.results_excel_path),
        config.agent.model_name
    )
    runner.evaluator.export_detailed_report(
        evaluation,
        os.path.join(runner.logger.run_dir, "detailed_reports"),
        config.agent.model_name
    )
    return evaluated


def _group_results(task_results: Iterable[TaskResult]) -> Dict[str, List[TaskResult]]:
    grouped: Dict[str, List[TaskResult]] = {}
    for result in task_results:
        group_key = result.metadata.get("group_key", result.task_id)
        grouped.setdefault(group_key, []).append(result)
    return grouped


def _calculate_metrics(
    task_results: List[TaskResult],
    pricing: Pricing,
    k_values: Optional[List[int]] = None,
) -> MetricSummary:
    if k_values is None:
        k_values = [1, 3, 5]

    calculator = MetricsCalculator()
    total_tasks = len(task_results)
    solved_results = [r for r in task_results if r.success]
    solved_tasks = len(solved_results)
    accuracy = calculator.calculate_accuracy(task_results)
    pass_at_k = calculator.calculate_pass_at_k(task_results, k_values)
    avg_at_k = calculator.calculate_avg_at_k(task_results, k_values)
    avg_steps_solved = calculator.calculate_step_efficiency(task_results)
    distance_to_optimal = calculator.calculate_distance_to_optimal(task_results)
    tokens_per_solved = calculator.calculate_token_efficiency(task_results)
    tokens_per_step = calculator.calculate_tokens_per_step(task_results)
    total_cost = calculator.calculate_cost(
        task_results,
        price_in=pricing.input_per_1k,
        price_out=pricing.output_per_1k,
    )
    usd_per_solved = calculator.calculate_usd_per_solved(
        task_results,
        price_in=pricing.input_per_1k,
        price_out=pricing.output_per_1k,
    )

    return MetricSummary(
        accuracy=accuracy,
        pass_at_k=pass_at_k,
        avg_at_k=avg_at_k,
        avg_steps_solved=avg_steps_solved,
        distance_to_optimal=distance_to_optimal,
        tokens_per_solved=tokens_per_solved,
        tokens_per_step=tokens_per_step,
        cost_total=total_cost,
        usd_per_solved=usd_per_solved,
        total_tasks=total_tasks,
        solved_tasks=solved_tasks,
    )


def _print_metrics(title: str, summary: MetricSummary) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("-" * 80)
    print(f"Tasks: {summary.total_tasks}, Solved: {summary.solved_tasks}, Accuracy: {summary.accuracy:.2%}")
    if summary.pass_at_k:
        pass_k = ", ".join([f"Pass@{k}: {v:.2%}" for k, v in summary.pass_at_k.items()])
        print(pass_k)
    if summary.avg_at_k:
        avg_k = ", ".join([f"Avg@{k}: {v:.2%}" for k, v in summary.avg_at_k.items()])
        print(avg_k)
    print(f"AvgSteps (solved): {summary.avg_steps_solved:.2f}")
    print(f"Distance-to-Optimal: {summary.distance_to_optimal:.2f}")
    print(f"Tokens/Solved: {summary.tokens_per_solved:.2f}")
    print(f"Tokens/Step: {summary.tokens_per_step:.2f}")
    print(f"Total Cost (USD): {summary.cost_total:.4f}")
    print(f"USD/Solved: {summary.usd_per_solved:.4f}")
    print("=" * 80)

def _metrics_to_dict(summary: MetricSummary) -> Dict[str, Any]:
    """Convert MetricSummary to a JSON-serializable dict.
    
    Replaces Infinity values with None (which becomes null in JSON).
    """
    import math
    result = asdict(summary)
    
    # Recursively replace infinity values with None
    def replace_inf(obj):
        if isinstance(obj, float):
            if math.isinf(obj):
                return None
            return obj
        elif isinstance(obj, dict):
            return {k: replace_inf(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [replace_inf(item) for item in obj]
        else:
            return obj
    
    return replace_inf(result)

def _save_metrics_to_file(summary: MetricSummary, file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(_metrics_to_dict(summary), f,indent=2,ensure_ascii=False)

def run_parallel_kxk(
    config_path: str,
    k: int,
    pricing: Pricing,
    max_workers: Optional[int] = None,
    seed_base: Optional[int] = None,
) -> List[TaskResult]:
    os.environ.setdefault("MPLBACKEND", "Agg")
    all_results: List[TaskResult] = []
    workers = max_workers or k
    for batch_id in range(1, k + 1):
        print(f"\nStarting batch {batch_id}/{k} with {k} parallel runs...")
        batch_results: List[TaskResult] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for run_id in range(1, k + 1):
                seed = seed_base + (batch_id * 1000 + run_id) if seed_base is not None else None
                futures.append(
                    executor.submit(
                        _run_single_task,
                        config_path=config_path,
                        batch_id=batch_id,
                        run_id=run_id,
                        seed=seed,
                    )
                )
            for future in as_completed(futures):
                batch_results.append(future.result())

        batch_results.sort(key=lambda r: r.task_id)
        all_results.extend(batch_results)
        summary = _calculate_metrics(batch_results, pricing, k_values=[k])
        _print_metrics(f"Batch {batch_id} Metrics (K={k})", summary)

    overall = _calculate_metrics(all_results, pricing, k_values=[k])
    _print_metrics(f"Overall Metrics (KxK, K={k})", overall)
    return all_results


def run_parallel_difficulty_tasks(
    config_paths: List[str],
    pricing: Pricing,
    max_workers: Optional[int] = None,
    runs_per_config: int = 1,
    seed_base: Optional[int] = None,
) -> List[TaskResult]:
    os.environ.setdefault("MPLBACKEND", "Agg")
    workers = max_workers or len(config_paths)
    all_results: List[TaskResult] = []
    print(f"\nStarting parallel runs for {len(config_paths)} configs...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for config_index, config_path in enumerate(config_paths):
            for run_id in range(1, runs_per_config + 1):
                seed = seed_base + (config_index * 1000 + run_id) if seed_base is not None else None
                futures.append(
                    executor.submit(
                        _run_single_task,
                        config_path=config_path,
                        batch_id=config_index + 1,
                        run_id=run_id,
                        seed=seed,
                    )
                )
        for future in as_completed(futures):
            all_results.append(future.result())

    summary = _calculate_metrics(all_results, pricing, k_values=[runs_per_config])
    _print_metrics("Overall Metrics (Different Difficulties)", summary)

    difficulty_groups: Dict[str, List[TaskResult]] = {}
    for result in all_results:
        difficulty = result.metadata.get("difficulty", "unknown")
        difficulty_groups.setdefault(difficulty, []).append(result)
    for difficulty, results in difficulty_groups.items():
        diff_summary = _calculate_metrics(results, pricing, k_values=[runs_per_config])
        _print_metrics(f"Difficulty '{difficulty}' Metrics", diff_summary)

    return all_results

def _build_level_list(levels: Optional[List[int]], level_range: Optional[str], default_level: int) -> List[int]:
    if levels:
        return levels
    if level_range:
        start_str, end_str = level_range.split("-", 1)
        start = int(start_str)
        end = int(end_str)
        if end < start:
            raise ValueError("level-range must be in ascending order, e.g., 0-31")
        return list(range(start, end + 1))
    return [default_level]
def run_parallel_levels(
    config_path: str,
    levels: Optional[List[int]],
    level_range: Optional[str],
    runs_per_level: int,
    pricing: Pricing,
    max_workers: Optional[int] = None,
    seed_base: Optional[int] = None,
) -> List[TaskResult]:

    import os
    from datetime import datetime

    config = load_config(config_path)
    level_list = _build_level_list(levels, level_range, getattr(config.task, "level_index", 0))
    workers = min(max_workers or min(8, len(level_list) * runs_per_level), 8)
    env_pool: Queue[int] = Queue()
    for env_id in range(1, workers + 1):
        env_pool.put(env_id)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = os.path.join(config.runner.log_dir, f"stacking_{timestamp}")
    os.makedirs(base_log_dir, exist_ok=True)
    level_log_dirs: Dict[int, str] = {}
    for level_index in level_list:
        _, _, level_name = _parse_level_mapping(level_index)
        level_log_dir = os.path.join(base_log_dir, f"level_{level_index}_{level_name}")
        os.makedirs(level_log_dir, exist_ok=True)
        level_log_dirs[level_index] = level_log_dir
        print("Created log directory for level", level_index, level_name, "at", level_log_dir)

    all_results: List[TaskResult] = []
    futures = []
    print(f"\nStarting parallel runs for {len(level_list)} levels with up to {workers} workers...")

    def _run_with_env(batch_id: int, run_id: int, level_index: int, seed: Optional[int]) -> TaskResult:
        env_id = env_pool.get()
        try:
            level_log_dir = level_log_dirs.get(level_index)
            return _run_single_task(
                config_path=config_path,
                batch_id=batch_id,
                run_id=run_id,
                level_index=level_index,
                level_log_dir=level_log_dir,
                env_id=env_id,
                seed=seed,
            )
        finally:
            env_pool.put(env_id)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        batch_id = 1
        for level_index in level_list:
            for run_id in range(1, runs_per_level + 1):
                seed = seed_base + (level_index * 1000 + run_id) if seed_base is not None else None
                futures.append(
                    executor.submit(_run_with_env, batch_id, run_id, level_index, seed)
                )
        for future in as_completed(futures):
            all_results.append(future.result())

    from chainbench.evaluation.metrics import MetricsCalculator
    calculator = MetricsCalculator()

    for level_index in level_list:
        level_results = [r for r in all_results if r.metadata.get("level_index") == level_index]
        if level_results:
            level_log_dir = level_log_dirs[level_index]
            level_summary = _calculate_metrics(level_results, pricing, k_values=[runs_per_level])
           
            # Save level-specific Excel report
            from chainbench.evaluation.evaluator import BenchmarkEvaluator
            evaluator = BenchmarkEvaluator(config.judgement)
            evaluation_result = evaluator.evaluate_metrics(level_results)
           
            level_excel_path = os.path.join(level_log_dir, f"level_{level_index}_results.xlsx")
            evaluator.export_results_to_excel(
                evaluation_result,
                level_excel_path,
                config.agent.model_name
            )
           
            level_report_dir = os.path.join(level_log_dir, "detailed_reports")
            evaluator.export_detailed_report(
                evaluation_result,
                level_report_dir,
                config.agent.model_name
            )
           
            print(f"\n📊 Level {level_index} Summary:")
            _print_metrics(f"Level {level_index} Metrics", level_summary)
            level_metrics_path = os.path.join(level_log_dir, f"level_{level_index}_metrics.json")
            _save_metrics_to_file(level_summary, level_metrics_path)


    # Overall summary
    summary = _calculate_metrics(all_results, pricing, k_values=[runs_per_level])
    _print_metrics("Overall Metrics (Luban Levels)", summary)
   
    # Save overall summary Excel
    from chainbench.evaluation.evaluator import BenchmarkEvaluator
    evaluator = BenchmarkEvaluator(config.judgement)
    overall_evaluation = evaluator.evaluate_metrics(all_results)
    overall_excel_path = os.path.join(base_log_dir, "overall_results.xlsx")
    evaluator.export_results_to_excel(
        overall_evaluation,
        overall_excel_path,
        config.agent.model_name
    )
    overall_report_dir = os.path.join(base_log_dir, "detailed_reports")
    evaluator.export_detailed_report(
        overall_evaluation,
        overall_report_dir,
        config.agent.model_name
    )
    overall_metrics_path = os.path.join(base_log_dir, "overall_metrics.json")
    _save_metrics_to_file(summary, overall_metrics_path)


    print(f"\n📁 All logs saved to: {base_log_dir}")
    print(f"   - Each level has its own directory: level_0/, level_1/, etc.")
    print(f"   - Per-level metrics JSON: level_<N>/level_<N>_metrics.json")
    print(f"   - Overall summary: {overall_excel_path}")
    print(f"   - Overall metrics JSON: {overall_metrics_path}")
   


    return all_results

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parallel runner for stacking_game tasks.")
    parser.add_argument("--config", help="Path to a single stacking_game YAML config.")
    # parser.add_argument(
    #     "--configs",
    #     nargs="+",
    #     help="List of stacking_game YAML configs (different difficulties).",
    # )
    # parser.add_argument("--k", type=int, default=1, help="K for KxK parallel testing.")
    parser.add_argument("--levels", nargs="+", type=int, help="Level indices to run (0-n).")
    parser.add_argument("--level-range", help="Inclusive level range, e.g., 0-7.")
    parser.add_argument("--workers", type=int, default=None, help="Max parallel workers.")
    parser.add_argument("--runs-per-level", type=int, default=1, help="Runs per config.")
    parser.add_argument("--price-in", type=float, default=0.1, help="USD per 1K input tokens.")
    parser.add_argument("--price-out", type=float, default=0.1, help="USD per 1K output tokens.")
    parser.add_argument("--seed-base", type=int, default=None, help="Base seed for deterministic runs.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    pricing = Pricing(input_per_1k=args.price_in, output_per_1k=args.price_out)

    # if args.config:
    #     run_parallel_kxk(
    #         config_path=args.config,
    #         k=max(1, args.k),
    #         pricing=pricing,
    #         max_workers=args.workers,
    #         seed_base=args.seed_base,
    #     )
    #     return 0
    run_parallel_levels(
        config_path=args.config,
        levels=args.levels,
        level_range=args.level_range,
        runs_per_level=max(1, args.runs_per_level),
        pricing=pricing,
        max_workers=args.workers,
        seed_base=args.seed_base,
    )

    # if args.configs:
    #     run_parallel_difficulty_tasks(
    #         config_paths=args.configs,
    #         pricing=pricing,
    #         max_workers=args.workers,
    #         runs_per_config=max(1, args.runs_per_config),
    #         seed_base=args.seed_base,
    #     )
    #     return 0

    # parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python -m chainbench.stacking_runner --config eval_configs/stacking.yaml --levels 0 1 --runs-per-level 1 --workers 2
