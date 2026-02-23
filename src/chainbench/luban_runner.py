"""Luban Lock benchmark runner with parallel level evaluation."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from queue import Queue
from typing import Dict, Iterable, List, Optional, Tuple

from chainbench import load_config, validate_config
from chainbench.core.base import TaskResult
from chainbench.evaluation.metrics import MetricsCalculator
from chainbench.runner import BenchmarkRunner

class LubanRunner(BenchmarkRunner):
    def _build_prompt_history(self) -> str:
        """Build prompt history from interaction history.
        
        Builds history in alternating format:
        - Action 1
        - Observation 1
        - Action 2
        - Observation 2
        - ...
        
        Includes ALL steps in the interaction history (no truncation).
        """
        user_prompt = self.task.get_user_prompt()
        
        # Add object mapping information so the agent knows which object_id corresponds to which color/object
        object_mapping = self.get_object_mapping()
        
        # Include ALL steps (skip only the initial reset step at idx 0)
        prompt_history = ""
        for idx, step_data in enumerate(self.interaction_history):
            if idx == 0:  # Skip initial reset step
                continue
                
            prompt_history += f"\n{'='*60}\nStep {idx}:\n"
            
            # Get actions and observations
            actions = step_data.get("actions", [])
            observations = step_data.get("observations", [])
            # print("length of actions: ", len(actions))
            # print("length of observations: ", len(observations))
            
            # If both are empty, show agent response (text-only response)
            if not actions and not observations:
                prompt_history += f"No actions and observations\n"
                # response = step_data.get("response", "")
                # if response:
                #     prompt_history += f"\n[Agent Response]: {response}\n"
            else:
                # Pair actions with observations and display them alternately
                max_count = max(len(actions), len(observations))
                
                for i in range(max_count):
                    # Display action if available
                    if i < len(actions):
                        action = actions[i]
                        # Handle both Action objects and dicts
                        if hasattr(action, 'action_type'):
                            action_type = action.action_type
                            parameters = action.parameters
                        else:
                            action_type = action.get("action_type", "unknown")
                            parameters = action.get("parameters", {})
                        prompt_history += f"\n[Action {i+1}]: {action_type}({parameters})\n"
                    
                    # Display observation if available
                    if i < len(observations):
                        obs = observations[i]
                        # Handle both Observation objects and dicts
                        if hasattr(obs, 'description'):
                            description = obs.description
                        else:
                            description = obs.get("description", "")
                        if description:
                            prompt_history += f"[Observation {i+1}]: {description}\n"
        
        # Include object mapping at the beginning so agent can reference it
        full_prompt = f"{user_prompt}\n\n{object_mapping}\n"
        
        if prompt_history:
            full_prompt += f"\nInteraction History:{prompt_history}\n"
            
        full_prompt += "\nNow, what's your next action?"
        
        return full_prompt
     
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


def _apply_luban_overrides(config, level_index: int, env_id: int, seed: Optional[int]) -> None:
    if hasattr(config.environment, "env_id"):
        config.environment.env_id = env_id
    if hasattr(config.environment, "level_index"):
        config.environment.level_index = level_index
    if hasattr(config.task, "level_index"):
        config.task.level_index = level_index
    if seed is not None:
        if hasattr(config.task, "init_seed"):
            config.task.init_seed = seed
        if hasattr(config.environment, "init_seed"):
            config.environment.init_seed = seed


def _run_single_task(
    config_path: str,
    batch_id: int,
    run_id: int,
    level_index: int,
    env_id: int,
    seed: Optional[int] = None,
    level_log_dir: Optional[str] = None,
) -> TaskResult:
    """
    Run a single task for a specific level.
    
    Args:
        config_path: Path to config file
        batch_id: Batch ID
        run_id: Run ID within the level
        level_index: Level index
        env_id: Environment ID
        seed: Optional seed
        level_log_dir: Optional base log directory for this level (for organizing logs by level)
    
    Returns:
        TaskResult with evaluation
    """
    suffix = f"b{batch_id}_r{run_id}_l{level_index}_e{env_id}_{int(time.time() * 1000)}"
    config = _clone_config_with_suffix(config_path, suffix)
    _apply_luban_overrides(config, level_index=level_index, env_id=env_id, seed=seed)
    
    # If level_log_dir is provided, organize logs by level
    if level_log_dir:
        # Create level-specific log directory structure
        import os
        level_base_name = f"level_{level_index}"
        config.runner.log_dir = level_log_dir
        config.runner.experiment_name = f"{level_base_name}_run_{run_id}"
    else:
        # Use default behavior (original experiment_name includes all info)
        pass
    
    runner = LubanRunner(config)
    runner.setup()
    result = runner.run_single_task()
    evaluation = runner.evaluate([result])
    evaluated = evaluation.task_results[0]
    evaluated.metadata["group_key"] = f"level_{level_index}"
    evaluated.metadata["config_path"] = config_path
    evaluated.metadata["env_id"] = env_id
    evaluated.metadata["level_index"] = level_index
    
    # Export results to Excel and detailed reports (same as run_benchmark)
    # This ensures luban_task has the same logging capabilities as stacking_task
    import os
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


def _metrics_to_dict(summary: MetricSummary) -> dict:
    """Convert MetricSummary to a JSON-serializable dict."""
    return asdict(summary)


def _save_metrics_to_file(summary: MetricSummary, path: str) -> None:
    """Persist final metrics to a JSON file."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_metrics_to_dict(summary), f, indent=2, ensure_ascii=False)


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

    # Create a base log directory for this parallel run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = os.path.join(config.runner.log_dir, f"luban_parallel_{timestamp}")
    os.makedirs(base_log_dir, exist_ok=True)
    
    # Create level-specific log directories
    level_log_dirs: Dict[int, str] = {}
    for level_index in level_list:
        level_log_dir = os.path.join(base_log_dir, f"level_{level_index}")
        os.makedirs(level_log_dir, exist_ok=True)
        level_log_dirs[level_index] = level_log_dir
        print(f"📁 Created log directory for level {level_index}: {level_log_dir}")

    all_results: List[TaskResult] = []
    futures = []
    print(f"\nStarting parallel runs for {len(level_list)} levels with up to {workers} workers...")
    print(f"📁 Base log directory: {base_log_dir}")

    def _run_with_env(batch_id: int, run_id: int, level_index: int, seed: Optional[int]) -> TaskResult:
        env_id = env_pool.get()
        try:
            level_log_dir = level_log_dirs.get(level_index)
            return _run_single_task(
                config_path=config_path,
                batch_id=batch_id,
                run_id=run_id,
                level_index=level_index,
                env_id=env_id,
                seed=seed,
                level_log_dir=level_log_dir,
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

    # Aggregate results by level and save level-specific summaries
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
    parser = argparse.ArgumentParser(description="Parallel runner for Luban Lock tasks.")
    parser.add_argument("--config", required=True, help="Path to a Luban YAML config.")
    parser.add_argument("--levels", nargs="+", type=int, help="Level indices to run (0-31).")
    parser.add_argument("--level-range", help="Inclusive level range, e.g., 0-7.")
    parser.add_argument("--runs-per-level", type=int, default=1, help="Runs per level.")
    parser.add_argument("--workers", type=int, default=None, help="Max parallel workers (<= 8).")
    parser.add_argument("--price-in", type=float, default=0.1, help="USD per 1K input tokens.")
    parser.add_argument("--price-out", type=float, default=0.1, help="USD per 1K output tokens.")
    parser.add_argument("--seed-base", type=int, default=None, help="Base seed for deterministic runs.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    pricing = Pricing(input_per_1k=args.price_in, output_per_1k=args.price_out)
    run_parallel_levels(
        config_path=args.config,
        levels=args.levels,
        level_range=args.level_range,
        runs_per_level=max(1, args.runs_per_level),
        pricing=pricing,
        max_workers=args.workers,
        seed_base=args.seed_base,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python -m chainbench.luban_runner --config eval_configs/luban.yaml --level-range 0-3 --runs-per-level 1 --price-in 0.0004 --price-out 0.001
# python -m chainbench.luban_runner --config eval_configs/luban.yaml --levels 1 --runs-per-level 1 --price-in 0.0004 --price-out 0.001