"""
Metrics calculation for chainbench benchmark evaluation.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import math

from chainbench.core.base import TaskResult, EvaluationResult


class MetricsCalculator:
    """Calculator for benchmark evaluation metrics."""
    
    def __init__(self):
        self.supported_metrics = [
            "accuracy",
            "pass_at_k", 
            "avg_at_k",
            "distance_to_optimal",
            "token_efficiency",
            "tokens_per_step",
            "cost_total",
            "usd_per_solved",
            "step_efficiency",
            "time_efficiency",
            "success_rate_by_difficulty"
        ]
        
    def calculate_accuracy(self, task_results: List[TaskResult]) -> float:
        """
        Calculate overall accuracy (success rate).
        
        Args:
            task_results: List of task execution results
            
        Returns:
            Accuracy as float between 0 and 1
        """
        if not task_results:
            return 0.0
            
        successful = sum(1 for result in task_results if result.success)
        return successful / len(task_results)
        
    def calculate_pass_at_k(self, task_results: List[TaskResult], k_values: List[int] = None) -> Dict[int, float]:
        """
        Calculate pass@k metrics directly over all task results (no grouping).
        
        Args:
            task_results: List of task execution results
            k_values: List of k values to calculate (default: [1, 3, 5])
            
        Returns:
            Dictionary mapping k to pass@k rate
        """
        if k_values is None:
            k_values = [1, 3, 5]
            
        if not task_results:
            return {k: 0.0 for k in k_values}

        accuracy = self.calculate_accuracy(task_results)
        return {k: accuracy for k in k_values}

    def calculate_avg_at_k(self, task_results: List[TaskResult], k_values: List[int] = None) -> Dict[int, float]:
        """
        Calculate Avg@k metrics directly over all task results (no grouping).

        Args:
            task_results: List of task execution results
            k_values: List of k values to calculate (default: [1, 3, 5])

        Returns:
            Dictionary mapping k to Avg@k rate
        """
        if k_values is None:
            k_values = [1, 3, 5]

        if not task_results:
            return {k: 0.0 for k in k_values}

        accuracy = self.calculate_accuracy(task_results)
        return {k: accuracy for k in k_values}
        
    def calculate_distance_to_optimal(self, task_results: List[TaskResult]) -> float:
        """
        Calculate average distance from optimal number of steps.
        
        Args:
            task_results: List of task execution results
            
        Returns:
            Average normalized distance to optimal steps
        """
        if not task_results:
            return float('inf')
            
        distances = []
        
        for result in task_results:
            if result.success:
                optimal_steps = result.metadata.get("optimal_steps", result.total_steps)
                if optimal_steps > 0:
                    # Normalized distance: (actual - optimal) / optimal
                    distance = max(0, result.total_steps - optimal_steps) / optimal_steps
                    distances.append(distance)
                elif optimal_steps == 0 and result.total_steps > 0:
                    # Handle edge case where optimal is 0 but actual > 0
                    distances.append(float('inf'))
                # If both optimal and actual are 0, distance is 0 (perfect)
                    
        return np.mean(distances) if distances else float('inf')
        
    def calculate_token_efficiency(self, task_results: List[TaskResult]) -> float:
        """
        Calculate token efficiency (tokens per successful task).
        
        Args:
            task_results: List of task execution results
            
        Returns:
            Average tokens per successful task
        """
        successful_results = [r for r in task_results if r.success]
        
        if not successful_results:
            return float('inf')
            
        token_counts = []
        
        for result in successful_results:
            tokens = result.metadata.get("total_tokens", 0)
            if tokens > 0:
                token_counts.append(tokens)
                
        return np.mean(token_counts) if token_counts else float('inf')

    def calculate_tokens_per_step(self, task_results: List[TaskResult]) -> float:
        """
        Calculate tokens per step across all tasks.

        Args:
            task_results: List of task execution results

        Returns:
            Average tokens per step
        """
        total_steps = sum(result.total_steps for result in task_results)
        if total_steps == 0:
            return float('inf')
        total_tokens = sum(result.metadata.get("total_tokens", 0) for result in task_results)
        return total_tokens / total_steps

    def calculate_cost(self, task_results: List[TaskResult], price_in: float = 0.0, price_out: float = 0.0) -> float:
        """
        Calculate total cost in USD.

        Args:
            task_results: List of task execution results
            price_in: USD per 1K input tokens
            price_out: USD per 1K output tokens

        Returns:
            Total cost in USD
        """
        total_cost = 0.0
        for result in task_results:
            # tokens_in = result.metadata.get("tokens_in")
            # tokens_out = result.metadata.get("tokens_out")
            # total_tokens = result.metadata.get("total_tokens", 0)
            tokens_in = result.metadata.get("total_tokens_in", 0)
            tokens_out = result.metadata.get("total_tokens_out", 0)
            total_cost += (price_in * tokens_in + price_out * tokens_out) / 1000.0
        return total_cost

    def calculate_usd_per_solved(self, task_results: List[TaskResult], price_in: float = 0.0, price_out: float = 0.0) -> float:
        """
        Calculate USD per solved task.

        Args:
            task_results: List of task execution results
            price_in: USD per 1K input tokens
            price_out: USD per 1K output tokens

        Returns:
            Average USD per solved task
        """
        solved = [r for r in task_results if r.success]
        if not solved:
            return float('inf')
        total_cost = self.calculate_cost(task_results, price_in, price_out)
        return total_cost / len(solved)
        
    def calculate_step_efficiency(self, task_results: List[TaskResult]) -> float:
        """
        Calculate step efficiency (steps per successful task).
        
        Args:
            task_results: List of task execution results
            
        Returns:
            Average steps per successful task
        """
        successful_results = [r for r in task_results if r.success]
        
        if not successful_results:
            return float('inf')
            
        steps = [result.total_steps for result in successful_results]
        return np.mean(steps) if steps else float('inf')
        
    def calculate_time_efficiency(self, task_results: List[TaskResult]) -> float:
        """
        Calculate time efficiency (seconds per successful task).
        
        Args:
            task_results: List of task execution results
            
        Returns:
            Average time per successful task
        """
        successful_results = [r for r in task_results if r.success]
        
        if not successful_results:
            return float('inf')
            
        times = [result.execution_time for result in successful_results]
        return np.mean(times) if times else float('inf')
        
    def calculate_success_rate_by_difficulty(self, task_results: List[TaskResult]) -> Dict[str, float]:
        """
        Calculate success rate broken down by difficulty level.
        
        Args:
            task_results: List of task execution results
            
        Returns:
            Dictionary mapping difficulty to success rate
        """
        difficulty_results = defaultdict(list)
        
        for result in task_results:
            difficulty = result.metadata.get("difficulty", "unknown")
            difficulty_results[difficulty].append(result)
            
        success_by_difficulty = {}
        
        for difficulty, results in difficulty_results.items():
            successful = sum(1 for r in results if r.success)
            success_by_difficulty[difficulty] = successful / len(results) if results else 0.0
            
        return success_by_difficulty
        
    def calculate_success_rate_by_task_type(self, task_results: List[TaskResult]) -> Dict[str, float]:
        """
        Calculate success rate broken down by task type.
        
        Args:
            task_results: List of task execution results
            
        Returns:
            Dictionary mapping task type to success rate
        """
        type_results = defaultdict(list)
        
        for result in task_results:
            type_results[result.task_type].append(result)
            
        success_by_type = {}
        
        for task_type, results in type_results.items():
            successful = sum(1 for r in results if r.success)
            success_by_type[task_type] = successful / len(results) if results else 0.0
            
        return success_by_type
        
    def calculate_trajectory_analysis(self, task_results: List[TaskResult]) -> Dict[str, Any]:
        """
        Analyze trajectory patterns and common failure modes.
        
        Args:
            task_results: List of task execution results
            
        Returns:
            Dictionary with trajectory analysis
        """
        analysis = {
            "avg_trajectory_length": 0,
            "common_first_actions": {},
            "common_failure_points": {},
            "action_sequence_analysis": {}
        }
        
        if not task_results:
            return analysis
            
        # Calculate average trajectory length
        trajectory_lengths = [len(result.trajectory) for result in task_results]
        analysis["avg_trajectory_length"] = np.mean(trajectory_lengths)
        
        # Analyze first actions (support both old and new formats)
        first_actions = defaultdict(int)
        for result in task_results:
            if result.trajectory:
                first_step = result.trajectory[0]
                
                if isinstance(first_step, dict):
                    # New format: {"response": str, "actions": List[Action], "observations": List[Observation]}
                    actions = first_step.get("actions", [])
                    if actions:
                        first_action = actions[0]
                        first_action_type = first_action.action_type if hasattr(first_action, 'action_type') else first_action.get("action_type", "unknown")
                        first_actions[first_action_type] += 1
                else:
                    # Old format: (action, observation)
                    first_action = first_step[0]
                    first_action_type = first_action.action_type if hasattr(first_action, 'action_type') else str(first_action.get("action_type", "unknown"))
                    first_actions[first_action_type] += 1
                
        total_results = len(task_results)
        analysis["common_first_actions"] = {
            action: count / total_results 
            for action, count in first_actions.items()
        }
        
        # Analyze failure points
        failed_results = [r for r in task_results if not r.success]
        failure_points = defaultdict(int)
        
        for result in failed_results:
            failure_step = len(result.trajectory)
            failure_points[f"step_{failure_step}"] += 1
            
        if failed_results:
            analysis["common_failure_points"] = {
                point: count / len(failed_results)
                for point, count in failure_points.items()
            }
            
        return analysis
        
    def calculate_comprehensive_metrics(
        self,
        task_results: List[TaskResult],
        price_in: float = 0.1,
        price_out: float = 0.1,
    ) -> EvaluationResult:
        """
        Calculate all metrics and return comprehensive evaluation result.
        
        Args:
            task_results: List of task execution results
            
        Returns:
            EvaluationResult with all calculated metrics
        """
        # Basic metrics
        accuracy = self.calculate_accuracy(task_results)
        pass_at_k = self.calculate_pass_at_k(task_results)
        avg_at_k = self.calculate_avg_at_k(task_results)
        distance_to_optimal = self.calculate_distance_to_optimal(task_results)
        token_efficiency = self.calculate_token_efficiency(task_results)
        tokens_per_step = self.calculate_tokens_per_step(task_results)
        cost_total = self.calculate_cost(task_results, price_in, price_out)
        usd_per_solved = self.calculate_usd_per_solved(task_results, price_in, price_out)
        
        # Detailed metrics
        detailed_metrics = {
            "avg_at_k": avg_at_k,
            "tokens_per_step": tokens_per_step,
            "cost_total": cost_total,
            "usd_per_solved": usd_per_solved,
            "step_efficiency": self.calculate_step_efficiency(task_results),
            "time_efficiency": self.calculate_time_efficiency(task_results),
            "success_by_difficulty": self.calculate_success_rate_by_difficulty(task_results),
            "success_by_task_type": self.calculate_success_rate_by_task_type(task_results),
            "trajectory_analysis": self.calculate_trajectory_analysis(task_results),
            "total_tasks": len(task_results),
            "successful_tasks": sum(1 for r in task_results if r.success),
            "failed_tasks": sum(1 for r in task_results if not r.success)
        }
        
        return EvaluationResult(
            accuracy=accuracy,
            pass_at_k=pass_at_k,
            distance_to_optimal=distance_to_optimal,
            token_efficiency=token_efficiency,
            detailed_metrics=detailed_metrics,
            task_results=task_results
        )
