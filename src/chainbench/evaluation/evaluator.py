"""
Main evaluator for chainbench benchmark system.
"""

from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
from chainbench.core.base import BaseEvaluator, Observation, TaskResult, EvaluationResult
from chainbench.evaluation.metrics import MetricsCalculator
from chainbench.evaluation.judge import LLMJudge
from chainbench.core import JudgementConfig


class BenchmarkEvaluator(BaseEvaluator):
    """Main evaluator for chainbench benchmarks."""
    
    def __init__(self, config: JudgementConfig):
        super().__init__(config)
        self.metrics_calculator = MetricsCalculator()
        self.judge = LLMJudge(config)
        
    def evaluate_metrics(self, task_results: List[TaskResult]) -> EvaluationResult:
        """Evaluate multiple task results and aggregate metrics."""
        # Then calculate comprehensive metrics with updated success status
        return self.metrics_calculator.calculate_comprehensive_metrics(task_results)
        
    def _evaluate_with_judge(self, task_result: TaskResult, task_success_criteria: str = None) -> Dict[str, float]:
        """Evaluate task result using LLM judge."""
        if not task_result.trajectory:
            return {}
            
        # Get final image from trajectory (support both old and new formats)
        final_image = None
        if task_result.trajectory:
            last_step = task_result.trajectory[-1]
            
            # Check if new format (dict) or old format (tuple)
            if isinstance(last_step, dict):
                # New format: {"response": str, "actions": List[Action], "observations": List[Observation]}
                observations = last_step.get("observations", [])
                if observations:
                    final_observation = observations[-1]
                    # observations are now Observation objects, can get image directly
                    final_image = getattr(final_observation, 'image', None)
            else:
                # Old format: (action, observation)
                final_observation = last_step[1]
                final_image = getattr(final_observation, 'image', None)
            
        if not final_image:
            return {}
            
        # Create task description - use custom criteria if provided
        if task_success_criteria:
            task_description = task_success_criteria
        else:
            task_description = f"Task: {task_result.task_type} puzzle"
            if task_result.metadata.get("description"):
                task_description += f". {task_result.metadata['description']}"
            
        # Create trajectory summary (support both formats)
        trajectory_summary = []
        for i, step_data in enumerate(task_result.trajectory):
            if isinstance(step_data, dict):
                # New format: actions and observations are objects
                actions = step_data.get("actions", [])
                observations = step_data.get("observations", [])
                for j, (action, obs) in enumerate(zip(actions, observations)):
                    # Handle Action and Observation objects
                    action_type = action.action_type if hasattr(action, 'action_type') else action.get("action_type", "unknown")
                    obs_desc = obs.description if hasattr(obs, 'description') else obs.get("description", "")
                    trajectory_summary.append(f"Step {i+1}.{j+1}: {action_type} - {obs_desc}")
            else:
                # Old format: (action, observation)
                action, observation = step_data
                trajectory_summary.append(f"Step {i+1}: {action.action_type} - {observation.description}")
            
        # Get judge evaluation
        judge_success, confidence, reasoning = self.judge.judge_success(
            final_image, task_description, trajectory_summary
        )
        
        return {
            "judge_success": judge_success,
            "judge_confidence": confidence,
            "judge_reasoning": reasoning
        }
        
    def export_results_to_excel(self, evaluation_result: EvaluationResult, output_path: str, model_name: str = "unknown") -> None:
        """Export evaluation results into two Excel files:
        1. Detailed results (one row per task)
        2. Difficulty-level statistics
        """

        # ========= 1. Detailed Results =========
        results_data = []
        for task_result in evaluation_result.task_results:
            optimal_steps = task_result.metadata.get("optimal_steps", 0)
            step_efficiency = (
                optimal_steps / task_result.total_steps
                if task_result.total_steps > 0 and optimal_steps > 0
                else 0
            )

            row_data = {
                "Model": model_name,
                "Task ID": task_result.task_id,
                "Task Type": task_result.task_type.replace("_", " ").title(),
                "Difficulty": task_result.metadata.get("difficulty", "Unknown").title(),
                "Success": 1 if task_result.success else 0,
                "Steps": task_result.total_steps,
                "Optimal Steps": optimal_steps,
                "Step Efficiency": step_efficiency,
                "Execution Time (s)": task_result.execution_time,
                "Token Usage": task_result.metadata.get("total_tokens", 0),
                "Total Tokens In": task_result.metadata.get("total_tokens_in", 0),
                "Total Tokens Out": task_result.metadata.get("total_tokens_out", 0),
                "Eval Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            results_data.append(row_data)

        df = pd.DataFrame(results_data)
        detailed_path = output_path.replace(".xlsx", "_Detailed.xlsx")
        df.to_excel(detailed_path, index=False)

        # ========= 2. Difficulty Statistics =========
        difficulty_stats = []
        difficulties = set(r.metadata.get("difficulty", "Unknown") for r in evaluation_result.task_results)

        for diff in difficulties:
            diff_tasks = [r for r in evaluation_result.task_results if r.metadata.get("difficulty", "Unknown") == diff]
            if diff_tasks:
                diff_success = sum(1 for r in diff_tasks if r.success)
                diff_success_rate = diff_success / len(diff_tasks)
                avg_steps = np.mean([r.total_steps for r in diff_tasks])
                avg_time = np.mean([r.execution_time for r in diff_tasks])

                difficulty_stats.append({
                    "Difficulty": diff.title(),
                    "Num Tasks": len(diff_tasks),
                    "Num Success": diff_success,
                    "Success Rate": f"{diff_success_rate:.1%}",
                    "Avg Steps": f"{avg_steps:.1f}",
                    "Avg Time (s)": f"{avg_time:.2f}",
                })

        if difficulty_stats:
            diff_df = pd.DataFrame(difficulty_stats)
            diff_path = output_path.replace(".xlsx", "_Difficulty.xlsx")
            diff_df.to_excel(diff_path, index=False)

        print(f"✅ Detailed results saved to {detailed_path}")
        print(f"✅ Difficulty stats saved to {diff_path}")
        
    def export_detailed_report(self, evaluation_result: EvaluationResult, 
                              output_dir: str, model_name: str = "unknown") -> None:
        """Export detailed evaluation report."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Main report
        report = {
            "model_name": model_name,
            "evaluation_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tasks": len(evaluation_result.task_results),
                "successful_tasks": sum(1 for r in evaluation_result.task_results if r.success),
                "accuracy": evaluation_result.accuracy,
                "pass_at_k": evaluation_result.pass_at_k,
                "distance_to_optimal": evaluation_result.distance_to_optimal,
                "token_efficiency": evaluation_result.token_efficiency
            },
            "detailed_metrics": evaluation_result.detailed_metrics,
            "task_breakdown": []
        }
        
        # Add individual task results
        for task_result in evaluation_result.task_results:
            task_data = {
                "task_id": task_result.task_id,
                "task_type": task_result.task_type,
                "success": task_result.success,
                "total_steps": task_result.total_steps,
                "execution_time": task_result.execution_time,
                "metadata": task_result.metadata,
                "trajectory_length": len(task_result.trajectory),
                "error_message": task_result.error_message
            }
            report["task_breakdown"].append(task_data)
            
        # Save main report
        if "/" in model_name:
            model_name = model_name.split("/")[-1]
        # Ensure output directory exists (including subdirectories from model_name)
        report_path = os.path.join(output_dir, f"{model_name}_evaluation_report.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        # Save trajectory details separately
        trajectories_path = os.path.join(output_dir, f"{model_name}_trajectories.json")
        os.makedirs(os.path.dirname(trajectories_path), exist_ok=True)
        trajectories_data = {}
        
        for task_result in evaluation_result.task_results:
            trajectory = []
            
            # Support both old and new trajectory formats
            for step_data in task_result.trajectory:
                if isinstance(step_data, dict):
                    # New format: dict with Action and Observation objects
                    step_dict = {
                        "response": step_data.get("response", ""),
                        "actions": [],
                        "observations": []
                    }
                    
                    # Serialize actions
                    for action in step_data.get("actions", []):
                        step_dict["actions"].append(action.to_dict() if hasattr(action, 'to_dict') else action)
                    
                    # Serialize observations
                    for obs in step_data.get("observations", []):
                        step_dict["observations"].append(obs.to_dict() if hasattr(obs, 'to_dict') else obs)
                    
                    trajectory.append(step_dict)
                else:
                    # Old format: (action, observation) tuple
                    action, observation = step_data
                    trajectory.append({
                        "action": action.to_dict() if hasattr(action, 'to_dict') else action,
                        "observation": observation.to_dict() if hasattr(observation, 'to_dict') else observation,
                    })
            
            trajectories_data[task_result.task_id] = trajectory
            
        with open(trajectories_path, 'w') as f:
            json.dump(trajectories_data, f, indent=2, default=str)
            
        print(f"Detailed report exported to {output_dir}")
        
    def compare_models(self, evaluation_results: Dict[str, EvaluationResult]) -> Dict[str, Any]:
        """Compare evaluation results across multiple models."""
        comparison = {
            "models": list(evaluation_results.keys()),
            "metrics_comparison": {},
            "rankings": {},
            "statistical_significance": {}
        }
        
        # Compare key metrics
        metrics_to_compare = ["accuracy", "distance_to_optimal", "token_efficiency"]
        
        for metric in metrics_to_compare:
            comparison["metrics_comparison"][metric] = {}
            
            for model_name, result in evaluation_results.items():
                if metric == "accuracy":
                    value = result.accuracy
                elif metric == "distance_to_optimal":
                    value = result.distance_to_optimal
                elif metric == "token_efficiency":
                    value = result.token_efficiency
                else:
                    continue
                    
                comparison["metrics_comparison"][metric][model_name] = value
                
        # Create rankings
        for metric in metrics_to_compare:
            metric_values = comparison["metrics_comparison"][metric]
            
            # Sort by metric value (higher is better for accuracy, lower is better for others)
            if metric == "accuracy":
                sorted_models = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
            else:
                # For distance_to_optimal and token_efficiency, lower is better
                sorted_models = sorted(metric_values.items(), key=lambda x: x[1])
                
            comparison["rankings"][metric] = [model for model, _ in sorted_models]
            
        return comparison
