"""
Evaluation system for the CHAIN benchmark (chainbench).

This package contains evaluation components including:
- Metrics calculation (accuracy, pass@k, etc.)
- LLM-as-judge evaluation
- Result aggregation and analysis
"""

from chainbench.evaluation.metrics import MetricsCalculator
from chainbench.evaluation.evaluator import BenchmarkEvaluator
from chainbench.evaluation.judge import LLMJudge

__all__ = [
    "MetricsCalculator",
    "BenchmarkEvaluator", 
    "LLMJudge"
]
