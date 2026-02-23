"""
Core modules for the CHAIN benchmark (chainbench).

This package contains the fundamental components:
- Base classes for environments, tasks, and agents
- Configuration management
- Registry for component discovery
- Action parsing and execution
- Result aggregation and logging
"""

from chainbench.core.base import (
    BaseEnvironment,
    BaseTask,
    BaseAgent,
    BaseEvaluator,
    BaseJudge,
)

from chainbench.core.config import Config, load_config, create_default_config, validate_config, EnvironmentConfig, TaskConfig, AgentConfig, JudgementConfig, CameraConfig, StepSelectionConfig

from chainbench.core.registry import register_environment, register_environment_config, register_agent, register_task_config, register_task, ENVIRONMENT_REGISTRY, TASK_REGISTRY, AGENT_REGISTRY, TASK_CONFIG_REGISTRY

__all__ = [
    "BaseEnvironment",
    "BaseTask",
    "BaseAgent",
    "BaseEvaluator",
    "BaseJudge",
    "Config",
    "load_config",
    "create_default_config", 
    "validate_config",
    "EnvironmentConfig",
    "TaskConfig",
    "AgentConfig",
    "JudgementConfig",
    "StepSelectionConfig",
    "register_environment",
    "register_environment_config",
    "register_task_config",
    "register_task",
    "register_agent",
    "CameraConfig",
    "ENVIRONMENT_REGISTRY",
    "TASK_REGISTRY",
    "AGENT_REGISTRY",
    "TASK_CONFIG_REGISTRY"
]
