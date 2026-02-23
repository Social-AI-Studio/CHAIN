"""
Configuration management for the CHAIN benchmark (chainbench).

This module handles loading and validation of configuration files,
environment variables, and provides typed configuration objects.
"""

import os
import yaml
from typing import Any, Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from chainbench.core.base import TaskDifficulty
from chainbench.core.registry import ENVIRONMENT_CONFIG_REGISTRY, TASK_CONFIG_REGISTRY


@dataclass
class CameraConfig:
    """Camera configuration for environment rendering."""
    position: Tuple[float, float, float] = (0.0, -1.0, 1.0)
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    up_vector: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov: float = 60.0
    aspect_ratio: float = 1.0
    near_plane: float = 0.1
    far_plane: float = 10.0
    image_width: int = 512
    image_height: int = 512

    def __post_init__(self):
        if not isinstance(self.position, (tuple, list)) or len(self.position) != 3:
            raise ValueError("position must be a tuple of 3 floats")
        if not isinstance(self.target, (tuple, list)) or len(self.target) != 3:
            raise ValueError("target must be a tuple of 3 floats")
        if not isinstance(self.up_vector, (tuple, list)) or len(self.up_vector) != 3:
            raise ValueError("up_vector must be a tuple of 3 floats")
        if not isinstance(self.fov, (float, int)) or self.fov <= 0:
            raise ValueError("fov must be a positive number")
        if not isinstance(self.aspect_ratio, (float, int)) or self.aspect_ratio <= 0:
            raise ValueError("aspect_ratio must be a positive number")
        if not isinstance(self.near_plane, (float, int)) or self.near_plane <= 0:
            raise ValueError("near_plane must be a positive number")
        if not isinstance(self.far_plane, (float, int)) or self.far_plane <= 0:
            raise ValueError("far_plane must be a positive number")
        if not isinstance(self.image_width, int) or self.image_width <= 0:
            raise ValueError("image_width must be a positive integer")
        if not isinstance(self.image_height, int) or self.image_height <= 0:
            raise ValueError("image_height must be a positive integer")

@dataclass
class AgentConfig:
    """Configuration for VLM agents."""
    type: str
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 500
    max_content_size: int = 32000
    timeout: float = 30.0
    device: str = "auto"
    torch_dtype: str = "auto"
    max_retries: int = 5
    
    def __post_init__(self):
        # Load from environment variables if not provided
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                import warnings
                warnings.warn(
                    "No OPENAI_API_KEY found in environment variables or config. "
                    "Please set OPENAI_API_KEY or configure api_key in the YAML file."
                )
        if self.base_url is None:
            self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not isinstance(self.max_retries, int) or self.max_retries <= 0:
            raise ValueError("max_retries must be a positive integer")

@dataclass
class JudgementConfig:
    """Configuration for judgement (LLM-as-judge)."""
    type: str
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 50
    timeout: float = 30.0
    max_retries: int = 5

    def __post_init__(self):
        # Load from environment variables if not provided
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                import warnings
                warnings.warn(
                    "No OPENAI_API_KEY found in environment variables or config. "
                    "Please set OPENAI_API_KEY or configure api_key in the YAML file."
                )
        if self.base_url is None:
            self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not isinstance(self.max_retries, int) or self.max_retries <= 0:
            raise ValueError("max_retries must be a positive integer")


@dataclass
class StepSelectionConfig:
    """Configuration for Process Step Selection using Beam Search."""
    
    # Whether to enable step selection
    enabled: bool = False
    
    # Selection method: "reward_model" or "pairwise_judge"
    selection_method: str = "reward_model"
    
    # Number of candidate actions to generate per step (for each beam)
    num_candidates: int = 5
    
    # Beam width: number of top actions to keep (parallel paths to maintain)
    # - top_k = 1: Greedy search (single best path)
    # - top_k > 1: Beam search (multiple parallel paths)
    top_k: int = 1
    
    # Reward model configuration (used when selection_method="reward_model")
    reward_model_type: str = "openai"  # "openai" or "transformers"
    reward_model_name: str = "gpt-4o"
    reward_model_mode: str = "generative"  # "generative" or "discriminative"
    reward_api_key: Optional[str] = None
    reward_base_url: Optional[str] = None
    reward_temperature: float = 0.1
    reward_max_tokens: int = 200
    reward_device: str = "auto"
    reward_torch_dtype: str = "auto"
    
    # Pairwise judge configuration (used when selection_method="pairwise_judge")  
    pairwise_judge_type: str = "openai"  # "openai" or "transformers"
    pairwise_judge_model: str = "gpt-4o"
    pairwise_api_key: Optional[str] = None
    pairwise_base_url: Optional[str] = None
    pairwise_temperature: float = 0.1
    pairwise_max_tokens: int = 100
    pairwise_device: str = "auto"
    pairwise_torch_dtype: str = "auto"
    
    # Sorting method for pairwise comparison: "bubble", "merge", or "full"
    # "full" means comparing all pairs and computing total wins for global ranking
    pairwise_sort_method: str = "merge"
    
    # Agent generation settings for multiple candidates
    candidate_temperature: float = 1.0  # Higher temperature for diversity
    
    def __post_init__(self):
        # Validate selection method
        if self.selection_method not in ["reward_model", "pairwise_judge"]:
            raise ValueError(f"selection_method must be 'reward_model' or 'pairwise_judge', got '{self.selection_method}'")
        
        # Validate reward model mode
        if self.reward_model_mode not in ["generative", "discriminative"]:
            raise ValueError(f"reward_model_mode must be 'generative' or 'discriminative', got '{self.reward_model_mode}'")
        
        # Validate numeric parameters
        if not isinstance(self.num_candidates, int) or self.num_candidates < 1:
            raise ValueError("num_candidates must be a positive integer")
        if not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError("top_k must be a positive integer")
        
        # Validate top_k <= num_candidates
        if self.top_k > self.num_candidates:
            raise ValueError(
                f"top_k ({self.top_k}) cannot be greater than num_candidates ({self.num_candidates}). "
                f"top_k represents the beam width (number of parallel paths to maintain)."
            )
        
        # Validate pairwise sort method
        if self.pairwise_sort_method not in ["bubble", "merge", "full"]:
            raise ValueError(f"pairwise_sort_method must be 'bubble', 'merge', or 'full', got '{self.pairwise_sort_method}'")
        
        # Performance warnings
        if self.num_candidates > 10:
            import warnings
            warnings.warn(
                f"num_candidates={self.num_candidates} is very large. "
                f"This will result in many LLM calls per step."
            )
        
        if self.top_k > 5:
            import warnings
            warnings.warn(
                f"top_k={self.top_k} (beam width) is very large. "
                f"This may result in exponential growth of computation."
            )
        
        # Estimate and warn about computation cost
        estimated_calls_per_step = self.num_candidates * self.top_k
        if estimated_calls_per_step > 25:
            import warnings
            warnings.warn(
                f"Estimated {estimated_calls_per_step} LLM calls per step "
                f"(num_candidates × top_k). This may be very slow and expensive!"
            )
        
        # Load API keys from environment if not provided
        if self.reward_api_key is None:
            self.reward_api_key = os.getenv("OPENAI_API_KEY")
        if self.reward_base_url is None:
            self.reward_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if self.pairwise_api_key is None:
            self.pairwise_api_key = os.getenv("OPENAI_API_KEY")
        if self.pairwise_base_url is None:
            self.pairwise_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

@dataclass
class EnvironmentConfig:
    """Configuration for physics environment."""
    type: str
    gui: bool = False
    urdf_local_path: str = "assets/pybullet/phobos_models"
    table_position: Tuple[float, float, float] = (0, 0, 0.4)
    real_time: bool = False
    gravity: float = -9.81
    time_step: float = 1.0/240.0
    render_width: int = 512
    render_height: int = 512
    multi_view: bool = True
    load_table: bool = False
    max_steps: int = 50
    max_settle_steps: int = 2000
    lin_vel_tol: float = 1e-3
    ang_vel_tol: float = 1e-3
    
    # Collision softening parameters
    enable_soft_collision: bool = True
    soft_collision_damping: float = 0.8  # 更大阻尼快速消能
    soft_collision_erp: float = 0.05     # 更低 ERP 减少暴力修正
    contact_breaking_threshold: float = 0.001  # 1mm 内自动断开接触
    
    # Overlap resolution (gentle separation after move)
    enable_overlap_resolution: bool = True
    overlap_threshold: float = -0.0005  # 检测 >0.5mm 的穿透
    overlap_separation_step: float = 0.001  # 每次分离 1mm
    overlap_max_iterations: int = 10    # 最多尝试 10 次
    
    def __post_init__(self):
        # Convert to absolute path if relative
        if not isinstance(self.urdf_local_path, str) or not self.urdf_local_path:
            raise ValueError("urdf_local_path must be a non-empty string")
        if not os.path.isabs(self.urdf_local_path):
            self.urdf_local_path = os.path.abspath(self.urdf_local_path)
        if not isinstance(self.table_position, (tuple, list)) or len(self.table_position) != 3:
            raise ValueError("table_position must be a tuple of 3 floats")
        if not isinstance(self.time_step, (float, int)) or self.time_step <= 0:
            raise ValueError("time_step must be a positive number")
        if not isinstance(self.gravity, (float, int)):
            raise ValueError("gravity must be a number")
        if not isinstance(self.render_width, int) or self.render_width <= 0:
            raise ValueError("render_width must be a positive integer")
        if not isinstance(self.render_height, int) or self.render_height <= 0:
            raise ValueError("render_height must be a positive integer")
        if not isinstance(self.max_steps, int) or self.max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if not isinstance(self.max_settle_steps, int) or self.max_settle_steps <= 0:
            raise ValueError("max_settle_steps must be a positive integer")
        if not isinstance(self.lin_vel_tol, (float, int)) or self.lin_vel_tol < 0:
            raise ValueError("lin_vel_tol must be a non-negative number")
        if not isinstance(self.ang_vel_tol, (float, int)) or self.ang_vel_tol < 0:
            raise ValueError("ang_vel_tol must be a non-negative number")
    
    @classmethod
    def from_dict(cls, env_data: Dict[str, Any]) -> "EnvironmentConfig":
        config_type = env_data.get("type", "default")
        config_cls = ENVIRONMENT_CONFIG_REGISTRY.get(config_type, EnvironmentConfig)
        return config_cls(**env_data)

@dataclass  
class TaskConfig:
    """Configuration for specific tasks."""
    type: str
    name: str
    difficulty: Optional[TaskDifficulty] = None
    
    def __post_init__(self):
        if isinstance(self.difficulty, str):
            self.difficulty = TaskDifficulty(self.difficulty)

    @classmethod
    def from_dict(cls, task_data: Dict[str, Any]) -> "TaskConfig":
        config_type = task_data.get("type", "default")
        config_cls = TASK_CONFIG_REGISTRY.get(config_type, TaskConfig)
        return config_cls(**task_data)


@dataclass
class RunnerConfig:
    """Configuration for experiment runner."""
    experiment_name: str
    log_dir: str = "logs"
    results_excel_path: str = "experiment_results.xlsx"
    max_steps: int = 50
    history_length: int = 5
    
    def __post_init__(self):
        # Directory creation is deferred to runner.setup() to avoid side effects on import
        pass


@dataclass
class Config:
    """Main configuration object."""
    runner: RunnerConfig
    agent: AgentConfig
    judgement: Optional[JudgementConfig] = None
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    step_selection: Optional[StepSelectionConfig] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        runner_data = data.get("runner", {})
        runner = RunnerConfig(**runner_data)
        
        agent_data = data.get("agent", {})
        agent = AgentConfig(**agent_data)
        
        judgement_data = data.get("judgement", {})
        judgement = JudgementConfig(**judgement_data)

        task_data = data.get("task", {})
        task = TaskConfig.from_dict(task_data)
        
        env_data = data.get("environment", {})
        environment = EnvironmentConfig.from_dict(env_data)
        
        # Parse step selection config if present
        step_selection_data = data.get("step_selection", {})
        step_selection = StepSelectionConfig(**step_selection_data) if step_selection_data else None
        
        return cls(
            runner=runner,
            agent=agent,
            judgement=judgement,
            task=task,
            environment=environment,
            step_selection=step_selection,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary."""
        result = {
            "runner": {
                **{k: v for k, v in self.runner.__dict__.items()}
            },
            "agent": {
                **{k: v for k, v in self.agent.__dict__.items()}
            },
            "judgement": {
                **{k: v for k, v in self.judgement.__dict__.items()}
            },
            "environment": {
                **{k: v for k, v in self.environment.__dict__.items()}
            },
            "task": {
                **{k: v for k, v in self.task.__dict__.items()}
            },
        }
        
        # Add step selection config if present
        if self.step_selection:
            result["step_selection"] = {
                **{k: v for k, v in self.step_selection.__dict__.items()}
        }
        
        return result


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
        ValueError: If required fields are missing
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML config: {e}")
    
    if not data:
        raise ValueError("Configuration file is empty")
    
    try:
        return Config.from_dict(data)
    except Exception as e:
        raise ValueError(f"Error creating config from data: {e}")


def create_default_config(output_path: str = "config.yaml") -> Config:
    """
    Create a default configuration file.
    
    Args:
        output_path: Path where to save the default config
        
    Returns:
        Default Config object
    """
    config = Config(
        runner=RunnerConfig(
            experiment_name="default_experiment"
        ),
        agent=AgentConfig(
            model_name="gpt-4o"
        ),
        judgement=JudgementConfig(
            model_name="gpt-4o"
        ),
        environment=EnvironmentConfig(),
        task=TaskConfig(
            name="luban_test",
            type="luban_disassembly",
            difficulty=TaskDifficulty.EASY
        )
    )
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
    
    return config


def validate_config(config: Config) -> List[str]:
    """
    Validate configuration and return list of warnings/errors.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation messages
    """
    issues = []
    
    # Check required fields
    if not config.agent.model_name:
        issues.append("ERROR: Agent model_name is required")
    
    if not config.runner.experiment_name:
        issues.append("ERROR: Experiment name is required")
    
    # Check paths exist
    if not os.path.exists(config.environment.urdf_local_path):
        issues.append(f"WARNING: URDF base path does not exist: {config.environment.urdf_local_path}")
    
    # Check API configuration
    if not config.agent.api_key:
        issues.append("WARNING: No API key found for agent. Set OPENAI_API_KEY environment variable.")
    
    if config.judgement and not config.judgement.api_key:
        issues.append("WARNING: No API key found for judgement agent.")
    
    # Check numeric ranges
    if config.agent.temperature < 0 or config.agent.temperature > 2:
        issues.append("WARNING: Agent temperature should be between 0 and 2")
    
    if config.agent.max_tokens <= 0:
        issues.append("ERROR: Agent max_tokens must be positive")
    
    if config.runner.max_steps <= 0:
        issues.append("ERROR: max_steps must be positive")
    
    return issues
