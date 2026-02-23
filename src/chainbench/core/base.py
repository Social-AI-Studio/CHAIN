"""
Base classes and interfaces for the chainbench benchmark system.

This module defines the fundamental abstractions for environments, tasks,
agents, and evaluators that form the backbone of the benchmark.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING, Callable, cast
from dataclasses import dataclass, field
from enum import Enum
import time
from PIL import Image
from abc import ABC, abstractmethod
if TYPE_CHECKING:
    from chainbench.core.config import EnvironmentConfig, TaskConfig, AgentConfig, JudgementConfig


class TaskDifficulty(Enum):
    """Difficulty levels for tasks."""
    EASY: str = "easy"
    MEDIUM: str = "medium"
    HARD: str = "hard"

@dataclass
class Action:
    """Represents an action to be executed in the environment."""
    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary representation."""
        return {
            "action_type": self.action_type,
            "parameters": self.parameters,
        }


@dataclass
class ObjectInfo:
    """Information about an object in the environment."""
    object_id: int
    name: str
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]
    object_type: str
    properties: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert object info to dictionary representation."""
        return {
            "object_id": self.object_id,
            "name": self.name,
            "position": list(self.position),
            "orientation": list(self.orientation),
            "object_type": self.object_type,
            "properties": self.properties,
        }


@dataclass
class State:
    """Represents the state of the environment at a given time."""
    step: int
    objects: List[ObjectInfo]
    time_stamp: float
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary representation."""
        return {
            "step": self.step,
            "objects": [obj.to_dict() for obj in self.objects],
            "time_stamp": self.time_stamp,
            "metadata": self.metadata,
        }


@dataclass
class Observation:
    """Observation data provided to the agent."""
    image: Image.Image
    state: State
    description: str
   
    def to_dict(self) -> Dict[str, Any]:
        """Convert observation to dictionary (excluding images)."""
        return {
            "state": self.state.to_dict(),
            "description": self.description,
        }


@dataclass
class TaskResult:
    """Results from task execution."""
    task_id: str
    task_type: str
    total_steps: int
    execution_time: float
    trajectory: Union[List[Tuple[Action, Observation]], List[Dict[str, Any]]]  # Support both old and new formats
    success: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task result to dictionary representation."""
        # Handle trajectory based on format
        trajectory_dict = []
        
        if self.trajectory:
            first_item = self.trajectory[0]
            
            if isinstance(first_item, dict):
                # New format: dict with response, actions (Action objects), observations (Observation objects)
                for step_data in self.trajectory:
                    step_dict = {
                        "response": step_data.get("response", ""),
                        "actions": [],
                        "observations": []
                    }
                    
                    # Serialize actions
                    for action in step_data.get("actions", []):
                        if hasattr(action, "to_dict"):
                            step_dict["actions"].append(action.to_dict())
                        else:
                            step_dict["actions"].append(action)
                    
                    # Serialize observations (excluding images for JSON compatibility)
                    for obs in step_data.get("observations", []):
                        if hasattr(obs, "to_dict"):
                            step_dict["observations"].append(obs.to_dict())
                        else:
                            step_dict["observations"].append(obs)
                    
                    trajectory_dict.append(step_dict)
            else:
                # Old format: list of (action, observation) tuples
                for action, observation in self.trajectory:
                    trajectory_dict.append({
                        "action": action.to_dict() if hasattr(action, "to_dict") else action,
                        "observation": observation.to_dict() if hasattr(observation, "to_dict") else observation,
                    })
        
        return {
            "task_id": self.task_id,
            "task_type": str(self.task_type),  # Ensure string conversion for enum types
            "total_steps": self.total_steps,
            "execution_time": self.execution_time,
            "success": bool(self.success),
            "trajectory": trajectory_dict,
            "metadata": self.metadata,
            "error_message": self.error_message,
        }


@dataclass
class EvaluationResult:
    """Results from evaluation of task performance."""
    accuracy: float
    pass_at_k: Dict[int, float]  # k -> success rate
    distance_to_optimal: float
    token_efficiency: float
    detailed_metrics: Dict[str, float]
    task_results: List[TaskResult]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation result to dictionary representation."""
        return {
            "accuracy": self.accuracy,
            "pass_at_k": self.pass_at_k,
            "distance_to_optimal": self.distance_to_optimal,
            "token_efficiency": self.token_efficiency,
            "detailed_metrics": self.detailed_metrics,
            "num_tasks": len(self.task_results)
        }


class BaseEnvironment(ABC):
    """Base class for physics simulation environments."""

     # registry for tool functions
    tool_registry: dict[str, Callable] = {}
    
    def __init__(self, config: EnvironmentConfig):
        self.config: EnvironmentConfig = config

    @classmethod
    def register_tool(cls, name: str):
        """
        Decorator to register a new tool function to the environment.
        Usage:
            @BaseEnvironment.register_tool("pick")
            def pick(self, ...): ...
        """
        def deco(func):
            cls.tool_registry[name] = cast(Callable, func)
            return func
        return deco
        
    @abstractmethod
    def reset(self) -> Observation:
        """Reset environment to initial state."""
        pass
    
    @abstractmethod
    def step(self, action: Action) -> Observation:
        """Execute action and return new observation, feedback, and done flag."""
        pass
    
    @abstractmethod
    def render(self, multi_view: bool = True) -> Union[Image.Image, Dict[str, Image.Image]]:
        """Render the current environment state."""
        pass
    
    @abstractmethod
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get JSON schemas for tool functions that VLM can call."""
        pass
    
    @abstractmethod
    def execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call from VLM."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Clean up environment resources."""
        pass


class BaseTask(ABC):
    """Base class for benchmark tasks."""
    
    def __init__(self, config: TaskConfig):
        self.config: TaskConfig = config
        self.task_id: str = f"{self.config.type}_{int(time.time())}"

    @abstractmethod
    def configure_environment(self, environment: BaseEnvironment) -> Observation:
        """Configure environment for this task."""
        pass

    @abstractmethod
    def evaluate_tasks(self, evaluator: BaseEvaluator, task_results: List[TaskResult]) -> EvaluationResult:
        """Evaluate success of the task."""
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get system prompt for this task."""
        pass
    
    @abstractmethod
    def get_user_prompt(self) -> str:
        """Get user prompt to start the task."""
        pass


class BaseAgent(ABC):
    """Base class for VLM agents."""
    
    def __init__(self, config: AgentConfig):
        self.config: AgentConfig = config

    @abstractmethod
    def process_observation(self, history: List[Observation], prompt: str, tools: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process observations and return response and tool calls.
        
        Returns:
            Tuple of (text_response, tool_calls_list)
        """
        pass
    
    @abstractmethod
    def get_token_count(self) -> int:
        """Get total tokens used by this agent."""
        pass


class BaseEvaluator(ABC):
    """Base class for evaluating task performance."""
    
    def __init__(self, config: JudgementConfig):
        self.config: JudgementConfig = config

    @abstractmethod
    def evaluate_metrics(self, task_results: List[TaskResult]) -> EvaluationResult:
        """Evaluate multiple task results and aggregate metrics."""
        pass


class BaseJudge(ABC):
    """Base class for LLM-as-judge evaluation."""
    
    def __init__(self, config: JudgementConfig):
        self.config: JudgementConfig = config
        
    @abstractmethod
    def judge_success(self, final_image: Image.Image, task_description: str, 
                     trajectory: List[str]) -> Tuple[bool, float, str]:
        """
        Judge if task was completed successfully.
        
        Returns:
            Tuple of (success, confidence_score, reasoning)
        """
        pass
