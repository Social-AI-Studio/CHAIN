"""
Base task implementation for the CHAIN benchmark (chainbench).
"""

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple

from chainbench.core import BaseTask, TaskConfig, BaseEnvironment
from chainbench.core.base import (Action, BaseEvaluator, ObjectInfo, Observation, State,
                                  TaskResult, EvaluationResult)

# Conditional pybullet import
try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    # Mock pybullet for testing
    class MockPyBullet:
        GUI = "GUI"
        DIRECT = "DIRECT"
        GEOM_BOX = 0
        GEOM_SPHERE = 1
        GEOM_CYLINDER = 2
        WORLD_FRAME = 0
        ER_BULLET_HARDWARE_OPENGL = 0
        
        @staticmethod
        def connect(mode): return 0
        @staticmethod
        def setGravity(x, y, z): pass
        @staticmethod
        def setTimeStep(dt): pass
        @staticmethod
        def setRealTimeSimulation(enable): pass
        @staticmethod
        def setAdditionalSearchPath(path): pass
        @staticmethod
        def loadURDF(path, **kwargs): return 1
        @staticmethod
        def stepSimulation(): pass
        @staticmethod
        def disconnect(client): pass
        @staticmethod
        def resetSimulation(): pass
        @staticmethod
        def createCollisionShape(*args, **kwargs): return 0
        @staticmethod
        def createVisualShape(*args, **kwargs): return 0
        @staticmethod
        def createMultiBody(*args, **kwargs): return 1
        @staticmethod
        def resetBasePositionAndOrientation(obj, pos, orn): pass
        @staticmethod
        def getBasePositionAndOrientation(obj): return ([0,0,0], [0,0,0,1])
        @staticmethod
        def removeBody(obj): pass
        @staticmethod
        def applyExternalForce(*args, **kwargs): pass
        @staticmethod
        def computeViewMatrix(*args, **kwargs): return []
        @staticmethod
        def computeProjectionMatrixFOV(*args, **kwargs): return []
        @staticmethod
        def getCameraImage(*args, **kwargs): return (512, 512, np.zeros((512, 512, 3), dtype=np.uint8), None, None)
        @staticmethod
        def getEulerFromQuaternion(q): return [0, 0, 0]
        @staticmethod
        def getQuaternionFromEuler(euler): return [0, 0, 0, 1]
        @staticmethod
        def getBaseVelocity(obj): return ([0,0,0], [0,0,0])
        @staticmethod
        def getContactPoints(*args, **kwargs): return []
        @staticmethod
        def getNumJoints(obj): return 0
    
    p = MockPyBullet()
    
    class MockPyBulletData:
        @staticmethod
        def getDataPath(): return ""
    
    pybullet_data = MockPyBulletData()


class PhysicsTask(BaseTask):
    """Base class for physics tasks."""
    
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.optimal_steps = self._calculate_optimal_steps()
        self.environment = None
        self.evaluator = None
    
    def configure_environment(self, environment: BaseEnvironment) -> Observation:
        """Configure environment for this task."""
        self.environment = environment
        self.environment.reset()
        self._configure_environment()
        settle_used = self.environment._wait_until_stable()
        print(f"Settle used: {settle_used}")
        observation = self.environment._create_observation()
        return observation

    def evaluate_tasks(self, evaluator: BaseEvaluator, task_results: List[TaskResult]) -> EvaluationResult:
        """Evaluate comprehensive metrics of the task."""
        self.evaluator = evaluator
        task_results = self._evaluate_success(task_results)
        evaluation_result = self.evaluator.evaluate_metrics(task_results)
        return evaluation_result
        
    def get_system_prompt(self) -> str:
        """Get system prompt for this task type."""
        system_prompt = self._get_initial_system_prompt()
        return system_prompt
        
    def get_user_prompt(self) -> str:
        """Get user prompt to start the task."""
        user_prompt = self._get_initial_instruction()
        return user_prompt
    
    @abstractmethod
    def _calculate_optimal_steps(self) -> int:
        """Calculate optimal number of steps for this specific task."""
        pass
    
    @abstractmethod
    def _configure_environment(self) -> None:
        """Configure environment for this task.
        
        Subclasses must override this method to set up environment-specific objects,
        load models, and define the initial state for the task.
        """
        pass

    @abstractmethod
    def _evaluate_success(self, task_results: List[TaskResult]) -> List[TaskResult]:
        """Evaluate success of the task."""
        pass

    # Abstract methods for subclasses
    @abstractmethod
    def _get_initial_system_prompt(self) -> str:
        """Get initial system prompt for this task type."""
        pass
        
    @abstractmethod
    def _get_initial_instruction(self) -> str:
        """Get initial instruction for starting the task."""
        pass