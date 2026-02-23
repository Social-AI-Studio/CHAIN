# PhyVPuzzle Developer Guide

## 🏗️ Architecture Overview

```
src/phyvpuzzle/
├── core/           # Base classes, registry, config
├── environment/    # Physics environments (PyBullet-based)
├── tasks/          # Task definitions and prompts
├── agents/         # VLM agent implementations
├── evaluation/     # Metrics and evaluation logic
├── utils/          # Helper utilities
├── runner.py       # Main execution runner
└── cli.py          # Command-line interface
```

**Key Design Patterns:**
- Decorator-based registration (`@register_task`, `@register_environment`)
- Abstract base classes for extension
- YAML configuration-driven setup
- Tool-based agent-environment interaction

## 🌍 Add New Environment

### Step 1: Create Environment Class

```python
# src/phyvpuzzle/environment/my_env.py
from typing import Dict, List, Any, Tuple
from .base_env import PhysicsEnvironment
from ..core.base import State
from ..core import register_environment

@register_environment("my_env")
class MyEnvironment(PhysicsEnvironment):
    """Custom environment implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.my_objects = {}

    def _setup_task_environment(self) -> None:
        """Setup environment objects."""
        # Load objects, setup physics world
        pass

    def _get_task_specific_tool_schemas(self) -> List[Dict[str, Any]]:
        """Define custom tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "my_custom_tool",
                    "description": "Custom tool description",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "param1": {"type": "string", "description": "Parameter 1"}
                        },
                        "required": ["param1"]
                    }
                }
            }
        ]

    def _execute_task_specific_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute custom tools."""
        if tool_name == "my_custom_tool":
            return self._my_custom_tool(arguments.get("param1"))
        return super()._execute_task_specific_tool(tool_name, arguments)

    def _my_custom_tool(self, param1: str) -> Dict[str, Any]:
        """Implement custom tool logic."""
        try:
            # Tool implementation
            return {"status": "success", "message": f"Processed {param1}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _evaluate_success(self) -> bool:
        """Evaluate task completion."""
        # Implement success criteria
        return False

    def _get_current_state(self) -> State:
        """Get current environment state."""
        return State(
            step=self.step_count,
            objects=self.my_objects,
            completed=False,
            success=False,
            metadata={}
        )
```

### Step 2: Update __init__.py

```python
# src/phyvpuzzle/environment/__init__.py
from .my_env import MyEnvironment

__all__ = [
    # ... existing environments
    "MyEnvironment",
]
```

## 📝 Add New Task

### Step 1: Create Task Class

```python
# src/phyvpuzzle/tasks/my_task.py
from typing import Dict, Any
from .base_task import PhysicsTask
from ..core.base import TaskDifficulty, State
from ..core import register_task, register_task_config

@register_task_config("my_task")
@dataclass
class MyTaskConfig(TaskConfig):
    """Task configuration."""
    my_param: int = 5

@register_task("my_task")
class MyTask(PhysicsTask):
    """Custom task implementation."""

    def _configure_environment(self, environment) -> None:
        """Configure environment for this task."""
        if not isinstance(environment, MyEnvironment):
            raise ValueError("Requires MyEnvironment")
        # Apply task-specific config
        pass

    def _get_initial_system_prompt(self) -> str:
        """System prompt for the task."""
        return "You are solving a custom physics task..."

    def _get_initial_instruction(self) -> str:
        """Initial user instruction."""
        return "Solve this task using available tools..."

    def _evaluate_success(self, final_state: State, trajectory: list) -> str:
        """Success evaluation criteria."""
        return "Task completed when [criteria met]."
```

### Step 2: Update __init__.py

```python
# src/phyvpuzzle/tasks/__init__.py
from .my_task import MyTask

__all__ = [
    # ... existing tasks
    "MyTask",
]
```

## 🤖 Add New Agent

### Step 1: Create Agent Class

```python
# src/phyvpuzzle/agents/my_agent.py
from typing import List, Dict, Any, Tuple, Optional
from .base_agent import VLMAgent
from ..core import register_agent

@register_agent("my_agent")
class MyAgent(VLMAgent):
    """Custom agent implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_url = config.get("api_url", "https://api.example.com")

    def _get_model_response(self, messages: List[Dict[str, Any]],
                          tools: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """Get response from custom model."""
        # Implement API call to your model
        # Return (content, tool_calls)
        pass

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        # Implement token counting
        return len(text.split())
```

### Step 2: Update __init__.py

```python
# src/phyvpuzzle/agents/__init__.py
from .my_agent import MyAgent

__all__ = [
    # ... existing agents
    "MyAgent",
]
```

## ⚙️ Configuration

Create YAML config file:

```yaml
# eval_configs/my_config.yaml
runner:
  experiment_name: "my_experiment"
  log_dir: "logs"

agent:
  type: "my_agent"
  model_name: "custom-model"
  api_url: "https://api.example.com"

environment:
  type: "my_env"
  max_steps: 20

task:
  type: "my_task"
  difficulty: "medium"
  my_param: 10
```

## 🧪 Testing

```python
# tests/test_my_env.py
import unittest
from src.phyvpuzzle.environment.my_env import MyEnvironment

class TestMyEnvironment(unittest.TestCase):
    def setUp(self):
        self.config = {"gui": False}
        self.env = MyEnvironment(self.config)

    def test_initialization(self):
        self.assertIsNotNone(self.env)

    def test_custom_tool(self):
        result = self.env.execute_tool_call("my_custom_tool", {"param1": "test"})
        self.assertEqual(result["status"], "success")

    def tearDown(self):
        self.env.close()
```

## 🚀 Quick Commands

```bash
# Run with custom config
python -m phyvpuzzle.cli run --config eval_configs/my_config.yaml

# Test new components
python -m pytest tests/test_my_env.py -v

# Debug with GUI
python -m phyvpuzzle.cli run --config eval_configs/my_config.yaml --gui
```

**Remember:** Always test thoroughly and follow the existing code patterns for consistency!
