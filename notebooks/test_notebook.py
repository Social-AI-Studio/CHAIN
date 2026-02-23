#!/usr/bin/env python3
"""Test script to verify the Jupyter notebook code works correctly."""

import sys
from pathlib import Path

# Add the src directory to Python path
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root / "src"))

print("Testing notebook code...")
print(f"Project root: {project_root}")

# Test imports
try:
    from phyvpuzzle import load_config, validate_config
    from phyvpuzzle.core import ENVIRONMENT_REGISTRY, TASK_REGISTRY
    from phyvpuzzle.core.base import Action
    print("✅ PhyVPuzzle imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test PyBullet
try:
    import pybullet as p
    print("✅ PyBullet imported successfully")
except ImportError:
    from phyvpuzzle.environment.base_env import p
    print("✅ Using alternative PyBullet import")

# Test configuration loading
config_path = project_root / "eval_configs" / "puzzle_quick.yaml"
if not config_path.exists():
    print(f"❌ Config file not found: {config_path}")
    sys.exit(1)

config = load_config(str(config_path))
print(f"✅ Configuration loaded: {config.runner.experiment_name}")

# Test task creation
task_cls = TASK_REGISTRY.get(config.task.type)
if task_cls is None:
    print(f"❌ Unknown task type: {config.task.type}")
    sys.exit(1)

task = task_cls(config.task)
print(f"✅ Task created: {task.task_id}")

# Test environment creation
env_cls = ENVIRONMENT_REGISTRY.get(config.environment.type)
if env_cls is None:
    print(f"❌ Unknown environment type: {config.environment.type}")
    sys.exit(1)

# Fix URDF path
if hasattr(config.environment, 'urdf_local_path'):
    urdf_path = project_root / config.environment.urdf_local_path
    config.environment.urdf_local_path = str(urdf_path)

environment = env_cls(config.environment)
print("✅ Environment created")

# Configure environment
initial_obs = task.configure_environment(environment)
print(f"✅ Environment configured: {initial_obs.description}")

# Test getting tools
tools = environment.get_tool_schemas()
print(f"✅ Found {len(tools)} tools")

# Test simple action
action = Action(action_type="observe", parameters={"angle": 90})
obs = environment.step(action)
print(f"✅ Action executed: {obs.description}")

# Clean up
environment.close()
print("✅ Environment closed")

print("\n🎉 All tests passed! The notebook code should work correctly.")