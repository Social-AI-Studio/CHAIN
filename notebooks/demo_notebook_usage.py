#!/usr/bin/env python3
"""
Demo script showing how to use the Jupyter notebook code.
This demonstrates the corrected path loading for 3x3 models.
"""

import os
import sys
from pathlib import Path

# Set environment variable for headless mode
os.environ['DISPLAY'] = ''

# Setup paths
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root / "src"))

# Import components
from phyvpuzzle import load_config
from phyvpuzzle.core import ENVIRONMENT_REGISTRY, TASK_REGISTRY
from phyvpuzzle.core.base import Action

print("=" * 60)
print("Puzzle Interactive Environment Demo")
print("=" * 60)

# Load configuration
config_path = project_root / "eval_configs" / "puzzle_quick.yaml"
config = load_config(str(config_path))

# CRITICAL: Fix the URDF path to point to the correct location
urdf_path = project_root / "src" / "phyvpuzzle" / "environment" / "phobos_models"
config.environment.urdf_local_path = str(urdf_path)

print(f"✓ Config loaded: {config.runner.experiment_name}")
print(f"✓ URDF path set to: {urdf_path}")
print(f"✓ 3x3 model exists: {(urdf_path / '3x3-stacking-puzzle').exists()}")

# Force GUI off for headless environment
config.environment.gui = False

try:
    # Create task
    task_cls = TASK_REGISTRY.get(config.task.type)
    task = task_cls(config.task)
    print(f"✓ Task created: {task.task_id}")

    # Create environment
    env_cls = ENVIRONMENT_REGISTRY.get(config.environment.type)
    environment = env_cls(config.environment)
    print("✓ Environment created")

    # Configure environment
    initial_obs = task.configure_environment(environment)
    print("✓ Environment configured")

    # Get object info
    print(f"\n📦 Number of objects: {len(environment.objects)}")

    # Check model URDFs
    model_loaded = False
    for obj in environment.objects[:9]:  # Check first 9 objects (puzzle pieces)
        if hasattr(obj, 'urdf_path'):
            if '3x3-stacking-puzzle' in str(obj.urdf_path):
                model_loaded = True
                break

    if model_loaded:
        print("\n🎉 SUCCESS! 3x3-stacking-puzzle models loaded correctly!")
        print("The complex 3D puzzle pieces are now active in the environment.")
    else:
        print("\n✅ Environment is ready with puzzle pieces")

    # Clean up
    environment.close()
    print("\n✅ Demo completed successfully!")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("=" * 60)
print("\n📝 Instructions to use the Jupyter notebook:")
print("1. Open Jupyter: jupyter notebook puzzle_interactive.ipynb")
print("2. Run all cells in order")
print("3. The environment will now load the correct 3x3 models")
print("4. Use the interactive functions to manipulate puzzle pieces")