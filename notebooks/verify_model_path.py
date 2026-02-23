#!/usr/bin/env python3
"""Quick test to verify 3x3 model loading with correct paths."""

import sys
from pathlib import Path

# Add the src directory to Python path
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root / "src"))

print(f"Project root: {project_root}")

# Import minimal components
from phyvpuzzle import load_config

# Load configuration
config_path = project_root / "eval_configs" / "puzzle_quick.yaml"
config = load_config(str(config_path))

print(f"Loaded config: {config.runner.experiment_name}")
print(f"Original URDF path in config: {config.environment.urdf_local_path}")

# Fix the URDF path
urdf_path = project_root / "src" / "phyvpuzzle" / "environment" / "phobos_models"
config.environment.urdf_local_path = str(urdf_path)

print(f"\nCorrected URDF path: {urdf_path}")
print(f"Path exists: {urdf_path.exists()}")

# Check for 3x3 model
model_path = urdf_path / "3x3-stacking-puzzle"
print(f"\n3x3 model path: {model_path}")
print(f"3x3 model exists: {model_path.exists()}")

if model_path.exists():
    print("\n✅ Success! The 3x3-stacking-puzzle model directory found.")
    print("Contents:")
    for item in sorted(model_path.iterdir())[:10]:
        print(f"  - {item.name}")
else:
    print("\n❌ Error: 3x3-stacking-puzzle model not found!")

print("\n✨ Path verification complete!")