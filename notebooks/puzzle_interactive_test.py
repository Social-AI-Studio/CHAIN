#!/usr/bin/env python3
"""
Interactive Puzzle Environment Test Script
This script extracts the core components from puzzle_quick.py to enable
interactive environment manipulation and visualization.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phyvpuzzle import load_config, validate_config
from phyvpuzzle.core import ENVIRONMENT_REGISTRY, TASK_REGISTRY, AGENT_REGISTRY
from phyvpuzzle.core.base import Action
try:
    import pybullet as p
except ImportError:
    from phyvpuzzle.environment.base_env import p

# Try to load environment variables from a .env file (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Note: python-dotenv not available. Using system environment variables only.")

class InteractivePuzzleEnvironment:
    """Interactive wrapper for puzzle environment with visualization capabilities."""

    def __init__(self, config_path=None):
        """Initialize the interactive environment."""
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent / "eval_configs" / "puzzle_quick.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load config and fix the URDF path
        self.config = load_config(str(config_path))

        # Fix the URDF path to be absolute - correct path
        if hasattr(self.config.environment, 'urdf_local_path'):
            # The correct path to phobos_models
            base_dir = Path(__file__).resolve().parent.parent
            urdf_path = base_dir / "src" / "phyvpuzzle" / "environment" / "phobos_models"
            self.config.environment.urdf_local_path = str(urdf_path)
            print(f"URDF path set to: {urdf_path}")

            # Verify the 3x3 model exists
            model_path = urdf_path / "3x3-stacking-puzzle"
            if model_path.exists():
                print(f"✓ 3x3 model found at: {model_path}")
            else:
                print(f"⚠️ 3x3 model not found at: {model_path}")

        print(f"Loaded configuration: {self.config.runner.experiment_name}")

        # Initialize components
        self.task = None
        self.environment = None
        self.current_observation = None
        self.step_count = 0
        self.max_steps = self.config.environment.max_steps
        self.image_history = []

    def setup(self):
        """Setup the environment and task."""
        print("Setting up task...")
        # Create task
        task_cls = TASK_REGISTRY.get(self.config.task.type)
        if task_cls is None:
            raise RuntimeError(f"Unknown task type: {self.config.task.type}")
        self.task = task_cls(self.config.task)

        print("Setting up environment...")
        # Create environment
        env_cls = ENVIRONMENT_REGISTRY.get(self.config.environment.type)
        if env_cls is None:
            raise RuntimeError(f"Unknown environment type: {self.config.environment.type}")
        self.environment = env_cls(self.config.environment)

        print("Configuring environment with task...")
        # Configure environment with task
        self.current_observation = self.task.configure_environment(self.environment)

        print(f"Environment ready! Task: {self.task.task_id}")
        print(f"Task Type: {self.config.task.type}")
        print(f"Difficulty: {self.config.task.difficulty.value}")
        print(f"Number of pieces: {self.config.task.num_pieces}")
        print(f"Max steps: {self.max_steps}")

        return self.current_observation

    def get_current_image(self):
        """Get the current observation image."""
        if self.current_observation and self.current_observation.image:
            return self.current_observation.image
        return None

    def save_image(self, filename=None, step=None):
        """Save the current observation image."""
        if self.current_observation and self.current_observation.image:
            if filename is None:
                if step is None:
                    step = self.step_count
                filename = f"puzzle_step_{step:03d}.png"

            # Ensure the image is in the correct format
            img = self.current_observation.image
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)

            img.save(filename)
            print(f"Image saved to {filename}")
            return filename
        else:
            print("No image to save")
            return None

    def display_image(self, figsize=(10, 10)):
        """Display the current observation image using matplotlib."""
        if self.current_observation and self.current_observation.image:
            plt.figure(figsize=figsize)
            plt.imshow(self.current_observation.image)
            plt.title(f"Step {self.step_count}: {self.current_observation.description}")
            plt.axis('off')
            plt.show()
        else:
            print("No image to display")

    def get_available_tools(self):
        """Get the list of available tools/actions."""
        if self.environment:
            return self.environment.get_tool_schemas()
        return []

    def get_object_mapping(self):
        """Get object ID to color/property mapping."""
        if not self.environment:
            return "Environment not initialized"

        lines = ["OBJECT MAPPING (object_id → properties):"]
        lines.append("=" * 60)

        non_container_count = 0

        for obj_info in self.environment.objects:
            # Skip container objects
            if obj_info.properties.get('is_container', False):
                continue

            obj_id = obj_info.object_id
            non_container_count += 1

            # Get visual shape data to retrieve color
            try:
                visual_shapes = p.getVisualShapeData(obj_id)

                if visual_shapes:
                    rgba_color = visual_shapes[0][7]  # Index 7 is rgbaColor

                    # Format color as RGB values (0-255 scale for readability)
                    r = int(rgba_color[0] * 255)
                    g = int(rgba_color[1] * 255)
                    b = int(rgba_color[2] * 255)

                    lines.append(f"object_id={obj_id}, RGB=({r}, {g}, {b})")
                else:
                    lines.append(f"object_id={obj_id}, color=unknown")

            except Exception as e:
                lines.append(f"object_id={obj_id}, color=error ({str(e)})")

        lines.append("=" * 60)
        lines.append(f"Total movable objects: {non_container_count}")

        return "\n".join(lines)

    def execute_action(self, action_type, parameters):
        """Execute an action in the environment."""
        if not self.environment:
            print("Environment not initialized. Call setup() first.")
            return None

        if self.step_count >= self.max_steps:
            print(f"Maximum steps ({self.max_steps}) reached!")
            return None

        # Create action
        action = Action(action_type=action_type, parameters=parameters)

        print(f"\nExecuting action: {action_type}({parameters})")

        # Execute action
        self.current_observation = self.environment.step(action)
        self.step_count += 1

        # Store image in history
        if self.current_observation.image:
            self.image_history.append(self.current_observation.image)

        print(f"Step {self.step_count}/{self.max_steps}")
        print(f"Result: {self.current_observation.description}")

        if 'tool_result' in self.current_observation.state.metadata:
            print(f"Tool result: {self.current_observation.state.metadata['tool_result']}")

        return self.current_observation

    def reset(self):
        """Reset the environment."""
        if self.environment:
            self.environment.close()

        self.step_count = 0
        self.image_history = []
        self.setup()
        print("Environment reset complete")

    def close(self):
        """Close the environment."""
        if self.environment:
            self.environment.close()
            print("Environment closed")

    def get_task_info(self):
        """Get information about the current task."""
        if not self.task:
            return "Task not initialized"

        info = {
            "Task ID": self.task.task_id,
            "Task Type": self.config.task.type,
            "Task Name": self.config.task.name,
            "Difficulty": self.config.task.difficulty.value,
            "Number of pieces": self.config.task.num_pieces,
            "Puzzle size": self.config.task.puzzle_size,
            "Piece size": self.config.task.piece_size,
            "Current step": f"{self.step_count}/{self.max_steps}",
            "Optimal steps": getattr(self.task, 'optimal_steps', 'Unknown')
        }
        return info

    def save_trajectory(self, filename="trajectory.json"):
        """Save the current trajectory to a JSON file."""
        trajectory = {
            "task_info": self.get_task_info(),
            "steps": self.step_count,
            "max_steps": self.max_steps,
            "image_count": len(self.image_history)
        }

        with open(filename, 'w') as f:
            json.dump(trajectory, f, indent=2, default=str)

        print(f"Trajectory saved to {filename}")


def main():
    """Main function for testing the interactive environment."""
    print("=" * 80)
    print("Interactive Puzzle Environment Test")
    print("=" * 80)

    # Create interactive environment
    env = InteractivePuzzleEnvironment()

    try:
        # Setup environment
        initial_obs = env.setup()

        # Display initial state
        print("\n" + "=" * 60)
        print("Initial State:")
        print(env.get_object_mapping())

        # Save initial image
        env.save_image("initial_state.png")

        # Get available tools
        tools = env.get_available_tools()
        print("\n" + "=" * 60)
        print("Available Tools:")
        for i, tool in enumerate(tools, 1):
            print(f"{i}. {tool['function']['name']}: {tool['function']['description']}")

        # Example interactions (you can modify these)
        print("\n" + "=" * 60)
        print("Example Interactions:")

        # Example 1: Get objects
        print("\n1. Getting all objects:")
        obs = env.execute_action("get_objects", {})

        # Example 2: Move an object (you would need to adjust object_id based on actual objects)
        # print("\n2. Moving an object:")
        # obs = env.execute_action("move_object", {"object_id": 1, "position": [0.0, 0.0, 0.1]})
        # env.save_image("after_move.png")

        # Display task info
        print("\n" + "=" * 60)
        print("Task Information:")
        for key, value in env.get_task_info().items():
            print(f"  {key}: {value}")

        # Save trajectory
        env.save_trajectory("test_trajectory.json")

    finally:
        # Clean up
        env.close()

    print("\n" + "=" * 80)
    print("Test completed successfully!")


if __name__ == "__main__":
    main()