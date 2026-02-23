import os
import json
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List

from PIL import Image


class ExperimentLogger:
    def __init__(self, log_dir: str, experiment_name: str):
        """
        Initializes the logger for an experiment.

        Args:
            log_dir (str): The base directory for logs.
            experiment_name (str): A unique name for the experiment.
        """
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"{experiment_name}_{self.timestamp}"
        self.run_dir = os.path.join(log_dir, self.experiment_name)
        self.images_dir = os.path.join(self.run_dir, "images")
        self.logs = []

        # Create directories
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

    def log_step(self, step: int, data: Dict[str, Any], verbose: bool = True):
        """
        Logs a single step of the experiment.

        Args:
            step (int): The current step number.
            data (Dict[str, Any]): A dictionary of data to log for the step.
            verbose (bool): Whether to print step information to console.
        """
        log_entry = {"step": step, "timestamp": datetime.now().isoformat(), **data}

        # Handle image saving
        if "image" in log_entry and isinstance(log_entry["image"], Image.Image):
            image_path = os.path.join(self.images_dir, f"step_{step}.png")
            log_entry["image"].save(image_path)
            log_entry["image_path"] = image_path
            del log_entry["image"]
            
            if verbose:
                print(f"  📷 Saved image: step_{step}.png")
        
        # Console output for different step types
        if verbose:
            step_type = data.get("step_type", "unknown")
            if step_type == "initial":
                print(f"🚀 Step {step}: Initial observation captured")
            elif step_type == "action":
                action_type = data.get("action", {}).get("action_type", "unknown")
                print(f"⚡ Step {step}: Executing action '{action_type}'")
                if data.get("tool_result"):
                    print(f"  📋 Result: {data['tool_result']}")
            elif step_type == "response":
                print(f"💭 Step {step}: Agent response (no action)")
            elif step_type == "error":
                print(f"❌ Step {step}: Error occurred")
                print(f"  🔍 Details: {data.get('error', 'Unknown error')}")

        self.logs.append(log_entry)

    def save_logs(self):
        """Saves all collected logs to a JSON file."""
        log_file = os.path.join(self.run_dir, "experiment_log.json")
        with open(log_file, "w") as f:
            json.dump(self.logs, f, indent=2, default=str)
        
        # Create a summary file
        summary_file = os.path.join(self.run_dir, "summary.txt")
        self._create_summary_file(summary_file)
        
        print(f"📁 Logs saved to: {log_file}")
        print(f"📋 Summary saved to: {summary_file}")
        
    def _create_summary_file(self, summary_file: str):
        """Create a human-readable summary file."""
        total_steps = len([log for log in self.logs if log.get("step_type") in ["action", "response"]])
        actions_taken = len([log for log in self.logs if log.get("step_type") == "action"])
        errors_occurred = len([log for log in self.logs if log.get("step_type") == "error"])
        
        with open(summary_file, "w") as f:
            f.write(f"Experiment Summary: {self.experiment_name}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Total Steps: {total_steps}\n")
            f.write(f"Actions Taken: {actions_taken}\n")
            f.write(f"Errors Occurred: {errors_occurred}\n")
            f.write(f"Images Saved: {len([log for log in self.logs if 'image_path' in log])}\n")
            f.write("\nStep-by-step breakdown:\n")
            f.write("-" * 30 + "\n")
            
            for log in self.logs:
                step = log.get("step", "?")
                step_type = log.get("step_type", "unknown")
                timestamp = log.get("timestamp", "")
                
                if step_type == "initial":
                    f.write(f"Step {step}: Initial setup\n")
                elif step_type == "action":
                    action = log.get("action", {})
                    action_type = action.get("action_type", "unknown")
                    f.write(f"Step {step}: Action '{action_type}'\n")
                    if log.get("tool_result"):
                        f.write(f"  Result: {log['tool_result']}\n")
                elif step_type == "error":
                    f.write(f"Step {step}: ERROR - {log.get('error', 'Unknown')}\n")

    def save_results_to_excel(self, results: Dict[str, Any], excel_path: str):
        """
        Saves the final evaluation results to an Excel file.
        If the file exists, it appends the new results.

        Args:
            results (Dict[str, Any]): A dictionary of evaluation results.
            excel_path (str): The path to the output Excel file.
        """
        results_df = pd.DataFrame([results])

        if os.path.exists(excel_path):
            try:
                existing_df = pd.read_excel(excel_path)
                updated_df = pd.concat([existing_df, results_df], ignore_index=True)
            except Exception as e:
                print(f"Could not read existing excel file: {e}. Creating a new one.")
                updated_df = results_df
        else:
            updated_df = results_df

        updated_df.to_excel(excel_path, index=False)
        print(f"Results saved to {excel_path}")
