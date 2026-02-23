"""
User-friendly display utilities for chainbench.
"""

import time
import sys
from typing import Optional, Dict, Any
from datetime import datetime


class ProgressDisplay:
    """Handles progress display and user-friendly output."""
    
    def __init__(self, total_steps: int = 100):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        
    def update(self, step: int, description: str = ""):
        """Update progress display."""
        self.current_step = step
        progress = min(step / self.total_steps, 1.0)
        
        # Calculate timing
        elapsed = time.time() - self.start_time
        if step > 0:
            eta = (elapsed / step) * (self.total_steps - step)
        else:
            eta = 0
            
        # Create progress bar
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        # Format time
        elapsed_str = self._format_time(elapsed)
        eta_str = self._format_time(eta)
        
        # Print progress
        progress_line = f"\r⏳ Progress: [{bar}] {progress:.1%} ({step}/{self.total_steps}) | Elapsed: {elapsed_str} | ETA: {eta_str}"
        if description:
            progress_line += f" | {description}"
            
        print(progress_line, end="", flush=True)
        
    def finish(self, success: bool = True):
        """Finish progress display."""
        total_time = time.time() - self.start_time
        total_time_str = self._format_time(total_time)
        
        if success:
            print(f"\n✅ Complete! Total time: {total_time_str}")
        else:
            print(f"\n❌ Failed! Total time: {total_time_str}")
            
    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"


class StatusDisplay:
    """Handles status display for different operations."""
    
    @staticmethod
    def print_header(title: str, width: int = 80):
        """Print a formatted header."""
        print("\n" + "=" * width)
        print(f"{title:^{width}}")
        print("=" * width)
        
    @staticmethod
    def print_section(title: str, width: int = 60):
        """Print a section header."""
        print(f"\n📋 {title}")
        print("-" * width)
        
    @staticmethod
    def print_config(config_dict: Dict[str, Any], title: str = "Configuration"):
        """Print configuration in a nice format."""
        StatusDisplay.print_section(title)
        for key, value in config_dict.items():
            print(f"  {key:<20} : {value}")
            
    @staticmethod
    def print_status(message: str, status: str = "info"):
        """Print a status message with appropriate emoji."""
        icons = {
            "info": "ℹ️",
            "success": "✅", 
            "warning": "⚠️",
            "error": "❌",
            "loading": "⏳",
            "processing": "🔄"
        }
        icon = icons.get(status, "ℹ️")
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{icon} [{timestamp}] {message}")
        
    @staticmethod
    def print_results(results: Dict[str, Any], title: str = "Results"):
        """Print results in a nice format."""
        StatusDisplay.print_section(title)
        for key, value in results.items():
            if isinstance(value, float):
                if 0 <= value <= 1:
                    # Probably a percentage or ratio
                    print(f"  {key:<20} : {value:.1%}")
                else:
                    print(f"  {key:<20} : {value:.3f}")
            elif isinstance(value, bool):
                icon = "✅" if value else "❌"
                print(f"  {key:<20} : {icon} {value}")
            else:
                print(f"  {key:<20} : {value}")
                
    @staticmethod
    def print_step_summary(step: int, action: str, result: str, timing: float = None):
        """Print a step summary."""
        timing_str = f" ({timing:.2f}s)" if timing else ""
        print(f"  Step {step:2d}: {action:<20} → {result}{timing_str}")
        
    @staticmethod 
    def print_separator(char: str = "-", length: int = 60):
        """Print a separator line."""
        print(char * length)
        
    @staticmethod
    def ask_confirmation(message: str) -> bool:
        """Ask for user confirmation."""
        response = input(f"❓ {message} (y/N): ").strip().lower()
        return response in ['y', 'yes']


class LiveLogger:
    """Live logging with real-time updates."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.step_times = {}
        
    def log_step_start(self, step: int, description: str):
        """Log the start of a step."""
        if self.verbose:
            StatusDisplay.print_status(f"Starting Step {step}: {description}", "processing")
        self.step_times[step] = time.time()
        
    def log_step_end(self, step: int, result: str, success: bool = True):
        """Log the end of a step."""
        if self.verbose:
            elapsed = time.time() - self.step_times.get(step, time.time())
            status = "success" if success else "error"
            StatusDisplay.print_status(
                f"Step {step} completed: {result} ({elapsed:.2f}s)", 
                status
            )
            
    def log_action(self, action_name: str, details: str = ""):
        """Log an action being performed."""
        if self.verbose:
            message = f"Executing: {action_name}"
            if details:
                message += f" - {details}"
            StatusDisplay.print_status(message, "processing")
            
    def log_result(self, message: str, success: bool = True):
        """Log a result."""
        if self.verbose:
            status = "success" if success else "error"
            StatusDisplay.print_status(message, status)
            
    def log_info(self, message: str):
        """Log an info message."""
        if self.verbose:
            StatusDisplay.print_status(message, "info")
            
    def log_warning(self, message: str):
        """Log a warning message."""
        if self.verbose:
            StatusDisplay.print_status(message, "warning")
            
    def log_error(self, message: str):
        """Log an error message."""
        if self.verbose:
            StatusDisplay.print_status(message, "error")
