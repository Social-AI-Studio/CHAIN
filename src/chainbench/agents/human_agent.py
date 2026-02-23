"""
Human agent implementation for interactive testing.

This agent allows human users to directly interact with the environment
by providing commands through the console interface.
"""

import json
import re
from typing import Any, Dict, List, Tuple, Optional, Union
from PIL import Image
import tempfile
import os
from pathlib import Path

from chainbench.agents import VLMAgent
from chainbench.core.base import Observation
from chainbench.core import register_agent, AgentConfig


@register_agent("human")
class HumanAgent(VLMAgent):
    """Agent that allows human interaction through console interface or predefined action list."""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.step_count = 0
        self.action_list: Optional[List[str]] = None
        self.action_index = 0
        self.interactive_mode = True
        # Store original action list to restore after reset
        self._original_action_list: Optional[List[str]] = None
    
    def set_action_list(self, actions: Union[List[str], str]) -> None:
        """Set action list for automated execution.
        
        Args:
            actions: List of action strings or path to action list file
        """
        if isinstance(actions, str):
            # It's a file path
            self._load_action_list(actions)
        elif isinstance(actions, list):
            # It's a list of actions
            self.action_list = actions.copy()  # Make a copy to avoid reference issues
            self._original_action_list = actions.copy()  # Store original for reset
            self.action_index = 0
            self.interactive_mode = False
            print(f"📋 Set {len(self.action_list)} actions for automated execution")
        else:
            raise ValueError("actions must be a list of strings or a file path")
    
    def reset_to_interactive(self) -> None:
        """Reset agent to interactive mode."""
        self.action_list = None
        self._original_action_list = None
        self.action_index = 0
        self.interactive_mode = True
        print("🎮 Switched to interactive mode")
    
    def reset_action_list(self) -> None:
        """Reset action list to beginning for re-execution."""
        if self._original_action_list:
            self.action_list = self._original_action_list.copy()
            self.action_index = 0
            self.interactive_mode = False
            print(f"🔄 Reset action list to beginning ({len(self.action_list)} actions)")
        else:
            print("⚠️  No original action list to reset to")
    
    def _load_action_list(self, action_list_path: str) -> None:
        """Load action list from file."""
        try:
            path = Path(action_list_path)
            if not path.exists():
                print(f"⚠️  Action list file not found: {action_list_path}")
                print("   Falling back to interactive mode.")
                return
            
            if path.suffix.lower() == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.action_list = data
                    elif isinstance(data, dict) and 'actions' in data:
                        self.action_list = data['actions']
                    else:
                        raise ValueError("Invalid JSON format. Expected list of actions or dict with 'actions' key.")
            else:
                # Plain text file, one action per line
                with open(path, 'r', encoding='utf-8') as f:
                    self.action_list = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
            
            if self.action_list:
                self._original_action_list = self.action_list.copy()  # Store original
                self.action_index = 0
                self.interactive_mode = False
                print(f"📋 Loaded {len(self.action_list)} actions from {action_list_path}")
            else:
                print(f"⚠️  No valid actions found in {action_list_path}")
                
        except Exception as e:
            print(f"❌ Error loading action list from {action_list_path}: {e}")
            print("   Falling back to interactive mode.")
        
    def process_observation(self, history: List[Observation], prompt: str, tools: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """Process observation and get next action (from list or human input)."""
        self.step_count += 1
        
        # Display current state
        self._display_state(history, prompt, tools)
        
        # Get action (from list or human input)
        if self.interactive_mode:
            response, tool_calls = self._get_human_input(tools)
        else:
            response, tool_calls = self._get_next_action_from_list(tools)
        
        # Don't count tokens for human agent
        # self.total_tokens += 0
        
        return response, tool_calls
    
    def _get_next_action_from_list(self, tools: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """Get next action from predefined action list."""
        if not self.action_list or self.action_index >= len(self.action_list):
            print("📋 Action list completed. Finishing task.")
            return "All predefined actions completed.", [{"id": "finish", "type": "function", "function": {"name": "finish", "arguments": "{}"}}]
        
        action_str = self.action_list[self.action_index]
        self.action_index += 1
        
        print(f"\n🤖 Executing action {self.action_index}/{len(self.action_list)}: {action_str}")
        
        # Handle special commands
        if action_str.lower() == 'finish':
            return "Task completed by action list.", [{"id": "finish", "type": "function", "function": {"name": "finish", "arguments": "{}"}}]
        
        if action_str.lower() == 'wait' or action_str.lower() == 'skip':
            return f"Action {self.action_index}: {action_str}", []
        
        # Parse tool call from action string
        tool_calls = self._parse_user_input(action_str, tools)
        if tool_calls:
            return f"Action {self.action_index}: {action_str}", tool_calls
        else:
            print(f"⚠️  Invalid action format: {action_str}")
            print("   Skipping this action and continuing...")
            return f"Skipped invalid action: {action_str}", []
    
    def _display_state(self, history: List[Observation], prompt: str, tools: Optional[List[Dict[str, Any]]] = None) -> None:
        """Display current state information to the user."""
        print("\n" + "="*80)
        mode_str = "AUTOMATED MODE" if not self.interactive_mode else "INTERACTIVE MODE"
        print(f"🎮 HUMAN AGENT - Step {self.step_count} ({mode_str})")
        print("="*80)
        
        # Show the latest observation
        if history:
            latest_obs = history[-1]
            if latest_obs.description:
                if self.interactive_mode:
                    print(f"\n📋 Current State:")
                    print(latest_obs.description)
                else:
                    # In automated mode, show abbreviated state
                    lines = latest_obs.description.split('\n')
                    print(f"\n📋 Current State (abbreviated):")
                    print('\n'.join(lines[:3]))  # Show first 3 lines
                    if len(lines) > 3:
                        print("   ... (truncated)")
            
            # Save and display image if available
            if latest_obs.image:
                self._display_image(latest_obs.image)
        
        # Show prompt (only in interactive mode)
        if self.interactive_mode and prompt:
            print(f"\n💬 System Prompt:")
            print(prompt)
        
        # Show available tools (only in interactive mode)
        if self.interactive_mode and tools:
            print(f"\n🛠️  Available Tools:")
            for i, tool in enumerate(tools, 1):
                func_info = tool.get("function", {})
                name = func_info.get("name", "unknown")
                desc = func_info.get("description", "No description")
                params = func_info.get("parameters", {}).get("properties", {})
                
                print(f"  {i}. {name}")
                print(f"     {desc}")
                if params:
                    print(f"     Parameters: {list(params.keys())}")
                print()
        elif not self.interactive_mode and tools:
            # In automated mode, just show tool count
            print(f"\n🛠️  {len(tools)} tools available")
    
    def _display_image(self, image: Image.Image) -> None:
        """Display image to the user (save to temp file)."""
        try:
            # Save image to temporary file
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"chainbench_step_{self.step_count}.png")
            image.save(temp_path)
            print(f"\n🖼️  Current scene saved to: {temp_path}")
            print("   (Open this file to see the current state)")
        except Exception as e:
            print(f"⚠️  Could not save image: {e}")
    
    def _get_human_input(self, tools: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """Get input from human user."""
        print("\n" + "-"*50)
        print("🎯 Your turn! Enter your action:")
        print("   - Type a tool call: tool_name(param1=value1, param2=value2)")
        print("   - Type 'help' to see available tools")
        print("   - Type 'finish' to complete the task")
        print("   - Type 'quit' to exit")
        print("-"*50)
        
        while True:
            try:
                user_input = input(">>> ").strip()
                
                if not user_input:
                    print("Please enter a command.")
                    continue
                
                if user_input.lower() == 'quit':
                    return "Exiting task.", [{"id": "quit", "type": "function", "function": {"name": "finish", "arguments": "{}"}}]
                
                if user_input.lower() == 'help':
                    self._show_help(tools)
                    continue
                
                if user_input.lower() == 'finish':
                    return "Task completed by user.", [{"id": "finish", "type": "function", "function": {"name": "finish", "arguments": "{}"}}]
                
                # Parse tool call from user input
                tool_calls = self._parse_user_input(user_input, tools)
                if tool_calls:
                    return f"User action: {user_input}", tool_calls
                else:
                    print("❌ Invalid command format. Try again or type 'help'.")
                    continue
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                return "Task interrupted by user.", [{"id": "quit", "type": "function", "function": {"name": "finish", "arguments": "{}"}}]
            except Exception as e:
                print(f"❌ Error: {e}. Please try again.")
                continue
    
    def _show_help(self, tools: Optional[List[Dict[str, Any]]] = None) -> None:
        """Show help information to the user."""
        print("\n📚 HELP - Available Commands:")
        print("="*40)
        
        if tools:
            for tool in tools:
                func_info = tool.get("function", {})
                name = func_info.get("name", "unknown")
                desc = func_info.get("description", "No description")
                params = func_info.get("parameters", {}).get("properties", {})
                required = func_info.get("parameters", {}).get("required", [])
                
                print(f"\n🔧 {name}")
                print(f"   {desc}")
                
                if params:
                    print("   Parameters:")
                    for param_name, param_info in params.items():
                        param_type = param_info.get("type", "unknown")
                        param_desc = param_info.get("description", "No description")
                        is_required = "required" if param_name in required else "optional"
                        print(f"     - {param_name} ({param_type}, {is_required}): {param_desc}")
                
                # Show example usage
                example_params = []
                for param_name, param_info in params.items():
                    if param_name in required:
                        param_type = param_info.get("type", "string")
                        if param_type == "string":
                            example_params.append(f'{param_name}="example"')
                        elif param_type == "number":
                            example_params.append(f'{param_name}=1.0')
                        elif param_type == "boolean":
                            example_params.append(f'{param_name}=true')
                        elif param_type == "array":
                            example_params.append(f'{param_name}=[1, 2, 3]')
                
                if example_params:
                    example = f"{name}({', '.join(example_params)})"
                    print(f"   Example: {example}")
        
        print(f"\n💡 Special Commands:")
        print(f"   - help: Show this help")
        print(f"   - finish: Complete the task")
        print(f"   - quit: Exit the program")
        print("="*40)
    
    def _parse_user_input(self, user_input: str, tools: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Parse user input into tool calls."""
        if not tools:
            return []
        
        # Extract tool names for validation
        available_tools = {tool["function"]["name"] for tool in tools}
        
        # Pattern to match: tool_name(param1=value1, param2=value2)
        pattern = r'(\w+)\((.*?)\)'
        match = re.match(pattern, user_input.strip())
        
        if not match:
            return []
        
        tool_name = match.group(1)
        args_str = match.group(2)
        
        # Validate tool name
        if tool_name not in available_tools:
            print(f"❌ Unknown tool: {tool_name}")
            print(f"   Available tools: {', '.join(available_tools)}")
            return []
        
        # Parse arguments
        try:
            args_dict = {}
            if args_str.strip():
                # Simple argument parsing
                # Handle: param1="value1", param2=123, param3=true, param4=[1,2,3]
                arg_pairs = []
                current_pair = ""
                paren_depth = 0
                bracket_depth = 0
                in_quotes = False
                quote_char = None
                
                for char in args_str:
                    if char in ['"', "'"] and not in_quotes:
                        in_quotes = True
                        quote_char = char
                    elif char == quote_char and in_quotes:
                        in_quotes = False
                        quote_char = None
                    elif not in_quotes:
                        if char == '(':
                            paren_depth += 1
                        elif char == ')':
                            paren_depth -= 1
                        elif char == '[':
                            bracket_depth += 1
                        elif char == ']':
                            bracket_depth -= 1
                        elif char == ',' and paren_depth == 0 and bracket_depth == 0:
                            arg_pairs.append(current_pair.strip())
                            current_pair = ""
                            continue
                    
                    current_pair += char
                
                if current_pair.strip():
                    arg_pairs.append(current_pair.strip())
                
                for pair in arg_pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Parse value
                        try:
                            # Try to parse as JSON first (handles numbers, booleans, arrays, objects)
                            parsed_value = json.loads(value)
                        except json.JSONDecodeError:
                            # If not valid JSON, treat as string (remove quotes if present)
                            if (value.startswith('"') and value.endswith('"')) or \
                               (value.startswith("'") and value.endswith("'")):
                                parsed_value = value[1:-1]
                            else:
                                parsed_value = value
                        
                        args_dict[key] = parsed_value
            
            return [{
                "id": f"call_{self.step_count}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(args_dict)
                }
            }]
            
        except Exception as e:
            print(f"❌ Error parsing arguments: {e}")
            return []
    
    def _prepare_messages(self, history: List[Observation], prompt: str) -> List[Dict[str, Any]]:
        """Prepare messages (not used for human agent, but required by interface)."""
        return []
    
    def _get_model_response(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """Get model response (not used for human agent, but required by interface)."""
        return "", []
    
    def get_token_count(self) -> int:
        """Get total tokens used (always 0 for human agent)."""
        return 0
