"""
OpenAI API-based agent implementation.
"""

import json
import re
from typing import Any, Dict, List, Tuple, Optional
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from chainbench.agents import VLMAgent
from chainbench.core.base import Observation
from chainbench.core import register_agent, AgentConfig

@register_agent("openai")
class OpenAIAgent(VLMAgent):
    """Agent using OpenAI API (or compatible APIs)."""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )

    def _prepare_messages(self, history: List[Observation], prompt: str) -> List[Dict[str, Any]]:
        """Prepare message list for model API call."""
        messages = []
        
        # Add system message
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })

        # Add current observation with image
        content = [
            {"type": "text", "text": prompt}
        ]
        
        # Add main image
        for observation in history:
            if observation.image:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": self._image_to_data_url(observation.image)}
                })
            
        messages.append({
            "role": "user",
            "content": content
        })
        
        return messages
        
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _get_model_response(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """Get response from OpenAI API."""
        try:
            # Make API call
            kwargs = {
                "model": self.config.model_name,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "timeout": self.config.timeout
            }
            
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
                
            if "gpt-5" in self.config.model_name:
                kwargs["reasoning_effort"] = "high"
            
            # print(messages)
            response = self.client.chat.completions.create(**kwargs)
            
            # Validate response structure
            if not response or not response.choices or len(response.choices) == 0:
                raise RuntimeError("Empty or invalid response from OpenAI API")
            
            # Extract response content
            message = response.choices[0].message
            if not message:
                raise RuntimeError("Empty message in OpenAI API response")
            if hasattr(message, "reasoning") and message.reasoning:
                content = message.reasoning.strip()
                if message.content:
                    content += "\n\n" + message.content.strip()
            else:
                content = message.content.strip()
            
            # Extract tool calls
            tool_calls = []
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_calls.append({
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    })
            else:
                tool_calls = self.parse_tool_calls_from_text(content)
                # if not tool_calls:
            
            return content, tool_calls
            
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return f"Error: {e}", []
        
    def _image_to_data_url(self, image) -> str:
        """Convert PIL Image to data URL."""
        import base64
        from io import BytesIO
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    
    def parse_tool_calls_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Parse tool calls from text response when formal tool calling isn't used."""
        tool_calls = []
        
        # Look for action patterns in text
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Pattern 1: Action: function_name(args)
            if line.startswith("Action:"):
                try:
                    action_part = line[7:].strip()  # Remove "Action: "
                    
                    # Extract function name and arguments
                    if '(' in action_part and ')' in action_part:
                        func_name = action_part.split('(')[0].strip()
                        args_part = action_part[action_part.find('(')+1:action_part.rfind(')')]
                        
                        # Simple argument parsing (would need more robust parsing)
                        args = {}
                        if args_part:
                            # Very basic parsing - in real implementation would use proper parser
                            try:
                                args = json.loads(f"{{{args_part}}}")
                            except:
                                # Fallback to simple key=value parsing
                                for arg in args_part.split(','):
                                    if '=' in arg:
                                        key, val = arg.split('=', 1)
                                        key = key.strip().strip("\"'")
                                        val = val.strip().strip("\"'")
                                        args[key] = val
                        
                        tool_calls.append({
                            "id": f"call_{len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": func_name,
                                "arguments": json.dumps(args)
                            }
                        })
                except Exception as e:
                    print(f"Error parsing action from text: {e}")
                    continue
            
            # Pattern 2: {"action": "func_name", "parameters": {...}}
            elif line.startswith('{') and '"action"' in line:
                try:
                    action_data = json.loads(line)
                    if "action" in action_data:
                        tool_calls.append({
                            "id": f"call_{len(tool_calls)}",
                            "type": "function", 
                            "function": {
                                "name": action_data["action"],
                                "arguments": json.dumps(action_data.get("parameters", {}))
                            }
                        })
                except Exception as e:
                    print(f"Error parsing JSON action: {e}")
                    continue
        
        idx = 0
        if not tool_calls:
            # Pattern 3: <action>JSON</action>
            # Use DOTALL to match across newlines
            pattern = r"<action>(.*?)</action>"
            rematches = re.findall(pattern, text, re.DOTALL)
            
            # Take the last match as it's likely the final decision
            if rematches:
                match = rematches[-1].strip()
                
                # Pre-processing to fix common JSON syntax errors from LLMs
                # Replace Python None/True/False with JSON null/true/false
                match_fixed = match.replace("None", "null").replace("True", "true").replace("False", "false")
                
                tool_data = {}
                success = False
                
                # Attempt 1: Direct JSON parse
                try:
                    tool_data = json.loads(match_fixed)
                    success = True
                except json.JSONDecodeError:
                    # Attempt 2: Try replacing single quotes with double quotes
                    try:
                        # Be careful not to replace apostrophes in text, but this is a heuristic
                        fixed_quotes = match_fixed.replace("'", '"')
                        tool_data = json.loads(fixed_quotes)
                        success = True
                    except json.JSONDecodeError:
                        # Attempt 3: Try to parse if it looks like python dict string
                        try:
                            import ast
                            # ast.literal_eval is safer than eval
                            tool_data = ast.literal_eval(match)
                            if isinstance(tool_data, dict):
                                success = True
                        except:
                            print(f"Failed to parse action content: {match}")
                            pass
                
                if success and isinstance(tool_data, dict):
                    # Normalize fields
                    # Check for tool name in various common fields
                    tool_name = tool_data.get("name") or tool_data.get("tool") or tool_data.get("action") or tool_data.get("function")
                    
                    if tool_name:
                        # Check for parameters
                        parameters = tool_data.get("parameters") or tool_data.get("arguments") or tool_data.get("args") or tool_data.get("params")
                        
                        # If no explicit parameters dict, assume rest of keys are parameters
                        if parameters is None:
                            parameters = {k: v for k, v in tool_data.items() 
                                        if k not in ["name", "tool", "action", "function", "id", "type"]}
                        
                        tool_calls.append({
                            "id": f"re_match_{idx}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(parameters if parameters else {})
                            }
                        })
                        idx += 1
        return tool_calls
