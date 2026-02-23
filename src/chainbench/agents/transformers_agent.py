"""
HuggingFace Transformers-based agent implementation for local model inference.
"""

import json
import torch
from typing import Any, Dict, List, Tuple
from transformers import AutoProcessor, AutoModel
from PIL import Image
import base64
from io import BytesIO

from chainbench.agents import VLMAgent
from chainbench.core import register_agent, AgentConfig
from chainbench.core.base import Observation

@register_agent("transformers") 
class TransformersAgent(VLMAgent):
    """Agent using Transformers for local model inference."""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.model = None
        self.processor = None
        self.device = config.device

        # Load model and tokenizer
        self._load_model_and_tokenizer()
        
    def _load_model_and_tokenizer(self) -> None:
        """Load model and tokenizer from HuggingFace."""
        try:
            print(f"Loading model: {self.config.model_name}")
                    
            self.processor = AutoProcessor.from_pretrained(self.config.model_name)
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

            self.model = AutoModel.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=getattr(torch, self.config.torch_dtype),
            ).eval()
            print("Loaded as vision-capable model")

            # Move to device if not using device_map
            if self.device != "cuda" or not hasattr(self.model, 'hf_device_map'):
                self.model = self.model.to(self.device)

            print(f"Model loaded successfully on {self.device}")

        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.config.model_name}: {e}")
        
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
                    "type": "image",
                    "image": observation.image,
                })
            
        messages.append({
            "role": "user",
            "content": content
        })
        
        return messages
    
    def _get_model_response(self, messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Get response from local Transformers model."""
        # Handle vision vs text-only models differently
        return self._get_vision_model_response(messages)

    def _get_vision_model_response(self, messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Get response from vision-capable model."""
        try:
            # Process with processor
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.processor(
                images=[msg["image"] for msg in messages if msg["type"] == "image"],
                text=text, 
                return_tensors="pt"
            ).to(self.device)

            # Generate response
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    do_sample=self.config.temperature > 0,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True
                )

            # Decode response
            generated_text = self.processor.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            text_response = generated_text.strip()

            # Parse tool calls from text
            tool_calls = self.parse_tool_calls_from_text(text_response)
            
            return text_response, tool_calls
            
        except Exception as e:
            print(f"Error in vision model response: {e}")
            return f"Error: {e}", []
    
    def parse_tool_calls_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract tool calls from model text response."""
        tool_calls = []
        
        # Look for function call patterns
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Pattern 1: Function call format
            if line.startswith("Function:") or line.startswith("Tool:") or line.startswith("Action:"):
                try:
                    # Extract function name and arguments
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        func_part = parts[1].strip()
                        
                        # Look for function_name(arguments) format
                        if '(' in func_part and ')' in func_part:
                            func_name = func_part.split('(')[0].strip()
                            args_str = func_part[func_part.find('(')+1:func_part.rfind(')')].strip()
                            
                            # Parse arguments
                            args = self._parse_function_args(args_str)
                            
                            tool_calls.append({
                                "id": f"call_{len(tool_calls)}",
                                "type": "function",
                                "function": {
                                    "name": func_name,
                                    "arguments": json.dumps(args)
                                }
                            })
                            
                except Exception as e:
                    print(f"Error parsing function call: {e}")
                    continue
            
            # Pattern 2: JSON format
            elif line.startswith('{') and ('"function"' in line or '"action"' in line):
                try:
                    call_data = json.loads(line)
                    
                    func_name = call_data.get("function") or call_data.get("action")
                    args = call_data.get("arguments") or call_data.get("parameters", {})
                    
                    if func_name:
                        tool_calls.append({
                            "id": f"call_{len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": func_name,
                                "arguments": json.dumps(args) if isinstance(args, dict) else args
                            }
                        })
                        
                except Exception as e:
                    print(f"Error parsing JSON function call: {e}")
                    continue
        
        return tool_calls
    
    def _parse_function_args(self, args_str: str) -> Dict[str, Any]:
        """Parse function arguments from string."""
        args = {}
        
        if not args_str.strip():
            return args
        
        try:
            # Try JSON parsing first
            if args_str.startswith('{') and args_str.endswith('}'):
                return json.loads(args_str)
            
            # Try simple key=value parsing
            for arg in args_str.split(','):
                if '=' in arg:
                    key, val = arg.split('=', 1)
                    key = key.strip().strip("\"'")
                    val = val.strip().strip("\"'")
                    
                    # Try to convert to appropriate type
                    try:
                        # Try number conversion
                        if '.' in val:
                            args[key] = float(val)
                        else:
                            args[key] = int(val)
                    except ValueError:
                        # Keep as string
                        args[key] = val
                        
        except Exception as e:
            print(f"Error parsing function arguments '{args_str}': {e}")
            
        return args
