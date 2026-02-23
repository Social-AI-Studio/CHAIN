"""
LLM-as-judge evaluation system for chainbench.
"""

import json
from typing import Tuple, List, Dict, Any
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from PIL import Image
import base64
from io import BytesIO
from transformers import AutoModelForCausalLM, AutoTokenizer
from chainbench.core.base import BaseJudge
from chainbench.core import JudgementConfig
import tiktoken
import re

class LLMJudge(BaseJudge):
    """LLM-based judge for evaluating task completion."""
    
    def __init__(self, config: JudgementConfig):
        super().__init__(config)
        
        # Initialize OpenAI client
        if config.type == "openai":
            self.client = OpenAI(
                api_key=config.api_key,
                base_url=config.base_url
            )
            try:
                self.tokenizer = tiktoken.encoding_for_model(config.model_name)
            except KeyError:
                self.tokenizer = tiktoken.get_encoding("o200k_base")
        elif config.type == "transformers":
            # For transformers, we expect a model and tokenizer to be provided in config
            self.clinet = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype="auto", device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        else:
            raise ValueError(f"Unsupported judgement type: {config.type}")
        
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.timeout = config.timeout
                
    def judge_success(self, final_image: Image.Image, task_description: str, 
                     trajectory: List[str]) -> Tuple[bool, float, str]:
        """
        Judge if task was completed successfully.
        
        Args:
            final_image: Final state image
            task_description: Description of the task
            trajectory: List of action descriptions
            
        Returns:
            Tuple of (success, confidence_score, reasoning)
        """
        # Prepare the evaluation prompt
        evaluation_prompt = self._create_evaluation_prompt(task_description, trajectory)
        
        # Get judgment from LLM
        try:
            response = self._get_judge_response(final_image, evaluation_prompt)
            return self._parse_judge_response(response)
        except Exception as e:
            print(f"Judge evaluation failed: {e}")
            return False, 0.0, f"Evaluation failed: {e}"
            
    def _create_evaluation_prompt(self, task_description: str, trajectory: List[str]) -> str:
        """Create evaluation prompt for the judge."""
        trajectory_text = "\n".join(trajectory) if trajectory else "No actions recorded"
        
        prompt = f"""You are an expert evaluator for physics puzzle tasks. Your job is to determine if the task was completed successfully based on the final state image.

TASK DESCRIPTION:
{task_description}

ACTION TRAJECTORY:
{trajectory_text}

Please analyze the final state image and determine if the puzzle task was completed successfully. 

For your evaluation, consider:
1. Whether the puzzle objective has been achieved (e.g., pieces disassembled, box fully packed, etc.)
2. The stability and correctness of the final configuration
3. Whether the result matches the expected outcome for this type of puzzle

Respond with a JSON object containing:
- "success": boolean (true if task completed successfully, false otherwise)
- "confidence": float between 0.0 and 1.0 (how confident you are in your judgment)  
- "reasoning": string (detailed explanation of your decision)

Example response format:
{{"success": true, "confidence": 0.95, "reasoning": "The target piece has been successfully separated from the interlocking puzzle structure."}}

Your response:"""
        
        return prompt
        
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        reraise=True
    )
    def _get_judge_response(self, image: Image.Image, prompt: str) -> str:
        """Get response from judge LLM."""
        # Convert image to base64
        image_data = self._image_to_base64(image)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"}
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        if self.config.type == "openai":
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout
            )
            content = response.choices[0].message.content
        elif self.config.type == "transformers":
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.clinet.device)
            generated_ids = self.clinet.generate(
                **model_inputs,
                max_length=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        else:
            raise ValueError(f"Unsupported judge backend: {self.config.type}")
        
        return content
        
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
        
    def _parse_judge_response(self, response: str) -> Tuple[bool, float, str]:
        """Parse the judge's response into structured format."""
        try:
            # Remove BOM and strip whitespace
            response = response.strip().lstrip("\ufeff")
            
            # Try to extract JSON block inside triple backticks or standalone
            code_block_match = re.search(r"```(?:json)?\s*({.*})\s*```", response, re.DOTALL|re.IGNORECASE)
            if code_block_match:
                json_str = code_block_match.group(1).strip()
            else:
                # Try to find first {...} JSON object in the text
                json_match = re.search(r"\{[\s\S]*\}", response)
                if json_match:
                    json_str = json_match.group(0).strip()
                else:
                    json_str = response  # maybe pure JSON
            
            # Parse JSON
            judgment = json.loads(json_str)
            
            success = bool(judgment.get("success", False))
            confidence = float(judgment.get("confidence", 0.0))
            # clamp to [0, 1]
            confidence = max(0.0, min(1.0, confidence))
            
            reasoning = judgment.get("reasoning", "No reasoning provided")
            return success, confidence, reasoning
        
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback parsing for non-JSON responses
            response_lower = response.lower()
            
            if "success" in response_lower and "true" in response_lower:
                success = True
            elif "success" in response_lower and "false" in response_lower:
                success = False
            else:
                # Try to infer from keywords
                positive_keywords = ["completed", "successful", "solved", "correct", "achieved"]
                negative_keywords = ["failed", "incomplete", "unsuccessful", "wrong", "not solved"]
                
                positive_count = sum(1 for word in positive_keywords if word in response_lower)
                negative_count = sum(1 for word in negative_keywords if word in response_lower)
                
                success = positive_count > negative_count
                
            confidence = 0.5  # default moderate confidence
            
            # Try to extract confidence if mentioned
            conf_match = re.search(r'confidence[^\d]*(\d+(?:\.\d+)?)', response_lower)
            if conf_match:
                try:
                    conf_value = float(conf_match.group(1))
                    if conf_value <= 1.0:
                        confidence = conf_value
                    elif conf_value <= 100:
                        confidence = conf_value / 100.0
                except ValueError:
                    pass
            
            reasoning = f"Parsed from non-JSON response: {response[:200]}..."
            return success, confidence, reasoning
            
    def judge_multiple_tasks(self, task_data: List[Dict[str, Any]]) -> List[Tuple[bool, float, str]]:
        """Judge multiple tasks in batch."""
        results = []
        
        for task in task_data:
            final_image = task.get("final_image")
            task_description = task.get("task_description", "")
            trajectory = task.get("trajectory", [])
            
            if final_image:
                result = self.judge_success(final_image, task_description, trajectory)
                results.append(result)
            else:
                results.append((False, 0.0, "No final image provided"))
                
        return results
        
    def get_judge_statistics(self, judgments: List[Tuple[bool, float, str]]) -> Dict[str, Any]:
        """Get statistics about judge evaluations."""
        if not judgments:
            return {}
            
        success_count = sum(1 for success, _, _ in judgments if success)
        avg_confidence = sum(conf for _, conf, _ in judgments) / len(judgments)
        
        confidence_by_success = {
            "successful_tasks": [conf for success, conf, _ in judgments if success],
            "failed_tasks": [conf for success, conf, _ in judgments if not success]
        }
        
        return {
            "total_judgments": len(judgments),
            "success_rate": success_count / len(judgments),
            "average_confidence": avg_confidence,
            "confidence_distribution": {
                "successful_avg": sum(confidence_by_success["successful_tasks"]) / len(confidence_by_success["successful_tasks"]) if confidence_by_success["successful_tasks"] else 0,
                "failed_avg": sum(confidence_by_success["failed_tasks"]) / len(confidence_by_success["failed_tasks"]) if confidence_by_success["failed_tasks"] else 0
            }
        }
