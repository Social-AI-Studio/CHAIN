"""
Reward Agent for Process Step Selection in chainbench.

This module provides two selection methods for choosing the best action(s):
1. Reward Model (VLM): Score each action (1-5) based on its contribution to task completion
2. Pairwise Judge (LLM): Compare actions pairwise and rank using bubble/merge/full sort
"""

import json
import base64
from io import BytesIO
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from abc import abstractmethod
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from openai import OpenAI

from chainbench.agents.base_agent import VLMAgent
from chainbench.core.config import StepSelectionConfig, AgentConfig
from chainbench.core.base import Observation


@dataclass
class ActionCandidate:
    """Represents a candidate action with its metadata."""
    action_type: str
    parameters: Dict[str, Any]
    response: str  # The reasoning/explanation from the agent
    tool_call: Dict[str, Any]  # Original tool call format
    score: float = 0.0  # Reward score or comparison score
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "parameters": self.parameters,
            "response": self.response,
            "tool_call": self.tool_call,
            "score": self.score
        }


class BaseRewardAgent(VLMAgent):
    """
    Base class for reward agents that score and select actions.
    
    Inherits from VLMAgent to maintain consistent interface with other agents.
    Supports two selection methods:
    1. reward_model: VLM scores each action independently (1-5 scale)
    2. pairwise_judge: LLM compares pairs of actions, then ranks using sorting
    """
    
    def __init__(self, config: StepSelectionConfig):
        # Create an AgentConfig from StepSelectionConfig for parent class
        agent_config = self._create_agent_config(config)
        super().__init__(agent_config)
        self.step_selection_config = config
    
    @abstractmethod
    def _create_agent_config(self, config: StepSelectionConfig) -> AgentConfig:
        """Create AgentConfig from StepSelectionConfig for the specific implementation."""
        pass
    
    def select_best_actions(
        self,
        candidates: List[ActionCandidate],
        current_image: Image.Image,
        task_description: str,
        object_mapping: str,
        interaction_history: str
    ) -> List[ActionCandidate]:
        """
        Select the top-k best actions from candidates.
        
        Args:
            candidates: List of candidate actions to evaluate
            current_image: Current observation image
            task_description: Description of the task goal
            object_mapping: Object mapping information
            interaction_history: History of previous interactions
            
        Returns:
            List of top-k best actions, sorted by score (highest first)
        """
        if not candidates:
            return []
        
        if self.step_selection_config.selection_method == "reward_model":
            scored_candidates = self._score_with_reward_model(
                candidates, current_image, task_description, object_mapping, interaction_history
            )
        else:  # pairwise_judge
            scored_candidates = self._rank_with_pairwise_comparison(
                candidates, current_image, task_description, object_mapping, interaction_history
            )
        
        return scored_candidates
    
    def _score_with_reward_model(
        self,
        candidates: List[ActionCandidate],
        current_image: Image.Image,
        task_description: str,
        object_mapping: str,
        interaction_history: str
    ) -> List[ActionCandidate]:
        """Score each candidate action using reward model (1-5 scale)."""
        for candidate in candidates:
            score = self._get_reward_score(
                candidate, current_image, task_description, object_mapping, interaction_history
            )
            candidate.score = score
        return candidates
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        reraise=True
    )
    def _get_reward_score(
        self,
        candidate: ActionCandidate,
        current_image: Image.Image,
        task_description: str,
        object_mapping: str,
        interaction_history: str
    ) -> float:
        """Get reward score (1-5) for a single action from the reward model."""
        prompt = self._create_reward_prompt(
            candidate, task_description, object_mapping, interaction_history
        )
        return self._call_reward_model(prompt, current_image)
    
    @abstractmethod
    def _call_reward_model(self, prompt: str, image: Image.Image) -> float:
        """Call the reward model and return the score (1-5)."""
        pass
    
    def _create_reward_prompt(
        self,
        candidate: ActionCandidate,
        task_description: str,
        object_mapping: str,
        interaction_history: str
    ) -> str:
        """Create prompt for reward model scoring based on model mode."""
        # Check if discriminative or generative mode
        mode = self.step_selection_config.reward_model_mode
        
        if mode == "discriminative":
            # Discriminative prompt: binary +/- response
            prompt = f"""You are a process supervision model for evaluating actions in physics-based tasks.

## Task Description:
{task_description}

## Current Object Information:
{object_mapping}

## Previous Interaction History:
{interaction_history}

## Proposed Action:
- Action Type: {candidate.action_type}
- Parameters: {json.dumps(candidate.parameters, indent=2)}
- Agent's Reasoning: {candidate.response}

## Your Task:
Evaluate whether this action is correct and helpful for completing the task.

Consider:
1. Visual Accuracy: Are visual elements from the image correctly identified (shapes, colors, positions, quantities, spatial relationships)?
2. Physical Validity: Does this action make physical sense given the current state?
3. Task Progress: Does it move toward the task goal efficiently?
4. Logical Reasoning: Is the agent's reasoning sound and logical?
5. Risk Assessment: Are there any risks of making the task harder to complete?

Response:
• "+" if the action is correct and helpful for task completion
• "-" if the action has any errors or is unhelpful

Only respond with "+" or "-". No explanations."""
        else:
            # Generative prompt: 1-5 scale response
            prompt = f"""You are evaluating how helpful an action is for completing a physics-based task.

## Task Description:
{task_description}

## Current Object Information:
{object_mapping}

## Previous Interaction History:
{interaction_history}

## Proposed Action:
- Action Type: {candidate.action_type}
- Parameters: {json.dumps(candidate.parameters, indent=2)}
- Agent's Reasoning: {candidate.response}

## Your Task:
Evaluate this action on a scale of 1 to 5, where:
- 1: Action is harmful or completely irrelevant to the task
- 2: Action is unlikely to help with the task
- 3: Action might help but is not optimal
- 4: Action is good and likely to help complete the task
- 5: Action is excellent and directly advances toward task completion

Consider:
1. Does this action make physical sense given the current state?
2. Does it move toward the task goal efficiently?
3. Is the reasoning sound and logical?
4. Are there any risks of making the task harder to complete?

Respond with ONLY a JSON object in this exact format:
{{"score": <number between 1 and 5>, "reasoning": "<brief explanation>"}}"""
        
        return prompt
    
    def _parse_reward_score(self, content: str) -> float:
        """Parse reward score from model response (1-5 scale)."""
        import re
        try:
            # Try to parse as JSON
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
                data = json.loads(json_str)
                score = float(data.get("score", 3.0))
                return max(1.0, min(5.0, score))  # Clamp to [1, 5]
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        
        # Fallback: try to find a number in the response
        numbers = re.findall(r'\b([1-5](?:\.\d+)?)\b', content)
        for num_str in numbers:
            num = float(num_str)
            if 1 <= num <= 5:
                return num
        
        # Default score if parsing fails
        return 3.0
    
    def _rank_with_pairwise_comparison(
        self,
        candidates: List[ActionCandidate],
        current_image: Image.Image,
        task_description: str,
        object_mapping: str,
        interaction_history: str
    ) -> List[ActionCandidate]:
        """Rank candidates using pairwise comparison and sorting."""
        if len(candidates) <= 1:
            if candidates:
                candidates[0].score = 1.0
            return candidates
        
        # Use the configured sorting method
        sort_method = self.step_selection_config.pairwise_sort_method
        if sort_method == "bubble":
            return self._bubble_sort_by_comparison(
                candidates, current_image, task_description, object_mapping, interaction_history
            )
        elif sort_method == "merge":
            return self._merge_sort_by_comparison(
                candidates, current_image, task_description, object_mapping, interaction_history
            )
        else:  # full
            return self._full_sort_by_comparison(
                candidates, current_image, task_description, object_mapping, interaction_history
            )
    
    def _bubble_sort_by_comparison(
        self,
        candidates: List[ActionCandidate],
        current_image: Image.Image,
        task_description: str,
        object_mapping: str,
        interaction_history: str
    ) -> List[ActionCandidate]:
        """Sort candidates using bubble sort with pairwise comparisons."""
        n = len(candidates)
        sorted_candidates = candidates.copy()
        
        for i in range(n):
            for j in range(0, n - i - 1):
                winner = self._pairwise_compare(
                    sorted_candidates[j],
                    sorted_candidates[j + 1],
                    current_image,
                    task_description,
                    object_mapping,
                    interaction_history
                )
                
                # If right candidate is better, swap
                if winner == 2:
                    sorted_candidates[j], sorted_candidates[j + 1] = sorted_candidates[j + 1], sorted_candidates[j]
        
        # Assign scores based on position (higher position = higher score)
        for i, candidate in enumerate(sorted_candidates):
            candidate.score = (n - i) / n
        
        return sorted_candidates
    
    def _merge_sort_by_comparison(
        self,
        candidates: List[ActionCandidate],
        current_image: Image.Image,
        task_description: str,
        object_mapping: str,
        interaction_history: str
    ) -> List[ActionCandidate]:
        """Sort candidates using merge sort with pairwise comparisons."""
        if len(candidates) <= 1:
            return candidates
        
        # Split
        mid = len(candidates) // 2
        left = self._merge_sort_by_comparison(
            candidates[:mid], current_image, task_description, object_mapping, interaction_history
        )
        right = self._merge_sort_by_comparison(
            candidates[mid:], current_image, task_description, object_mapping, interaction_history
        )
        
        # Merge
        sorted_candidates = self._merge(
            left, right, current_image, task_description, object_mapping, interaction_history
        )
        
        # Assign scores based on position
        n = len(sorted_candidates)
        for i, candidate in enumerate(sorted_candidates):
            candidate.score = (n - i) / n
        
        return sorted_candidates
    
    def _merge(
        self,
        left: List[ActionCandidate],
        right: List[ActionCandidate],
        current_image: Image.Image,
        task_description: str,
        object_mapping: str,
        interaction_history: str
    ) -> List[ActionCandidate]:
        """Merge two sorted lists using pairwise comparison."""
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            winner = self._pairwise_compare(
                left[i], right[j],
                current_image, task_description, object_mapping, interaction_history
            )
            
            if winner == 1:  # left is better
                result.append(left[i])
                i += 1
            else:  # right is better or equal
                result.append(right[j])
                j += 1
        
        # Add remaining elements
        result.extend(left[i:])
        result.extend(right[j:])
        
        return result
    
    def _full_sort_by_comparison(
        self,
        candidates: List[ActionCandidate],
        current_image: Image.Image,
        task_description: str,
        object_mapping: str,
        interaction_history: str
    ) -> List[ActionCandidate]:
        """
        Full pairwise comparison: compare all pairs and compute total wins for global ranking.
        
        This method compares every pair of candidates and counts wins for each candidate.
        The final score is the win count normalized by total comparisons.
        
        Complexity: O(n^2) comparisons, but provides most accurate global ranking.
        """
        n = len(candidates)
        
        # Initialize win counts for each candidate
        win_counts = {i: 0 for i in range(n)}
        
        # Compare all pairs
        for i in range(n):
            for j in range(i + 1, n):
                winner = self._pairwise_compare(
                    candidates[i],
                    candidates[j],
                    current_image,
                    task_description,
                    object_mapping,
                    interaction_history
                )
                
                if winner == 1:  # candidates[i] wins
                    win_counts[i] += 1
                else:  # candidates[j] wins
                    win_counts[j] += 1
        
        # Assign scores based on win counts (normalized)
        max_possible_wins = n - 1  # Each candidate can win against n-1 others
        for i, candidate in enumerate(candidates):
            candidate.score = win_counts[i] / max_possible_wins if max_possible_wins > 0 else 0.0
        
        # Sort by win count (descending)
        sorted_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
        
        return sorted_candidates
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        reraise=True
    )
    def _pairwise_compare(
        self,
        action_a: ActionCandidate,
        action_b: ActionCandidate,
        current_image: Image.Image,
        task_description: str,
        object_mapping: str,
        interaction_history: str
    ) -> int:
        """
        Compare two actions and determine which is better.
        
        Returns:
            1 if action_a is better, 2 if action_b is better
        """
        prompt = self._create_pairwise_prompt(
            action_a, action_b, task_description, object_mapping, interaction_history
        )
        return self._call_pairwise_judge(prompt, current_image)
    
    @abstractmethod
    def _call_pairwise_judge(self, prompt: str, image: Image.Image) -> int:
        """Call the pairwise judge and return 1 or 2."""
        pass
    
    def _create_pairwise_prompt(
        self,
        action_a: ActionCandidate,
        action_b: ActionCandidate,
        task_description: str,
        object_mapping: str,
        interaction_history: str
    ) -> str:
        """Create prompt for pairwise comparison."""
        prompt = f"""You are comparing two proposed actions for a physics-based task. Determine which action is more likely to help complete the task.

## Task Description:
{task_description}

## Current Object Information:
{object_mapping}

## Previous Interaction History:
{interaction_history}

## Action A:
- Type: {action_a.action_type}
- Parameters: {json.dumps(action_a.parameters, indent=2)}
- Reasoning: {action_a.response}

## Action B:
- Type: {action_b.action_type}
- Parameters: {json.dumps(action_b.parameters, indent=2)}
- Reasoning: {action_b.response}

## Your Task:
Compare these two actions and decide which one is BETTER for completing the task.

Consider:
1. Physical feasibility and correctness
2. Progress toward the task goal
3. Efficiency of the action
4. Risk of making the task harder

Respond with ONLY a JSON object in this exact format:
{{"winner": <1 or 2>, "reasoning": "<brief explanation>"}}

where winner=1 means Action A is better, winner=2 means Action B is better."""
        
        return prompt
    
    def _parse_pairwise_result(self, content: str) -> int:
        """Parse pairwise comparison result from model response."""
        try:
            # Try to parse as JSON
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
                data = json.loads(json_str)
                winner = int(data.get("winner", 1))
                if winner in [1, 2]:
                    return winner
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        
        # Fallback: look for explicit mentions
        content_lower = content.lower()
        if "action a" in content_lower and "better" in content_lower:
            return 1
        if "action b" in content_lower and "better" in content_lower:
            return 2
        if "winner: 1" in content_lower or "winner\":1" in content_lower or "winner\": 1" in content_lower:
            return 1
        if "winner: 2" in content_lower or "winner\":2" in content_lower or "winner\": 2" in content_lower:
            return 2
        
        # Default to action A if parsing fails
        return 1
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def _image_to_data_url(self, image: Image.Image) -> str:
        """Convert PIL Image to data URL."""
        img_base64 = self._image_to_base64(image)
        return f"data:image/png;base64,{img_base64}"
    
    # Implement abstract methods from VLMAgent (not used directly but required)
    def _prepare_messages(self, history: List[Observation], prompt: str) -> List[Dict[str, Any]]:
        """Prepare message list for model API call."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        content = [{"type": "text", "text": prompt}]
        for observation in history:
            if observation.image:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": self._image_to_data_url(observation.image)}
                })
        
        messages.append({"role": "user", "content": content})
        return messages
    
    def _get_model_response(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """Get response from the specific model implementation."""
        # This method is not used for reward scoring, but required by base class
        raise NotImplementedError("RewardAgent does not use standard model response interface")


class OpenAIRewardAgent(BaseRewardAgent):
    """Reward agent using OpenAI API (or compatible APIs)."""
    
    def __init__(self, config: StepSelectionConfig):
        super().__init__(config)
        
        # Initialize OpenAI client
        
        # Use reward model config for reward scoring
        self.reward_client = OpenAI(
            api_key=config.reward_api_key,
            base_url=config.reward_base_url
        )
        
        # Use pairwise judge config for pairwise comparison
        self.judge_client = OpenAI(
            api_key=config.pairwise_api_key,
            base_url=config.pairwise_base_url
        )
    
    def _create_agent_config(self, config: StepSelectionConfig) -> AgentConfig:
        """Create AgentConfig from StepSelectionConfig."""
        return AgentConfig(
            type="openai",
            model_name=config.reward_model_name,
            api_key=config.reward_api_key,
            base_url=config.reward_base_url,
            temperature=config.reward_temperature,
            max_tokens=config.reward_max_tokens,
        )
    
    def _call_reward_model(self, prompt: str, image: Image.Image) -> float:
        """Call OpenAI API for reward scoring."""
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

        response = self.reward_client.chat.completions.create(
            model=self.step_selection_config.reward_model_name,
            messages=messages,
            temperature=self.step_selection_config.reward_temperature,
            max_tokens=self.step_selection_config.reward_max_tokens,
        )
        content = response.choices[0].message.content.strip()
        return self._parse_reward_score(content)
    
    def _call_pairwise_judge(self, prompt: str, image: Image.Image) -> int:
        """Call OpenAI API for pairwise comparison."""
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

        response = self.judge_client.chat.completions.create(
            model=self.step_selection_config.pairwise_judge_model,
            messages=messages,
            temperature=self.step_selection_config.pairwise_temperature,
            max_tokens=self.step_selection_config.pairwise_max_tokens,
        )
        
        content = response.choices[0].message.content.strip()
        return self._parse_pairwise_result(content)


class TransformersRewardAgent(BaseRewardAgent):
    """Reward agent using HuggingFace Transformers for local model inference."""
    
    def __init__(self, config: StepSelectionConfig):
        super().__init__(config)
        
        # Lazy load models
        self._reward_model = None
        self._reward_processor = None
        self._judge_model = None
        self._judge_processor = None

        self.use_api = False
    
    def _create_agent_config(self, config: StepSelectionConfig) -> AgentConfig:
        """Create AgentConfig from StepSelectionConfig."""
        return AgentConfig(
            type="transformers",
            model_name=config.reward_model_name,
            api_key="dummy",  # Not used for transformers
            base_url="dummy",  # Not used for transformers
            temperature=config.reward_temperature,
            max_tokens=config.reward_max_tokens,
            device=config.reward_device,
            torch_dtype=config.reward_torch_dtype,
        )
    
    def _load_reward_model(self) -> None:
        """Lazy load reward model."""
        if self.use_api:
            return
        if self._reward_model is None:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText
            
            config = self.step_selection_config
            
            self._reward_processor = AutoProcessor.from_pretrained(config.reward_model_name)
            if self._reward_processor.tokenizer.pad_token is None:
                self._reward_processor.tokenizer.pad_token = self._reward_processor.tokenizer.eos_token
            
            torch_dtype = getattr(torch, config.reward_torch_dtype) if config.reward_torch_dtype != "auto" else "auto"
            self._reward_model = AutoModelForImageTextToText.from_pretrained(
                config.reward_model_name,
                trust_remote_code=True,
                device_map="auto" if config.reward_device == "auto" else None,
                torch_dtype=torch_dtype,
            ).eval()
            
            if config.reward_device != "auto" and not hasattr(self._reward_model, 'hf_device_map'):
                self._reward_model = self._reward_model.to(config.reward_device)
    
    def _load_judge_model(self) -> None:
        """Lazy load judge model."""
        if self.use_api:
            return
        if self._judge_model is None:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText
            
            config = self.step_selection_config
            
            self._judge_processor = AutoProcessor.from_pretrained(config.pairwise_judge_model)
            if self._judge_processor.tokenizer.pad_token is None:
                self._judge_processor.tokenizer.pad_token = self._judge_processor.tokenizer.eos_token
            
            torch_dtype = getattr(torch, config.pairwise_torch_dtype) if config.pairwise_torch_dtype != "auto" else "auto"
            self._judge_model = AutoModelForImageTextToText.from_pretrained(
                config.pairwise_judge_model,
                trust_remote_code=True,
                device_map="auto" if config.pairwise_device == "auto" else None,
                torch_dtype=torch_dtype,
            ).eval()
            
            if config.pairwise_device != "auto" and not hasattr(self._judge_model, 'hf_device_map'):
                self._judge_model = self._judge_model.to(config.pairwise_device)
    def _pil_to_data_url(self, image, fmt: str = "PNG") -> str:
        import base64, io
        buf = io.BytesIO()
        image.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/{fmt.lower()};base64,{b64}"

    def _friendli_chat(self, messages, max_tokens: int, temperature: float, top_p: float = 0.95, timeout: int = 300) -> str:
        import os, requests

        token = os.environ.get("FRIENDLI_TOKEN")
        if not token:
            raise RuntimeError("FRIENDLI_TOKEN is not set in environment variables.")

        url = "https://api.friendli.ai/dedicated/v1/chat/completions"

        # 你在 Friendli Dedicated 上部署后的 model_id（强烈建议放 config）
        model_id = getattr(self.step_selection_config, "friendli_model_id", None) or os.environ.get("FRIENDLI_MODEL_ID")
        if not model_id:
            raise RuntimeError("Friendli model_id not found. Set step_selection_config.friendli_model_id or env FRIENDLI_MODEL_ID.")

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        try:
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Friendli API error: {e}, body={resp.text[:2000]}") from e

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            raise RuntimeError(f"Unexpected Friendli response format: {data}")


    def _parse_plus_minus_from_text(self, text: str) -> str:
        """
        从模型输出里抽取 + 或 -。
        优先匹配“独立 token”的 +/-；匹配不到再退化为找第一个出现的 +/-。
        """
        import re

        t = (text or "").strip()

        # 1) 独立 token：比如 "+" / "-" / " + " / "\n-\n"
        m = re.search(r'(?<!\S)([+-])(?!\S)', t)
        if m:
            return m.group(1)

        # 2) 退化：找第一个出现的 +/- 字符
        m = re.search(r'[+-]', t)
        if m:
            return m.group(0)

        raise ValueError(f"Cannot find '+' or '-' in model output: {text!r}")
    
    def _call_reward_model(self, prompt: str, image: Image.Image) -> float:
        """Call Transformers model for reward scoring."""
        if self.use_api:
            mode = self.step_selection_config.reward_model_mode
            if mode == "discriminative":
                # Friendli 走 image_url(data url) + text
                data_url = self._pil_to_data_url(image, fmt="PNG")

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

                out_text = self._friendli_chat(
                    messages=messages,
                    max_tokens=self.step_selection_config.reward_max_tokens,          # discriminative 一般很短
                    temperature=0.0,       # 稳定输出
                )

                sign = self._parse_plus_minus_from_text(out_text)
                return 5.0 if sign == "+" else 1.0
        import torch
        import torch.nn.functional as F
        
        self._load_reward_model()
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = self._reward_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = self._reward_processor(
            images=[image],
            text=text,
            return_tensors="pt"
        ).to(self._reward_model.device)
        
        # Check if discriminative or generative mode
        mode = self.step_selection_config.reward_model_mode
        
        if mode == "discriminative":
            # For discriminative mode: get logits of + and - tokens
            with torch.inference_mode():
                outputs = self._reward_model(**inputs)
                logits = outputs.logits
                
                # Get the logits for the next token position
                next_token_logits = logits[0, -1, :]
                
                # Get token IDs for "+" and "-"
                plus_token_id = self._reward_processor.tokenizer.encode("+", add_special_tokens=False)[0]
                minus_token_id = self._reward_processor.tokenizer.encode("-", add_special_tokens=False)[0]
                
                # Extract logits for + and - tokens
                plus_logit = next_token_logits[plus_token_id].item()
                minus_logit = next_token_logits[minus_token_id].item()
                
                # Apply softmax to get probabilities
                logits_tensor = torch.tensor([plus_logit, minus_logit])
                probs = F.softmax(logits_tensor, dim=0)
                
                # Return probability of + token as reward score (0-1 range)
                # Convert to 1-5 scale for consistency
                prob_plus = probs[0].item()
                return 1.0 + prob_plus * 4.0  # Map [0, 1] to [1, 5]
        else:
            # For generative mode: generate text and parse score
            with torch.inference_mode():
                outputs = self._reward_model.generate(
                    **inputs,
                    max_new_tokens=self.step_selection_config.reward_max_tokens,
                    temperature=self.step_selection_config.reward_temperature,
                    do_sample=self.step_selection_config.reward_temperature > 0,
                    pad_token_id=self._reward_processor.tokenizer.pad_token_id,
                    eos_token_id=self._reward_processor.tokenizer.eos_token_id,
                )
            
            generated_text = self._reward_processor.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return self._parse_reward_score(generated_text.strip())
    
    def _call_pairwise_judge(self, prompt: str, image: Image.Image) -> int:
        """Call Transformers model for pairwise comparison."""
        import torch
        
        self._load_judge_model()
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = self._judge_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = self._judge_processor(
            images=[image],
            text=text,
            return_tensors="pt"
        ).to(self._judge_model.device)
        
        with torch.inference_mode():
            outputs = self._judge_model.generate(
                **inputs,
                max_new_tokens=self.step_selection_config.pairwise_max_tokens,
                temperature=self.step_selection_config.pairwise_temperature,
                do_sample=self.step_selection_config.pairwise_temperature > 0,
                pad_token_id=self._judge_processor.tokenizer.pad_token_id,
                eos_token_id=self._judge_processor.tokenizer.eos_token_id,
            )
        
        generated_text = self._judge_processor.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return self._parse_pairwise_result(generated_text.strip())


def create_reward_agent(config: StepSelectionConfig) -> BaseRewardAgent:
    """
    Factory function to create the appropriate reward agent based on configuration.
    
    Args:
        config: StepSelectionConfig with model type and settings
        
    Returns:
        Appropriate reward agent instance (OpenAI or Transformers)
    """
    # Determine which model type to use based on selection method
    if config.selection_method == "reward_model":
        model_type = config.reward_model_type
    else:  # pairwise_judge
        model_type = config.pairwise_judge_type
    
    if model_type == "openai":
        return OpenAIRewardAgent(config)
    elif model_type == "transformers":
        return TransformersRewardAgent(config)
    else:
        raise ValueError(f"Unknown reward model type: {model_type}")


class StepSelectionManager:
    """
    Manages step selection state using beam search.
    
    - top_k = 1: Greedy search (maintains single best path)
    - top_k > 1: Beam search (maintains multiple parallel paths)
    """
    
    def __init__(self, config: StepSelectionConfig, reward_agent: BaseRewardAgent):
        self.config = config
        self.reward_agent = reward_agent
        
        # Beam state: list of (path_history, cumulative_score) tuples
        # Each path_history is a list of ActionCandidates
        self.beams: List[Tuple[List[ActionCandidate], float]] = []
        
    def initialize(self) -> None:
        """Initialize step selection with empty beam."""
        self.beams = [([], 0.0)]
    
    def expand_beams(
        self,
        candidate_generator_fn,  # Function that generates candidates
        current_image: Image.Image,
        task_description: str,
        object_mapping: str,
        interaction_history: str
    ) -> List[ActionCandidate]:
        """
        Expand all beams with new candidate actions using beam search.
        
        Process:
        1. For each current beam, generate num_candidates actions
        2. Score all candidates using reward model or pairwise judge
        3. Expand each beam with its top-scored actions
        4. Keep top_k best beams globally (beam width)
        5. Return the best action from the best beam
        
        Args:
            candidate_generator_fn: Function that returns List[ActionCandidate]
            current_image: Current observation
            task_description: Task description
            object_mapping: Object mapping info
            interaction_history: Interaction history string
            
        Returns:
            Best action to execute this step (from best beam)
        """
        all_expansions = []
        
        # Expand each current beam
        for beam_idx, (path, cum_score) in enumerate(self.beams):
            # Generate candidates for this beam
            candidates = candidate_generator_fn()
            
            if not candidates:
                continue
            
            # Score all candidates
            scored = self.reward_agent.select_best_actions(
                candidates,
                current_image,
                task_description,
                object_mapping,
                interaction_history
            )
            
            # Create new beam entries for each scored candidate
            # (not just top_k, we'll select globally later)
            for candidate in scored:
                new_path = path + [candidate]
                new_score = cum_score + candidate.score
                all_expansions.append((new_path, new_score, candidate))
        
        if not all_expansions:
            return []
        
        # Sort all expansions by cumulative score (descending)
        all_expansions.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top_k beams (beam width)
        self.beams = [(path, score) for path, score, _ in all_expansions[:self.config.top_k]]
        
        # Return the action from the best beam (to execute this step)
        best_action = all_expansions[0][2]
        return [best_action]
    
    def get_best_path(self) -> Tuple[List[ActionCandidate], float]:
        """Get the current best path and its score."""
        if not self.beams:
            return [], 0.0
        return max(self.beams, key=lambda x: x[1])


# Keep backward compatibility aliases
RewardAgent = BaseRewardAgent
BeamSearchManager = StepSelectionManager
