"""
Agent implementations for VLM models.

This package contains different agent implementations that can process
visual observations and generate actions, including:
- OpenAI API-based agents
- Local model agents
- Reward agents for process step selection
"""

from chainbench.agents.base_agent import VLMAgent
from chainbench.agents.openai_agent import OpenAIAgent
from chainbench.agents.transformers_agent import TransformersAgent
from chainbench.agents.human_agent import HumanAgent
from chainbench.agents.reward_agent import (
    BaseRewardAgent,
    OpenAIRewardAgent,
    TransformersRewardAgent,
    ActionCandidate,
    StepSelectionManager,
    create_reward_agent,
    # Backward compatibility aliases
    RewardAgent,
    BeamSearchManager,
)

__all__ = [
    "VLMAgent",
    "OpenAIAgent", 
    "TransformersAgent",
    "HumanAgent",
    "BaseRewardAgent",
    "OpenAIRewardAgent",
    "TransformersRewardAgent",
    "ActionCandidate",
    "StepSelectionManager",
    "create_reward_agent",
    # Backward compatibility
    "RewardAgent",
    "BeamSearchManager",
]
