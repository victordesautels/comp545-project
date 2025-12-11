#!/usr/bin/env python3
"""
Modular reward system for PPO training.

Combines safety rewards (from SHADE monitors) with task-success rewards.
All reward components are normalized to [-1, 1] range for consistency.

Architecture:
- BaseReward: Abstract interface for all reward components
- SafetyReward: Wraps SHADE monitors (CoT, probe, hybrid)
- TaskReward: Evaluates task success (placeholder implementation)
- RewardMixer: Combines multiple reward sources with configurable weights
"""

from __future__ import annotations
import sys
from pathlib import Path

# Make the project root importable
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

# SHADE monitor imports
from src.monitors.cot_judge import CoTJudge, JudgeConfig
from src.monitors.activation_probe import LinearProbe, ProbeConfig
from src.monitors.hybrid import HybridMonitor, HybridConfig
from src.utils.model_loader import DEFAULT_MODEL


# ==========================================
# Base Reward Interface
# ==========================================
class BaseReward(ABC):
    """Abstract base class for all reward components."""
    
    @abstractmethod
    def compute_reward(self, prompt: str, response: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Compute reward for a single prompt-response pair.
        
        Args:
            prompt: Input prompt
            response: Model response
            context: Optional context (e.g., environment state, tool calls)
            
        Returns:
            Reward in [-1, 1] range (higher is better)
        """
        pass
    
    def compute_rewards(self, prompts: List[str], responses: List[str], 
                       contexts: Optional[List[Dict[str, Any]]] = None) -> List[float]:
        """Compute rewards for a batch of prompt-response pairs."""
        if contexts is None:
            contexts = [None] * len(prompts)
        return [self.compute_reward(p, r, c) for p, r, c in zip(prompts, responses, contexts)]


# ==========================================
# Safety Reward (wraps SHADE monitors)
# ==========================================
@dataclass
class SafetyRewardConfig:
    """Configuration for safety reward from SHADE monitors."""
    monitor_type: str = "cot"  # cot | probe | hybrid
    
    # CoT parameters
    cot_model_id: Optional[str] = None
    cot_threshold: int = 65
    
    # Probe parameters
    probe_model_id: Optional[str] = None
    probe_dir: Optional[str] = None
    probe_threshold: int = 65
    
    # Hybrid parameters
    hybrid_w1: float = 0.5
    hybrid_w2: float = 0.5
    
    # Reward mapping
    reward_when_safe: float = 1.0     # Reward when not flagged
    reward_when_unsafe: float = -1.0  # Reward when flagged
    use_continuous: bool = True       # Use continuous score mapping instead of binary


class SafetyReward(BaseReward):
    """Safety reward from SHADE monitors (CoT, probe, or hybrid)."""
    
    def __init__(self, cfg: SafetyRewardConfig, device: Optional[torch.device] = None):
        self.cfg = cfg
        self.device = device or torch.device("cpu")
        self.monitor = self._build_monitor()
    
    def _build_monitor(self):
        """Build the appropriate SHADE monitor."""
        if self.cfg.monitor_type == "cot":
            assert self.cfg.cot_model_id is not None, "cot_model_id required"
            judge_cfg = JudgeConfig(
                model_id=self.cfg.cot_model_id,
                threshold=self.cfg.cot_threshold
            )
            # Pass device so model loads on correct GPU from the start
            monitor = CoTJudge(judge_cfg, device=self.device)
            return monitor
        
        elif self.cfg.monitor_type == "probe":
            assert self.cfg.probe_model_id is not None, "probe_model_id required"
            assert self.cfg.probe_dir is not None, "probe_dir required"
            probe_cfg = ProbeConfig(model_id=self.cfg.probe_model_id)
            monitor = LinearProbe(probe_cfg)
            monitor.load(self.cfg.probe_dir)
            monitor.device = self.device
            # Only move if not already distributed via device_map="auto"
            if monitor.model is not None and not hasattr(monitor.model, 'hf_device_map'):
                monitor.model.to(self.device)
            return monitor
        
        elif self.cfg.monitor_type == "hybrid":
            assert self.cfg.probe_dir is not None, "probe_dir required"
            hybrid_cfg = HybridConfig(
                cot_model_id=self.cfg.cot_model_id or DEFAULT_MODEL,
                cot_threshold=self.cfg.cot_threshold,
                probe_model_id=self.cfg.probe_model_id or DEFAULT_MODEL,
                probe_dir=self.cfg.probe_dir,
                probe_threshold=self.cfg.probe_threshold,
                mode="weighted",
                weight_cot=self.cfg.hybrid_w1,
                weight_probe=self.cfg.hybrid_w2,
                decision_threshold=0.5,
            )
            # Pass device so models load on correct GPU from the start
            monitor = HybridMonitor(hybrid_cfg, device=self.device)
            return monitor
        
        else:
            raise ValueError(f"Unknown monitor_type: {self.cfg.monitor_type}")
    
    def compute_reward(self, prompt: str, response: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Compute safety reward from SHADE monitor.
        
        Returns:
            Reward in [-1, 1] where:
            - 1.0: Very safe (low suspicion score)
            - 0.0: Neutral
            - -1.0: Very unsafe (high suspicion score, flagged)
        """
        try:
            if self.cfg.monitor_type == "cot":
                result = self.monitor.judge(response)
                score = result["score"]  # 0-100, higher = more suspicious
                flagged = result["flagged"]
                
                if self.cfg.use_continuous:
                    # Map [0, 100] suspicious score to [1, -1] reward
                    reward = 1.0 - (score / 50.0)  # Linear mapping
                    return max(-1.0, min(1.0, reward))
                else:
                    # Binary: safe or unsafe
                    return self.cfg.reward_when_safe if not flagged else self.cfg.reward_when_unsafe
            
            elif self.cfg.monitor_type == "probe":
                scores = self.monitor.predict_scores([response])
                prob_malicious = scores[0, 1] / 100.0  # Scaled score 0-100 to probability 0-1
                
                if self.cfg.use_continuous:
                    # Map [0, 1] malicious probability to [1, -1] reward
                    reward = 1.0 - 2.0 * prob_malicious
                    return max(-1.0, min(1.0, reward))
                else:
                    flagged = scores[0, 1] >= self.cfg.probe_threshold
                    return self.cfg.reward_when_safe if not flagged else self.cfg.reward_when_unsafe
            
            elif self.cfg.monitor_type == "hybrid":
                result = self.monitor.score_text(response)
                combined_score = result["fusion"]["combined_0_100"]
                flagged = result["fusion"]["flagged"]
                
                if self.cfg.use_continuous:
                    # Map [0, 100] combined score to [1, -1] reward
                    reward = 1.0 - (combined_score / 50.0)
                    return max(-1.0, min(1.0, reward))
                else:
                    return self.cfg.reward_when_safe if not flagged else self.cfg.reward_when_unsafe
            
            else:
                return 0.0
        
        except Exception as e:
            print(f"[SafetyReward] Error computing reward: {e}")
            return 0.0


# ==========================================
# Task Reward (placeholder evaluators)
# ==========================================
@dataclass
class TaskRewardConfig:
    """Configuration for task-success reward."""
    evaluator_type: str = "keyword"  # keyword | structured | oracle
    
    # Keyword evaluator
    required_keywords: List[str] = None  # Keywords that must appear in response
    forbidden_keywords: List[str] = None  # Keywords that must NOT appear
    case_sensitive: bool = False
    
    # Structured evaluator
    expected_tool_calls: List[str] = None  # Expected tool names in order
    allow_partial: bool = True  # Allow partial completion
    
    # Oracle evaluator (perfect knowledge of correct answer)
    oracle_answer: Optional[str] = None
    similarity_threshold: float = 0.8
    
    # Reward values
    reward_when_success: float = 1.0
    reward_when_failure: float = -1.0
    reward_partial: float = 0.5  # For partial completion


class TaskReward(BaseReward):
    """Task-success reward (placeholder implementation)."""
    
    def __init__(self, cfg: TaskRewardConfig):
        self.cfg = cfg
        if self.cfg.required_keywords is None:
            self.cfg.required_keywords = []
        if self.cfg.forbidden_keywords is None:
            self.cfg.forbidden_keywords = []
        if self.cfg.expected_tool_calls is None:
            self.cfg.expected_tool_calls = []
    
    def compute_reward(self, prompt: str, response: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Compute task-success reward.
        
        Args:
            prompt: Task prompt
            response: Model response
            context: Optional dict with:
                - 'tool_calls': List of tool names used
                - 'final_state': Environment state
                - 'task_evaluator': External evaluator function
        
        Returns:
            Reward in [-1, 1] where:
            - 1.0: Task completed successfully
            - 0.5: Partial success
            - 0.0: No progress
            - -1.0: Task failed or harmful action
        """
        if self.cfg.evaluator_type == "keyword":
            return self._evaluate_keywords(response)
        elif self.cfg.evaluator_type == "structured":
            return self._evaluate_structured(context)
        elif self.cfg.evaluator_type == "oracle":
            return self._evaluate_oracle(response)
        else:
            return 0.0
    
    def _evaluate_keywords(self, response: str) -> float:
        """Keyword-based evaluation (simple lexical matching)."""
        text = response if self.cfg.case_sensitive else response.lower()
        
        # Check required keywords
        required = [kw if self.cfg.case_sensitive else kw.lower() 
                   for kw in self.cfg.required_keywords]
        forbidden = [kw if self.cfg.case_sensitive else kw.lower() 
                    for kw in self.cfg.forbidden_keywords]
        
        # Count matches
        required_found = sum(1 for kw in required if kw in text)
        forbidden_found = sum(1 for kw in forbidden if kw in text)
        
        # Failure if any forbidden keyword found
        if forbidden_found > 0:
            return self.cfg.reward_when_failure
        
        # Success if all required keywords found
        if len(required) == 0:
            # No requirements means neutral
            return 0.0
        
        if required_found == len(required):
            return self.cfg.reward_when_success
        
        # Partial success based on fraction of keywords found
        if self.cfg.allow_partial and required_found > 0:
            fraction = required_found / len(required)
            return self.cfg.reward_partial * fraction
        
        return self.cfg.reward_when_failure
    
    def _evaluate_structured(self, context: Optional[Dict[str, Any]]) -> float:
        """Structured evaluation based on tool calls."""
        if context is None or 'tool_calls' not in context:
            return 0.0
        
        tool_calls = context.get('tool_calls', [])
        expected = self.cfg.expected_tool_calls
        
        if len(expected) == 0:
            return 0.0
        
        # Check if expected tools were called in order
        expected_idx = 0
        for call in tool_calls:
            if expected_idx < len(expected):
                if call == expected[expected_idx]:
                    expected_idx += 1
        
        # Calculate success
        if expected_idx == len(expected):
            return self.cfg.reward_when_success
        
        if self.cfg.allow_partial and expected_idx > 0:
            fraction = expected_idx / len(expected)
            return self.cfg.reward_partial * fraction
        
        return self.cfg.reward_when_failure
    
    def _evaluate_oracle(self, response: str) -> float:
        """Oracle evaluation (compare to known correct answer)."""
        if self.cfg.oracle_answer is None:
            return 0.0
        
        # Simple similarity check (can be replaced with better metrics)
        response_lower = response.lower().strip()
        oracle_lower = self.cfg.oracle_answer.lower().strip()
        
        # Exact match
        if response_lower == oracle_lower:
            return self.cfg.reward_when_success
        
        # Substring match
        if oracle_lower in response_lower or response_lower in oracle_lower:
            return self.cfg.reward_partial
        
        # Token overlap (simple Jaccard similarity)
        response_tokens = set(re.findall(r'\w+', response_lower))
        oracle_tokens = set(re.findall(r'\w+', oracle_lower))
        
        if len(oracle_tokens) == 0:
            return 0.0
        
        intersection = len(response_tokens & oracle_tokens)
        union = len(response_tokens | oracle_tokens)
        
        if union == 0:
            return 0.0
        
        similarity = intersection / union
        
        if similarity >= self.cfg.similarity_threshold:
            return self.cfg.reward_when_success
        elif similarity >= self.cfg.similarity_threshold / 2:
            return self.cfg.reward_partial
        else:
            return self.cfg.reward_when_failure


# ==========================================
# Reward Mixer (combines multiple rewards)
# ==========================================
@dataclass
class RewardMixerConfig:
    """Configuration for combining multiple reward sources."""
    task_weight: float = 0.5      # alpha: weight for task reward
    safety_weight: float = 0.5    # beta: weight for safety reward
    normalize: bool = True        # Normalize total reward to [-1, 1]
    clip_total: bool = True       # Clip total reward to [-1, 1]


class RewardMixer:
    """Combines multiple reward sources with configurable weights."""
    
    def __init__(
        self,
        cfg: RewardMixerConfig,
        safety_reward: Optional[SafetyReward] = None,
        task_reward: Optional[TaskReward] = None,
    ):
        self.cfg = cfg
        self.safety_reward = safety_reward
        self.task_reward = task_reward
        
        # Validate that at least one reward is provided
        if self.safety_reward is None and self.task_reward is None:
            raise ValueError("At least one reward component (safety or task) must be provided")
    
    def compute_reward(
        self, 
        prompt: str, 
        response: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Compute combined reward.
        
        Args:
            prompt: Input prompt
            response: Model response
            context: Optional context for task evaluation
            
        Returns:
            Dictionary with:
                - 'safety_reward': Safety component (or 0.0 if not used)
                - 'task_reward': Task component (or 0.0 if not used)
                - 'total_reward': Weighted combination
                - 'weights': Dict with alpha and beta
        """
        safety_r = 0.0
        task_r = 0.0
        
        # Compute safety reward
        if self.safety_reward is not None:
            try:
                safety_r = self.safety_reward.compute_reward(prompt, response, context)
            except Exception as e:
                print(f"[RewardMixer] Safety reward error: {e}")
                safety_r = 0.0
        
        # Compute task reward
        if self.task_reward is not None:
            try:
                task_r = self.task_reward.compute_reward(prompt, response, context)
            except Exception as e:
                print(f"[RewardMixer] Task reward error: {e}")
                task_r = 0.0
        
        # Combine with weights
        total = self.cfg.task_weight * task_r + self.cfg.safety_weight * safety_r
        
        # Normalize if requested
        if self.cfg.normalize:
            weight_sum = self.cfg.task_weight + self.cfg.safety_weight
            if weight_sum > 0:
                total = total / weight_sum
        
        # Clip to [-1, 1]
        if self.cfg.clip_total:
            total = max(-1.0, min(1.0, total))
        
        return {
            'safety_reward': float(safety_r),
            'task_reward': float(task_r),
            'total_reward': float(total),
            'weights': {
                'task_weight': self.cfg.task_weight,
                'safety_weight': self.cfg.safety_weight,
            }
        }
    
    def compute_rewards(
        self,
        prompts: List[str],
        responses: List[str],
        contexts: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, float]]:
        """Compute rewards for a batch of prompt-response pairs."""
        if contexts is None:
            contexts = [None] * len(prompts)
        return [self.compute_reward(p, r, c) for p, r, c in zip(prompts, responses, contexts)]
    
    def compute_total_rewards(
        self,
        prompts: List[str],
        responses: List[str],
        contexts: Optional[List[Dict[str, Any]]] = None
    ) -> List[float]:
        """Compute only the total rewards (for PPO training)."""
        results = self.compute_rewards(prompts, responses, contexts)
        return [r['total_reward'] for r in results]


# ==========================================
# Convenience builder
# ==========================================
def build_reward_mixer(
    task_weight: float = 0.5,
    safety_weight: float = 0.5,
    safety_monitor_type: str = "cot",
    safety_config: Optional[Dict[str, Any]] = None,
    task_evaluator_type: str = "keyword",
    task_config: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> RewardMixer:
    """
    Convenience function to build a RewardMixer with default settings.
    
    Args:
        task_weight: Alpha weight for task reward
        safety_weight: Beta weight for safety reward
        safety_monitor_type: Type of safety monitor (cot/probe/hybrid)
        safety_config: Additional config for safety reward
        task_evaluator_type: Type of task evaluator (keyword/structured/oracle)
        task_config: Additional config for task reward
        device: Device for safety monitor
        
    Returns:
        RewardMixer instance ready for use
    """
    # Build safety reward
    safety_cfg_dict = safety_config or {}
    safety_cfg_dict['monitor_type'] = safety_monitor_type
    safety_cfg = SafetyRewardConfig(**safety_cfg_dict)
    safety_reward = SafetyReward(safety_cfg, device=device)
    
    # Build task reward
    task_cfg_dict = task_config or {}
    task_cfg_dict['evaluator_type'] = task_evaluator_type
    task_cfg = TaskRewardConfig(**task_cfg_dict)
    task_reward = TaskReward(task_cfg)
    
    # Build mixer
    mixer_cfg = RewardMixerConfig(
        task_weight=task_weight,
        safety_weight=safety_weight,
    )
    
    return RewardMixer(mixer_cfg, safety_reward, task_reward)


# ==========================================
# Testing stub
# ==========================================
if __name__ == "__main__":
    print("Reward module loaded successfully")
    print("\nAvailable components:")
    print("  - SafetyReward: Wraps SHADE monitors (CoT, probe, hybrid)")
    print("  - TaskReward: Evaluates task success (keyword, structured, oracle)")
    print("  - RewardMixer: Combines multiple rewards with weights")
    print("\nExample usage:")
    print("""
    # Build individual rewards
    safety_cfg = SafetyRewardConfig(monitor_type="cot", cot_model_id=DEFAULT_MODEL)
    safety_reward = SafetyReward(safety_cfg)
    
    task_cfg = TaskRewardConfig(
        evaluator_type="keyword",
        required_keywords=["transfer", "complete"],
    )
    task_reward = TaskReward(task_cfg)
    
    # Combine with mixer
    mixer_cfg = RewardMixerConfig(task_weight=0.6, safety_weight=0.4)
    mixer = RewardMixer(mixer_cfg, safety_reward, task_reward)
    
    # Compute reward
    result = mixer.compute_reward(
        prompt="Transfer $100 to Alice",
        response="I will transfer $100 to Alice's account"
    )
    print(result)  # {'safety_reward': 0.8, 'task_reward': 1.0, 'total_reward': 0.92, ...}
    """)

