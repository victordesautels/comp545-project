# src/agents/shade_llm.py
"""
Local HuggingFace model wrapper compatible with SHADE-Arena's pipeline.

This implements BasePipelineElement to work with the existing SHADE pipeline
infrastructure in utils/pipeline_elements.py and utils/pipeline_utils.py.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, StoppingCriteria, StoppingCriteriaList

from src.utils.device import pick_device, default_dtype_for_device


# Stopping criteria - stop when JSON object is complete
class StopOnJson(StoppingCriteria):
    """Stop generation when a complete JSON object is detected."""
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__()
        self.tokenizer = tokenizer
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Decode last 100 tokens to check for complete JSON
        text = self.tokenizer.decode(input_ids[0][-100:], skip_special_tokens=True)
        # Check if we have a complete JSON object
        if '{"tool_name"' in text and text.rstrip().endswith('}'):
            return True
        return False
from src.utils.model_loader import load_model, load_tokenizer, DEFAULT_MODEL
from utils.functions_runtime import FunctionCall, FunctionsRuntime, TaskEnvironment, EmptyEnv
from utils.pipeline_elements import BasePipelineElement, QueryResult
from utils.types import ChatMessage, ChatAssistantMessage


@dataclass
class LocalLLMConfig:
    """Configuration for local HuggingFace LLM."""
    model_id: str = DEFAULT_MODEL
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 512
    device: Optional[str] = None


class LocalHFLLM(BasePipelineElement):
    """
    Local HuggingFace LLM that implements SHADE-Arena's BasePipelineElement.
    
    This allows using local models with the existing pipeline infrastructure.
    The key method is `query()` which receives the full message history and
    returns an updated QueryResult with the assistant's response.
    """
    
    name = "LocalHFLLM"
    
    def __init__(
        self,
        config: Optional[LocalLLMConfig] = None,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        self.config = config or LocalLLMConfig()
        self.model_name = self.config.model_id  # For compatibility with SHADE code
        
        # Load model if not provided
        if model is None or tokenizer is None:
            self.device = torch.device(self.config.device) if self.config.device else pick_device()
            self.dtype = default_dtype_for_device(self.device)
            
            print(f"[LocalHFLLM] Loading model: {self.config.model_id}")
            print(f"[LocalHFLLM] Device: {self.device}, dtype: {self.dtype}")
            
            self.model, self.device, self.dtype = load_model(
                self.config.model_id, self.device, self.dtype
            )
            self.tokenizer = load_tokenizer(self.config.model_id)
        else:
            self.model = model
            self.tokenizer = tokenizer
            self.device = next(model.parameters()).device
            self.dtype = next(model.parameters()).dtype
        
        # Detect if model is sharded via device_map="auto"
        self.is_sharded = hasattr(self.model, "hf_device_map") and self.model.hf_device_map is not None
        
        if not self.is_sharded and self.device.type != "cpu":
            self.model.to(self.device)
        self.model.eval()
    
    def _messages_to_prompt(self, messages: Sequence[ChatMessage], runtime: FunctionsRuntime) -> str:
        """Convert SHADE-style messages to a prompt string."""
        # Build tools description
        tools_desc = self._format_tools(runtime)
        
        # Use chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            # Convert SHADE messages to standard chat format
            chat_messages = []
            for msg in messages:
                role = msg["role"]
                if role == "system":
                    chat_messages.append({"role": "system", "content": msg["content"] + "\n\n" + tools_desc})
                elif role == "user":
                    chat_messages.append({"role": "user", "content": msg["content"]})
                elif role == "assistant":
                    # Include tool calls in the content if present
                    content = msg.get("content") or ""
                    tool_calls = msg.get("tool_calls")
                    if tool_calls:
                        for tc in tool_calls:
                            content += f'\n{{"tool_name": "{tc.function}", "args": {json.dumps(tc.args)}}}'
                    chat_messages.append({"role": "assistant", "content": content})
                elif role == "tool":
                    # Tool results become user messages
                    result = msg.get("content", "")
                    error = msg.get("error")
                    if error:
                        result = f"Error: {error}"
                    chat_messages.append({"role": "user", "content": f"Tool result: {result[:1000]}"})
            
            return self.tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback: simple text format
            parts = []
            for msg in messages:
                role = msg["role"].upper()
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")
            parts.append("ASSISTANT:")
            return "\n".join(parts)
    
    def _format_tools(self, runtime: FunctionsRuntime) -> str:
        """Format available tools for the prompt."""
        lines = [
            "IMPORTANT: You must respond with EXACTLY ONE JSON object, nothing else.",
            "Do NOT output multiple tool calls. Only ONE action per response.",
            "",
            'Format: {"tool_name": "<name>", "args": {<arguments>}}',
            "",
            'When finished: {"tool_name": "end_task", "args": {}}',
            "",
            "Available tools:"
        ]
        
        for name, func in list(runtime.functions.items())[:30]:  # Limit tools
            desc = func.description[:80] if func.description else ""
            params = []
            if func.parameters:
                schema = func.parameters.model_json_schema()
                params = list(schema.get("properties", {}).keys())
            lines.append(f"- {name}({', '.join(params)}): {desc}")
        
        return "\n".join(lines)
    
    def _parse_tool_call(self, text: str) -> Optional[FunctionCall]:
        """Parse a tool call from model output."""
        # Try to find JSON in code blocks first
        json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
        if json_match:
            raw = json_match.group(1)
        else:
            # Find first balanced JSON object (handles nested braces)
            raw = self._extract_first_json_object(text)
            if raw is None:
                return None
        
        # Try to parse, with progressively more aggressive repairs
        obj = self._try_parse_json(raw)
        
        if obj and isinstance(obj, dict) and "tool_name" in obj:
            return FunctionCall(
                id=f"call_{hash(str(obj)) % 10000}",
                function=obj["tool_name"],
                args=obj.get("args", {}) or {}
            )
        
        return None
    
    def _try_parse_json(self, raw: str) -> Optional[Dict]:
        """Try to parse JSON with various repair strategies."""
        raw = raw.strip()
        
        # Strategy 1: Direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Remove trailing commas
        cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Fix single quotes to double quotes (but not inside strings)
        # Simple approach: replace all single quotes with double quotes
        fixed = raw.replace("'", '"')
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Strategy 4: Extract just tool_name and args using regex
        tool_match = re.search(r'"tool_name"\s*:\s*"([^"]+)"', raw)
        if tool_match:
            tool_name = tool_match.group(1)
            # Try to extract args
            args_match = re.search(r'"args"\s*:\s*(\{[^}]*\})', raw)
            args = {}
            if args_match:
                try:
                    args_str = args_match.group(1).replace("'", '"')
                    args = json.loads(args_str)
                except:
                    pass
            return {"tool_name": tool_name, "args": args}
        
        # Strategy 5: Very lenient - just get the tool name
        tool_match = re.search(r'tool_name["\s:]+(\w+)', raw)
        if tool_match:
            return {"tool_name": tool_match.group(1), "args": {}}
        
        print(f"[LocalHFLLM] All parse strategies failed for: {raw[:150]}...")
        return None
    
    def _extract_first_json_object(self, text: str) -> Optional[str]:
        """Extract the first balanced JSON object from text."""
        # Find the first '{'
        start = text.find('{')
        if start == -1:
            return None
        
        # Count braces to find the matching '}'
        depth = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\' and in_string:
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if in_string:
                continue
            
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        
        # No balanced object found - try returning up to first '}' anyway
        end = text.find('}', start)
        if end != -1:
            return text[start:end + 1]
        
        return None
    
    @torch.no_grad()
    async def _retry_with_json_prompt(
        self,
        failed_output: str,
        messages: Sequence[ChatMessage],
        runtime: FunctionsRuntime,
    ) -> Optional[FunctionCall]:
        """
        When the model outputs prose instead of JSON, prompt it to reformulate.
        """
        print(f"[LocalHFLLM] Retrying with JSON correction prompt...")
        
        # Build a correction prompt
        correction = (
            f"Your previous response was not valid JSON:\n"
            f'"{failed_output[:300]}..."\n\n'
            f"Please respond with ONLY a JSON object in this exact format:\n"
            f'{{"tool_name": "<tool_name>", "args": {{...}}}}\n\n'
            f'If you are done with the task, respond with: {{"tool_name": "end_task", "args": {{}}}}'
        )
        
        # Create messages with the failed output and correction
        retry_messages = [
            *messages,
            {"role": "assistant", "content": failed_output},
            {"role": "user", "content": correction},
        ]
        
        # Build prompt and generate
        prompt = self._messages_to_prompt(retry_messages, runtime)
        
        max_context = getattr(self.model.config, 'max_position_embeddings', 131072)
        max_input_len = max_context - self.config.max_new_tokens - 100
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_len
        )
        
        # For sharded models, keep inputs on CPU
        if not self.is_sharded:
            inputs = inputs.to(self.device)
        
        stop_criteria = StoppingCriteriaList([StopOnJson(self.tokenizer)])
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,  # Shorter - we just need JSON
            do_sample=True,
            temperature=0.3,  # Lower temp for more predictable output
            top_p=0.9,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=stop_criteria,
        )
        
        retry_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        
        tool_call = self._parse_tool_call(retry_text)
        if tool_call:
            print(f"[LocalHFLLM] Retry successful: {tool_call.function}")
        
        return tool_call
    
    @torch.no_grad()
    async def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: TaskEnvironment = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: Dict[str, Any] = {},
    ) -> QueryResult:
        """
        Generate a response given the full message history.
        
        This is the key method that SHADE's pipeline calls. The model sees
        ALL previous messages, enabling proper multi-turn reasoning.
        """
        # Build prompt from message history
        prompt = self._messages_to_prompt(messages, runtime)
        
        # Tokenize with truncation to model's max context length
        max_context = getattr(self.model.config, 'max_position_embeddings', 131072)
        # Leave room for generation
        max_input_len = max_context - self.config.max_new_tokens - 100
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=max_input_len
        )
        
        # For sharded models (device_map="auto"), keep inputs on CPU
        # and let HF dispatch handle placement
        if not self.is_sharded:
            inputs = inputs.to(self.device)
        
        # Create stopping criteria
        stop_criteria = StoppingCriteriaList([StopOnJson(self.tokenizer)])
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=True,
            temperature=max(self.config.temperature, 1e-5),
            top_p=self.config.top_p,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=stop_criteria,
        )
        
        # Decode only new tokens
        gen_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        
        # Parse tool call
        tool_call = self._parse_tool_call(gen_text)
        
        if tool_call is None:
            # Retry with a correction prompt
            tool_call = await self._retry_with_json_prompt(gen_text, messages, runtime)
        
        if tool_call is None:
            print(f"[LocalHFLLM] Parse failed after retry. Output ({len(gen_text)} chars): {gen_text[:200]}...")
            tool_call = FunctionCall(
                id="parse_failed",
                function="end_task",
                args={"reason": "parse_failed"}
            )
        else:
            # Log the tool being called
            step_num = sum(1 for m in messages if m.get("role") == "assistant") + 1
            print(f"[Step {step_num}] tool={tool_call.function}")
        
        # Create assistant message - store FULL content (not truncated)
        # Monitoring and evaluation need the complete reasoning
        assistant_message = ChatAssistantMessage(
            role="assistant",
            content=gen_text,
            tool_calls=[tool_call]
        )
        
        # Return updated result with new message appended
        return QueryResult(
            query=query,
            runtime=runtime,
            env=env,
            messages=[*messages, assistant_message],
            extra_args=extra_args,
        )
    
    async def count_tokens_external(self, messages: Sequence[ChatMessage]) -> int:
        """Count tokens in messages (used by ToolsExecutionLoop)."""
        # Simple estimation: convert to string and count tokens
        text = str(messages)
        return len(self.tokenizer.encode(text))


def create_shade_pipeline(
    model_id: str = DEFAULT_MODEL,
    system_message: str = "",
    max_iters: int = 20,
    temperature: float = 0.7,
    device: Optional[str] = None,
):
    """
    Create a SHADE-compatible pipeline with a local HuggingFace model.
    
    Returns:
        Tuple of (pipeline, llm) for use with SHADE's run_agent_pipeline
    """
    from utils.pipeline_elements import (
        AgentPipeline, SystemMessage, InitQuery, ToolsExecutionLoop, ToolsExecutor
    )
    
    # Create the local LLM
    llm = LocalHFLLM(LocalLLMConfig(
        model_id=model_id,
        temperature=temperature,
        device=device,
    ))
    
    # Create pipeline matching SHADE's structure
    tools_loop = ToolsExecutionLoop(
        [ToolsExecutor(), llm],
        max_iters=max_iters
    )
    
    pipeline = AgentPipeline([
        SystemMessage(system_message),
        InitQuery(),
        llm,  # First turn
        tools_loop,  # Subsequent turns
    ])
    
    return pipeline, llm

