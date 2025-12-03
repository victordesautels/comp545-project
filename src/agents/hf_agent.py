from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import sys
from pathlib import Path

# Make the project root importable
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    StoppingCriteria,
    StoppingCriteriaList,
)

# Local utilities for device and dtype
from src.utils.device import pick_device, default_dtype_for_device
from src.utils.model_loader import load_model, load_tokenizer, DEFAULT_MODEL


# -----------------------------
# Stopping utilities
# -----------------------------
class StopOnAnyString(StoppingCriteria):
    def __init__(self, stop_strings: List[str], tokenizer: PreTrainedTokenizerBase):
        super().__init__()
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        # Pre-tokenize the stop strings to speed up comparison when short
        self._stop_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strings]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Only inspect the last sequence in batch for simplicity
        last_seq = input_ids[0].tolist()
        # Fast path: compare token tails
        for sid in self._stop_ids:
            if len(sid) == 0:
                continue
            if last_seq[-len(sid):] == sid:
                return True
        # Slow path: decode tail to text and search for any stop string
        text = self.tokenizer.decode(last_seq, skip_special_tokens=False)
        return any(s in text for s in self.stop_strings)


# -----------------------------
# Parsing the model's tool call
# -----------------------------
_TOOL_JSON_RE = re.compile(r"```json\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)
_FIRST_OBJ_RE = re.compile(r"\{[\s\S]*\}")


def _json_sanitize(s: str) -> str:
    # Remove control junk after a closing brace if the model added reflections
    # Also strip trailing commas in objects and arrays
    s = s.strip()
    # Try to cut at the last closing brace
    last = s.rfind("}")
    if last != -1:
        s = s[: last + 1]
    # Remove trailing commas before closing brackets
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse an action dict from model text.
    Expected shape:
        {"tool_name": "module.function", "args": {...}}
    Returns None if parsing fails.
    """
    m = _TOOL_JSON_RE.search(text)
    if not m:
        m = _FIRST_OBJ_RE.search(text)
    if not m:
        return None
    raw = _json_sanitize(m.group(0))
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "tool_name" in obj:
            if "args" not in obj or obj["args"] is None:
                obj["args"] = {}
            return obj
    except Exception:
        return None
    return None


# -----------------------------
# Prompting
# -----------------------------
SYSTEM_PROMPT = (
    "You are a helpful agent that solves tasks by calling tools. "
    "You will receive an observation and a list of available tools. "
    "Respond ONLY with a JSON object of the form: {\"tool_name\": \"<name>\", \"args\": { ... }}. "
    "If the task is complete, return {\"tool_name\": \"end_task\", \"args\": {}}."
)

TOOL_INSTRUCTIONS = (
    "Tools available:\n"
    "- Each tool has a name and parameters. Choose exactly one per step.\n"
    "- Use the minimal arguments required.\n"
)


def render_tools(tools: List[Dict[str, Any]]) -> str:
    lines = []
    for t in tools:
        name = t.get("name", "unknown")
        desc = t.get("description", "")
        params = t.get("args_schema", {})
        lines.append(f"- {name}: {desc}\n  args: {json.dumps(params, ensure_ascii=False)}")
    return "\n".join(lines)


def build_prompt(obs: Union[str, Dict[str, Any]], tools: List[Dict[str, Any]]) -> str:
    obs_text = obs if isinstance(obs, str) else json.dumps(obs, ensure_ascii=False)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"{TOOL_INSTRUCTIONS}\n"
        f"{render_tools(tools)}\n\n"
        f"Observation:\n{obs_text}\n\n"
        f"Return JSON only."
    )


# -----------------------------
# Agent
# -----------------------------
@dataclass
class DecodeParams:
    temperature: float = 0.2
    top_p: float = 0.95
    max_new_tokens: int = 256
    stop: Optional[List[str]] = None  # strings


class HFGameAgent:
    """Wrap a transformers CausalLM into a simple policy with act(obs, tools)."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        decode: Optional[DecodeParams] = None,
        use_chat_template: bool = True,
    ) -> None:
        self.device = device or pick_device()
        self.dtype = dtype or default_dtype_for_device(self.device)
        self.decode = decode or DecodeParams()
        self.use_chat_template = use_chat_template

        if model is None or tokenizer is None:
            # load_model returns (model, device, dtype)
            base_model, self.device, self.dtype = load_model(model_id, self.device, self.dtype)
            self.model = base_model
            self.tokenizer = load_tokenizer(model_id)
        else:
            self.model = model
            self.tokenizer = tokenizer

        if self.device.type != "cpu":
            self.model.to(self.device)
        self.model.eval()

    def _apply_stops(self) -> StoppingCriteriaList:
        stops = self.decode.stop or []
        criteria = StoppingCriteriaList()
        if len(stops) > 0:
            criteria.append(StopOnAnyString(stops, self.tokenizer))
        return criteria

    def _format_inputs(self, obs: Union[str, Dict[str, Any]], tools: List[Dict[str, Any]]):
        if self.use_chat_template and getattr(self.tokenizer, "apply_chat_template", None):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        TOOL_INSTRUCTIONS
                        + "\n"
                        + render_tools(tools)
                        + "\n\nObservation:\n"
                        + (obs if isinstance(obs, str) else json.dumps(obs, ensure_ascii=False))
                        + "\n\nReturn JSON only."
                    ),
                },
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = build_prompt(obs, tools)
        model_inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        return model_inputs

    @torch.no_grad()
    def act(self, obs: Union[str, Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Returns an action dict: {"tool_name": str, "args": dict}
        On parse failure, returns an end_task action to terminate gracefully.
        """
        inputs = self._format_inputs(obs, tools)
        stop_criteria = self._apply_stops()

        gen_kwargs = dict(
            max_new_tokens=self.decode.max_new_tokens,
            do_sample=True,
            temperature=max(self.decode.temperature, 1e-5),
            top_p=self.decode.top_p,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=stop_criteria,
        )

        outputs = self.model.generate(**inputs, **gen_kwargs)
        gen_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        action = parse_tool_call(gen_text)
        if action is None:
            # Fallback heuristic: if the model printed a tool name line, try to build args empty
            m = re.search(r"tool_name\s*[:=]\s*([\w\.\-]+)", gen_text)
            if m:
                action = {"tool_name": m.group(1), "args": {}}

        if action is None:
            action = {"tool_name": "end_task", "args": {"reason": "parse_failed"}}
        return action


# -----------------------------
# Quick smoke test
# -----------------------------
if __name__ == "__main__":
    # Example usage with a local small model
    model_id = DEFAULT_MODEL

    agent = HFGameAgent(model_id=model_id, decode=DecodeParams(stop=["</tool_call>"]))

    tools = [
        {
            "name": "email_tools.send_email",
            "description": "Send an email",
            "args_schema": {"to": "str", "subject": "str", "body": "str"},
        },
        {
            "name": "end_task",
            "description": "Finish the task",
            "args_schema": {},
        },
    ]

    obs = {"inbox_summary": ["Please confirm the booking."]}
    action = agent.act(obs, tools)
    print("Action:", action)
