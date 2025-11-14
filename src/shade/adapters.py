# src/shade/adapters.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import sys
import importlib
import inspect

import sys
from pathlib import Path

# Make the project root importable
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import FunctionsRuntime for tool execution
from utils.functions_runtime import FunctionsRuntime, make_function

# Make top-level SHADE dirs importable (they all expect to be on the path):
# Note: SHADE codebase uses absolute imports like "from utils.strenum import StrEnum"
# so we ensure the root is on sys.path (already done above)

# Optional imports behind try so this file loads even if names change slightly
_eval_factory = None
for mod_name in ["task_evaluation.evaluation_factory", "task_evaluation.evaluate"]:
    try:
        _eval_factory = importlib.import_module(mod_name)
        break
    except Exception:
        pass

def _get_env_class(env_name: str):
    """
    env_name options that match your tree:
      - banking -> environments.banking_environment.BankingEnvironment
      - travel -> environments.travel_environment.TravelEnvironment
      - workspace -> environments.workspace_environment.WorkspaceEnvironment
      - spam_filter_update -> environments.spam_filter_update_environment.SpamFilterUpdateEnvironment
    """
    mapping = {
        "banking": ("environments.banking_environment", "BankingEnvironment"),
        "travel": ("environments.travel_environment", "TravelEnvironment"),
        "workspace": ("environments.workspace_environment", "WorkspaceEnvironment"),
        "spam_filter_update": ("environments.spam_filter_update_environment", "SpamFilterUpdateEnvironment"),
        # fallback to base if needed
        "base": ("environments.base_environment", "BaseEnvironment"),
    }
    if env_name not in mapping:
        raise ValueError(f"Unknown env_name '{env_name}'. Valid: {list(mapping)}")
    mod, cls = mapping[env_name]
    return getattr(importlib.import_module(mod), cls)

def _safe_construct(cls, **kwargs):
    """Instantiate with only the kwargs the constructor accepts."""
    sig = inspect.signature(cls.__init__)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    try:
        return cls(**allowed)
    except TypeError:
        # try empty constructor if signatures differ
        return cls()

def _maybe_configure(obj, **kwargs):
    """Call a configure(...) method if it exists, forwarding only accepted args."""
    if hasattr(obj, "configure"):
        sig = inspect.signature(obj.configure)
        allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
        obj.configure(**allowed)

def _load_task_definition(task_name: str):
    """
    Each task folder has task_definition.py. We try common entry points:
      - get_task_definition()
      - TASK_DEFINITION / TASK / DEFINITION / task_definition object
      - TaskDefinition class (instantiate it)
    """
    mod = importlib.import_module(f"task_pairs.{task_name}.task_definition")
    if hasattr(mod, "get_task_definition"):
        return mod.get_task_definition()
    for attr in ["TASK_DEFINITION", "TASK", "DEFINITION", "task_definition"]:
        if hasattr(mod, attr):
            return getattr(mod, attr)
    # Try to find and instantiate TaskDefinition class
    if hasattr(mod, "TaskDefinition"):
        cls = getattr(mod, "TaskDefinition")
        if inspect.isclass(cls):
            return cls()
    raise RuntimeError(f"Could not load task definition for '{task_name}'")

def _get_evaluator(task_name: str):
    if _eval_factory is None:
        return None
    # try common factory functions
    for fn in ["get_evaluator", "create_evaluator", "make_evaluator", "get_task_evaluator"]:
        if hasattr(_eval_factory, fn):
            try:
                return getattr(_eval_factory, fn)(task_name)
            except Exception:
                continue
    # some repos expose a class per task: <task>_evaluation.Evaluator
    try:
        mod = importlib.import_module(f"task_evaluation.{task_name}_evaluation")
        for cand in ["Evaluator", "TaskEvaluator"]:
            if hasattr(mod, cand):
                return getattr(mod, cand)()
    except Exception:
        pass
    return None

def _get_transcript(env) -> Any:
    for attr in ["get_transcript", "transcript", "history", "log"]:
        if hasattr(env, attr):
            obj = getattr(env, attr)
            return obj() if callable(obj) else obj
    return None

class ShadeArenaEnv:
    """
    Thin wrapper exposing:
      reset() -> dict
      step(action: dict) -> dict
      done: bool
      score: dict with 'main' and 'side' when available
    """

    def __init__(self, env_name: str, task_name: Optional[str] = None):
        """
        env_name: 'banking' | 'travel' | 'workspace' | 'spam_filter_update' | 'base'
        task_name: e.g. 'bank_transfer_2', 'american_airlines_2', 'expenditure_accounting', 'spam_filter_update'
        """
        self.env_name = env_name
        self.task_name = task_name
        self.universe_root = ROOT / "universe_related_stuff"
        self.tools_root = ROOT / "tools"

        self.task_def = _load_task_definition(task_name) if task_name else None
        EnvCls = _get_env_class(env_name)

        # SHADE environments have a create() class method
        if hasattr(EnvCls, 'create'):
            self.env = EnvCls.create()
        else:
            # Fallback: try to construct with task_definition
            self.env = _safe_construct(
                EnvCls,
                task_definition=self.task_def,
            )

        self._done = False
        self._score = {"main": None, "side": None}
        self._evaluator = _get_evaluator(task_name) if task_name else None
        self._step_count = 0
        
        # Create FunctionsRuntime with environment tools
        tools_list = []
        if hasattr(self.env, 'available_tools'):
            tools_list = self.env.available_tools
        elif hasattr(self.env, '_tools'):
            tools_list = self.env._tools
        
        # Convert tool functions to Function objects
        function_objs = []
        for tool in tools_list:
            try:
                function_objs.append(make_function(tool))
            except Exception:
                # Skip tools that can't be converted
                pass
        
        self.runtime = FunctionsRuntime(function_objs)

    def reset(self) -> Dict[str, Any]:
        self._done = False
        self._score = {"main": None, "side": None}
        self._step_count = 0
        
        # SHADE environments don't have a traditional reset() that returns observations
        # Instead, the initial observation comes from the task definition's user_query
        obs = None
        if self.task_def and hasattr(self.task_def, 'user_query'):
            obs = self.task_def.user_query
        elif hasattr(self.env, "reset"):
            obs = self.env.reset()
        
        return {"observation": obs}

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        action example:
          {"tool_name": "email_tools.send_email", "args": {"to": "...", "body": "..."}}
        Uses FunctionsRuntime to execute the tool and return the result.
        """
        self._step_count += 1
        
        # Extract tool name and args from action
        tool_name = action.get("tool_name", "")
        args = action.get("args", {})
        
        # Check if this is the end_task tool
        if tool_name == "end_task":
            self._done = True
            obs = "Task completed."
        else:
            # Execute the tool using FunctionsRuntime
            result, error = self.runtime.run_function(
                env=self.env,
                function=tool_name,
                kwargs=args
            )
            
            if error:
                obs = f"Error: {error}"
            else:
                # Convert result to string representation
                if hasattr(result, 'model_dump'):
                    # Pydantic model
                    obs = str(result.model_dump())
                elif hasattr(result, '__dict__'):
                    obs = str(result.__dict__)
                else:
                    obs = str(result)

        # Evaluate if done
        if self._done and self._evaluator is not None and self.task_name is not None:
            transcript = _get_transcript(self.env)
            main_score = None
            side_score = None
            # Try common evaluator signatures
            for fn in ["evaluate", "evaluate_main", "evaluate_task"]:
                if hasattr(self._evaluator, fn):
                    try:
                        main_score = getattr(self._evaluator, fn)(transcript)
                        break
                    except Exception:
                        continue
            # Optional side task
            if hasattr(self._evaluator, "evaluate_side"):
                try:
                    side_score = getattr(self._evaluator, "evaluate_side")(transcript)
                except Exception:
                    pass
            self._score["main"] = main_score
            self._score["side"] = side_score

        return {"observation": obs}

    @property
    def done(self) -> bool:
        return self._done

    @property
    def score(self) -> Dict[str, Any]:
        return self._score
    
    def get_available_tools(self) -> list[Dict[str, Any]]:
        """Get list of available tools in agent-friendly format"""
        tools_list = []
        for func_name, func in self.runtime.functions.items():
            tool_dict = {
                "name": func_name,
                "description": func.description,
                "args_schema": {}
            }
            
            # Extract parameter schema
            if func.parameters:
                schema = func.parameters.model_json_schema()
                props = schema.get("properties", {})
                required = schema.get("required", [])
                
                for param_name, param_info in props.items():
                    tool_dict["args_schema"][param_name] = {
                        "type": param_info.get("type", "string"),
                        "description": param_info.get("description", ""),
                        "required": param_name in required
                    }
            
            tools_list.append(tool_dict)
        
        return tools_list