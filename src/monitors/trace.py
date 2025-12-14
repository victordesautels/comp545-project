from __future__ import annotations
import json
import os
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional



# JSONL trace writer
def _now_ms() -> int:
    return int(time.time() * 1000)


def _default_redactor(key: str, value: Any) -> Any:
    """Basic redaction for secrets and tokens.
    Override by passing a custom redactor to TraceLogger.
    """
    key_l = key.lower()
    if any(tok in key_l for tok in ["api_key", "apikey", "token", "password", "secret"]):
        return "***REDACTED***"
    return value


@dataclass
class TraceEvent:
    ts: int
    session_id: str
    step: int
    kind: str  # prompt | model_output | action | observation | thoughts | tool_result | score | meta
    data: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class TraceLogger:
    """Thread safe JSONL trace logger.

    Each session writes to a single JSONL file at logs/traces/<session_id>.jsonl
    """

    def __init__(
        self,
        session_id: str,
        root: Optional[os.PathLike] = None,
        redactor: Optional[callable] = None,
        flush: bool = True,
    ) -> None:
        self.session_id = session_id
        self.root = Path(root) if root is not None else Path("logs") / "traces"
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / f"{session_id}.jsonl"
        self._fh = open(self.path, "a", encoding="utf-8")
        self._lock = threading.Lock()
        self._step = 0
        self._redactor = redactor or _default_redactor
        self._flush = flush
        # mark file start
        self.log(kind="meta", data={"event": "session_start", "pid": os.getpid()})

    def close(self) -> None:
        self.log(kind="meta", data={"event": "session_end"})
        try:
            self._fh.close()
        except Exception:
            pass

    # Context manager support
    def __enter__(self) -> "TraceLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def step(self) -> int:
        return self._step

    def next_step(self) -> int:
        self._step += 1
        return self._step

    def log(self, kind: str, data: Dict[str, Any]) -> None:
        # apply key-based redaction shallowly
        safe: Dict[str, Any] = {}
        for k, v in (data or {}).items():
            safe[k] = self._redactor(k, v)
        evt = TraceEvent(ts=_now_ms(), session_id=self.session_id, step=self._step, kind=kind, data=safe)
        line = evt.to_json() + "\n"
        with self._lock:
            self._fh.write(line)
            if self._flush:
                self._fh.flush()


# High level helpers
def log_prompt(tr: TraceLogger, prompt: str, tools: Optional[Iterable[Dict[str, Any]]] = None) -> None:
    tr.log("prompt", {"text": prompt, "tools": list(tools) if tools is not None else None})


def log_model_output(tr: TraceLogger, text: str, raw_ids_len: Optional[int] = None, latency_ms: Optional[int] = None) -> None:
    tr.log("model_output", {"text": text, "token_count": raw_ids_len, "latency_ms": latency_ms})


def log_action(tr: TraceLogger, tool_name: str, args: Dict[str, Any]) -> None:
    tr.log("action", {"tool_name": tool_name, "args": args})


def log_observation(tr: TraceLogger, observation: Any) -> None:
    tr.log("observation", {"observation": observation})


def log_thoughts(tr: TraceLogger, content: str, tag: Optional[str] = None) -> None:
    tr.log("thoughts", {"content": content, "tag": tag})


def log_tool_result(tr: TraceLogger, tool_name: str, result: Any, ok: Optional[bool] = None) -> None:
    tr.log("tool_result", {"tool_name": tool_name, "result": result, "ok": ok})


def log_score(tr: TraceLogger, main: Any = None, side: Any = None, extra: Optional[Dict[str, Any]] = None) -> None:
    data = {"main": main, "side": side}
    if extra:
        data.update(extra)
    tr.log("score", data)


# Integration shims
def wrap_agent_act(agent, tracer: TraceLogger):
    """Return a proxy .act(obs, tools) that logs prompt, output, and action.

    This assumes the agent exposes:
      - tokenizer.apply_chat_template or builds a text prompt internally
      - act(obs, tools) -> {"tool_name": str, "args": dict}
    """

    def _proxy(obs, tools):
        step = tracer.next_step()
        # Best effort: try to reconstruct the text prompt like HFGameAgent does
        try:
            from src.agents.hf_agent import build_prompt
            prompt_text = build_prompt(obs, tools)
        except Exception:
            prompt_text = json.dumps({"obs": obs, "tools": tools}, ensure_ascii=False)
        log_prompt(tracer, prompt_text, tools)

        t0 = _now_ms()
        action = agent.act(obs, tools)
        dt = _now_ms() - t0

        # If the agent can expose last generated text, record it
        gen_text = getattr(agent, "_last_generation", None)
        if isinstance(gen_text, str):
            log_model_output(tracer, gen_text, latency_ms=dt)
        else:
            log_model_output(tracer, "", latency_ms=dt)

        log_action(tracer, action.get("tool_name", ""), action.get("args", {}))
        return action

    return _proxy


def wrap_env(env, tracer: TraceLogger):
    """Return a proxy env with reset(), step(action), done, score that logs observations and results."""

    class _EnvProxy:
        def __init__(self, env, tracer: TraceLogger):
            self._env = env
            self._tr = tracer

        def reset(self):
            obs = self._env.reset()
            log_observation(self._tr, obs)
            return obs

        def step(self, action):
            log_action(self._tr, action.get("tool_name", ""), action.get("args", {}))
            out = self._env.step(action)
            log_observation(self._tr, out)
            # try to log score when done
            done = getattr(self._env, "done", False)
            if callable(done):
                done = bool(done())
            if bool(done):
                score = getattr(self._env, "score", None)
                if callable(score):
                    score = score()
                if isinstance(score, dict):
                    log_score(self._tr, score.get("main"), score.get("side"), extra={k: v for k, v in score.items() if k not in {"main", "side"}})
            return out

        @property
        def done(self):
            d = getattr(self._env, "done", False)
            return d() if callable(d) else bool(d)

        @property
        def score(self):
            s = getattr(self._env, "score", None)
            return s() if callable(s) else s

    return _EnvProxy(env, tracer)


# CLI
if __name__ == "__main__":
    tr = TraceLogger(session_id="test_session")
    try:
        tr.next_step()
        log_prompt(tr, "Inbox has 3 emails. Choose a tool.", tools=[{"name": "email_tools.send_email", "args_schema": {"to": "str", "body": "str"}}])
        log_model_output(tr, '{"tool_name": "email_tools.send_email", "args": {"to": "alice@example.com", "body": "hi"}}')
        log_action(tr, "email_tools.send_email", {"to": "alice@example.com", "body": "hi"})
        log_observation(tr, {"status": "sent"})
        log_thoughts(tr, "I think the task is complete.", tag="reflection")
        tr.next_step()
        log_action(tr, "end_task", {})
        log_score(tr, main=1.0, side=0.0)
    finally:
        tr.close()
