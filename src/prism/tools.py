"""Layer 3 — Tools (structured acting).

Frontier models orchestrate tools cleanly in their heads. Small models fumble
the *mechanics*: malformed JSON, wrong tool names, missing/mistyped args. Prism
externalizes reliable tool use:

  - a Tool registry with explicit schemas (exposing Layers 1 & 2 as actions:
    search_code, get_context, inspect_code, analyze_impact, plan, read_file, …),
  - a lenient parser that extracts a tool call from messy small-model output
    (JSON, fenced JSON, or function-call syntax),
  - schema validation with type coercion and fuzzy tool-name repair,
  - a ReAct loop that feeds structured observations back and lets the model
    self-correct instead of derailing.

So a weak model acts through a hardened interface it can't easily break.
"""

from __future__ import annotations

import difflib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable

from . import context as ctx
from .planner import make_plan
from .verify import verify

MAX_OBS_CHARS = 1600


@dataclass
class Param:
    name: str
    type: str = "string"  # string | integer | number | boolean
    required: bool = False
    default: object = None
    description: str = ""


@dataclass
class Tool:
    name: str
    description: str
    params: list[Param]
    fn: Callable[..., object]

    def spec(self) -> str:
        args = ", ".join(
            f"{p.name}:{p.type}{'' if p.required else '?'}" for p in self.params
        )
        return f"{self.name}({args}) — {self.description}"


class ToolError(Exception):
    pass


def _coerce(param: Param, value: object) -> object:
    if value is None:
        return param.default
    try:
        if param.type == "integer":
            return int(value)
        if param.type == "number":
            return float(value)
        if param.type == "boolean":
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in ("1", "true", "yes", "on")
        return str(value)
    except (TypeError, ValueError) as e:
        raise ToolError(f"arg '{param.name}' expected {param.type}, got {value!r}") from e


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> str:
        return "\n".join(f"- {t.spec()}" for t in self._tools.values())

    def resolve_name(self, name: str) -> str | None:
        if name in self._tools:
            return name
        match = difflib.get_close_matches(name, self._tools, n=1, cutoff=0.6)
        return match[0] if match else None

    def validate(self, name: str, args: dict) -> dict:
        tool = self._tools[name]
        clean: dict[str, object] = {}
        args = args or {}
        for p in tool.params:
            if p.name not in args and p.required:
                raise ToolError(f"missing required arg '{p.name}' for {name}")
            clean[p.name] = _coerce(p, args.get(p.name, p.default))
        return clean

    def call(self, name: str, args: dict) -> object:
        resolved = self.resolve_name(name)
        if resolved is None:
            raise ToolError(
                f"unknown tool '{name}'. available: {', '.join(self.names())}"
            )
        clean = self.validate(resolved, args)
        return self._tools[resolved].fn(**clean)


# --------------------------------------------------------------------------
# Lenient parsing of a tool call out of small-model output.
# --------------------------------------------------------------------------

_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_FUNC_CALL = re.compile(r"(?P<name>[a-zA-Z_]\w*)\s*\((?P<args>.*?)\)", re.DOTALL)
_KWARG = re.compile(r"(\w+)\s*=\s*(\"[^\"]*\"|'[^']*'|[^,]+)")


def _balanced_json_objects(text: str) -> list[str]:
    out: list[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    out.append(text[start:i + 1])
    return out


def parse_tool_call(text: str) -> dict | None:
    """Return {"tool": name, "args": {...}} from messy model output, or None."""
    # 1. fenced json block
    m = _FENCED_JSON.search(text or "")
    candidates: list[str] = [m.group(1)] if m else []
    # 2. any balanced {...} objects, last first (models often end with the call)
    candidates.extend(reversed(_balanced_json_objects(text or "")))
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("tool") or obj.get("name") or obj.get("action")
        if not name:
            continue
        args = obj.get("args") or obj.get("arguments") or obj.get("input") or {}
        if not isinstance(args, dict):
            args = {}
        return {"tool": str(name), "args": args}

    # 3. function-call syntax:  search_code(query="login")
    fm = _FUNC_CALL.search(text or "")
    if fm:
        args = {}
        for k, v in _KWARG.findall(fm.group("args")):
            args[k] = v.strip().strip("'\"")
        return {"tool": fm.group("name"), "args": args}
    return None


# --------------------------------------------------------------------------
# Default registry — exposes Prism's own layers as tools.
# --------------------------------------------------------------------------

def _clip(obj: object) -> object:
    s = json.dumps(obj, default=str)
    if len(s) <= MAX_OBS_CHARS:
        return obj
    return {"truncated": True, "preview": s[:MAX_OBS_CHARS]}


def build_default_registry(repo_root: str, token_budget: int = 4000) -> ToolRegistry:
    reg = ToolRegistry()

    def t_search(query: str, top_k: int = 8):
        r = ctx.search_code(repo_root, query, top_k=top_k)
        return _clip({
            "symbols": [{"qualname": s["qualname"], "file": s["file"]} for s in r.get("symbols", [])[:top_k]],
            "files": [f["file"] for f in r.get("files", [])[:top_k]],
        })

    def t_get_context(task: str, budget: int = token_budget):
        r = ctx.get_context(repo_root, task, token_budget=budget)
        return _clip({
            "files_to_edit": [f["path"] for f in r.get("files_to_edit", [])],
            "periphery": [p["qualname"] for p in r.get("periphery", [])[:10]],
            "dependency_edges": [f"{e['from']} -> {e['to']}" for e in r.get("dependency_edges", [])[:10]],
            "fits_in": r.get("fits_in"),
        })

    def t_inspect(symbol: str = "", file: str = ""):
        return _clip(ctx.inspect_code(repo_root, symbol=symbol or None, file=file or None))

    def t_impact(symbol: str = "", file: str = "", hops: int = 2):
        return _clip(ctx.analyze_impact(repo_root, symbol=symbol or None, file=file or None, hops=hops))

    def t_plan(task: str):
        plan = make_plan(repo_root, task, token_budget=token_budget)
        return _clip({"source": plan.source, "steps": [
            {"index": s.index, "title": s.title, "target": s.target} for s in plan.steps
        ]})

    def t_read_file(path: str, start: int = 1, end: int = 200):
        full = os.path.join(repo_root, path)
        try:
            with open(full, "r", encoding="utf-8") as fh:
                lines = fh.read().splitlines()
        except OSError as e:
            raise ToolError(f"cannot read {path}: {e}")
        seg = "\n".join(lines[max(0, start - 1):end])
        return _clip({"path": path, "start": start, "end": min(end, len(lines)), "text": seg})

    def t_verify(paths: str):
        targets = [p.strip() for p in paths.split(",") if p.strip()]
        abs_targets = [os.path.join(repo_root, p) for p in targets]
        return _clip(verify(abs_targets))

    reg.register(Tool("search_code", "find symbols/files relevant to a query (Layer 1)",
                      [Param("query", "string", True), Param("top_k", "integer", False, 8)], t_search))
    reg.register(Tool("get_context", "budget-packed dossier for a task (Layer 1)",
                      [Param("task", "string", True), Param("budget", "integer", False, token_budget)], t_get_context))
    reg.register(Tool("inspect_code", "exact lines/neighbors/quality for a symbol or file",
                      [Param("symbol", "string", False, ""), Param("file", "string", False, "")], t_inspect))
    reg.register(Tool("analyze_impact", "blast radius before editing a symbol or file",
                      [Param("symbol", "string", False, ""), Param("file", "string", False, ""),
                       Param("hops", "integer", False, 2)], t_impact))
    reg.register(Tool("plan", "decompose a task into ordered steps (Layer 2)",
                      [Param("task", "string", True)], t_plan))
    reg.register(Tool("read_file", "read a line window of a file",
                      [Param("path", "string", True), Param("start", "integer", False, 1),
                       Param("end", "integer", False, 200)], t_read_file))
    reg.register(Tool("verify", "compile/verify comma-separated files",
                      [Param("paths", "string", True)], t_verify))
    return reg


# --------------------------------------------------------------------------
# ReAct loop.
# --------------------------------------------------------------------------

TOOLS_SYSTEM = (
    "You are an agent that acts by calling tools. Each turn, respond with EXACTLY "
    "one JSON object and nothing else:\n"
    '  {"tool": "<name>", "args": {...}}\n'
    "to call a tool, or\n"
    '  {"tool": "final", "args": {"answer": "..."}}\n'
    "when you have enough information. Available tools:\n{specs}\n"
    "Call one tool at a time. Use observations to decide the next call."
)


@dataclass
class ToolStep:
    tool: str
    args: dict
    ok: bool
    observation: object


@dataclass
class ToolTrace:
    task: str
    steps: list[ToolStep] = field(default_factory=list)
    answer: str = ""
    stopped: str = ""  # "final" | "max_steps" | "no_call"


# Constrained-decoding schema for a tool call. Per the research pivot, trained
# tool-use + structured/constrained decoding is preferred over lenient parsing +
# fuzzy repair; we enforce this JSON shape and keep parse_tool_call as fallback.
TOOL_CALL_FORMAT = {
    "type": "object",
    "properties": {
        "tool": {"type": "string"},
        "args": {"type": "object"},
    },
    "required": ["tool", "args"],
}


def _chat(client, system: str, user: str, structured: bool):
    """Call chat with constrained decoding when supported; degrade gracefully.

    Tool use is ACTING, not deep reasoning, so we request think=False: a hybrid
    model should emit the tool-call JSON directly (fast), not spend its budget in
    <think>. Falls back cleanly for clients without `think`/`format` kwargs.
    """
    attempts = []
    if structured:
        attempts.append(dict(format=TOOL_CALL_FORMAT, think=False))
        attempts.append(dict(format=TOOL_CALL_FORMAT))
    attempts.append(dict(think=False))
    attempts.append({})
    for kw in attempts:
        try:
            return client.chat(system, user, **kw)
        except TypeError:
            continue
    return client.chat(system, user)


def run_tools(
    repo_root: str,
    task: str,
    client,
    registry: ToolRegistry | None = None,
    max_steps: int = 8,
    token_budget: int = 4000,
    structured: bool = True,
    on_event=None,
) -> ToolTrace:
    """Drive a model through a hardened tool-use loop.

    `structured=True` uses constrained decoding (Ollama `format`) to force valid
    tool-call JSON; parse_tool_call remains the fallback for models/clients that
    can't constrain output.
    """
    def emit(msg: str) -> None:
        if on_event:
            on_event(msg)

    reg = registry or build_default_registry(repo_root, token_budget=token_budget)
    # NB: the prompt contains literal JSON braces, so use replace(), not format().
    system = TOOLS_SYSTEM.replace("{specs}", reg.specs())
    trace = ToolTrace(task=task)
    transcript = f"Task: {task}"

    for step in range(1, max_steps + 1):
        reply = _chat(client, system, transcript, structured)
        call = parse_tool_call(reply)
        if call is None:
            emit(f"[tools] step {step}: no tool call parsed; nudging model")
            transcript += "\n\nYour last reply had no valid JSON tool call. Respond with one JSON object only."
            trace.stopped = "no_call"
            continue
        if call["tool"] == "final":
            trace.answer = str(call["args"].get("answer", ""))
            trace.stopped = "final"
            emit(f"[tools] step {step}: final answer")
            return trace

        try:
            result = reg.call(call["tool"], call["args"])
            ok = True
        except ToolError as e:
            result = {"error": str(e)}
            ok = False
        trace.steps.append(ToolStep(call["tool"], call["args"], ok, result))
        emit(f"[tools] step {step}: {call['tool']}({call['args']}) -> {'ok' if ok else 'error'}")
        obs = json.dumps(result, default=str)[:MAX_OBS_CHARS]
        transcript += f"\n\nCalled {call['tool']}({json.dumps(call['args'])})\nObservation: {obs}"

    trace.stopped = trace.stopped or "max_steps"
    return trace
