"""Layer 2 — Reasoning / Planning.

Frontier models plan long multi-step changes *inside their heads*. Small models
can't hold that. So Prism externalizes the plan: the harness decomposes a task
into ordered, bounded steps and holds them as explicit state; the small model
only ever reasons about one step at a time, each step grounded by Layer 1
(Context). Global reasoning lives in the harness, local reasoning in the model.

Two planning modes:
  - graph-derived (no model): derive steps from the code graph + Context. Fully
    deterministic, runs offline — proof that "planning is infrastructure."
  - model-refined: a small model refines the plan, grounded by the dossier
    structure (not full source, so it stays cheap).

Execution walks the plan one step at a time, giving each step its own tight
Context dossier instead of dumping the whole task on the model at once.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from .agent import _parse_files, _verify_files
from .context import get_context
from .llm import call_chat
from .render import render_markdown

PLAN_SYSTEM = (
    "You are a planning module. Decompose the coding task into the SMALLEST "
    "ordered list of concrete steps. Each step must be independently doable and "
    "name the file/area it touches. Do not write code. Respond ONLY with a JSON "
    'array: [{"title": "...", "detail": "...", "target": "path or symbol", '
    '"kind": "edit|investigate"}]. 2 to 6 steps.'
)

EXEC_SYSTEM = (
    "You are a coding agent executing ONE step of a larger plan. You are given "
    "the step and a budget-sized context dossier for it. Make only the change "
    "this step requires. Respond ONLY with changed files, each as:\n\n"
    "FILE: <relative/path.ext>\n"
    "```<language>\n<full new file contents>\n```\n\n"
    "No prose, no unchanged files."
)

_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)
_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Step:
    index: int
    title: str
    detail: str
    kind: str = "edit"  # "edit" | "investigate"
    target: str = ""
    status: str = "pending"  # pending | done | verified | failed | skipped
    proposed_files: list[dict] = field(default_factory=list)  # [{path, chars}]
    context_tokens: int = 0
    notes: str = ""


@dataclass
class Plan:
    task: str
    steps: list[Step]
    source: str  # "graph" | "model"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _plan_items(raw: str) -> list[dict]:
    """Extract a list of step-dicts from messy model JSON, tolerating shapes:
    a bare array, an object with a steps/plan/files key, or a single step object."""
    # 1. a JSON array anywhere in the text
    m = _JSON_ARRAY.search(raw or "")
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return [x for x in arr if isinstance(x, dict)]
        except json.JSONDecodeError:
            pass
    # 2. a JSON object: {"steps":[...]} / {"plan":[...]} / a single step {...}
    m = _JSON_OBJ.search(raw or "")
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
        if isinstance(obj, dict):
            for key in ("steps", "plan", "files", "items"):
                if isinstance(obj.get(key), list):
                    return [x for x in obj[key] if isinstance(x, dict)]
            if obj.get("title") or obj.get("target"):  # a single step object
                return [obj]
    return []


def _parse_plan(raw: str, task: str) -> list[Step]:
    """Parse a model's JSON plan, tolerating surrounding prose and array/object shapes."""
    steps: list[Step] = []
    for i, it in enumerate(_plan_items(raw), start=1):
        steps.append(
            Step(
                index=i,
                title=str(it.get("title", f"Step {i}"))[:120],
                detail=str(it.get("detail", it.get("title", ""))),
                kind="investigate" if it.get("kind") == "investigate" else "edit",
                target=str(it.get("target", it.get("file", ""))),
            )
        )
    if not steps:  # fallback: one step = the whole task
        steps = [Step(index=1, title=task[:120], detail=task, kind="edit")]
    return steps


def make_plan(
    repo_root: str,
    task: str,
    token_budget: int = 8000,
    client=None,
    max_steps: int = 6,
    think: bool = True,
) -> Plan:
    """Produce an ordered plan. With no client, derive it from the code graph."""
    payload = get_context(repo_root, task, token_budget=token_budget)
    edit_files = [f["path"] for f in payload.get("files_to_edit", [])]

    if client is None:
        # Deterministic, graph-derived plan — global reasoning with zero model.
        steps: list[Step] = []
        for i, f in enumerate(edit_files[:max_steps], start=1):
            steps.append(
                Step(
                    index=i,
                    title=f"Update {f}",
                    detail=f"Apply '{task}' within {f}. Keep existing call sites working.",
                    kind="edit",
                    target=f,
                )
            )
        if not steps:
            steps = [Step(index=1, title=task[:120], detail=task, kind="edit")]
        notes = [
            "graph-derived plan: steps = the files Context ranked as edit targets",
            f"{len(payload.get('dependency_edges', []))} dependency edges considered",
        ]
        return Plan(task=task, steps=steps, source="graph", notes=notes)

    # Model-refined plan, grounded by the dossier STRUCTURE (cheap, no full source).
    # Planning is REASONING -> request thinking mode on hybrid models by default.
    # Benchmarks pass think=False to isolate Prism's grounding from the model's
    # innate thinking (and to stay fast/scorable on slow local hardware).
    dossier = render_markdown(payload, include_source=False)
    # Bound output to JSON + a token cap so small models emit a compact plan and
    # STOP, instead of rambling to the context limit (minutes per call).
    raw = call_chat(client, PLAN_SYSTEM, f"Task: {task}\n\n{dossier}",
                    think=think, format="json", num_predict=512)
    steps = _parse_plan(raw, task)[:max_steps]
    for i, s in enumerate(steps, start=1):
        s.index = i
    return Plan(task=task, steps=steps, source="model", notes=["model-refined plan"])


def run_plan(
    plan: Plan,
    repo_root: str,
    client,
    token_budget: int = 8000,
    verify: bool = True,
    stream: bool = False,
    on_event=None,
    on_token=None,
) -> Plan:
    """Execute the plan one bounded step at a time, each with its own dossier."""
    def emit(msg: str) -> None:
        if on_event:
            on_event(msg)

    for step in plan.steps:
        if step.kind != "edit":
            step.status = "skipped"
            emit(f"[plan] step {step.index} ({step.title}) — investigate, skipped execution")
            continue

        emit(f"[plan] step {step.index}/{len(plan.steps)}: {step.title}")
        focus = step.detail if not step.target else f"{step.detail} (focus: {step.target})"
        payload = get_context(repo_root, focus, token_budget=token_budget)
        step.context_tokens = payload.get("fits_in", 0)
        dossier = render_markdown(payload, include_source=True)
        user = f"## Step {step.index}: {step.title}\n{step.detail}\n\n{dossier}"

        # Executing a step is EDIT generation -> disable thinking for speed.
        reply = call_chat(client, EXEC_SYSTEM, user, think=False,
                          stream=stream, on_token=on_token or (lambda t: None))

        files = _parse_files(reply)
        step.proposed_files = [{"path": f.path, "chars": len(f.code)} for f in files]
        if not files:
            step.status = "failed"
            step.notes = "no file blocks produced"
            emit(f"[plan] step {step.index}: no edits produced")
            continue

        if verify:
            check = _verify_files(repo_root, files)
            step.status = "verified" if check["ok"] else "failed"
            if not check["ok"]:
                step.notes = "; ".join(e["message"] for e in check["errors"])
            emit(f"[plan] step {step.index}: {'verified' if check['ok'] else 'FAILED ' + step.notes}")
        else:
            step.status = "done"

    return plan


def plan_and_execute(
    repo_root: str,
    task: str,
    client=None,
    token_budget: int = 8000,
    verify: bool = True,
    stream: bool = False,
    on_event=None,
    on_token=None,
) -> Plan:
    plan = make_plan(repo_root, task, token_budget=token_budget, client=client)
    if client is None:
        return plan  # planning only; no model to execute
    return run_plan(
        plan, repo_root, client, token_budget=token_budget, verify=verify,
        stream=stream, on_event=on_event, on_token=on_token,
    )
