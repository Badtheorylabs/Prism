"""Domain packs (build-order item 6) — the harness, beyond coding.

Proves the core thesis with a working non-coding vertical: the SAME runtime loop
(context -> model -> CHECK -> repair -> trace) works when the truth signal isn't
tests but a DOMAIN checker. Here: a data task graded by *reconciliation* —
the harness computes ground truth from the input and rejects any answer whose
numbers don't add up, then feeds the exact discrepancy back for repair.

`solve_task` is the domain-agnostic harness core (what verify_and_repair is for
code). Swap the `check` and you have a new vertical: research (citations
resolve), ops (records reconcile), docs (rules pass), etc.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .checkers import CheckResult
from .llm import call_chat
from .trace import NULL_TRACER

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class DomainResult:
    resolved: bool
    result: CheckResult
    artifact: object
    attempts: int


def solve_task(*, prompt, context, produce, check, max_attempts=3, tracer=None, on_event=None) -> DomainResult:
    """Domain-agnostic solve loop: produce an artifact, CHECK it, repair, retry."""
    import time

    def emit(m):
        if on_event:
            on_event(m)

    tr = tracer or NULL_TRACER
    tr.phase("domain_solve", task=prompt)
    tr.context(fits_in=len(context) // 4, budget=0)

    feedback = ""
    best: tuple[CheckResult, object] | None = None
    for attempt in range(1, max_attempts + 1):
        t0 = time.time()
        artifact = produce(feedback)
        tr.model_call(role="produce", think=False, latency=time.time() - t0,
                      output_chars=len(str(artifact)), attempt=attempt)
        res = check(artifact)
        tr.check(kind=res.kind, ok=res.ok, signal=res.signal, attempt=attempt)
        emit(f"[domain] attempt {attempt}: ok={res.ok} signal={res.signal}")
        if best is None or res.signal > best[0].signal:
            best = (res, artifact)
        if res.ok:
            tr.outcome(resolved=True, won_by="domain", attempts=attempt)
            return DomainResult(True, res, artifact, attempt)
        feedback = "\n".join(f"- {e['message']}" for e in res.errors) or "Try again."
        tr.repair(attempt=attempt, feedback=feedback)

    tr.outcome(resolved=False, won_by="domain", attempts=max_attempts)
    return DomainResult(False, best[0], best[1], max_attempts)


# --------------------------------------------------------------------------
# Data reconciliation pack — "sum <value> by <group>", graded by ground truth.
# --------------------------------------------------------------------------

def data_reconciliation(rows: list[dict], group: str, value: str):
    """Build (context, prompt, check) for a group-and-sum task with a truth loop."""
    expected: dict[str, float] = {}
    for r in rows:
        expected[str(r[group])] = expected.get(str(r[group]), 0) + r[value]

    context = "Records (one per line, `%s: %s`):\n" % (group, value) + "\n".join(
        f"{r[group]}: {r[value]}" for r in rows
    )
    prompt = (f"Sum the '{value}' for each distinct '{group}'. "
              f"Output ONLY a JSON object mapping each {group} to its numeric total.")

    def check(artifact) -> CheckResult:
        if not isinstance(artifact, dict):
            return CheckResult(False, 0.0, "reconcile",
                               errors=[{"message": "output was not a JSON object"}])
        errors = []
        for g, tot in expected.items():
            got = artifact.get(g)
            if got is None:
                errors.append({"message": f"missing {group}={g} (should total {tot})"})
            elif got != tot:
                errors.append({"message": f"{group}={g}: expected {tot}, got {got}"})
        for g in artifact:
            if g not in expected:
                errors.append({"message": f"unexpected {group}={g}"})
        ok = not errors
        return CheckResult(ok=ok, signal=2.0 if ok else 0.0, kind="reconcile",
                           detail={"expected": expected}, errors=errors)

    return context, prompt, check, expected


def _make_produce(client, system: str, prompt: str, context: str):
    def produce(feedback: str):
        user = f"{context}\n\n{prompt}"
        if feedback:
            user += f"\n\n## Your previous answer was wrong; fix exactly these:\n{feedback}"
        reply = call_chat(client, system, user, think=False)
        m = _JSON_OBJ.search(reply or "")
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return produce


def run_data_reconciliation(client, rows, group, value, max_attempts=3, tracer=None, on_event=None) -> DomainResult:
    """End-to-end non-coding task through the harness: model answers, reconciliation grades."""
    context, prompt, check, _expected = data_reconciliation(rows, group, value)
    produce = _make_produce(client, "You are a precise data assistant. Output only JSON.", prompt, context)
    return solve_task(prompt=prompt, context=context, produce=produce, check=check,
                      max_attempts=max_attempts, tracer=tracer, on_event=on_event)


# a tiny built-in dataset for the CLI demo
DEMO_ROWS = [
    {"region": "east", "revenue": 100},
    {"region": "west", "revenue": 50},
    {"region": "east", "revenue": 25},
    {"region": "north", "revenue": 70},
    {"region": "west", "revenue": 30},
]
