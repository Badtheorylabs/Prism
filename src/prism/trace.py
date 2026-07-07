"""Prism Eval/Trace spine — the runtime's memory.

Every capability layer plugs into this. It records, per run, the full loop:
  - what CONTEXT the harness gave the model,
  - each MODEL_CALL (role, think mode, latency, output size),
  - each ACTION (tool/command) and its observation,
  - each CHECK (the verifier — tests for code, schema/citations/reconciliation
    for other domains) with its signal,
  - each REPAIR (what feedback drove the retry),
  - the OUTCOME (resolved?, which attempt/layer won).

Deliberately domain-agnostic: `check(kind=...)` is "tests" for coding today and
"schema"/"citation"/"reconcile" for future domain packs — same spine.

Zero dependencies (json/os/time). A disabled Tracer is a no-op, so it never
slows the hot path. Persists to `<root>/.prism/runs/<run_id>/` as:
  - trace.jsonl  (one event per line, replayable)
  - summary.json (rolled-up verdict)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field


_RUN_COUNTER = 0


def _next_run_id(t0: float) -> str:
    """Unique run id even for runs created within the same millisecond."""
    global _RUN_COUNTER
    _RUN_COUNTER += 1
    return f"run-{int(t0 * 1000)}-{os.getpid()}-{_RUN_COUNTER}"


@dataclass
class Event:
    seq: int
    t: float          # seconds since run start
    kind: str         # phase|context|model_call|action|check|repair|outcome|note
    data: dict = field(default_factory=dict)


class Tracer:
    def __init__(self, root: str | None = None, task: str = "", label: str = "",
                 enabled: bool = True, run_id: str | None = None):
        self.enabled = enabled
        self.task = task
        self.label = label
        self.events: list[Event] = []
        self._seq = 0
        self._t0 = time.time()
        self.run_id = run_id or _next_run_id(self._t0)
        self.run_dir: str | None = None
        if enabled and root is not None:
            self.run_dir = os.path.join(root, ".prism", "runs", self.run_id)
            os.makedirs(self.run_dir, exist_ok=True)

    # ---- recording ---------------------------------------------------------

    def event(self, kind: str, **data) -> None:
        if not self.enabled:
            return
        self._seq += 1
        ev = Event(seq=self._seq, t=round(time.time() - self._t0, 3), kind=kind, data=data)
        self.events.append(ev)
        if self.run_dir is not None:
            with open(os.path.join(self.run_dir, "trace.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(ev)) + "\n")

    def phase(self, name: str, **data) -> None:
        self.event("phase", name=name, **data)

    def context(self, *, fits_in: int = 0, budget: int = 0, files_to_edit=None,
                periphery: int = 0, **data) -> None:
        self.event("context", fits_in=fits_in, budget=budget,
                   files_to_edit=files_to_edit or [], periphery=periphery, **data)

    def model_call(self, *, role: str, think: bool | None = None, latency: float = 0.0,
                   output_chars: int = 0, **data) -> None:
        self.event("model_call", role=role, think=think,
                   latency=round(latency, 3), output_chars=output_chars, **data)

    def action(self, *, tool: str, ok: bool, **data) -> None:
        self.event("action", tool=tool, ok=ok, **data)

    def check(self, *, kind: str, ok: bool, signal: float = 0.0, **data) -> None:
        """Domain-agnostic verifier event. kind: tests|schema|citation|reconcile|..."""
        self.event("check", check_kind=kind, ok=ok, signal=signal, **data)

    def repair(self, *, attempt: int, feedback: str = "", **data) -> None:
        self.event("repair", attempt=attempt, feedback=feedback[:500], **data)

    def outcome(self, *, resolved: bool, won_by: str = "", attempts: int = 0, **data) -> None:
        self.event("outcome", resolved=resolved, won_by=won_by, attempts=attempts, **data)

    def note(self, msg: str, **data) -> None:
        self.event("note", msg=msg, **data)

    # ---- summary -----------------------------------------------------------

    def summarize(self) -> dict:
        kinds: dict[str, int] = {}
        for e in self.events:
            kinds[e.kind] = kinds.get(e.kind, 0) + 1
        checks = [e for e in self.events if e.kind == "check"]
        outcome = next((e for e in reversed(self.events) if e.kind == "outcome"), None)
        model_calls = [e for e in self.events if e.kind == "model_call"]
        return {
            "run_id": self.run_id,
            "task": self.task,
            "label": self.label,
            "duration_s": round(time.time() - self._t0, 3),
            "event_counts": kinds,
            "model_calls": len(model_calls),
            "model_latency_s": round(sum(e.data.get("latency", 0) for e in model_calls), 3),
            "checks": len(checks),
            "checks_passed": sum(1 for e in checks if e.data.get("ok")),
            "resolved": bool(outcome and outcome.data.get("resolved")),
            "won_by": (outcome.data.get("won_by") if outcome else ""),
            "attempts": (outcome.data.get("attempts") if outcome else 0),
            "files": (outcome.data.get("files", []) if outcome else []),
        }

    def close(self, **extra) -> dict:
        summary = {**self.summarize(), **extra}
        if self.run_dir is not None:
            with open(os.path.join(self.run_dir, "summary.json"), "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2)
        return summary


# --- module-level no-op default so callers can trace unconditionally ---------

NULL_TRACER = Tracer(enabled=False)


def load_run(run_dir: str) -> dict:
    """Replay a persisted run: {summary, events}."""
    events = []
    tp = os.path.join(run_dir, "trace.jsonl")
    if os.path.exists(tp):
        with open(tp, encoding="utf-8") as fh:
            events = [json.loads(line) for line in fh if line.strip()]
    summary = {}
    sp = os.path.join(run_dir, "summary.json")
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as fh:
            summary = json.load(fh)
    return {"summary": summary, "events": events}
