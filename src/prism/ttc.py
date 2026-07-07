"""Test-time compute — difficulty-aware, verifier-guided (Layers 2+4).

Research findings this implements:
  - Adaptive/compute-optimal test-time allocation is >4x more efficient than
    fixed best-of-N ([Snell et al. 2408.03314]) — so DON'T sample a fixed N.
  - Test-time compute only helps ABOVE a competence threshold — so if the model
    can't even compile after a couple tries, escalating further is wasted; bail.
  - Selection must be VERIFIER-guided (execution), not self-scored.

Strategy: start cheap (1 sample). If execution passes, stop — easy task, no
compute wasted. If it fails, escalate: diversify temperature, feed execution
errors back, sample again — up to a budget. Early-abort when attempts show no
progress (stuck below the competence threshold).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .context import get_context
from .render import render_markdown
from .trace import NULL_TRACER
from .verifier import Verdict, _default_generate, make_verdict


@dataclass
class TTCResult:
    verdict: Verdict
    attempts: int
    samples: list[Verdict]
    stopped: str  # "solved" | "budget" | "stuck"


def adaptive_solve(
    repo_root: str,
    task: str,
    client,
    max_samples: int = 6,
    token_budget: int = 6000,
    run_tests: bool = True,
    stuck_after: int = 2,
    on_event=None,
    tracer=None,
) -> TTCResult:
    """Escalate compute only as difficulty demands; select by execution."""
    def emit(msg: str) -> None:
        if on_event:
            on_event(msg)

    tr = tracer or NULL_TRACER
    tr.phase("adaptive_ttc", task=task)
    payload = get_context(repo_root, task, token_budget=token_budget)
    tr.context(fits_in=payload.get("fits_in", 0), budget=token_budget,
               files_to_edit=[f["path"] for f in payload.get("files_to_edit", [])],
               periphery=len(payload.get("periphery", [])))
    dossier = render_markdown(payload, include_source=True)
    gen = _default_generate(client, dossier)

    samples: list[Verdict] = []
    best: Verdict | None = None
    feedback = ""
    temp = 0.1
    zero_streak = 0

    for i in range(1, max_samples + 1):
        t0 = time.time()
        files = gen(feedback, temp)
        tr.model_call(role="edit", think=False, latency=time.time() - t0,
                      output_chars=sum(len(f.code) for f in files), sample=i, temperature=round(temp, 2))
        v = make_verdict(repo_root, files, task=task, run_tests=run_tests)
        er = v.exec_report
        tr.check(kind="tests" if (er and er.tests_ran) else "compile",
                 ok=v.ok, signal=v.signal, sample=i,
                 tests_ran=bool(er and er.tests_ran), tests_passed=bool(er and er.tests_passed))
        samples.append(v)
        if best is None or v.signal > best.signal:
            best = v
        emit(f"[ttc] sample {i}: exec_ok={v.ok} signal={v.signal} (temp={round(temp, 2)})")

        if v.accepted:  # execution passed — easy enough, stop escalating
            tr.outcome(resolved=True, won_by="ttc", attempts=i, files=[f.path for f in v.files])
            return TTCResult(v, i, samples, "solved")

        # difficulty signal: no compilation progress at all
        zero_streak = zero_streak + 1 if v.signal == 0.0 else 0
        if zero_streak >= stuck_after and i >= stuck_after:
            emit(f"[ttc] no progress after {zero_streak} zero-signal samples — below competence threshold, stopping")
            tr.outcome(resolved=False, won_by="ttc", attempts=i, stopped="stuck")
            return TTCResult(best, i, samples, "stuck")

        # escalate: diversify + feed the execution errors back
        temp = min(1.0, temp + 0.2)
        problems = [f"{e.get('file')}:{e.get('line')}: {e['message']}" for e in v.errors]
        feedback = "\n".join(f"- {p}" for p in problems) or "Try a different approach."
        tr.repair(attempt=i, feedback=feedback)

    tr.outcome(resolved=bool(best and best.accepted), won_by="ttc", attempts=max_samples, stopped="budget")
    return TTCResult(best, max_samples, samples, "budget")
