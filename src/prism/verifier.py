"""Layer 4 — Verification / Reliability (execution-grounded).

REDESIGNED per the frontier-vs-Prism research audit
(docs/research/frontier-vs-prism.md):

  - The research is unambiguous: reliability comes from EXECUTION (tests,
    compile, type-check, diagnostics) and separately-trained verifiers — NOT
    from a weak model critiquing itself (which can *degrade* accuracy and
    suffers self-preference bias).
  - So execution is now the PRIMARY signal. `best_of_n` ranks candidates by the
    execution signal, not by self-judgment.
  - Same-model self-critique is DEMOTED to an optional hint (`adversarial=True`,
    off by default) that only supplies repair suggestions and tie-breaks — it
    can never gate acceptance.
  - A SEPARATE/stronger verifier model may be supplied (`verifier_client`); the
    generator never judges itself.

`accepted` == execution passed (compiles, and tests pass when tests exist).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .agent import ProposedFile, _parse_files
from .agent import SYSTEM_PROMPT as EDIT_SYSTEM
from .context import get_context
from .execution import ExecReport, ground_check
from .llm import call_chat
from .render import render_markdown
from .trace import NULL_TRACER

CRITIC_SYSTEM = (
    "You are an adversarial code reviewer. Given a task and a proposed change, "
    "try hard to find why it is WRONG, incomplete, or unsafe for the task. "
    "Respond ONLY with JSON: "
    '{"satisfies": true|false, "issues": ["..."]}. '
    "Default to satisfies=false when uncertain."
)

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Critique:
    satisfies: bool
    issues: list[str] = field(default_factory=list)


@dataclass
class Verdict:
    ok: bool  # execution-grounded pass (compiles + tests pass if any exist)
    signal: float = 0.0  # execution signal in [0, 3] — the primary ranking key
    errors: list[dict] = field(default_factory=list)
    exec_report: Optional[ExecReport] = None
    critique: Optional[Critique] = None  # optional HINT only; never gates
    files: list[ProposedFile] = field(default_factory=list)
    score: float = 0.0  # = signal, plus a tiny critique tie-break

    @property
    def accepted(self) -> bool:
        """Trusted iff EXECUTION passed. Self-critique cannot accept or reject."""
        return self.ok


def _chat(client, system: str, user: str, temperature: float = 0.1) -> str:
    try:
        return client.chat(system, user, temperature=temperature)
    except TypeError:
        return client.chat(system, user)


def check(
    repo_root: str,
    files: list[ProposedFile],
    run_tests: bool = True,
    run_types: bool = False,
    test_target: str | None = None,
) -> ExecReport:
    """The single source of truth for correctness: run the code, don't ask a model."""
    return ground_check(repo_root, files, run_tests=run_tests, run_types=run_types,
                        test_target=test_target)


def _render_files(files: list[ProposedFile], limit: int = 4000) -> str:
    text = "\n\n".join(f"FILE: {f.path}\n{f.code}" for f in files)
    return text[:limit]


def critique(client, task: str, files: list[ProposedFile]) -> Critique:
    """Optional self/separate-model critique. HINT ONLY — see module docstring."""
    if not files:
        return Critique(satisfies=False, issues=["no files proposed"])
    user = f"Task: {task}\n\nProposed change:\n{_render_files(files)}"
    # Critique is REASONING -> request thinking mode on the (separate) verifier.
    raw = call_chat(client, CRITIC_SYSTEM, user, think=True, temperature=0.0)
    m = _JSON_OBJ.search(raw or "")
    if m:
        try:
            obj = json.loads(m.group(0))
            issues = obj.get("issues") or []
            if not isinstance(issues, list):
                issues = [str(issues)]
            return Critique(satisfies=bool(obj.get("satisfies", False)),
                            issues=[str(i) for i in issues][:8])
        except json.JSONDecodeError:
            pass
    return Critique(satisfies=False, issues=["critic response unparseable"])


def _score(report: ExecReport, crit: Optional[Critique], n_files: int) -> float:
    """Ranking score is EXECUTION-DOMINATED. Critique is at most a small tie-break."""
    s = report.signal  # [0, 3], the real signal
    if n_files == 0:
        return -1.0
    if crit is not None:  # tie-break only, tiny weight — never flips a pass/fail
        s += 0.15 if crit.satisfies else -0.05 * len(crit.issues)
    return round(s, 3)


def make_verdict(
    repo_root: str,
    files: list[ProposedFile],
    *,
    task: str = "",
    adversarial: bool = False,
    verifier_client=None,
    run_tests: bool = True,
    run_types: bool = False,
) -> Verdict:
    """Grade a candidate by execution first; critique only as an optional hint."""
    report = check(repo_root, files, run_tests=run_tests, run_types=run_types)
    crit = None
    if adversarial and files and verifier_client is not None:
        # Only a SEPARATE verifier model may critique — never the generator itself.
        crit = critique(verifier_client, task, files)
    return Verdict(
        ok=report.ok, signal=report.signal, errors=report.errors,
        exec_report=report, critique=crit, files=files,
        score=_score(report, crit, len(files)),
    )


def _default_generate(client, dossier: str):
    def gen(feedback: str, temperature: float) -> list[ProposedFile]:
        user = dossier if not feedback else f"{dossier}\n\n## Fix these problems\n{feedback}"
        # Edit generation is not reasoning -> disable thinking for speed.
        reply = call_chat(client, EDIT_SYSTEM, user, think=False, temperature=temperature)
        return _parse_files(reply)
    return gen


def best_of_n(
    repo_root: str,
    task: str,
    client,
    n: int = 3,
    token_budget: int = 6000,
    adversarial: bool = False,
    verifier_client=None,
    run_tests: bool = True,
    generate: Optional[Callable[[str, float], list[ProposedFile]]] = None,
    on_event=None,
    tracer=None,
) -> tuple[Verdict, list[Verdict]]:
    """Sample N candidates, rank by EXECUTION signal, return (winner, all)."""
    def emit(msg: str) -> None:
        if on_event:
            on_event(msg)

    tr = tracer or NULL_TRACER
    tr.phase("best_of_n", task=task, n=n)
    payload = get_context(repo_root, task, token_budget=token_budget)
    tr.context(fits_in=payload.get("fits_in", 0), budget=token_budget,
               files_to_edit=[f["path"] for f in payload.get("files_to_edit", [])],
               periphery=len(payload.get("periphery", [])))
    dossier = render_markdown(payload, include_source=True)
    gen = generate or _default_generate(client, dossier)

    verdicts: list[Verdict] = []
    for i in range(n):
        t0 = time.time()
        files = gen("", 0.1 + 0.25 * i)  # diversify
        tr.model_call(role="edit", think=False, latency=time.time() - t0,
                      output_chars=sum(len(f.code) for f in files), candidate=i + 1)
        v = make_verdict(repo_root, files, task=task, adversarial=adversarial,
                         verifier_client=verifier_client, run_tests=run_tests)
        er = v.exec_report
        tr.check(kind="tests" if (er and er.tests_ran) else "compile",
                 ok=v.ok, signal=v.signal, candidate=i + 1,
                 tests_ran=bool(er and er.tests_ran), tests_passed=bool(er and er.tests_passed))
        verdicts.append(v)
        emit(f"[verify] candidate {i + 1}/{n}: exec_ok={v.ok} signal={v.signal} score={v.score}")

    winner = max(verdicts, key=lambda v: (v.signal, v.ok, v.score, -len(v.errors)))
    tr.outcome(resolved=winner.accepted, won_by="best_of_n", attempts=n,
               files=[f.path for f in winner.files])
    return winner, verdicts


def verify_and_repair(
    repo_root: str,
    task: str,
    client,
    max_attempts: int = 3,
    token_budget: int = 6000,
    adversarial: bool = False,
    verifier_client=None,
    run_tests: bool = True,
    generate: Optional[Callable[[str, float], list[ProposedFile]]] = None,
    on_event=None,
    tracer=None,
) -> Verdict:
    """Generate -> EXECUTE -> feed execution errors back -> retry.

    Repair feedback is driven by execution errors first; critique issues are
    secondary hints. Accepts on the first execution pass. Every attempt, check,
    and repair is recorded to the trace spine (`tracer`).
    """
    def emit(msg: str) -> None:
        if on_event:
            on_event(msg)

    tr = tracer or NULL_TRACER
    tr.phase("verify_and_repair", task=task)
    payload = get_context(repo_root, task, token_budget=token_budget)
    tr.context(fits_in=payload.get("fits_in", 0), budget=token_budget,
               files_to_edit=[f["path"] for f in payload.get("files_to_edit", [])],
               periphery=len(payload.get("periphery", [])))
    dossier = render_markdown(payload, include_source=True)
    gen = generate or _default_generate(client, dossier)

    feedback = ""
    best: Optional[Verdict] = None
    for attempt in range(1, max_attempts + 1):
        t0 = time.time()
        files = gen(feedback, 0.1)
        tr.model_call(role="edit", think=False, latency=time.time() - t0,
                      output_chars=sum(len(f.code) for f in files), attempt=attempt)
        v = make_verdict(repo_root, files, task=task, adversarial=adversarial,
                         verifier_client=verifier_client, run_tests=run_tests)
        er = v.exec_report
        tr.check(kind="tests" if (er and er.tests_ran) else "compile",
                 ok=v.ok, signal=v.signal, attempt=attempt,
                 tests_ran=bool(er and er.tests_ran),
                 tests_passed=bool(er and er.tests_passed))
        emit(f"[verify] attempt {attempt}: exec_ok={v.ok} signal={v.signal}")
        if best is None or v.signal > best.signal:
            best = v
        if v.accepted:
            tr.outcome(resolved=True, won_by="repair", attempts=attempt,
                       files=[f.path for f in v.files])
            return v
        # repair feedback: EXECUTION errors are the hard signal
        problems: list[str] = [f"{e.get('file')}:{e.get('line')}: {e['message']}" for e in v.errors]
        if v.critique and not v.critique.satisfies:  # secondary hints
            problems.extend(v.critique.issues)
        if not files:
            problems.append("Respond in the FILE: + code-fence format.")
        feedback = "\n".join(f"- {p}" for p in problems) or "Try again."
        tr.repair(attempt=attempt, feedback=feedback)

    tr.outcome(resolved=bool(best and best.accepted), won_by="repair", attempts=max_attempts,
               files=[f.path for f in best.files] if best else [])
    return best or Verdict(ok=False, errors=[{"message": "no attempts"}])
