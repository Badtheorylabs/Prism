"""Checkers — the domain-agnostic truth loop.

Prism's core bet: a small model becomes a useful worker when it has an EXTERNAL
truth signal to check its work against. For coding that's tests/compile. But the
harness doesn't care what the signal is — it just needs `ok` + a `signal` score.
This module makes the verifier pluggable so the SAME runtime (context -> model ->
check -> repair -> trace) works across domains:

  - coding:      run tests / compile            (CodingChecker)
  - data:        validate a produced artifact    (FunctionChecker / SchemaChecker)
  - research:    citations resolve, sources agree (FunctionChecker)
  - ops:         records reconcile, API 200s      (FunctionChecker)

A Checker turns "did the work succeed?" into an execution-grounded verdict — the
same shape best_of_n / verify_and_repair / TTC already rank by. `kind` flows
straight into the trace spine's `check(kind=...)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol


@dataclass
class CheckResult:
    ok: bool
    signal: float  # [0, 3]-ish, higher = stronger evidence of success
    kind: str
    detail: dict = field(default_factory=dict)
    errors: list[dict] = field(default_factory=list)


class Checker(Protocol):
    kind: str

    def check(self, workdir: str, artifact) -> CheckResult:
        """Verify an artifact produced in `workdir`. `artifact` is domain-specific
        (edited files for code, a dict/df for data, a report for research)."""
        ...


class CodingChecker:
    """Coding truth loop: overlay edits and run tests/compile (wraps execution)."""

    kind = "tests"

    def __init__(self, run_tests: bool = True, run_types: bool = False):
        self.run_tests = run_tests
        self.run_types = run_types

    def check(self, workdir: str, artifact) -> CheckResult:
        from .execution import ground_check  # local import: keeps checkers light

        report = ground_check(workdir, artifact, run_tests=self.run_tests, run_types=self.run_types)
        kind = "tests" if report.tests_ran else "compile"
        return CheckResult(ok=report.ok, signal=report.signal, kind=kind,
                           detail={"tests_ran": report.tests_ran,
                                   "tests_passed": report.tests_passed},
                           errors=report.errors)


class FunctionChecker:
    """Generic checker for ANY domain: supply a predicate returning a verdict.

    The predicate gets (workdir, artifact) and returns either a bool, or a
    (ok, signal, detail) tuple. This is how a non-coding domain pack plugs its
    own truth signal into the exact same harness.
    """

    def __init__(self, kind: str, fn: Callable[[str, object], object]):
        self.kind = kind
        self.fn = fn

    def check(self, workdir: str, artifact) -> CheckResult:
        try:
            out = self.fn(workdir, artifact)
        except Exception as e:  # a failing check is a verdict, not a crash
            return CheckResult(ok=False, signal=0.0, kind=self.kind,
                               errors=[{"message": f"{type(e).__name__}: {e}"}])
        if isinstance(out, tuple):
            ok, signal, detail = (list(out) + [{}])[:3]
            return CheckResult(ok=bool(ok), signal=float(signal), kind=self.kind, detail=detail or {})
        return CheckResult(ok=bool(out), signal=2.0 if out else 0.0, kind=self.kind)


class SchemaChecker:
    """Validate a produced dict/record against required keys + type predicates —
    a minimal data-domain truth signal with no external deps."""

    kind = "schema"

    def __init__(self, required: dict[str, type]):
        self.required = required

    def check(self, workdir: str, artifact) -> CheckResult:
        if not isinstance(artifact, dict):
            return CheckResult(False, 0.0, self.kind, errors=[{"message": "artifact is not a record"}])
        errors = []
        for key, typ in self.required.items():
            if key not in artifact:
                errors.append({"message": f"missing key: {key}"})
            elif not isinstance(artifact[key], typ):
                errors.append({"message": f"{key} expected {typ.__name__}, got {type(artifact[key]).__name__}"})
        ok = not errors
        return CheckResult(ok=ok, signal=2.0 if ok else 0.0, kind=self.kind, errors=errors)
