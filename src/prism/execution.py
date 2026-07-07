"""Execution grounding — the reliable external verifier (Layer 4 core).

The research verdict: reliability comes from EXECUTION (tests, compile,
type-check, diagnostics), not from a weak model judging itself. This module is
that external signal. It applies a proposed edit to a throwaway overlay of the
repo and actually runs checks against it, returning a structured report the rest
of the system ranks by.

Checks (Python-first, best-effort for others):
  - compile:   py_compile every edited/overlaid .py  (always available)
  - tests:     pytest in the overlay                 (if pytest present)
  - types:     mypy on edited files                  (if mypy present, opt-in)

Non-Python files get a light structural check (non-empty, balanced braces) since
we can't run their toolchains from stdlib.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

from .pathsafe import safe_join
from .verify import verify

_IGNORE = {"__pycache__", ".prism", ".git", "node_modules", ".venv", "venv", "build", "dist"}


@dataclass
class ExecReport:
    ok: bool
    compiled: bool
    tests_ran: bool
    tests_passed: bool
    types_ok: bool | None  # None = not run
    errors: list[dict] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    @property
    def signal(self) -> float:
        """A single execution-grounded reliability score in [0, 3].

        Execution evidence dominates: passing tests > compiles-clean > broken.
        This is what best-of-N should rank by (NOT a self-critique score).
        """
        if not self.compiled:
            return 0.0
        s = 1.0  # compiles
        if self.tests_ran:
            s += 1.5 if self.tests_passed else -0.5
        if self.types_ok is True:
            s += 0.5
        elif self.types_ok is False:
            s -= 0.25
        return round(max(0.0, s), 3)


def _copy_repo(repo_root: str, dst: str) -> None:
    def ignore(_dir, names):
        return [n for n in names if n in _IGNORE]
    for entry in os.listdir(repo_root):
        if entry in _IGNORE:
            continue
        src = os.path.join(repo_root, entry)
        d = os.path.join(dst, entry)
        if os.path.isdir(src):
            shutil.copytree(src, d, ignore=ignore, dirs_exist_ok=True)
        else:
            shutil.copy2(src, d)


def _mypy_check(paths: list[str], cwd: str) -> tuple[bool | None, str]:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "mypy", "--no-error-summary", "--ignore-missing-imports", *paths],
            cwd=cwd, capture_output=True, text=True, timeout=120,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None, "mypy not available"
    return proc.returncode == 0, (proc.stdout + proc.stderr)[-2000:]


def ground_check(
    repo_root: str,
    files: list,  # list of objects with .path and .code (ProposedFile-like)
    run_tests: bool = True,
    run_types: bool = False,
    test_target: str | None = None,
) -> ExecReport:
    """Overlay `files` onto a copy of the repo and run execution checks."""
    errors: list[dict] = []
    py_files = [f for f in files if f.path.endswith(".py")]
    other_files = [f for f in files if not f.path.endswith(".py")]

    with tempfile.TemporaryDirectory() as overlay:
        _copy_repo(repo_root, overlay)
        # apply the proposed edit into the overlay — model paths are untrusted,
        # so every write is sandboxed via safe_join (no absolute / `..` escapes).
        written_py: list[str] = []
        escaped = False
        for f in files:
            dst = safe_join(overlay, f.path)
            if dst is None:
                escaped = True
                errors.append({"file": f.path, "line": None,
                               "message": "rejected unsafe path (absolute or traversal)",
                               "kind": "unsafe_path"})
                continue
            os.makedirs(os.path.dirname(dst) or overlay, exist_ok=True)
            with open(dst, "w", encoding="utf-8") as fh:
                fh.write(f.code)
            if f.path.endswith(".py"):
                written_py.append(dst)

        # 1. compile check (stdlib, always) + light non-python check
        compiled = not escaped  # an unsafe path fails the candidate outright
        if written_py:
            res = verify(written_py)
            if not res["ok"]:
                compiled = False
                for e in res["errors"]:
                    # remap overlay path back to repo-relative
                    e = dict(e)
                    e["file"] = os.path.relpath(e["file"], overlay) if e.get("file", "").startswith(overlay) else e.get("file")
                    errors.append(e)
        for f in other_files:
            if not f.code.strip():
                compiled = False
                errors.append({"file": f.path, "line": None, "message": "empty file", "kind": "empty"})
            elif f.code.count("{") != f.code.count("}"):
                compiled = False
                errors.append({"file": f.path, "line": None, "message": "unbalanced braces", "kind": "syntax"})

        # 2. tests (only if compile passed and pytest is present)
        tests_ran = tests_passed = False
        test_detail = ""
        if compiled and run_tests:
            tinfo = _run_pytest(overlay, test_target)
            tests_ran = tinfo["ran"]
            tests_passed = tinfo["ran"] and (tinfo["failed"] in (0, None))
            test_detail = tinfo["output"]
            if tests_ran and not tests_passed:
                errors.append({"file": test_target or ".", "line": None,
                               "message": f"{tinfo['failed']} test(s) failed", "kind": "test"})

        # 3. types (opt-in)
        types_ok: bool | None = None
        types_detail = ""
        if compiled and run_types and written_py:
            types_ok, types_detail = _mypy_check([os.path.basename(p) for p in written_py], overlay)

    report = ExecReport(
        ok=compiled and (not tests_ran or tests_passed) and (types_ok is not False),
        compiled=compiled,
        tests_ran=tests_ran,
        tests_passed=tests_passed,
        types_ok=types_ok,
        errors=errors,
        detail={"tests": test_detail[-2000:], "types": types_detail},
    )
    return report


def _run_pytest(cwd: str, target: str | None) -> dict:
    """Run pytest in `cwd`. Critically: only report ran=True when pytest ACTUALLY
    executed collected tests. Missing pytest, no-tests-collected, and internal
    errors all report ran=False so they can NEVER be mistaken for a passing run.
    """
    import re

    # Use THIS interpreter, not whatever `python` resolves to on PATH — otherwise
    # a system python without pytest silently makes every run report "no tests".
    cmd = [sys.executable, "-m", "pytest", "-q", "-x"]
    if target:
        cmd.append(target)
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=300)
    except FileNotFoundError:
        return {"ran": False, "passed": None, "failed": None, "output": "python not available"}
    except subprocess.TimeoutExpired:
        return {"ran": True, "passed": None, "failed": 1, "output": "pytest timed out"}

    out = proc.stdout + proc.stderr
    low = out.lower()

    # pytest not installed: `python -m pytest` exits non-zero with this message.
    if "no module named pytest" in low:
        return {"ran": False, "passed": None, "failed": None, "output": "pytest not installed"}
    # pytest exit codes: 5 = no tests collected; 2/3/4 = interrupt/internal/usage.
    if proc.returncode == 5 or "no tests ran" in low:
        return {"ran": False, "passed": None, "failed": None, "output": out[-2000:]}
    if proc.returncode in (2, 3, 4):
        return {"ran": False, "passed": None, "failed": None, "output": out[-2000:]}

    passed = failed = 0
    for line in out.splitlines():
        mp = re.search(r"(\d+) passed", line)
        mf = re.search(r"(\d+) (?:failed|error)", line)
        if mp:
            passed = int(mp.group(1))
        if mf:
            failed = int(mf.group(1))

    # Require POSITIVE evidence that collected tests actually executed: either a
    # parsed summary count, or a clean exit (0) / test-failure exit (1) that also
    # produced a pytest-style summary. Absent that, we did NOT verify tests.
    summary_seen = bool(re.search(r"\d+ (passed|failed|error|skipped|xfailed|deselected)", low))
    ran = summary_seen and proc.returncode in (0, 1)
    if not ran:
        return {"ran": False, "passed": None, "failed": None, "output": out[-2000:]}
    return {"ran": True, "passed": passed, "failed": failed, "output": out[-2000:]}
