"""Verify hook: run cheap deterministic checks on an edit and return structured
errors the agent loop can feed back to a small model.

The compiler/test runner is a free "big model": it catches what a small model
misses. v1 supports Python compile-checking (always available) and pytest
(if installed / requested).
"""

from __future__ import annotations

import os
import py_compile
import subprocess
import tempfile
from typing import Optional


def _compile_check(path: str) -> list[dict]:
    errors: list[dict] = []
    try:
        py_compile.compile(path, doraise=True)
    except py_compile.PyCompileError as e:
        exc = e.exc_value
        line = getattr(exc, "lineno", None)
        errors.append(
            {"file": path, "line": line, "message": str(exc), "kind": "syntax"}
        )
    except FileNotFoundError:
        errors.append({"file": path, "line": None, "message": "file not found", "kind": "io"})
    return errors


def verify(
    paths: list[str],
    run_tests: bool = False,
    test_target: Optional[str] = None,
    cwd: Optional[str] = None,
) -> dict:
    """Verify one or more edited files.

    Args:
        paths: files to compile-check.
        run_tests: if True, also run pytest.
        test_target: optional pytest target (file/dir/node id).
        cwd: working dir for pytest.

    Returns:
        {ok, errors: [{file,line,message,kind}], tests: {ran, passed, failed, output}}
    """
    errors: list[dict] = []
    for p in paths:
        errors.extend(_compile_check(p))

    tests_info = {"ran": False, "passed": None, "failed": None, "output": ""}
    if run_tests and not errors:
        tests_info = _run_pytest(test_target, cwd)
        if tests_info["failed"]:
            errors.append(
                {
                    "file": test_target or ".",
                    "line": None,
                    "message": f"{tests_info['failed']} test(s) failed",
                    "kind": "test",
                }
            )

    return {"ok": len(errors) == 0, "errors": errors, "tests": tests_info}


def _run_pytest(target: Optional[str], cwd: Optional[str]) -> dict:
    import sys
    cmd = [sys.executable, "-m", "pytest", "-q"]
    if target:
        cmd.append(target)
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=300
        )
    except FileNotFoundError:
        return {"ran": False, "passed": None, "failed": None,
                "output": "pytest not available"}
    except subprocess.TimeoutExpired:
        return {"ran": True, "passed": None, "failed": 1,
                "output": "pytest timed out"}

    import re

    out = proc.stdout + proc.stderr
    low = out.lower()
    # Never report a phantom pass: missing pytest, no tests collected, or an
    # internal/usage error all mean tests did NOT run.
    if "no module named pytest" in low:
        return {"ran": False, "passed": None, "failed": None, "output": "pytest not installed"}
    if proc.returncode == 5 or "no tests ran" in low or proc.returncode in (2, 3, 4):
        return {"ran": False, "passed": None, "failed": None, "output": out[-4000:]}

    passed = failed = 0
    for line in out.splitlines():
        m_p = re.search(r"(\d+) passed", line)
        m_f = re.search(r"(\d+) (?:failed|error)", line)
        if m_p:
            passed = int(m_p.group(1))
        if m_f:
            failed = int(m_f.group(1))
    summary_seen = bool(re.search(r"\d+ (passed|failed|error|skipped|xfailed|deselected)", low))
    if not (summary_seen and proc.returncode in (0, 1)):
        return {"ran": False, "passed": None, "failed": None, "output": out[-4000:]}
    return {"ran": True, "passed": passed, "failed": failed, "output": out[-4000:]}


def verify_snippet(code: str) -> dict:
    """Convenience: compile-check a raw code string (used by agent loops)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(code)
        tmp = fh.name
    try:
        return verify([tmp])
    finally:
        os.unlink(tmp)
