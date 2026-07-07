"""Checker tests — proves the truth loop is domain-agnostic (coding AND beyond)."""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prism.agent import ProposedFile  # noqa: E402
from prism.checkers import CodingChecker, FunctionChecker, SchemaChecker  # noqa: E402
from prism.trace import Tracer  # noqa: E402


def test_coding_checker():
    with tempfile.TemporaryDirectory() as root:
        with open(os.path.join(root, "m.py"), "w") as fh:
            fh.write("x = 1\n")
        c = CodingChecker(run_tests=False)
        good = c.check(root, [ProposedFile("m.py", "def f():\n    return 1\n")])
        bad = c.check(root, [ProposedFile("m.py", "def f(:\n")])
        assert good.ok and good.signal >= 1.0 and good.kind == "compile"
        assert not bad.ok and bad.signal == 0.0


def test_schema_checker_data_domain():
    c = SchemaChecker(required={"name": str, "age": int})
    ok = c.check(".", {"name": "ada", "age": 36})
    bad = c.check(".", {"name": "ada"})  # missing age
    wrong = c.check(".", {"name": "ada", "age": "old"})  # wrong type
    assert ok.ok and ok.kind == "schema"
    assert not bad.ok and any("missing" in e["message"] for e in bad.errors)
    assert not wrong.ok


def test_function_checker_any_domain():
    # a non-coding "task": produce the reversed list; the checker is the truth loop
    def is_reversed(workdir, artifact):
        expected = [3, 2, 1]
        ok = artifact == expected
        return ok, 2.0 if ok else 0.0, {"expected": expected, "got": artifact}

    c = FunctionChecker(kind="data", fn=is_reversed)
    assert c.check(".", [3, 2, 1]).ok
    v = c.check(".", [1, 2, 3])
    assert not v.ok and v.detail["got"] == [1, 2, 3]
    # a predicate that raises is a verdict, not a crash
    boom = FunctionChecker(kind="x", fn=lambda w, a: 1 / 0)
    assert not boom.check(".", None).ok


def test_checker_feeds_trace_spine_generically():
    # the SAME runtime spine records a non-coding check
    with tempfile.TemporaryDirectory() as root:
        tr = Tracer(root=root, task="reverse a list", label="domain:data")
        checker = FunctionChecker(kind="data", fn=lambda w, a: a == [3, 2, 1])
        tr.context(fits_in=0, budget=0)
        res = checker.check(root, [3, 2, 1])
        tr.check(kind=res.kind, ok=res.ok, signal=res.signal)
        tr.outcome(resolved=res.ok, won_by="domain-check", attempts=1)
        s = tr.close()
        assert s["resolved"] and s["checks"] == 1 and s["checks_passed"] == 1


if __name__ == "__main__":
    test_coding_checker()
    test_schema_checker_data_domain()
    test_function_checker_any_domain()
    test_checker_feeds_trace_spine_generically()
    print("all checker tests passed")
