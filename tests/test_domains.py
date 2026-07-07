"""Domain pack tests — the harness works beyond coding (data reconciliation)."""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prism.domains import data_reconciliation, run_data_reconciliation  # noqa: E402
from prism.trace import Tracer  # noqa: E402

ROWS = [
    {"region": "east", "amt": 10},
    {"region": "east", "amt": 20},
    {"region": "west", "amt": 20},
]
# ground truth: east=30, west=20


def test_reconciliation_check_is_ground_truth():
    _ctx, _prompt, check, expected = data_reconciliation(ROWS, "region", "amt")
    assert expected == {"east": 30, "west": 20}
    assert check({"east": 30, "west": 20}).ok
    bad = check({"east": 25, "west": 20})
    assert not bad.ok and any("east" in e["message"] for e in bad.errors)
    assert not check({"east": 30}).ok  # missing west
    assert not check("not json").ok


class _WrongThenRight:
    """Model gets the totals wrong first, then fixes them from the discrepancy."""

    def __init__(self):
        self.n = 0

    def chat(self, system, user, temperature=0.1, think=None):
        self.n += 1
        if self.n == 1:
            return 'Here you go: {"east": 10, "west": 5}'  # wrong
        return '{"east": 30, "west": 20}'  # correct


def test_non_coding_task_runs_through_harness_and_repairs():
    with tempfile.TemporaryDirectory() as root:
        tr = Tracer(root=root, task="sum amt by region", label="domain:data")
        res = run_data_reconciliation(_WrongThenRight(), ROWS, "region", "amt",
                                      max_attempts=3, tracer=tr, on_event=lambda m: None)
        summary = tr.close(resolved=res.resolved)
        # same harness: it repaired a wrong answer into a reconciled one
        assert res.resolved and res.attempts == 2
        assert res.artifact == {"east": 30, "west": 20}
        # the trace spine recorded the non-coding run identically to a coding one
        assert summary["resolved"] and summary["checks"] == 2
        # the check kind is the DOMAIN signal, not tests
        events = [e for e in _read(tr.run_dir) if e["kind"] == "check"]
        assert all(e["data"]["check_kind"] == "reconcile" for e in events)


def _read(run_dir):
    import json
    with open(os.path.join(run_dir, "trace.jsonl")) as fh:
        return [json.loads(x) for x in fh if x.strip()]


def test_gives_up_when_never_correct():
    class AlwaysWrong:
        def chat(self, system, user, temperature=0.1, think=None):
            return '{"east": 1, "west": 1}'
    res = run_data_reconciliation(AlwaysWrong(), ROWS, "region", "amt", max_attempts=2)
    assert not res.resolved and res.attempts == 2
    assert res.result.errors  # reports the remaining discrepancies


if __name__ == "__main__":
    test_reconciliation_check_is_ground_truth()
    test_non_coding_task_runs_through_harness_and_repairs()
    test_gives_up_when_never_correct()
    print("all domain tests passed")
