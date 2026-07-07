"""Eval/Trace spine tests — the runtime memory every layer plugs into."""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prism.agent import ProposedFile  # noqa: E402
from prism.trace import NULL_TRACER, Tracer, load_run  # noqa: E402
from prism.verifier import verify_and_repair  # noqa: E402


def test_tracer_records_and_persists():
    with tempfile.TemporaryDirectory() as root:
        tr = Tracer(root=root, task="demo", label="unit")
        tr.phase("start")
        tr.context(fits_in=812, budget=4000, files_to_edit=["api.py"], periphery=5)
        tr.model_call(role="edit", think=False, latency=1.2, output_chars=340)
        tr.check(kind="tests", ok=True, signal=2.5, tests_ran=True, tests_passed=True)
        tr.outcome(resolved=True, won_by="repair", attempts=1)
        summary = tr.close()

        assert summary["resolved"] is True
        assert summary["won_by"] == "repair"
        assert summary["checks"] == 1 and summary["checks_passed"] == 1
        assert summary["model_calls"] == 1
        # persisted + replayable
        loaded = load_run(tr.run_dir)
        assert loaded["summary"]["resolved"] is True
        kinds = [e["kind"] for e in loaded["events"]]
        assert "context" in kinds and "check" in kinds and "outcome" in kinds


def test_null_tracer_is_noop():
    # a disabled tracer records nothing and writes nothing
    NULL_TRACER.phase("x")
    NULL_TRACER.check(kind="tests", ok=True)
    assert NULL_TRACER.summarize()["checks"] == 0
    assert NULL_TRACER.run_dir is None


class _BadThenGood:
    def __init__(self):
        self.calls = 0

    def chat(self, system, user, temperature=0.1, think=None):
        self.calls += 1
        if self.calls == 1:
            return "FILE: api.py\n```python\ndef f(:\n```"
        return "FILE: api.py\n```python\ndef f():\n    return True\n```"


def test_repair_loop_emits_trace():
    with tempfile.TemporaryDirectory() as root:
        with open(os.path.join(root, "api.py"), "w") as fh:
            fh.write("def f():\n    return 1\n")
        tr = Tracer(root=root, task="fix f", label="repair")
        v = verify_and_repair(root, "fix f", _BadThenGood(), max_attempts=3,
                              run_tests=False, tracer=tr, on_event=lambda m: None)
        summary = tr.close()
        assert v.accepted
        # the loop recorded: 1 failed attempt + repair + a winning attempt
        assert summary["model_calls"] >= 2
        assert summary["checks"] >= 2
        assert summary["resolved"] is True
        assert summary["won_by"] == "repair"


if __name__ == "__main__":
    test_tracer_records_and_persists()
    test_null_tracer_is_noop()
    test_repair_loop_emits_trace()
    print("all trace tests passed")
