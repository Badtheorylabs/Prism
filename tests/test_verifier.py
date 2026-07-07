"""Layer 4 (execution-grounded Verification) tests.

Execution is the primary signal; self-critique is demoted to an optional,
separate-verifier-only hint. Tests use run_tests=False for speed/determinism
(execution-with-pytest is covered in test_pivot.py)."""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prism.agent import ProposedFile  # noqa: E402
from prism.verifier import best_of_n, make_verdict, verify_and_repair  # noqa: E402

SAMPLE = '''
def login(user, password):
    """Authenticate a user."""
    return True
'''


def _repo(root: str) -> None:
    with open(os.path.join(root, "api.py"), "w") as fh:
        fh.write(SAMPLE)


def test_execution_is_primary_signal():
    with tempfile.TemporaryDirectory() as root:
        _repo(root)
        good = make_verdict(root, [ProposedFile("api.py", "def f():\n    return 1\n")], run_tests=False)
        bad = make_verdict(root, [ProposedFile("api.py", "def f(:\n")], run_tests=False)
        assert good.ok and good.accepted and good.signal >= 1.0
        assert not bad.ok and not bad.accepted and bad.signal == 0.0
        # signal, not self-judgment, drives the score
        assert good.score > bad.score


def test_self_critique_cannot_gate_and_needs_separate_verifier():
    with tempfile.TemporaryDirectory() as root:
        _repo(root)
        files = [ProposedFile("api.py", "def f():\n    return 1\n")]
        # adversarial=True but NO verifier_client -> critique stays None (generator
        # is never allowed to judge itself)
        v = make_verdict(root, files, task="t", adversarial=True, verifier_client=None, run_tests=False)
        assert v.critique is None
        assert v.accepted is True  # acceptance is purely execution-based


class _BadThenGood:
    def __init__(self):
        self.calls = 0

    def chat(self, system, user, temperature=0.1):
        self.calls += 1
        if self.calls == 1:
            return "FILE: api.py\n```python\ndef f(:\n```"  # syntax error
        return "FILE: api.py\n```python\ndef f():\n    return True\n```"


def test_repair_loop_recovers_via_execution():
    with tempfile.TemporaryDirectory() as root:
        _repo(root)
        v = verify_and_repair(root, "make login safe", _BadThenGood(),
                              max_attempts=3, run_tests=False, on_event=lambda m: None)
        assert v.ok and v.accepted
        assert v.files and "return True" in v.files[0].code


def test_best_of_n_ranks_by_execution_not_selfscore():
    with tempfile.TemporaryDirectory() as root:
        _repo(root)

        def gen(feedback, temperature):
            if temperature < 0.2:
                return [ProposedFile("api.py", "def f(:\n")]  # broken
            return [ProposedFile("api.py", "def f():\n    return True\n")]  # valid

        winner, all_v = best_of_n(root, "task", client=None, n=3, run_tests=False,
                                  generate=gen, on_event=lambda m: None)
        assert winner.ok and winner.accepted
        assert winner.signal >= 1.0
        assert any(not v.ok for v in all_v)  # a broken candidate existed and lost


if __name__ == "__main__":
    test_execution_is_primary_signal()
    test_self_critique_cannot_gate_and_needs_separate_verifier()
    test_repair_loop_recovers_via_execution()
    test_best_of_n_ranks_by_execution_not_selfscore()
    print("all verifier tests passed")
