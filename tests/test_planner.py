"""Layer 2 (Reasoning/Planning) tests. Pure stdlib; no model required for the
graph-derived path. A tiny fake client exercises the model-refined + execution
path without a network."""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prism.planner import Plan, make_plan, run_plan  # noqa: E402

SAMPLE_AUTH = '''
def hash_password(password: str) -> str:
    """Hash a plaintext password for storage."""
    return password[::-1]
'''

SAMPLE_API = '''
from auth import hash_password


def login(user, password, stored_hash):
    """Authenticate a user by password."""
    return hash_password(password) == stored_hash
'''


def _make_repo(root: str) -> None:
    with open(os.path.join(root, "auth.py"), "w") as fh:
        fh.write(SAMPLE_AUTH)
    with open(os.path.join(root, "api.py"), "w") as fh:
        fh.write(SAMPLE_API)


class _FakeClient:
    """Stands in for Ollama: plans in JSON, then emits a trivial valid edit."""

    model = "fake"

    def available(self) -> bool:
        return True

    def chat(self, system: str, user: str) -> str:
        if "planning module" in system:
            return (
                '[{"title":"Update login","detail":"add hashing","target":"api.py",'
                '"kind":"edit"},{"title":"Check hashing","detail":"review",'
                '"target":"auth.py","kind":"investigate"}]'
            )
        return "FILE: api.py\n```python\ndef login():\n    return True\n```"


def test_graph_derived_plan_needs_no_model():
    with tempfile.TemporaryDirectory() as root:
        _make_repo(root)
        plan = make_plan(root, "change password hashing", token_budget=2000)
        assert isinstance(plan, Plan)
        assert plan.source == "graph"
        assert len(plan.steps) >= 1
        # every step targets a real file Context ranked as an edit surface
        targets = {s.target for s in plan.steps}
        assert "auth.py" in targets or "api.py" in targets
        assert all(s.status == "pending" for s in plan.steps)


def test_model_plan_and_bounded_execution():
    with tempfile.TemporaryDirectory() as root:
        _make_repo(root)
        client = _FakeClient()
        plan = make_plan(root, "add rate limiting to login", token_budget=2000, client=client)
        assert plan.source == "model"
        assert len(plan.steps) == 2

        plan = run_plan(plan, root, client, token_budget=2000, verify=True, on_event=lambda m: None)
        edit_steps = [s for s in plan.steps if s.kind == "edit"]
        invest_steps = [s for s in plan.steps if s.kind == "investigate"]
        # edit steps run and verify; each gets its OWN bounded dossier
        assert all(s.status == "verified" for s in edit_steps)
        assert all(s.context_tokens > 0 for s in edit_steps)
        assert all(s.context_tokens <= 2000 for s in edit_steps)
        # investigate steps are not executed as edits
        assert all(s.status == "skipped" for s in invest_steps)


if __name__ == "__main__":
    test_graph_derived_plan_needs_no_model()
    test_model_plan_and_bounded_execution()
    print("all planner tests passed")
