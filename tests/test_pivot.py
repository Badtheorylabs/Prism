"""Tests for the research-driven pivot: execution grounding, agentic exploration,
LSP navigation, adaptive TTC, and <think>-tag handling. Pure stdlib + fakes."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prism.agent import ProposedFile  # noqa: E402
from prism.context import build_repo_index, index_path_for  # noqa: E402
from prism.execution import ground_check  # noqa: E402
from prism.explorer import discovered_files, explore, rank_lines  # noqa: E402
from prism.graph import CodeGraph  # noqa: E402
from prism.llm import strip_think  # noqa: E402
from prism.lsp import GraphLSP  # noqa: E402
from prism.ttc import adaptive_solve  # noqa: E402

AUTH = '''
def hash_password(password):
    """Hash a password."""
    return password[::-1]
'''

API = '''
from auth import hash_password


def login(user, password, stored):
    """Log a user in via password hashing."""
    if hash_password(password) == stored:
        return audit_trail(user)
    return False


def audit_trail(user):
    """Persist an entry to the ledger."""
    return {"user": user}
'''

TEST_OK = '''
from api import login


def test_login():
    assert login("u", "p", "p"[::-1])
'''


def _repo(root: str, with_test: bool = True) -> None:
    with open(os.path.join(root, "auth.py"), "w") as fh:
        fh.write(AUTH)
    with open(os.path.join(root, "api.py"), "w") as fh:
        fh.write(API)
    if with_test:
        os.makedirs(os.path.join(root, "tests"), exist_ok=True)
        with open(os.path.join(root, "tests", "test_api.py"), "w") as fh:
            fh.write(TEST_OK)


# ---- L2: think-tag stripping -------------------------------------------------

def test_strip_think_removes_reasoning():
    assert strip_think("<think>reasoning here</think>FILE: a.py") == "FILE: a.py"
    # unbalanced (opening lost by truncation) -> keep post-reasoning
    assert strip_think("blah blah</think>ANSWER") == "ANSWER"
    assert strip_think("no tags") == "no tags"


# ---- L4: execution grounding actually runs tests -----------------------------

def test_ground_check_runs_tests_in_overlay():
    import importlib.util
    if importlib.util.find_spec("pytest") is None:
        return  # pytest not available in this env; skip the test-run assertion
    with tempfile.TemporaryDirectory() as root:
        _repo(root, with_test=True)
        # a valid edit that keeps the test green
        good = ground_check(root, [ProposedFile("auth.py", AUTH)], run_tests=True)
        assert good.compiled
        # an edit that BREAKS the login contract -> test fails -> execution catches it
        broken = ground_check(
            root,
            [ProposedFile("auth.py", "def hash_password(password):\n    return 'WRONG'\n")],
            run_tests=True,
        )
        assert broken.compiled  # it compiles fine...
        if broken.tests_ran:
            assert not broken.tests_passed  # ...but execution catches the regression
            assert broken.signal < good.signal


# ---- L1: agentic exploration + LSP + line ranking ----------------------------

def test_exploration_discovers_via_graph_edges():
    with tempfile.TemporaryDirectory() as root:
        _repo(root, with_test=False)
        db = index_path_for(root)
        build_repo_index(root, db)
        graph = CodeGraph(db)
        # query mentions hashing; login should surface as a structural discovery
        disc = explore(graph, "change password hashing", rounds=2)
        quals = {d.qualname for d in disc}
        assert any("hash_password" in q for q in quals)  # anchor
        files = {f["file"] for f in discovered_files(disc)}
        assert "auth.py" in files
        # some discovery came from a graph hop, not just retrieval
        assert any(d.hops > 0 for d in disc)
        graph.close()


def test_lsp_find_references():
    with tempfile.TemporaryDirectory() as root:
        _repo(root, with_test=False)
        db = index_path_for(root)
        build_repo_index(root, db)
        graph = CodeGraph(db)
        lsp = GraphLSP(graph)
        assert lsp.find_definition("hash_password") is not None
        refs = lsp.find_referencing_symbols("auth.hash_password")
        # api.login calls hash_password -> it must show up as a referencing symbol
        assert any("login" in r.qualname for r in refs)
        graph.close()


def test_rank_lines_finds_relevant_line():
    with tempfile.TemporaryDirectory() as root:
        _repo(root, with_test=False)
        db = index_path_for(root)
        build_repo_index(root, db)
        graph = CodeGraph(db)
        windows = rank_lines(graph, "auth.py", "hash password", top_k=2)
        assert windows and any("hash_password" in w["text"] for w in windows)
        graph.close()


# ---- L2+L4: adaptive test-time compute --------------------------------------

class _OneShotGood:
    """Solves on the first try — TTC should stop immediately (no wasted compute)."""

    def chat(self, system, user, temperature=0.1):
        return "FILE: api.py\n```python\ndef f():\n    return 1\n```"


def test_ttc_stops_early_when_easy():
    with tempfile.TemporaryDirectory() as root:
        _repo(root, with_test=False)
        res = adaptive_solve(root, "trivial change", _OneShotGood(),
                             max_samples=5, run_tests=False, on_event=lambda m: None)
        assert res.stopped == "solved"
        assert res.attempts == 1  # early exit — did not burn the full budget


class _AlwaysBroken:
    def chat(self, system, user, temperature=0.1):
        return "FILE: api.py\n```python\ndef f(:\n```"


def test_ttc_bails_when_stuck():
    with tempfile.TemporaryDirectory() as root:
        _repo(root, with_test=False)
        res = adaptive_solve(root, "impossible", _AlwaysBroken(),
                             max_samples=6, stuck_after=2, run_tests=False, on_event=lambda m: None)
        assert res.stopped == "stuck"
        assert res.attempts < 6  # gave up below the competence threshold, saved compute


# ---- Regressions for Codex-found P1 bugs ------------------------------------

def test_missing_pytest_is_never_a_false_green():
    """Regression: a missing/failed pytest must NOT be reported as tests passing."""
    import importlib.util
    with tempfile.TemporaryDirectory() as root:
        _repo(root, with_test=True)  # has a tests/ dir
        # an import-preserving edit so the test file can still collect + run
        r = ground_check(root, [ProposedFile("auth.py", AUTH)], run_tests=True)
        assert r.compiled
        if importlib.util.find_spec("pytest") is None:
            # the exact false-green Codex caught: pytest absent -> must be ran=False
            assert r.tests_ran is False
            assert r.tests_passed is False
            assert r.signal < 2.5  # NOT the phantom "tests passed" bonus
        else:
            assert r.tests_ran and r.tests_passed  # real run, real pass


def test_execution_rejects_unsafe_paths():
    """Regression: absolute / traversal model paths must not escape the overlay."""
    import os as _os
    probe = _os.path.join(tempfile.gettempdir(), "prism_escape_probe_test.py")
    if _os.path.exists(probe):
        _os.remove(probe)
    with tempfile.TemporaryDirectory() as root:
        _repo(root, with_test=False)
        r = ground_check(root, [ProposedFile(probe, "print('escaped')\n")], run_tests=False)
        assert not _os.path.exists(probe), "unsafe absolute path escaped the sandbox!"
        assert not r.compiled
        assert any(e.get("kind") == "unsafe_path" for e in r.errors)
        # traversal form too
        r2 = ground_check(root, [ProposedFile("../../evil.py", "x=1\n")], run_tests=False)
        assert not r2.compiled
        assert any(e.get("kind") == "unsafe_path" for e in r2.errors)


def test_safe_join():
    from prism.pathsafe import safe_join
    assert safe_join("/repo", "a/b.py") == os.path.abspath("/repo/a/b.py")
    assert safe_join("/repo", "/etc/passwd") is None
    assert safe_join("/repo", "../../x") is None
    assert safe_join("/repo", "") is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all pivot tests passed")
