"""End-to-end smoke test on a tiny synthetic repo. Pure stdlib + pytest-free
(uses assert; run with `python -m pytest` or directly)."""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prism.attention import build_attention_manifest  # noqa: E402
from prism.context import (  # noqa: E402
    analyze_impact,
    build_repo_index,
    get_context,
    inspect_code,
    search_code,
)
from prism.graph import CodeGraph  # noqa: E402
from prism.verify import verify_snippet  # noqa: E402

SAMPLE_AUTH = '''
def hash_password(password: str) -> str:
    """Hash a plaintext password for storage."""
    return password[::-1]


def verify_password(password: str, hashed: str) -> bool:
    """Check a password against its stored hash."""
    return hash_password(password) == hashed
'''

SAMPLE_API = '''
from auth import verify_password


class RateLimiter:
    """Limits requests per client."""

    def allow(self, client_id: str) -> bool:
        return True


def login(user, password, stored_hash):
    """Authenticate a user by password."""
    if verify_password(password, stored_hash):
        return {"ok": True}
    return {"ok": False}
'''

SAMPLE_TEST = '''
from api import login


def test_login_ok():
    assert login("u", "p", "p"[::-1])["ok"]
'''


def _make_repo(root: str) -> None:
    with open(os.path.join(root, "auth.py"), "w") as fh:
        fh.write(SAMPLE_AUTH)
    with open(os.path.join(root, "api.py"), "w") as fh:
        fh.write(SAMPLE_API)
    os.makedirs(os.path.join(root, "tests"), exist_ok=True)
    with open(os.path.join(root, "tests", "test_api.py"), "w") as fh:
        fh.write(SAMPLE_TEST)


def test_index_and_context():
    with tempfile.TemporaryDirectory() as root:
        _make_repo(root)
        db = os.path.join(root, ".prism", "index.db")
        build_repo_index(root, db)

        g = CodeGraph(db)
        assert g.count() >= 5  # 4 funcs + RateLimiter + allow
        g.close()

        payload = get_context(root, "add rate limiting to login", token_budget=4000, db_path=db)
        assert payload["fits_in"] <= 4000
        assert payload["symbol_count"] >= 5
        # the api file (owns login + RateLimiter) should be an edit target
        edit_files = {f["path"] for f in payload["files_to_edit"]}
        assert "api.py" in edit_files
        # budget is respected
        assert payload["token_budget"] == 4000


def test_dependency_edges():
    with tempfile.TemporaryDirectory() as root:
        _make_repo(root)
        payload = get_context(root, "change password hashing", token_budget=4000)
        # auth.py owns hash/verify_password which api.login calls -> edge expected
        edit_files = {f["path"] for f in payload["files_to_edit"]}
        assert "auth.py" in edit_files


def test_import_aliases_and_self_calls_resolve():
    with tempfile.TemporaryDirectory() as root:
        with open(os.path.join(root, "helpers.py"), "w") as fh:
            fh.write(
                '''
def normalize(value: str) -> str:
    """Normalize a user-provided value."""
    return value.strip().lower()
'''
            )
        with open(os.path.join(root, "service.py"), "w") as fh:
            fh.write(
                '''
import helpers as h
from helpers import normalize as norm


class Service:
    def run(self, value: str) -> str:
        return self.clean(value)

    def clean(self, value: str) -> str:
        return h.normalize(norm(value))
'''
            )
        db = os.path.join(root, ".prism", "index.db")
        build_repo_index(root, db)
        g = CodeGraph(db)
        edges = {
            (src.qualname, dst.qualname, kind)
            for src, dst, kind in g.edges_touching_files(["service.py", "helpers.py"])
        }
        g.close()

        assert ("service.Service.run", "service.Service.clean", "calls") in edges
        assert ("service.Service.clean", "helpers.normalize", "calls") in edges


def test_line_level_inspection_and_quality():
    with tempfile.TemporaryDirectory() as root:
        with open(os.path.join(root, "risky.py"), "w") as fh:
            fh.write(
                '''
API_TOKEN = "secret-value"


def run(value):
    print(value)
    return eval(value)
'''
            )

        payload = inspect_code(root, symbol="risky.run", radius=1)
        assert payload["ok"]
        assert payload["symbol"]["qualname"] == "risky.run"
        line_roles = {line["line_no"]: line["role"] for line in payload["lines"]}
        assert line_roles[6] == "call_or_expression"
        codes = {finding["code"] for finding in payload["quality_findings"]}
        assert "debug_print" in codes
        assert "dynamic_exec" in codes

        file_payload = inspect_code(root, file="risky.py")
        assert file_payload["ok"]
        assert file_payload["chunks"]
        file_codes = {finding["code"] for finding in file_payload["quality_findings"]}
        assert "possible_secret" in file_codes


def test_search_and_impact_tools():
    with tempfile.TemporaryDirectory() as root:
        _make_repo(root)
        results = search_code(root, "rate limiting login", top_k=5)
        assert results["ok"]
        assert any(hit["qualname"] == "api.login" for hit in results["symbols"])
        assert any(hit["file"] == "api.py" for hit in results["files"])
        assert results["chunks"]

        impact = analyze_impact(root, symbol="api.login", hops=2)
        assert impact["ok"]
        assert impact["target"]["symbol"] == "api.login"
        callers = {item["qualname"] for item in impact["direct_callers"]}
        callees = {item["qualname"] for item in impact["direct_callees"]}
        assert "tests.test_api.test_login_ok" in callers
        assert "auth.verify_password" in callees
        assert impact["risk"]["relevant_tests"] >= 1


def test_attention_manifest_covers_repo_with_handles():
    with tempfile.TemporaryDirectory() as root:
        _make_repo(root)
        payload = build_attention_manifest(
            root,
            task="add rate limiting to login",
            token_budget=2000,
        )
        assert payload["ok"]
        assert payload["mode"] == "virtual_whole_codebase_attention"
        assert payload["coverage"]["files"] >= 3
        assert payload["coverage"]["symbols"] >= 5
        assert payload["coverage"]["lines"] > 0
        assert payload["attention"]["symbols"]
        assert payload["attention"]["chunks"]
        handles = {item["handle"] for item in payload["attention"]["symbols"]}
        assert "symbol:api.login" in handles
        assert payload["navigation_protocol"]


def test_verify_catches_syntax():
    good = verify_snippet("def f():\n    return 1\n")
    assert good["ok"]
    bad = verify_snippet("def f(:\n    return\n")
    assert not bad["ok"]
    assert bad["errors"][0]["kind"] == "syntax"


if __name__ == "__main__":
    test_index_and_context()
    test_dependency_edges()
    test_import_aliases_and_self_calls_resolve()
    test_line_level_inspection_and_quality()
    test_search_and_impact_tools()
    test_attention_manifest_covers_repo_with_handles()
    test_verify_catches_syntax()
    print("all smoke tests passed")
