"""Memory tests — runtime/skill memory built on persisted traces."""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prism.memory import Memory  # noqa: E402
from prism.trace import Tracer  # noqa: E402


def _run(root, task, resolved, won_by, attempts, files):
    tr = Tracer(root=root, task=task, label="test")
    tr.context(fits_in=100, budget=4000, files_to_edit=files)
    tr.check(kind="tests", ok=resolved, signal=2.5 if resolved else 0.0)
    tr.outcome(resolved=resolved, won_by=won_by, attempts=attempts, files=files)
    tr.close()


def test_memory_stats_and_recall():
    with tempfile.TemporaryDirectory() as root:
        _run(root, "add rate limiting to login", True, "ttc", 2, ["api.py", "limits.py"])
        _run(root, "add rate limit for signup", True, "repair", 1, ["api.py", "limits.py"])
        _run(root, "fix off by one in pagination", False, "repair", 3, ["paginate.py"])
        _run(root, "add caching to search", True, "repair", 1, ["search.py"])

        mem = Memory(root)
        stats = mem.stats()
        assert stats["runs"] == 4
        assert stats["resolved"] == 3
        assert stats["resolved_rate"] == 0.75
        assert "repair" in stats["wins_by_layer"]

        # recall for a NEW but similar task
        hint = mem.recall("add rate limiting to the login endpoint")
        assert hint["similar_tasks"], "should find similar past tasks"
        top = hint["similar_tasks"][0]["task"]
        assert "rate limit" in top
        # files that worked on the similar resolved tasks bubble up
        assert "limits.py" in hint["suggested_files"] or "api.py" in hint["suggested_files"]
        assert hint["suggested_approach"] in ("ttc", "repair")


def test_memory_co_changes():
    with tempfile.TemporaryDirectory() as root:
        _run(root, "t1", True, "repair", 1, ["api.py", "limits.py"])
        _run(root, "t2", True, "ttc", 2, ["api.py", "limits.py"])
        _run(root, "t3", True, "repair", 1, ["api.py"])  # single file, no pair
        mem = Memory(root)
        cc = mem.co_changes(min_count=2)
        assert cc and cc[0]["files"] == ["api.py", "limits.py"] and cc[0]["count"] == 2


def test_memory_empty_repo():
    with tempfile.TemporaryDirectory() as root:
        mem = Memory(root)
        assert mem.stats() == {"runs": 0}
        assert mem.recall("anything")["similar_tasks"] == []
        assert mem.render_hint("anything") == ""


if __name__ == "__main__":
    test_memory_stats_and_recall()
    test_memory_co_changes()
    test_memory_empty_repo()
    print("all memory tests passed")
