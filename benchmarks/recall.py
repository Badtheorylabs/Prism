"""Retrieval-recall benchmark — the north-star metric.

Replays real git commits: for each commit, use its message as the task, run
get_context against the repo state *before* the commit, and measure whether the
files the commit actually changed appear in the returned context (files_to_edit
+ periphery). High recall => small models downstream get what they need.

Usage:
    python benchmarks/recall.py <git-repo> [--n 20] [--budget 8000]

Note: this checks out past commits, so run it on a clean repo you don't mind
having its HEAD moved (it restores HEAD at the end).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile

from prism.context import get_context


def _git(repo: str, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", repo, *args], capture_output=True, text=True
    ).stdout.strip()


def _recent_commits(repo: str, n: int) -> list[str]:
    out = _git(repo, "log", "--format=%H", "-n", str(n + 1))
    return out.splitlines()


def _commit_message(repo: str, sha: str) -> str:
    return _git(repo, "log", "-1", "--format=%s%n%b", sha)


def _changed_py_files(repo: str, sha: str) -> list[str]:
    out = _git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", sha)
    return [f for f in out.splitlines() if f.endswith(".py")]


def run(repo: str, n: int, budget: int) -> int:
    original_head = _git(repo, "rev-parse", "HEAD")
    commits = _recent_commits(repo, n)
    if len(commits) < 2:
        print("need at least 2 commits to benchmark")
        return 1

    total_hits = 0
    total_expected = 0
    evaluated = 0

    try:
        for sha in commits[:n]:
            changed = _changed_py_files(repo, sha)
            if not changed:
                continue
            task = _commit_message(repo, sha)
            # index the state *before* this commit
            _git(repo, "checkout", "-q", f"{sha}~1")
            with tempfile.TemporaryDirectory() as td:
                db = f"{td}/index.db"
                payload = get_context(repo, task, token_budget=budget, db_path=db, reindex=True)
            returned_files = {f["path"] for f in payload["files_to_edit"]}
            returned_files |= {p["file"] for p in payload["periphery"]}

            hits = sum(1 for f in changed if f in returned_files)
            total_hits += hits
            total_expected += len(changed)
            evaluated += 1
            recall = hits / len(changed)
            print(f"{sha[:8]}  recall {recall:5.0%}  ({hits}/{len(changed)})  {task.splitlines()[0][:50]}")
    finally:
        _git(repo, "checkout", "-q", original_head)

    if total_expected == 0:
        print("no python-changing commits found")
        return 1
    print(f"\nOVERALL FILE RECALL: {total_hits/total_expected:.1%} "
          f"over {evaluated} commits ({total_hits}/{total_expected} files)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--budget", type=int, default=8000)
    args = ap.parse_args()
    raise SystemExit(run(args.repo, args.n, args.budget))
