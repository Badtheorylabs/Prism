"""Prism Memory (build-order item 5) — runtime/skill memory, built on the trace spine.

IMPORTANT framing: this is NOT memory baked into the context lens (that stays
stateless by design). This is the RUNTIME learning across runs. It reads the
persisted traces (`.prism/runs/*/summary.json`) — nothing new to maintain — and
surfaces evidence-based signals for a new task:

  - stats:      resolved rate, avg attempts, which layer wins most
  - similar:    past tasks close to this one (by term overlap) + their outcomes
  - co_changes: files that were edited together in SUCCESSFUL runs
  - recall:     a compact hint for a new task — "similar task X was solved by
                <layer> editing <files>" — that an agent MAY consult (opt-in),
                never auto-injected into context.

So the harness gets better as it accumulates runs, without making the context
tool stateful.
"""

from __future__ import annotations

import os
from collections import Counter

from .semantic import tokenize
from .trace import load_run


class Memory:
    def __init__(self, repo_root: str):
        self.repo_root = repo_root
        self.runs_dir = os.path.join(repo_root, ".prism", "runs")

    def _runs(self) -> list[dict]:
        if not os.path.isdir(self.runs_dir):
            return []
        out = []
        for rid in sorted(os.listdir(self.runs_dir)):
            data = load_run(os.path.join(self.runs_dir, rid))
            s = data.get("summary") or {}
            if s.get("task") is not None:
                out.append(s)
        return out

    def stats(self) -> dict:
        runs = self._runs()
        if not runs:
            return {"runs": 0}
        resolved = [r for r in runs if r.get("resolved")]
        wins = Counter(r.get("won_by", "") for r in resolved)
        attempts = [r.get("attempts", 0) for r in resolved if r.get("attempts")]
        return {
            "runs": len(runs),
            "resolved": len(resolved),
            "resolved_rate": round(len(resolved) / len(runs), 3),
            "avg_attempts_on_win": round(sum(attempts) / len(attempts), 2) if attempts else None,
            "wins_by_layer": dict(wins),
        }

    def similar(self, task: str, k: int = 3) -> list[dict]:
        """Past runs whose task string overlaps this one, best first."""
        q = set(tokenize(task))
        if not q:
            return []
        scored = []
        for r in self._runs():
            toks = set(tokenize(r.get("task", "")))
            if not toks:
                continue
            overlap = len(q & toks) / len(q | toks)  # Jaccard
            if overlap > 0:
                scored.append((overlap, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{**r, "similarity": round(o, 3)} for o, r in scored[:k]]

    def co_changes(self, min_count: int = 2) -> list[dict]:
        """File pairs edited together in successful runs (files-that-change-together)."""
        pair_counts: Counter = Counter()
        for r in self._runs():
            if not r.get("resolved"):
                continue
            files = sorted(set(r.get("files", [])))
            for i in range(len(files)):
                for j in range(i + 1, len(files)):
                    pair_counts[(files[i], files[j])] += 1
        return [
            {"files": list(pair), "count": c}
            for pair, c in pair_counts.most_common()
            if c >= min_count
        ]

    def recall(self, task: str, k: int = 3) -> dict:
        """A compact, opt-in memory hint for a new task (evidence, not injection)."""
        sims = self.similar(task, k=k)
        wins = [s for s in sims if s.get("resolved")]
        suggested_files = Counter()
        for s in wins:
            for f in s.get("files", []):
                suggested_files[f] += 1
        approach = Counter(s.get("won_by", "") for s in wins).most_common(1)
        return {
            "similar_tasks": [
                {"task": s["task"], "resolved": s.get("resolved"),
                 "won_by": s.get("won_by"), "files": s.get("files", []),
                 "similarity": s.get("similarity")}
                for s in sims
            ],
            "suggested_files": [f for f, _ in suggested_files.most_common(5)],
            "suggested_approach": approach[0][0] if approach else None,
            "co_changes": self.co_changes(),
        }

    def render_hint(self, task: str) -> str:
        """Human/agent-readable hint string (empty if no useful memory)."""
        r = self.recall(task)
        if not r["similar_tasks"]:
            return ""
        lines = ["## Memory (past runs — advisory, not authoritative)"]
        for s in r["similar_tasks"]:
            mark = "solved" if s["resolved"] else "failed"
            lines.append(f"- {mark} similar task: {s['task']!r} "
                         f"(won_by={s['won_by']}, files={s['files']}, sim={s['similarity']})")
        if r["suggested_files"]:
            lines.append(f"- files that worked before: {r['suggested_files']}")
        if r["suggested_approach"]:
            lines.append(f"- approach that tends to win here: {r['suggested_approach']}")
        return "\n".join(lines)
