"""Agentic iterative exploration + line-level ranking (Layer 1 redesign).

The research verdict on context: the SOTA pattern is NOT one-shot retrieval and
NOT pure long-context — it's **agentic exploration layered on retrieval + graph
structure**. Bigger context windows don't help because the failure mode is
*navigational salience*: the model never DISCOVERS that a structurally-connected
file is relevant. And the open axis beyond file-level localization is
**line-level ranking**.

So this module:
  - uses BM25/hybrid retrieval as ANCHORS (kept — validated as defensible),
  - then iteratively expands along call-graph / reference edges to surface
    structurally-relevant code that term-matching alone would miss,
  - and ranks individual LINES within a file by relevance.

It is fully deterministic (no model) — an agentic *search*, not an LLM loop.
"""

from __future__ import annotations

from dataclasses import dataclass

from .graph import CodeGraph
from .lsp import GraphLSP
from .semantic import build_index, tokenize

HOP_DECAY = 0.55  # relevance decay per graph hop away from an anchor


@dataclass
class Discovery:
    qualname: str
    file: str
    score: float
    reason: str
    hops: int  # 0 = retrieval anchor; >0 = discovered via graph traversal


def explore(
    graph: CodeGraph,
    task: str,
    rounds: int = 2,
    anchors_k: int = 8,
    max_discoveries: int = 40,
) -> list[Discovery]:
    """Retrieval anchors -> iterative graph expansion -> ranked discovery set."""
    index = build_index(graph)
    seeds = index.search(task, top_k=anchors_k)  # [(symbol_id, score)]
    if not seeds:
        return []

    top = seeds[0][1] or 1.0
    discovered: dict[int, Discovery] = {}
    frontier: list[tuple[int, float]] = []

    for sid, sc in seeds:
        sym = graph.get(sid)
        if not sym:
            continue
        norm = round(sc / top, 3)
        discovered[sid] = Discovery(sym.qualname, sym.file, norm, "retrieval anchor", 0)
        frontier.append((sid, norm))

    for r in range(1, rounds + 1):
        nxt: list[tuple[int, float]] = []
        for sid, base in frontier:
            origin = graph.get(sid)
            for neighbor, kind in graph.neighbors(sid):  # both directions
                if neighbor.id in discovered:
                    continue
                sc = round(base * HOP_DECAY, 3)
                discovered[neighbor.id] = Discovery(
                    neighbor.qualname, neighbor.file, sc,
                    f"{kind}-edge from {origin.name if origin else '?'} (hop {r})", r,
                )
                nxt.append((neighbor.id, sc))
        frontier = nxt
        if not frontier:
            break

    ranked = sorted(discovered.values(), key=lambda d: d.score, reverse=True)
    return ranked[:max_discoveries]


def discovered_files(discoveries: list[Discovery], limit: int = 12) -> list[dict]:
    """Aggregate discoveries into a ranked, de-duplicated file list."""
    by_file: dict[str, dict] = {}
    for d in discoveries:
        b = by_file.setdefault(d.file, {"file": d.file, "score": 0.0, "min_hops": 99, "why": ""})
        b["score"] = round(b["score"] + d.score, 3)
        if d.hops < b["min_hops"]:
            b["min_hops"] = d.hops
            b["why"] = d.reason
    ordered = sorted(by_file.values(), key=lambda x: (x["score"], -x["min_hops"]), reverse=True)
    return ordered[:limit]


def rank_lines(
    graph: CodeGraph,
    file: str,
    query: str,
    top_k: int = 5,
    window: int = 3,
) -> list[dict]:
    """Line-level ranking: score each line by query overlap, return top windows.

    Addresses the research's 'line-level coverage and efficient ranking' gap —
    file-level localization is already strong; this is the open axis.
    """
    rows = graph.conn.execute(
        "SELECT line_no, text FROM lines WHERE file=? ORDER BY line_no", (file,)
    ).fetchall()
    if not rows:
        return []
    q = set(tokenize(query))
    if not q:
        return []

    scored: list[tuple[int, float, str]] = []
    for r in rows:
        toks = set(tokenize(r["text"]))
        if not toks:
            continue
        overlap = len(q & toks)
        if overlap:
            scored.append((r["line_no"], overlap / (1 + len(toks) * 0.05), r["text"]))
    scored.sort(key=lambda x: x[1], reverse=True)

    # merge nearby top lines into windows
    picked: list[dict] = []
    used: set[int] = set()
    for line_no, sc, _text in scored:
        if line_no in used or len(picked) >= top_k:
            continue
        start = max(1, line_no - window)
        end = line_no + window
        lines = graph.line_window(file, start, end)
        for ln in lines:
            used.add(ln.line_no)
        picked.append({
            "file": file,
            "focus_line": line_no,
            "start_line": start,
            "end_line": end,
            "score": round(sc, 3),
            "text": "\n".join(ln.text for ln in lines),
        })
    return picked
