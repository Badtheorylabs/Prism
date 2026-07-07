"""The orchestrator: ties indexer -> graph -> semantic -> packer into the single
public tool `get_context`. This is the function agents and small models call.
"""

from __future__ import annotations

import os

from .graph import CodeGraph
from .indexer import index_repo
from .packer import pack
from .semantic import build_index

DEFAULT_DB_NAME = "index.db"
INDEX_VERSION = "6"


def index_path_for(repo_root: str) -> str:
    return os.path.join(repo_root, ".prism", DEFAULT_DB_NAME)


def build_repo_index(repo_root: str, db_path: str | None = None) -> str:
    """Index a repo into a SQLite graph. Returns the db path."""
    repo_root = os.path.abspath(repo_root)
    db_path = db_path or index_path_for(repo_root)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    result = index_repo(repo_root)
    graph = CodeGraph(db_path)
    graph.build(result)
    graph.set_meta("repo_root", repo_root)
    graph.set_meta("index_version", INDEX_VERSION)
    graph.close()
    return db_path


def _needs_reindex(db_path: str) -> bool:
    if not os.path.exists(db_path):
        return True
    graph = CodeGraph(db_path)
    version = graph.get_meta("index_version")
    graph.close()
    return version != INDEX_VERSION


def get_context(
    repo_root: str,
    task: str,
    token_budget: int = 8000,
    db_path: str | None = None,
    reindex: bool = False,
) -> dict:
    """Return an optimally-packed context dossier for `task` within `token_budget`.

    This is the crown-jewel tool. Agents call it with a task string and their
    model's token budget; they get back exactly the slice that fits.

    Args:
        repo_root: root of the codebase.
        task: natural-language description of the change.
        token_budget: hard ceiling on returned context size (in tokens).
        db_path: override index location.
        reindex: rebuild the index even if one exists.

    Returns:
        dict payload — see packer.ContextPackage.to_dict().
    """
    repo_root = os.path.abspath(repo_root)
    db_path = db_path or index_path_for(repo_root)
    if reindex or _needs_reindex(db_path):
        build_repo_index(repo_root, db_path)

    graph = CodeGraph(db_path)
    stored_root = graph.get_meta("repo_root") or repo_root
    index = build_index(graph)
    pkg = pack(graph, index, task, token_budget, stored_root)
    payload = pkg.to_dict()
    payload["symbol_count"] = graph.count()
    payload["repo_stats"] = graph.stats()
    graph.close()
    return payload


def repo_stats(repo_root: str, db_path: str | None = None, reindex: bool = False) -> dict:
    """Return a lightweight architecture map for a repo."""
    repo_root = os.path.abspath(repo_root)
    db_path = db_path or index_path_for(repo_root)
    if reindex or _needs_reindex(db_path):
        build_repo_index(repo_root, db_path)
    graph = CodeGraph(db_path)
    stats = graph.stats()
    graph.close()
    return stats


def search_code(
    repo_root: str,
    query: str,
    *,
    top_k: int = 10,
    db_path: str | None = None,
    reindex: bool = False,
) -> dict:
    """Search symbols, chunks, and files using the repo substrate."""
    repo_root = os.path.abspath(repo_root)
    db_path = db_path or index_path_for(repo_root)
    if reindex or _needs_reindex(db_path):
        build_repo_index(repo_root, db_path)

    graph = CodeGraph(db_path)
    try:
        index = build_index(graph)
        symbol_hits = []
        for sid, score in index.search(query, top_k=top_k):
            sym = graph.get(sid)
            if not sym:
                continue
            symbol_hits.append(
                {
                    "id": sym.id,
                    "qualname": sym.qualname,
                    "file": sym.file,
                    "kind": sym.kind,
                    "signature": sym.signature,
                    "start_line": sym.start_line,
                    "end_line": sym.end_line,
                    "score": round(score, 6),
                }
            )
        chunk_hits = []
        for chunk_id, score in index.search_chunks(query, top_k=top_k):
            chunk = graph.get_chunk(chunk_id)
            if not chunk:
                continue
            chunk_hits.append(
                {
                    "id": chunk.id,
                    "file": chunk.file,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "tokens": chunk.tokens,
                    "score": round(score, 6),
                    "symbols": chunk.symbols.split(",") if chunk.symbols else [],
                    "quality_summary": chunk.quality_summary,
                    "preview": "\n".join(chunk.text.splitlines()[:12]),
                }
            )
        file_hits = [
            {"file": file, "score": round(score, 6), "symbol_ids": symbol_ids}
            for file, score, symbol_ids in index.search_files(query, top_k=top_k)
        ]
        return {
            "ok": True,
            "query": query,
            "symbols": symbol_hits,
            "chunks": chunk_hits,
            "files": file_hits,
        }
    finally:
        graph.close()


def analyze_impact(
    repo_root: str,
    *,
    file: str | None = None,
    symbol: str | None = None,
    hops: int = 2,
    db_path: str | None = None,
    reindex: bool = False,
) -> dict:
    """Return blast-radius facts before an agent edits code."""
    repo_root = os.path.abspath(repo_root)
    db_path = db_path or index_path_for(repo_root)
    if reindex or _needs_reindex(db_path):
        build_repo_index(repo_root, db_path)
    graph = CodeGraph(db_path)
    try:
        return graph.impact_analysis(file=file, symbol=symbol, hops=hops)
    finally:
        graph.close()


def inspect_code(
    repo_root: str,
    *,
    file: str | None = None,
    symbol: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
    radius: int = 2,
    db_path: str | None = None,
    reindex: bool = False,
) -> dict:
    """Return exact file/symbol/line detail from the repo substrate.

    This is the follow-up tool agents use after `get_context` when they need
    line-level detail without stuffing an entire huge file into the prompt.
    """
    repo_root = os.path.abspath(repo_root)
    db_path = db_path or index_path_for(repo_root)
    if reindex or _needs_reindex(db_path):
        build_repo_index(repo_root, db_path)

    graph = CodeGraph(db_path)
    try:
        if symbol:
            return graph.inspect_symbol(symbol, radius=radius)
        if file and start_line is not None:
            lines = graph.line_window(
                file,
                start_line=start_line,
                end_line=end_line or start_line,
                radius=radius,
            )
            return {
                "ok": True,
                "file": file,
                "lines": [line.__dict__ for line in lines],
                "quality_findings": [
                    finding.__dict__
                    for finding in graph.quality_findings(file=file, limit=100)
                    if start_line - radius <= finding.line_no <= (end_line or start_line) + radius
                ],
            }
        if file:
            payload = graph.file_outline(file)
            payload["ok"] = True
            return payload
        return {
            "ok": False,
            "error": "provide either symbol=... or file=...",
        }
    finally:
        graph.close()
