"""Virtual whole-codebase attention for small coding models.

This is the product layer: make a small model feel like it can "read the whole
repo" by giving it a compact architecture manifest plus stable handles it can
drill into through MCP/CLI tools.
"""

from __future__ import annotations

import os

from .context import _needs_reindex, build_repo_index, index_path_for
from .graph import CodeGraph, SymbolRow
from .semantic import build_index


def _symbol_ref(sym: SymbolRow) -> dict:
    return {
        "handle": f"symbol:{sym.qualname}",
        "qualname": sym.qualname,
        "file": sym.file,
        "kind": sym.kind,
        "signature": sym.signature,
        "lines": [sym.start_line, sym.end_line],
    }


def build_attention_manifest(
    repo_root: str,
    task: str | None = None,
    token_budget: int = 8000,
    db_path: str | None = None,
    reindex: bool = False,
) -> dict:
    """Return a compact, handle-based whole-codebase view.

    The manifest is intentionally not a source dump. It is a navigable index:
    a small model gets repo-wide orientation plus handles it can pass to
    inspect/search/impact tools.
    """
    repo_root = os.path.abspath(repo_root)
    db_path = db_path or index_path_for(repo_root)
    if reindex or _needs_reindex(db_path):
        build_repo_index(repo_root, db_path)

    graph = CodeGraph(db_path)
    try:
        stats = graph.stats()
        index = build_index(graph)
        query = task or "architecture entrypoints dependencies tests risk"

        ranked_symbols = []
        for sid, score in index.search(query, top_k=24):
            sym = graph.get(sid)
            if not sym:
                continue
            item = _symbol_ref(sym)
            item["score"] = round(score, 6)
            ranked_symbols.append(item)

        ranked_chunks = []
        for chunk_id, score in index.search_chunks(query, top_k=16):
            chunk = graph.get_chunk(chunk_id)
            if not chunk:
                continue
            ranked_chunks.append(
                {
                    "handle": f"chunk:{chunk.id}",
                    "file": chunk.file,
                    "lines": [chunk.start_line, chunk.end_line],
                    "tokens": chunk.tokens,
                    "score": round(score, 6),
                    "symbols": chunk.symbols.split(",") if chunk.symbols else [],
                    "quality_summary": chunk.quality_summary,
                }
            )

        file_handles = []
        for file_hit in stats.get("dependency_hotspots", [])[:20]:
            file_handles.append(
                {
                    "handle": f"file:{file_hit['file']}",
                    "file": file_hit["file"],
                    "incoming": file_hit.get("incoming", 0),
                    "outgoing": file_hit.get("outgoing", 0),
                }
            )
        known = {item["file"] for item in file_handles}
        for item in stats.get("largest_files", [])[:20]:
            if item["file"] not in known:
                file_handles.append(
                    {
                        "handle": f"file:{item['file']}",
                        "file": item["file"],
                        "symbols": item.get("symbols", 0),
                    }
                )
                known.add(item["file"])

        # Agent instructions are data, not prose hidden in a README. Any MCP
        # client can use this as the first system-facing payload.
        navigation = [
            {
                "need": "find relevant code",
                "tool": "search_codebase",
                "input": {"query": task or "<task>", "top_k": 10},
            },
            {
                "need": "open exact symbol",
                "tool": "inspect_code_detail",
                "input": {"symbol": "symbol qualname from a symbol handle"},
            },
            {
                "need": "open exact lines",
                "tool": "inspect_code_detail",
                "input": {"file": "path from file handle", "start_line": 1, "end_line": 80},
            },
            {
                "need": "check blast radius before edit",
                "tool": "analyze_change_impact",
                "input": {"symbol": "symbol qualname or file path", "hops": 2},
            },
            {
                "need": "produce bounded edit context",
                "tool": "get_code_context",
                "input": {"task": task or "<task>", "token_budget": token_budget},
            },
        ]

        return {
            "ok": True,
            "mode": "virtual_whole_codebase_attention",
            "task": task,
            "token_budget": token_budget,
            "claim": "The whole repo is indexed; this payload is the compact attention map, not the full source dump.",
            "coverage": {
                "files": stats.get("files", 0),
                "symbols": stats.get("symbols", 0),
                "lines": stats.get("lines", 0),
                "chunks": stats.get("chunks", 0),
                "edges": stats.get("edges", 0),
                "quality_findings": stats.get("quality_findings", 0),
            },
            "architecture": {
                "entrypoints": stats.get("entrypoints", []),
                "hubs": stats.get("hubs", []),
                "dependency_hotspots": stats.get("dependency_hotspots", []),
                "quality_hotspots": stats.get("quality_hotspots", []),
            },
            "attention": {
                "symbols": ranked_symbols,
                "chunks": ranked_chunks,
                "files": file_handles[:32],
            },
            "navigation_protocol": navigation,
            "handles": {
                "file": "file:<path> -> inspect_code_detail(file=path)",
                "symbol": "symbol:<qualname> -> inspect_code_detail(symbol=qualname)",
                "chunk": "chunk:<id> -> inspect_code_detail(file=chunk.file,start_line=chunk.start,end_line=chunk.end)",
                "line_window": "line:<path>:<start>-<end> -> inspect_code_detail(file=path,start_line=start,end_line=end)",
            },
        }
    finally:
        graph.close()
