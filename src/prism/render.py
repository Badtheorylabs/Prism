"""Render context packages into agent-readable dossiers."""

from __future__ import annotations


def render_markdown(payload: dict, include_source: bool = True) -> str:
    """Turn get_context JSON into a compact prompt payload for a coding agent."""
    lines: list[str] = []
    lines.append("# Prism Context Dossier")
    lines.append("")
    lines.append(f"Task: {payload.get('task', '').strip()}")
    lines.append(
        f"Budget: {payload.get('fits_in', 0)} / {payload.get('token_budget', 0)} estimated tokens"
    )
    if payload.get("symbol_count") is not None:
        lines.append(f"Indexed symbols: {payload['symbol_count']}")
    lines.append("")

    budget = payload.get("budget") or {}
    if budget:
        lines.append("## Budget Use")
        for key in (
            "edit_files",
            "periphery",
            "tests",
            "quality_signals",
            "relevant_chunks",
            "dependency_edges",
        ):
            item = budget.get(key)
            if not item:
                continue
            lines.append(f"- {key}: {item.get('used')} / {item.get('limit')}")
        lines.append(f"- remaining: {budget.get('remaining', 0)}")
        lines.append("")

    ranked = payload.get("ranked_symbols") or []
    if ranked:
        lines.append("## Why These Files")
        for sym in ranked[:8]:
            terms = ", ".join(sym.get("matched_terms") or [])
            suffix = f" matched: {terms}" if terms else ""
            lines.append(
                f"- {sym.get('qualname')} ({sym.get('file')}) score={sym.get('score')} - "
                f"{sym.get('why')}{suffix}"
            )
        lines.append("")

    edges = payload.get("dependency_edges") or []
    if edges:
        lines.append("## Dependency Edges")
        for edge in edges[:40]:
            lines.append(f"- {edge.get('from')} --{edge.get('kind')}--> {edge.get('to')}")
        lines.append("")

    signals = payload.get("quality_signals") or []
    if signals:
        lines.append("## Quality Signals")
        for signal in signals[:20]:
            symbol = f" ({signal.get('symbol')})" if signal.get("symbol") else ""
            lines.append(
                f"- {signal.get('severity')} {signal.get('code')} at "
                f"{signal.get('file')}:{signal.get('line_no')}{symbol}: "
                f"{signal.get('message')}"
            )
        lines.append("")

    chunks = payload.get("relevant_chunks") or []
    if chunks:
        lines.append("## Relevant Chunks")
        for chunk in chunks[:8]:
            symbols = ", ".join(chunk.get("symbols") or [])
            suffix = f" symbols: {symbols}" if symbols else ""
            lines.append(
                f"### {chunk.get('file')}:{chunk.get('start_line')}-{chunk.get('end_line')} "
                f"(score={chunk.get('score')}, {chunk.get('tokens')} tokens){suffix}"
            )
            if include_source:
                lines.append("```python")
                lines.append((chunk.get("text") or "").rstrip())
                lines.append("```")
        lines.append("")

    files = payload.get("files_to_edit") or []
    lines.append("## Files To Edit")
    if not files:
        lines.append("- No edit files selected. Re-index or try a more specific task.")
    for item in files:
        mode = item.get("mode", "full")
        lines.append(f"### {item.get('path')} ({item.get('tokens', 0)} tokens, {mode})")
        included = item.get("included_symbols") or []
        if included:
            lines.append(f"Included symbols: {', '.join(included)}")
        if include_source:
            lines.append("```python")
            lines.append(item.get("full_source", "").rstrip())
            lines.append("```")
        else:
            lines.append("- full source omitted")
    lines.append("")

    periphery = payload.get("periphery") or []
    if periphery:
        lines.append("## Periphery")
        for item in periphery:
            summary = item.get("one_line_summary") or item.get("signature")
            lines.append(
                f"- [{item.get('relation')}] {item.get('qualname')} "
                f"({item.get('file')}): `{item.get('signature')}` - {summary}"
            )
        lines.append("")

    tests = payload.get("relevant_tests") or []
    if tests:
        lines.append("## Relevant Tests")
        for item in tests:
            lines.append(
                f"- {item.get('qualname')} ({item.get('file')}): `{item.get('signature')}`"
            )
        lines.append("")

    notes = payload.get("notes") or []
    if notes:
        lines.append("## Notes")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_stats(stats: dict) -> str:
    lines = [
        "# Prism Repo Map",
        "",
        f"Symbols: {stats.get('symbols', 0)}",
        f"Files: {stats.get('files', 0)}",
        f"Edges: {stats.get('edges', 0)}",
        f"Imports: {stats.get('imports', 0)}",
        f"Lines: {stats.get('lines', 0)}",
        f"Chunks: {stats.get('chunks', 0)}",
        f"Quality findings: {stats.get('quality_findings', 0)}",
        "",
    ]
    by_kind = stats.get("by_kind") or {}
    if by_kind:
        lines.append("## Symbols By Kind")
        for kind, count in sorted(by_kind.items()):
            lines.append(f"- {kind}: {count}")
        lines.append("")
    largest = stats.get("largest_files") or []
    if largest:
        lines.append("## Largest Files")
        for item in largest:
            lines.append(f"- {item.get('file')}: {item.get('symbols')} symbols")
        lines.append("")
    hubs = stats.get("hubs") or []
    if hubs:
        lines.append("## Dependency Hubs")
        for item in hubs:
            lines.append(
                f"- {item.get('qualname')} ({item.get('file')}): "
                f"{item.get('incoming')} incoming, {item.get('outgoing')} outgoing"
            )
        lines.append("")
    entrypoints = stats.get("entrypoints") or []
    if entrypoints:
        lines.append("## Entrypoints")
        for item in entrypoints:
            lines.append(
                f"- {item.get('qualname')} ({item.get('file')}): "
                f"{item.get('outgoing')} outgoing"
            )
        lines.append("")
    dep_hotspots = stats.get("dependency_hotspots") or []
    if dep_hotspots:
        lines.append("## Dependency Hotspots")
        for item in dep_hotspots:
            lines.append(
                f"- {item.get('file')}: {item.get('incoming')} incoming, "
                f"{item.get('outgoing')} outgoing"
            )
        lines.append("")
    quality_hotspots = stats.get("quality_hotspots") or []
    if quality_hotspots:
        lines.append("## Quality Hotspots")
        for item in quality_hotspots:
            lines.append(
                f"- {item.get('file')}: {item.get('high')} high, "
                f"{item.get('medium')} medium, {item.get('total')} total"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
