"""Packer: assemble the optimal context package within a fixed token budget.

This is the crown jewel. Big-context models are wasteful; naive RAG misses
dependencies. The packer's job is: given a task and a token budget, return the
*best* use of that budget —
    - full source for the files the model will actually edit,
    - signatures + one-line summaries for the surrounding periphery,
    - the exact dependency edges ("touch X -> these break"),
    - relevant tests,
all guaranteed to fit within `token_budget`.

Budget policy (fractions of total budget):
    edit files : up to 60%   (full source)
    periphery  : up to 30%   (signature + one-line summary)
    tests      : up to 10%   (signature + summary)
    edges are tiny and always included; periphery is trimmed last to fit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .graph import CodeGraph, SymbolRow
from .semantic import HybridIndex, tokenize
from .tokens import estimate

EDIT_FRACTION = 0.60
PERIPHERY_FRACTION = 0.30
TEST_FRACTION = 0.10


@dataclass
class EditFile:
    path: str
    full_source: str
    tokens: int
    mode: str = "full"
    included_symbols: list[str] = field(default_factory=list)


@dataclass
class PeripheryItem:
    qualname: str
    file: str
    signature: str
    one_line_summary: str
    relation: str  # "callee" | "caller" | "related" | "test"


@dataclass
class DependencyEdge:
    frm: str
    to: str
    kind: str


@dataclass
class RankedSymbol:
    qualname: str
    file: str
    kind: str
    signature: str
    score: float
    matched_terms: list[str]
    why: str


@dataclass
class QualitySignal:
    file: str
    line_no: int
    severity: str
    code: str
    message: str
    symbol: str | None = None


@dataclass
class RelevantChunk:
    file: str
    start_line: int
    end_line: int
    tokens: int
    score: float
    symbols: list[str]
    quality_summary: str
    text: str


@dataclass
class ContextPackage:
    task: str
    files_to_edit: list[EditFile] = field(default_factory=list)
    periphery: list[PeripheryItem] = field(default_factory=list)
    dependency_edges: list[DependencyEdge] = field(default_factory=list)
    relevant_tests: list[PeripheryItem] = field(default_factory=list)
    ranked_symbols: list[RankedSymbol] = field(default_factory=list)
    quality_signals: list[QualitySignal] = field(default_factory=list)
    relevant_chunks: list[RelevantChunk] = field(default_factory=list)
    budget: dict = field(default_factory=dict)
    token_budget: int = 0
    fits_in: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        # friendlier key names in the public payload
        for e in d["dependency_edges"]:
            e["from"] = e.pop("frm")
        return d


def _is_test_symbol(sym: SymbolRow) -> bool:
    f = sym.file.lower()
    return (
        "test" in f.split("/")[-1]
        or sym.name.startswith("test_")
        or "/tests/" in f
    )


def _excerpt_for_file(
    graph: CodeGraph,
    file: str,
    repo_root: str,
    priority_symbol_ids: list[int],
    token_limit: int,
) -> tuple[str, int, list[str]]:
    """Return a compact source excerpt for an oversized file."""
    src = graph.file_source(file, repo_root)
    lines = src.splitlines()
    parts: list[str] = [f"# Excerpt from {file}; full file omitted to fit token budget."]

    # Keep imports/header because snippets without imports are low-value.
    header: list[str] = []
    for line in lines[:80]:
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith("#")
            or stripped.startswith("import ")
            or stripped.startswith("from ")
        ):
            header.append(line)
            continue
        break
    if header:
        parts.extend(header)

    symbols = graph.symbols_in_file(file)
    by_id = {sym.id: sym for sym in symbols}
    ordered: list[SymbolRow] = []
    for sid in priority_symbol_ids:
        sym = by_id.get(sid)
        if sym and sym not in ordered:
            ordered.append(sym)
    for sym in symbols:
        if sym not in ordered:
            ordered.append(sym)

    included: list[str] = []
    for sym in ordered:
        candidate = "\n".join(
            [
                "",
                f"# --- {sym.qualname} lines {sym.start_line}-{sym.end_line} ---",
                sym.source.rstrip(),
            ]
        )
        trial = "\n".join(parts) + candidate
        if estimate(trial) > token_limit and included:
            break
        if estimate(trial) > token_limit:
            # Last resort: include signature/doc instead of the whole body.
            compact = "\n".join(
                [
                    "",
                    f"# --- {sym.qualname} lines {sym.start_line}-{sym.end_line} ---",
                    sym.signature,
                    f"# {sym.one_line}",
                ]
            )
            if estimate("\n".join(parts) + compact) <= token_limit:
                parts.append(compact)
                included.append(sym.qualname)
            break
        parts.append(candidate)
        included.append(sym.qualname)

    text = "\n".join(parts).rstrip() + "\n"
    return text, estimate(text), included


def pack(
    graph: CodeGraph,
    index: HybridIndex,
    task: str,
    token_budget: int,
    repo_root: str,
    max_edit_files: int = 3,
) -> ContextPackage:
    pkg = ContextPackage(task=task, token_budget=token_budget)
    query_terms = sorted(set(tokenize(task)))

    # 1. seed with semantic search
    seeds = index.search(task, top_k=24)
    if not seeds:
        pkg.notes.append("no symbols matched the task; index may be empty or off-topic")
        return pkg
    seed_syms = [graph.get(sid) for sid, _ in seeds]
    seed_syms = [s for s in seed_syms if s]
    seed_score = {sid: sc for sid, sc in seeds}
    seed_rank = {sid: rank for rank, (sid, _score) in enumerate(seeds, start=1)}

    for sym in seed_syms[:10]:
        sym_terms = set(tokenize(" ".join([sym.qualname, sym.signature, sym.docstring, sym.source])))
        matches = [t for t in query_terms if t in sym_terms][:12]
        pkg.ranked_symbols.append(
            RankedSymbol(
                qualname=sym.qualname,
                file=sym.file,
                kind=sym.kind,
                signature=sym.signature,
                score=round(seed_score.get(sym.id, 0.0), 6),
                matched_terms=matches,
                why=f"BM25/code-field rank #{seed_rank.get(sym.id, '?')} for the task",
            )
        )

    # 2. choose edit-target files from aggregate file evidence, excluding tests
    #    unless the task is explicitly about tests.
    file_candidates = index.search_files(task, top_k=12)
    file_score = {file: score for file, score, _ids in file_candidates}
    file_symbol_ids = {file: ids for file, _score, ids in file_candidates}
    best_file_score = max(file_score.values(), default=0.0)
    wants_tests = any(t in query_terms for t in {"test", "tests", "pytest", "spec"})
    edit_files_order: list[str] = []
    for file, score, _ids in file_candidates:
        if not wants_tests and ("test" in file.lower().split("/")[-1] or "/tests/" in file):
            continue
        if best_file_score and score < best_file_score * 0.50 and edit_files_order:
            continue
        if file not in edit_files_order:
            edit_files_order.append(file)

    # Fallback: seed symbol files in rank order when file aggregation found
    # nothing useful.
    if not edit_files_order:
        for sym in seed_syms:
            if not wants_tests and _is_test_symbol(sym):
                continue
            if sym.file not in edit_files_order:
                edit_files_order.append(sym.file)

    edit_budget = int(token_budget * EDIT_FRACTION)
    used = 0
    chosen_files: set[str] = set()
    for f in edit_files_order:
        if len(chosen_files) >= max_edit_files:
            break
        src = graph.file_source(f, repo_root)
        t = estimate(src)
        mode = "full"
        included_symbols: list[str] = []
        remaining_edit_budget = max(300, edit_budget - used)
        if t > remaining_edit_budget:
            src, t, included_symbols = _excerpt_for_file(
                graph,
                f,
                repo_root,
                file_symbol_ids.get(f, []),
                remaining_edit_budget,
            )
            mode = "excerpt"
            pkg.notes.append(f"excerpted {f} to fit the edit-file budget")
        if used + t > edit_budget and chosen_files:
            continue  # skip a file too big to fit alongside what we have
        pkg.files_to_edit.append(
            EditFile(
                path=f,
                full_source=src,
                tokens=t,
                mode=mode,
                included_symbols=included_symbols,
            )
        )
        chosen_files.add(f)
        used += t
    pkg.fits_in = used

    # Surface deterministic risk/quality signals near likely edit files.
    q_used = 0
    quality_budget = max(80, int(token_budget * 0.05))
    for f in chosen_files:
        for finding in graph.quality_findings(file=f, limit=12):
            sym_name = None
            if finding.symbol_id is not None:
                sym = graph.get(finding.symbol_id)
                sym_name = sym.qualname if sym else None
            signal = QualitySignal(
                file=finding.file,
                line_no=finding.line_no,
                severity=finding.severity,
                code=finding.code,
                message=finding.message,
                symbol=sym_name,
            )
            signal_tokens = estimate(
                " ".join(
                    [
                        signal.file,
                        str(signal.line_no),
                        signal.severity,
                        signal.code,
                        signal.message,
                        signal.symbol or "",
                    ]
                )
            )
            if q_used + signal_tokens > quality_budget:
                break
            pkg.quality_signals.append(signal)
            q_used += signal_tokens
    pkg.fits_in += q_used

    # Include high-scoring chunks that are not already covered by chosen edit files.
    c_used = 0
    chunk_budget = max(120, int(token_budget * 0.10))
    for chunk_id, score in index.search_chunks(task, top_k=12):
        chunk = graph.get_chunk(chunk_id)
        if not chunk or chunk.file in chosen_files:
            continue
        text = chunk.text
        text_tokens = estimate(text)
        if text_tokens > chunk_budget:
            lines = text.splitlines()
            text = "\n".join(lines[:40])
            text_tokens = estimate(text)
        if c_used + text_tokens > chunk_budget:
            continue
        pkg.relevant_chunks.append(
            RelevantChunk(
                file=chunk.file,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                tokens=text_tokens,
                score=round(score, 6),
                symbols=chunk.symbols.split(",") if chunk.symbols else [],
                quality_summary=chunk.quality_summary,
                text=text,
            )
        )
        c_used += text_tokens
    pkg.fits_in += c_used

    # 3. periphery via graph expansion around both seed symbols and chosen files.
    seed_ids = [s.id for s in seed_syms]
    chosen_symbol_ids = [
        sym.id for f in chosen_files for sym in graph.symbols_in_file(f)
    ]
    expanded = graph.expand(chosen_symbol_ids or seed_ids, hops=1)

    # rank periphery: seeds not already in edit files, then graph neighbors
    periphery_candidates: list[tuple[SymbolRow, str, float]] = []
    seen_ids: set[int] = set()

    for sym in seed_syms:
        if sym.file in chosen_files or _is_test_symbol(sym):
            continue
        if sym.id in seen_ids:
            continue
        seen_ids.add(sym.id)
        periphery_candidates.append((sym, "related", seed_score.get(sym.id, 0.0)))

    for sid, (sym, ekind) in expanded.items():
        if sym.file in chosen_files or _is_test_symbol(sym) or sid in seen_ids:
            continue
        seen_ids.add(sid)
        relation = "graph-neighbor"
        if ekind == "calls":
            relation = "caller/callee"
        elif ekind == "inherits":
            relation = "inheritance"
        periphery_candidates.append((sym, relation, seed_score.get(sym.id, 0.0) + 0.75))

    periphery_candidates.sort(key=lambda x: x[2], reverse=True)

    periphery_budget = int(token_budget * PERIPHERY_FRACTION)
    pused = 0
    for sym, relation, _score in periphery_candidates:
        item = PeripheryItem(
            qualname=sym.qualname,
            file=sym.file,
            signature=sym.signature,
            one_line_summary=sym.one_line,
            relation=relation,
        )
        t = estimate(item.signature + item.one_line_summary) + 4
        if pused + t > periphery_budget:
            continue
        pkg.periphery.append(item)
        pused += t
    pkg.fits_in += pused

    # 4. dependency edges: exact symbol-to-symbol edges touching edit targets.
    edge_seen: set[tuple[str, str, str]] = set()
    for src, dst, kind in graph.edges_touching_files(chosen_files):
        key = (src.qualname, dst.qualname, kind)
        if key in edge_seen:
            continue
        edge_seen.add(key)
        pkg.dependency_edges.append(
            DependencyEdge(frm=src.qualname, to=dst.qualname, kind=kind)
        )

    # 5. relevant tests: test symbols among seeds/expansion, or tests referencing edits
    test_budget = int(token_budget * TEST_FRACTION)
    tused = 0
    test_syms: list[SymbolRow] = []
    for sym in seed_syms:
        if _is_test_symbol(sym):
            test_syms.append(sym)
    for _sid, (sym, _k) in expanded.items():
        if _is_test_symbol(sym):
            test_syms.append(sym)
    seen_test: set[int] = set()
    for sym in test_syms:
        if sym.id in seen_test:
            continue
        seen_test.add(sym.id)
        item = PeripheryItem(
            qualname=sym.qualname,
            file=sym.file,
            signature=sym.signature,
            one_line_summary=sym.one_line,
            relation="test",
        )
        t = estimate(item.signature + item.one_line_summary) + 4
        if tused + t > test_budget:
            continue
        pkg.relevant_tests.append(item)
        tused += t
    pkg.fits_in += tused

    def edge_cost(edge: DependencyEdge) -> int:
        return estimate(f"{edge.frm} {edge.kind} {edge.to}") + 2

    edge_tokens = sum(edge_cost(e) for e in pkg.dependency_edges)
    pkg.fits_in += edge_tokens
    pkg.budget = {
        "edit_files": {"used": used, "limit": edit_budget},
        "periphery": {"used": pused, "limit": periphery_budget},
        "tests": {"used": tused, "limit": test_budget},
        "quality_signals": {"used": q_used, "limit": quality_budget},
        "relevant_chunks": {"used": c_used, "limit": chunk_budget},
        "dependency_edges": {"used": edge_tokens, "limit": "included"},
        "remaining": max(0, token_budget - pkg.fits_in),
    }

    # 6. final guard: trim lower-value context if we overshot the total budget.
    while pkg.fits_in > token_budget and pkg.periphery:
        dropped = pkg.periphery.pop()
        dropped_tokens = estimate(dropped.signature + dropped.one_line_summary) + 4
        pkg.fits_in -= dropped_tokens
        pkg.budget["periphery"]["used"] = max(0, pkg.budget["periphery"]["used"] - dropped_tokens)
        pkg.budget["remaining"] = max(0, token_budget - pkg.fits_in)
        pkg.notes.append("trimmed periphery to fit budget")
    while pkg.fits_in > token_budget and pkg.relevant_tests:
        dropped = pkg.relevant_tests.pop()
        dropped_tokens = estimate(dropped.signature + dropped.one_line_summary) + 4
        pkg.fits_in -= dropped_tokens
        pkg.budget["tests"]["used"] = max(0, pkg.budget["tests"]["used"] - dropped_tokens)
        pkg.budget["remaining"] = max(0, token_budget - pkg.fits_in)
        pkg.notes.append("trimmed tests to fit budget")
    while pkg.fits_in > token_budget and pkg.quality_signals:
        dropped = pkg.quality_signals.pop()
        dropped_tokens = estimate(
            f"{dropped.file} {dropped.line_no} {dropped.severity} {dropped.code} {dropped.message}"
        )
        pkg.fits_in -= dropped_tokens
        pkg.budget["quality_signals"]["used"] = max(
            0, pkg.budget["quality_signals"]["used"] - dropped_tokens
        )
        pkg.budget["remaining"] = max(0, token_budget - pkg.fits_in)
        pkg.notes.append("trimmed quality signals to fit budget")
    while pkg.fits_in > token_budget and pkg.relevant_chunks:
        dropped = pkg.relevant_chunks.pop()
        pkg.fits_in -= dropped.tokens
        pkg.budget["relevant_chunks"]["used"] = max(
            0, pkg.budget["relevant_chunks"]["used"] - dropped.tokens
        )
        pkg.budget["remaining"] = max(0, token_budget - pkg.fits_in)
        pkg.notes.append("trimmed relevant chunks to fit budget")
    while pkg.fits_in > token_budget and pkg.dependency_edges:
        dropped = pkg.dependency_edges.pop()
        dropped_tokens = edge_cost(dropped)
        pkg.fits_in -= dropped_tokens
        pkg.budget["dependency_edges"]["used"] = max(
            0, pkg.budget["dependency_edges"]["used"] - dropped_tokens
        )
        pkg.budget["remaining"] = max(0, token_budget - pkg.fits_in)
        pkg.notes.append("trimmed dependency edges to fit budget")
    if pkg.fits_in > token_budget:
        pkg.notes.append(
            "could not fully fit edit excerpts; increase token_budget for this file"
        )

    return pkg
