"""Semantic retrieval over symbols and chunks.

v1 is a dependency-free BM25 hybrid retriever. It builds weighted documents for
symbols and overlapping chunks, then ranks them against a task string. This is
deliberately swappable: any function with the same search signatures can replace
it, including a small on-device embedding model. The interface, not the
algorithm, is the product.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional

from .graph import CodeGraph

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "into", "is", "it", "of", "on", "or", "the", "this", "to", "using",
    "with", "without", "add", "change", "update", "make", "keep", "per",
}


def _split_identifier(tok: str) -> list[str]:
    """snake_case and CamelCase -> component words, plus the whole token."""
    parts = tok.split("_")
    words: list[str] = []
    for p in parts:
        if not p:
            continue
        # split CamelCase
        camel = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", p)
        words.extend(camel if camel else [p])
    out = [w.lower() for w in words]
    out.append(tok.lower())
    return out


def tokenize(text: str) -> list[str]:
    toks: list[str] = []
    for m in _TOKEN_RE.findall(text or ""):
        toks.extend(t for t in _split_identifier(m) if t not in STOPWORDS)
    return toks


class HybridIndex:
    """BM25 hybrid index.

    The class keeps its old name for API compatibility with the rest of the
    package, but the scoring is BM25 plus code-aware boosts.
    """

    def __init__(self, graph: CodeGraph):
        self.graph = graph
        self.doc_tokens: dict[int, Counter] = {}
        self.doc_len: dict[int, int] = {}
        self.idf: dict[str, float] = {}
        self.fields: dict[int, dict[str, set[str]]] = {}
        self.symbol_file: dict[int, str] = {}
        self.avg_doc_len = 1.0
        self.chunk_tokens: dict[int, Counter] = {}
        self.chunk_len: dict[int, int] = {}
        self.chunk_idf: dict[str, float] = {}
        self.chunk_file: dict[int, str] = {}
        self.avg_chunk_len = 1.0
        self._build()
        self._build_chunks()

    def _doc_for(self, sym) -> str:
        # Weight durable code signals more than implementation bodies.
        return " ".join(
            [
                sym.file,
                sym.file,
                sym.qualname,
                sym.qualname,
                sym.qualname,
                sym.name,
                sym.name,
                sym.signature,
                sym.signature,
                sym.docstring,
                sym.docstring,
                sym.source,
            ]
        )

    def _build(self) -> None:
        df: Counter = Counter()
        total_len = 0
        for sym in self.graph.all_symbols():
            toks = tokenize(self._doc_for(sym))
            counts = Counter(toks)
            self.doc_tokens[sym.id] = counts
            self.doc_len[sym.id] = sum(counts.values()) or 1
            total_len += self.doc_len[sym.id]
            self.symbol_file[sym.id] = sym.file
            self.fields[sym.id] = {
                "file": set(tokenize(sym.file)),
                "qualname": set(tokenize(sym.qualname)),
                "signature": set(tokenize(sym.signature)),
                "docstring": set(tokenize(sym.docstring)),
                "source": set(tokenize(sym.source)),
            }
            for term in counts:
                df[term] += 1
        n = max(1, len(self.doc_tokens))
        self.avg_doc_len = max(1.0, total_len / n)
        for term, d in df.items():
            self.idf[term] = math.log(1.0 + (n - d + 0.5) / (d + 0.5))

    def _build_chunks(self) -> None:
        df: Counter = Counter()
        total_len = 0
        for chunk in self.graph.all_chunks():
            doc = " ".join(
                [
                    chunk.file,
                    chunk.file,
                    chunk.symbols,
                    chunk.symbols,
                    chunk.quality_summary,
                    chunk.text,
                ]
            )
            toks = tokenize(doc)
            counts = Counter(toks)
            self.chunk_tokens[chunk.id] = counts
            self.chunk_len[chunk.id] = sum(counts.values()) or 1
            self.chunk_file[chunk.id] = chunk.file
            total_len += self.chunk_len[chunk.id]
            for term in counts:
                df[term] += 1
        n = max(1, len(self.chunk_tokens))
        self.avg_chunk_len = max(1.0, total_len / n)
        for term, d in df.items():
            self.chunk_idf[term] = math.log(1.0 + (n - d + 0.5) / (d + 0.5))

    def _bm25(self, sid: int, q_counts: Counter) -> float:
        counts = self.doc_tokens[sid]
        dl = self.doc_len[sid]
        k1 = 1.4
        b = 0.72
        score = 0.0
        for term, qc in q_counts.items():
            tf = counts.get(term, 0)
            if tf <= 0:
                continue
            denom = tf + k1 * (1.0 - b + b * dl / self.avg_doc_len)
            score += qc * self.idf.get(term, 0.0) * ((tf * (k1 + 1.0)) / denom)
        return score

    def _bm25_counts(
        self,
        counts: Counter,
        doc_len: int,
        avg_len: float,
        idf: dict[str, float],
        q_counts: Counter,
    ) -> float:
        k1 = 1.4
        b = 0.72
        score = 0.0
        for term, qc in q_counts.items():
            tf = counts.get(term, 0)
            if tf <= 0:
                continue
            denom = tf + k1 * (1.0 - b + b * doc_len / avg_len)
            score += qc * idf.get(term, 0.0) * ((tf * (k1 + 1.0)) / denom)
        return score

    def _field_boost(self, sid: int, q_set: set[str], raw_query: str) -> float:
        fields = self.fields[sid]
        boost = 0.0
        weights = {
            "file": 1.25,
            "qualname": 1.75,
            "signature": 1.35,
            "docstring": 1.0,
            "source": 0.3,
        }
        for name, weight in weights.items():
            matched = len(q_set & fields[name])
            if matched:
                boost += weight * (matched / max(1, len(q_set)))

        # Exact phrase-ish boost for identifiers the tokenizer split apart.
        sym = self.graph.get(sid)
        if sym:
            haystacks = [sym.file.lower(), sym.qualname.lower(), sym.signature.lower()]
            raw = raw_query.lower().strip()
            for piece in raw.split():
                if len(piece) >= 4 and any(piece in h for h in haystacks):
                    boost += 0.2
        return boost

    def search(self, task: str, top_k: int = 12) -> list[tuple[int, float]]:
        q_tokens = tokenize(task)
        if not q_tokens:
            return []
        q_counts = Counter(q_tokens)
        q_set = set(q_tokens)

        scored: list[tuple[int, float]] = []
        for sid in self.doc_tokens:
            score = self._bm25(sid, q_counts)
            score += self._field_boost(sid, q_set, task)
            if score <= 0.0:
                continue
            counts = self.doc_tokens[sid]
            matched = sum(1 for t in q_set if counts.get(t, 0) > 0)
            coverage = matched / len(q_set)
            score *= 1.0 + coverage
            scored.append((sid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search_files(self, task: str, top_k: int = 8) -> list[tuple[str, float, list[int]]]:
        """Aggregate symbol and chunk search into file-level edit candidates."""
        ranked = self.search(task, top_k=max(24, top_k * 6))
        by_file: dict[str, dict] = {}
        for rank, (sid, score) in enumerate(ranked, start=1):
            file = self.symbol_file.get(sid)
            if not file:
                continue
            bucket = by_file.setdefault(file, {"score": 0.0, "symbols": []})
            # Reward both best hit and breadth, with a small top-rank prior.
            bucket["score"] += score / (rank ** 0.35)
            bucket["symbols"].append(sid)
        for rank, (chunk_id, score) in enumerate(
            self.search_chunks(task, top_k=max(24, top_k * 6)), start=1
        ):
            file = self.chunk_file.get(chunk_id)
            if not file:
                continue
            bucket = by_file.setdefault(file, {"score": 0.0, "symbols": []})
            bucket["score"] += (score * 0.65) / (rank ** 0.35)
        out = [
            (file, data["score"], data["symbols"])
            for file, data in by_file.items()
        ]
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:top_k]

    def search_chunks(self, task: str, top_k: int = 12) -> list[tuple[int, float]]:
        q_tokens = tokenize(task)
        if not q_tokens:
            return []
        q_counts = Counter(q_tokens)
        q_set = set(q_tokens)
        scored: list[tuple[int, float]] = []
        for chunk_id, counts in self.chunk_tokens.items():
            score = self._bm25_counts(
                counts,
                self.chunk_len[chunk_id],
                self.avg_chunk_len,
                self.chunk_idf,
                q_counts,
            )
            if score <= 0.0:
                continue
            matched = sum(1 for t in q_set if counts.get(t, 0) > 0)
            score *= 1.0 + (matched / len(q_set))
            scored.append((chunk_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


TfidfIndex = HybridIndex


def build_index(graph: CodeGraph, _cache: dict = {}) -> HybridIndex:
    """Build (and per-db-path cache) a BM25 hybrid index for a graph."""
    key = graph.db_path
    idx: Optional[HybridIndex] = _cache.get(key)
    if idx is None:
        idx = HybridIndex(graph)
        _cache[key] = idx
    else:
        # Rebind to the currently-open connection: get_context closes the graph
        # each call, so a cached index would otherwise query a closed DB on the
        # next call for the same repo (e.g. multi-step planning).
        idx.graph = graph
    return idx
