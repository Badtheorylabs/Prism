"""LSP-style semantic navigation over the code graph (Layer 1 redesign).

The research names an LSP abstraction (find_symbol / find_referencing_symbols /
document_symbols) as part of the SOTA context pattern, and separately validated
our call-graph for structural queries (hub detection, caller ranking). This
module exposes that graph as an LSP-flavored navigation API — no external LSP
server required — so agents can do exact symbol lookup and reference search
across languages. A real LSP backend can be slotted behind the same interface
later.
"""

from __future__ import annotations

from dataclasses import dataclass

from .graph import CodeGraph, SymbolRow


@dataclass
class Reference:
    qualname: str
    file: str
    kind: str  # "calls" | "inherits"
    signature: str
    start_line: int


class GraphLSP:
    def __init__(self, graph: CodeGraph):
        self.graph = graph

    def find_symbol(self, name: str, limit: int = 20) -> list[SymbolRow]:
        """Exact-or-suffix symbol lookup by short name or qualname."""
        rows = self.graph.conn.execute(
            """SELECT * FROM symbols
               WHERE name = ? OR qualname = ? OR qualname LIKE ?
               ORDER BY (name = ?) DESC, LENGTH(qualname) LIMIT ?""",
            (name, name, f"%.{name}", name, limit),
        ).fetchall()
        return [self.graph._row(r) for r in rows]

    def find_definition(self, name: str) -> SymbolRow | None:
        hits = self.find_symbol(name, limit=1)
        return hits[0] if hits else None

    def document_symbols(self, file: str) -> list[SymbolRow]:
        return self.graph.symbols_in_file(file)

    def find_referencing_symbols(self, qualname: str) -> list[Reference]:
        """Who references this symbol — the LSP 'find references' / reverse edges."""
        sym = self.graph.get_symbol_by_qualname(qualname)
        if not sym:
            return []
        refs: list[Reference] = []
        for other, kind in self.graph.neighbors(sym.id, direction="in"):
            refs.append(Reference(other.qualname, other.file, kind, other.signature, other.start_line))
        return refs

    def goto_callees(self, qualname: str) -> list[Reference]:
        sym = self.graph.get_symbol_by_qualname(qualname)
        if not sym:
            return []
        out: list[Reference] = []
        for other, kind in self.graph.neighbors(sym.id, direction="out"):
            out.append(Reference(other.qualname, other.file, kind, other.signature, other.start_line))
        return out

    def workspace_symbols(self, query: str, limit: int = 30) -> list[SymbolRow]:
        """Fuzzy workspace-wide symbol search by substring."""
        rows = self.graph.conn.execute(
            """SELECT * FROM symbols WHERE qualname LIKE ? OR name LIKE ?
               ORDER BY LENGTH(qualname) LIMIT ?""",
            (f"%{query}%", f"%{query}%", limit),
        ).fetchall()
        return [self.graph._row(r) for r in rows]
