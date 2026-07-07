"""Graph store: persist symbols + edges in SQLite and traverse them.

The graph is the *exact* part of the system — "touch X and these break" is a
deterministic query here, never something a model guesses. Call/inherit edges
are resolved best-effort by name after the whole repo is parsed.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Iterable, Optional

from .indexer import IndexResult, SourceFile, Symbol
from .tokens import estimate

COMMON_ATTRIBUTE_METHODS = {
    "add",
    "append",
    "clear",
    "close",
    "copy",
    "decode",
    "encode",
    "endswith",
    "extend",
    "fetchall",
    "fetchone",
    "get",
    "items",
    "join",
    "keys",
    "lower",
    "pop",
    "read",
    "replace",
    "rstrip",
    "setdefault",
    "split",
    "splitlines",
    "startswith",
    "strip",
    "update",
    "values",
    "write",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS symbols (
    id         INTEGER PRIMARY KEY,
    qualname   TEXT NOT NULL,
    name       TEXT NOT NULL,
    kind       TEXT NOT NULL,
    file       TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line   INTEGER NOT NULL,
    signature  TEXT NOT NULL,
    docstring  TEXT NOT NULL,
    source     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file);

CREATE TABLE IF NOT EXISTS edges (
    src_id INTEGER NOT NULL,
    dst_id INTEGER NOT NULL,
    kind   TEXT NOT NULL      -- "calls" | "inherits"
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_id);

CREATE TABLE IF NOT EXISTS imports (
    file   TEXT NOT NULL,
    module TEXT NOT NULL,
    name   TEXT NOT NULL,
    alias  TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_imports_file ON imports(file);

CREATE TABLE IF NOT EXISTS files (
    file       TEXT PRIMARY KEY,
    line_count INTEGER NOT NULL,
    tokens     INTEGER NOT NULL,
    source     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS lines (
    file            TEXT NOT NULL,
    line_no         INTEGER NOT NULL,
    text            TEXT NOT NULL,
    role            TEXT NOT NULL,
    owner_symbol_id INTEGER,
    quality_flags   TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (file, line_no)
);
CREATE INDEX IF NOT EXISTS idx_lines_file ON lines(file);
CREATE INDEX IF NOT EXISTS idx_lines_owner ON lines(owner_symbol_id);

CREATE TABLE IF NOT EXISTS chunks (
    id              INTEGER PRIMARY KEY,
    file            TEXT NOT NULL,
    start_line      INTEGER NOT NULL,
    end_line        INTEGER NOT NULL,
    tokens          INTEGER NOT NULL,
    text            TEXT NOT NULL,
    symbols         TEXT NOT NULL DEFAULT '',
    quality_summary TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file);

CREATE TABLE IF NOT EXISTS quality_findings (
    id        INTEGER PRIMARY KEY,
    file      TEXT NOT NULL,
    line_no   INTEGER NOT NULL,
    severity  TEXT NOT NULL,
    code      TEXT NOT NULL,
    message   TEXT NOT NULL,
    symbol_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_quality_file ON quality_findings(file);
CREATE INDEX IF NOT EXISTS idx_quality_symbol ON quality_findings(symbol_id);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


@dataclass
class SymbolRow:
    id: int
    qualname: str
    name: str
    kind: str
    file: str
    start_line: int
    end_line: int
    signature: str
    docstring: str
    source: str

    @property
    def one_line(self) -> str:
        """A one-line summary: first docstring line, else the signature."""
        if self.docstring:
            return self.docstring.strip().splitlines()[0]
        return self.signature


@dataclass
class LineRow:
    file: str
    line_no: int
    text: str
    role: str
    owner_symbol_id: int | None
    quality_flags: str


@dataclass
class ChunkRow:
    id: int
    file: str
    start_line: int
    end_line: int
    tokens: int
    text: str
    symbols: str
    quality_summary: str


@dataclass
class QualityFinding:
    id: int
    file: str
    line_no: int
    severity: str
    code: str
    message: str
    symbol_id: int | None


class CodeGraph:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        cols = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(imports)").fetchall()
        }
        if "alias" not in cols:
            self.conn.execute("ALTER TABLE imports ADD COLUMN alias TEXT NOT NULL DEFAULT ''")
            self.conn.commit()

    # ---- build -------------------------------------------------------------

    def build(self, result: IndexResult) -> None:
        """Populate the DB from an IndexResult, resolving edges by name."""
        cur = self.conn.cursor()
        cur.execute("DELETE FROM symbols")
        cur.execute("DELETE FROM edges")
        cur.execute("DELETE FROM imports")
        cur.execute("DELETE FROM files")
        cur.execute("DELETE FROM lines")
        cur.execute("DELETE FROM chunks")
        cur.execute("DELETE FROM quality_findings")

        name_to_ids: dict[str, list[int]] = {}
        qualname_to_id: dict[str, int] = {}
        file_name_to_ids: dict[tuple[str, str], list[int]] = {}
        symbol_meta: list[tuple[int, Symbol]] = []

        for sym in result.symbols:
            cur.execute(
                """INSERT INTO symbols
                   (qualname,name,kind,file,start_line,end_line,signature,docstring,source)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    sym.qualname, sym.name, sym.kind, sym.file,
                    sym.start_line, sym.end_line, sym.signature,
                    sym.docstring, sym.source,
                ),
            )
            sid = cur.lastrowid
            name_to_ids.setdefault(sym.name, []).append(sid)
            qualname_to_id[sym.qualname] = sid
            file_name_to_ids.setdefault((sym.file, sym.name), []).append(sid)
            symbol_meta.append((sid, sym))

        imports_by_file: dict[str, dict[str, tuple[str, str]]] = {}
        for imp in result.imports:
            imports_by_file.setdefault(imp.file, {})[imp.alias or imp.name] = (
                imp.module,
                imp.name,
            )
            cur.execute(
                "INSERT INTO imports (file,module,name,alias) VALUES (?,?,?,?)",
                (imp.file, imp.module, imp.name, imp.alias),
            )

        edge_seen: set[tuple[int, int, str]] = set()

        def module_name(file: str) -> str:
            if file.endswith(".py"):
                file = file[:-3]
            return file.replace("/", ".")

        def ids_for_qual_or_suffix(qual: str) -> list[int]:
            if qual in qualname_to_id:
                return [qualname_to_id[qual]]
            suffix = "." + qual
            matches = [
                sid
                for q, sid in qualname_to_id.items()
                if q.endswith(suffix) or q == qual
            ]
            return matches

        def resolve_ref(sym: Symbol, ref: str) -> list[int]:
            if not ref:
                return []
            parts = ref.split(".")
            file_imports = imports_by_file.get(sym.file, {})
            resolved: list[int] = []

            # Exact qualified name, e.g. package.module.func.
            resolved.extend(ids_for_qual_or_suffix(ref))

            # Same-class method calls, e.g. self.allow().
            if len(parts) == 2 and parts[0] in {"self", "cls"}:
                owner = sym.qualname.rsplit(".", 1)[0]
                resolved.extend(ids_for_qual_or_suffix(f"{owner}.{parts[1]}"))

            # Imported aliases:
            #   import limits as lim; lim.client_key()
            #   from limits import client_key as key; key()
            if parts[0] in file_imports:
                mod, imported_name = file_imports[parts[0]]
                rest = parts[1:]
                if rest:
                    if imported_name == mod:
                        target = ".".join([mod, *rest])
                    else:
                        target = ".".join([mod, imported_name, *rest])
                else:
                    target = imported_name if imported_name == mod else f"{mod}.{imported_name}"
                resolved.extend(ids_for_qual_or_suffix(target))

            # Same-file direct call beats global fallback.
            if len(parts) == 1:
                resolved.extend(file_name_to_ids.get((sym.file, ref), []))

            # Module-relative direct lookup, e.g. function in the same module.
            if len(parts) == 1:
                resolved.extend(ids_for_qual_or_suffix(f"{module_name(sym.file)}.{ref}"))

            # Controlled global fallback by terminal name.
            if len(parts) == 1:
                resolved.extend(name_to_ids.get(ref, []))
            elif not resolved and parts[-1] not in COMMON_ATTRIBUTE_METHODS:
                resolved.extend(name_to_ids.get(parts[-1], []))

            out: list[int] = []
            seen: set[int] = set()
            for dst in resolved:
                if dst != sid and dst not in seen:
                    seen.add(dst)
                    out.append(dst)
            return out

        def add_edge(src: int, dst: int, kind: str) -> None:
            key = (src, dst, kind)
            if key in edge_seen:
                return
            edge_seen.add(key)
            cur.execute(
                "INSERT INTO edges (src_id,dst_id,kind) VALUES (?,?,?)",
                (src, dst, kind),
            )

        # Resolve edges with scoped import-aware lookup.
        for sid, sym in symbol_meta:
            for callee in sym.calls:
                for dst in resolve_ref(sym, callee):
                    add_edge(sid, dst, "calls")
            for base in sym.bases:
                for dst in resolve_ref(sym, base):
                    add_edge(sid, dst, "inherits")
        self._build_line_tables(cur, result.files, symbol_meta)
        self.conn.commit()

    # ---- line/chunk/quality indexing --------------------------------------

    def _build_line_tables(
        self,
        cur: sqlite3.Cursor,
        files: list[SourceFile],
        symbol_meta: list[tuple[int, Symbol]],
    ) -> None:
        by_file: dict[str, list[tuple[int, Symbol]]] = {}
        for sid, sym in symbol_meta:
            by_file.setdefault(sym.file, []).append((sid, sym))

        for source_file in files:
            source = source_file.source
            lines = source.splitlines()
            cur.execute(
                "INSERT INTO files (file,line_count,tokens,source) VALUES (?,?,?,?)",
                (source_file.path, source_file.line_count, estimate(source), source),
            )
            file_symbols = sorted(
                by_file.get(source_file.path, []),
                key=lambda item: (item[1].end_line - item[1].start_line, item[1].start_line),
            )
            line_findings: dict[int, list[str]] = {}
            for line_no, text in enumerate(lines, start=1):
                owner_id = self._owner_for_line(file_symbols, line_no)
                findings = self._quality_for_line(text, line_no, owner_id)
                if findings:
                    line_findings[line_no] = [f[1] for f in findings]
                for severity, code, message in findings:
                    cur.execute(
                        """INSERT INTO quality_findings
                           (file,line_no,severity,code,message,symbol_id)
                           VALUES (?,?,?,?,?,?)""",
                        (source_file.path, line_no, severity, code, message, owner_id),
                    )
                cur.execute(
                    """INSERT INTO lines
                       (file,line_no,text,role,owner_symbol_id,quality_flags)
                       VALUES (?,?,?,?,?,?)""",
                    (
                        source_file.path,
                        line_no,
                        text,
                        self._line_role(text),
                        owner_id,
                        ",".join(line_findings.get(line_no, [])),
                    ),
                )
            self._insert_chunks(cur, source_file.path, lines, file_symbols, line_findings)

    def _owner_for_line(self, symbols: list[tuple[int, Symbol]], line_no: int) -> int | None:
        for sid, sym in symbols:
            if sym.start_line <= line_no <= sym.end_line:
                return sid
        return None

    def _line_role(self, text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return "blank"
        if stripped.startswith("#"):
            return "comment"
        if stripped.startswith(("import ", "from ")):
            return "import"
        if stripped.startswith("@"):
            return "decorator"
        if stripped.startswith(("def ", "async def ", "class ")):
            return "definition"
        if stripped.startswith(("if ", "elif ", "else:", "for ", "while ", "try:", "except ", "finally:", "with ", "match ", "case ")):
            return "control_flow"
        if stripped.startswith("return "):
            return "return"
        if "=" in stripped and "==" not in stripped:
            return "assignment"
        if stripped.endswith(")") or "(" in stripped:
            return "call_or_expression"
        return "expression"

    def _quality_for_line(
        self, text: str, line_no: int, owner_id: int | None
    ) -> list[tuple[str, str, str]]:
        stripped = text.strip()
        findings: list[tuple[str, str, str]] = []
        code_only = self._strip_string_literals(stripped)
        lowered = code_only.lower()
        if len(text) > 120:
            findings.append(("info", "long_line", "Line is longer than 120 characters."))
        if "todo" in lowered or "fixme" in lowered:
            findings.append(("info", "todo", "Unresolved TODO/FIXME marker."))
        if stripped.startswith("print("):
            findings.append(("info", "debug_print", "Debug print in application code."))
        if "eval(" in code_only or "exec(" in code_only:
            findings.append(("high", "dynamic_exec", "Uses eval/exec; review for safety."))
        if stripped.startswith("except:") or stripped.startswith("except Exception:"):
            findings.append(("medium", "broad_except", "Broad exception handler can hide failures."))
        secret_terms = ("api_key", "secret", "token", "password")
        if "=" in stripped:
            lhs, rhs = stripped.split("=", 1)
            lhs_lower = lhs.lower()
            lhs_name = lhs_lower.strip().rstrip(":")
            if (
                any(term in lhs_lower for term in secret_terms)
                and not lhs_name.endswith(("terms", "names", "keys"))
                and any(quote in rhs for quote in ("'", '"'))
            ):
                findings.append(("high", "possible_secret", "Possible hardcoded credential-like value."))
        if "assert False" in code_only:
            findings.append(("medium", "assert_false", "Unconditional failing assertion."))
        if line_no == 1 and stripped.startswith("#!") and "python" not in stripped:
            findings.append(("info", "non_python_shebang", "Unexpected shebang for a Python index."))
        return findings

    def _strip_string_literals(self, text: str) -> str:
        return re.sub(r"('([^'\\]|\\.)*'|\"([^\"\\]|\\.)*\")", '""', text)

    def _insert_chunks(
        self,
        cur: sqlite3.Cursor,
        file: str,
        lines: list[str],
        symbols: list[tuple[int, Symbol]],
        line_findings: dict[int, list[str]],
        max_lines: int = 80,
        overlap: int = 12,
    ) -> None:
        if not lines:
            return
        start = 1
        while start <= len(lines):
            end = min(len(lines), start + max_lines - 1)
            text = "\n".join(lines[start - 1:end])
            symbol_names = [
                sym.qualname
                for _sid, sym in symbols
                if not (sym.end_line < start or sym.start_line > end)
            ]
            flags: list[str] = []
            for line_no in range(start, end + 1):
                flags.extend(line_findings.get(line_no, []))
            quality_summary = ",".join(sorted(set(flags)))
            cur.execute(
                """INSERT INTO chunks
                   (file,start_line,end_line,tokens,text,symbols,quality_summary)
                   VALUES (?,?,?,?,?,?,?)""",
                (file, start, end, estimate(text), text, ",".join(symbol_names), quality_summary),
            )
            if end == len(lines):
                break
            start = max(start + 1, end - overlap + 1)

    # ---- read --------------------------------------------------------------

    def _row(self, r: sqlite3.Row) -> SymbolRow:
        return SymbolRow(**{k: r[k] for k in r.keys()})

    def all_symbols(self) -> list[SymbolRow]:
        rows = self.conn.execute("SELECT * FROM symbols").fetchall()
        return [self._row(r) for r in rows]

    def symbols_in_file(self, file: str) -> list[SymbolRow]:
        rows = self.conn.execute(
            "SELECT * FROM symbols WHERE file=? ORDER BY start_line", (file,)
        ).fetchall()
        return [self._row(r) for r in rows]

    def get(self, symbol_id: int) -> Optional[SymbolRow]:
        r = self.conn.execute("SELECT * FROM symbols WHERE id=?", (symbol_id,)).fetchone()
        return self._row(r) if r else None

    def get_symbol_by_qualname(self, qualname: str) -> Optional[SymbolRow]:
        r = self.conn.execute(
            "SELECT * FROM symbols WHERE qualname=?",
            (qualname,),
        ).fetchone()
        if r:
            return self._row(r)
        r = self.conn.execute(
            "SELECT * FROM symbols WHERE qualname LIKE ? ORDER BY LENGTH(qualname) LIMIT 1",
            (f"%.{qualname}",),
        ).fetchone()
        return self._row(r) if r else None

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) AS c FROM symbols").fetchone()["c"]

    def edge_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) AS c FROM edges").fetchone()["c"]

    def file_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) AS c FROM files").fetchone()["c"]

    def import_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) AS c FROM imports").fetchone()["c"]

    def stats(self) -> dict:
        """Small repo map for human demos and agent diagnostics."""
        kinds = self.conn.execute(
            "SELECT kind, COUNT(*) AS c FROM symbols GROUP BY kind ORDER BY kind"
        ).fetchall()
        files = self.conn.execute(
            """SELECT file, COUNT(*) AS symbols
               FROM symbols GROUP BY file ORDER BY symbols DESC, file LIMIT 12"""
        ).fetchall()
        hubs = self.conn.execute(
            """SELECT s.qualname, s.file,
                      COUNT(DISTINCT out.dst_id) AS outgoing,
                      COUNT(DISTINCT inc.src_id) AS incoming
               FROM symbols s
               LEFT JOIN edges out ON out.src_id = s.id
               LEFT JOIN edges inc ON inc.dst_id = s.id
               GROUP BY s.id
               ORDER BY (outgoing + incoming) DESC, incoming DESC, outgoing DESC
               LIMIT 12"""
        ).fetchall()
        entrypoints = self.conn.execute(
            """SELECT s.*, COUNT(out.dst_id) AS outgoing
               FROM symbols s
               LEFT JOIN edges out ON out.src_id = s.id
               WHERE s.id NOT IN (SELECT dst_id FROM edges WHERE kind='calls')
                 AND (s.name IN ('main','run','serve','handler','app')
                      OR s.file LIKE '%cli.py'
                      OR s.file LIKE '%server.py'
                      OR s.file LIKE '%app.py')
               GROUP BY s.id
               ORDER BY outgoing DESC, s.file, s.start_line
               LIMIT 12"""
        ).fetchall()
        quality_hotspots = self.conn.execute(
            """SELECT file,
                      SUM(CASE severity WHEN 'high' THEN 1 ELSE 0 END) AS high,
                      SUM(CASE severity WHEN 'medium' THEN 1 ELSE 0 END) AS medium,
                      COUNT(*) AS total
               FROM quality_findings
               GROUP BY file
               ORDER BY high DESC, medium DESC, total DESC, file
               LIMIT 12"""
        ).fetchall()
        dependency_hotspots = self.conn.execute(
            """SELECT s.file,
                      COUNT(DISTINCT inc.src_id) AS incoming,
                      COUNT(DISTINCT out.dst_id) AS outgoing
               FROM symbols s
               LEFT JOIN edges inc ON inc.dst_id = s.id
               LEFT JOIN edges out ON out.src_id = s.id
               GROUP BY s.file
               ORDER BY (incoming + outgoing) DESC, incoming DESC
               LIMIT 12"""
        ).fetchall()
        return {
            "symbols": self.count(),
            "files": self.file_count(),
            "edges": self.edge_count(),
            "imports": self.import_count(),
            "chunks": self.conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"],
            "lines": self.conn.execute("SELECT COUNT(*) AS c FROM lines").fetchone()["c"],
            "quality_findings": self.conn.execute(
                "SELECT COUNT(*) AS c FROM quality_findings"
            ).fetchone()["c"],
            "by_kind": {r["kind"]: r["c"] for r in kinds},
            "largest_files": [dict(r) for r in files],
            "hubs": [dict(r) for r in hubs],
            "entrypoints": [
                {
                    "qualname": row["qualname"],
                    "file": row["file"],
                    "signature": row["signature"],
                    "outgoing": row["outgoing"],
                }
                for row in entrypoints
            ],
            "quality_hotspots": [dict(r) for r in quality_hotspots],
            "dependency_hotspots": [dict(r) for r in dependency_hotspots],
        }

    def file_outline(self, file: str) -> dict:
        f = self.conn.execute("SELECT * FROM files WHERE file=?", (file,)).fetchone()
        symbols = self.symbols_in_file(file)
        findings = self.quality_findings(file=file, limit=50)
        chunks = self.chunks_for_file(file)
        return {
            "file": file,
            "line_count": f["line_count"] if f else 0,
            "tokens": f["tokens"] if f else 0,
            "symbols": [
                {
                    "id": sym.id,
                    "qualname": sym.qualname,
                    "kind": sym.kind,
                    "signature": sym.signature,
                    "start_line": sym.start_line,
                    "end_line": sym.end_line,
                    "one_line_summary": sym.one_line,
                }
                for sym in symbols
            ],
            "chunks": [
                {
                    "id": chunk.id,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "tokens": chunk.tokens,
                    "symbols": chunk.symbols.split(",") if chunk.symbols else [],
                    "quality_summary": chunk.quality_summary,
                }
                for chunk in chunks
            ],
            "quality_findings": [finding.__dict__ for finding in findings],
        }

    def line_window(
        self,
        file: str,
        start_line: int = 1,
        end_line: int | None = None,
        radius: int = 0,
    ) -> list[LineRow]:
        if end_line is None:
            end_line = start_line
        start = max(1, start_line - radius)
        end = end_line + radius
        rows = self.conn.execute(
            """SELECT * FROM lines
               WHERE file=? AND line_no BETWEEN ? AND ?
               ORDER BY line_no""",
            (file, start, end),
        ).fetchall()
        return [LineRow(**dict(row)) for row in rows]

    def chunks_for_file(self, file: str) -> list[ChunkRow]:
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE file=? ORDER BY start_line",
            (file,),
        ).fetchall()
        return [ChunkRow(**dict(row)) for row in rows]

    def all_chunks(self) -> list[ChunkRow]:
        rows = self.conn.execute("SELECT * FROM chunks ORDER BY file, start_line").fetchall()
        return [ChunkRow(**dict(row)) for row in rows]

    def get_chunk(self, chunk_id: int) -> ChunkRow | None:
        row = self.conn.execute("SELECT * FROM chunks WHERE id=?", (chunk_id,)).fetchone()
        return ChunkRow(**dict(row)) if row else None

    def chunks_touching_symbol(self, symbol_id: int) -> list[ChunkRow]:
        sym = self.get(symbol_id)
        if not sym:
            return []
        rows = self.conn.execute(
            """SELECT * FROM chunks
               WHERE file=? AND NOT (end_line < ? OR start_line > ?)
               ORDER BY start_line""",
            (sym.file, sym.start_line, sym.end_line),
        ).fetchall()
        return [ChunkRow(**dict(row)) for row in rows]

    def quality_findings(
        self,
        file: str | None = None,
        symbol_id: int | None = None,
        limit: int = 100,
    ) -> list[QualityFinding]:
        if symbol_id is not None:
            rows = self.conn.execute(
                """SELECT * FROM quality_findings
                   WHERE symbol_id=? ORDER BY
                   CASE severity WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                   line_no LIMIT ?""",
                (symbol_id, limit),
            ).fetchall()
        elif file is not None:
            rows = self.conn.execute(
                """SELECT * FROM quality_findings
                   WHERE file=? ORDER BY
                   CASE severity WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                   line_no LIMIT ?""",
                (file, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT * FROM quality_findings ORDER BY
                   CASE severity WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                   file, line_no LIMIT ?""",
                (limit,),
            ).fetchall()
        return [QualityFinding(**dict(row)) for row in rows]

    def inspect_symbol(self, qualname: str, radius: int = 2) -> dict:
        sym = self.get_symbol_by_qualname(qualname)
        if not sym:
            return {"ok": False, "error": f"symbol not found: {qualname}"}
        lines = self.line_window(sym.file, sym.start_line, sym.end_line, radius=radius)
        neighbors = self.neighbors(sym.id)
        findings = self.quality_findings(symbol_id=sym.id, limit=50)
        return {
            "ok": True,
            "symbol": {
                "id": sym.id,
                "qualname": sym.qualname,
                "name": sym.name,
                "kind": sym.kind,
                "file": sym.file,
                "start_line": sym.start_line,
                "end_line": sym.end_line,
                "signature": sym.signature,
                "docstring": sym.docstring,
                "one_line_summary": sym.one_line,
            },
            "lines": [line.__dict__ for line in lines],
            "neighbors": [
                {
                    "qualname": other.qualname,
                    "file": other.file,
                    "kind": kind,
                    "signature": other.signature,
                }
                for other, kind in neighbors
            ],
            "quality_findings": [finding.__dict__ for finding in findings],
            "chunks": [chunk.__dict__ for chunk in self.chunks_touching_symbol(sym.id)],
        }

    def impact_analysis(
        self,
        *,
        file: str | None = None,
        symbol: str | None = None,
        hops: int = 2,
    ) -> dict:
        """Return the likely blast radius for changing a file or symbol."""
        seed_symbols: list[SymbolRow] = []
        target: dict
        if symbol:
            sym = self.get_symbol_by_qualname(symbol)
            if not sym:
                return {"ok": False, "error": f"symbol not found: {symbol}"}
            seed_symbols = [sym]
            target = {
                "type": "symbol",
                "symbol": sym.qualname,
                "file": sym.file,
                "start_line": sym.start_line,
                "end_line": sym.end_line,
            }
        elif file:
            seed_symbols = self.symbols_in_file(file)
            target = {"type": "file", "file": file}
        else:
            return {"ok": False, "error": "provide file=... or symbol=..."}

        seed_ids = [sym.id for sym in seed_symbols]
        expanded = self.expand(seed_ids, hops=hops)
        direct_callers: list[SymbolRow] = []
        direct_callees: list[SymbolRow] = []
        inheritance: list[SymbolRow] = []
        for sid in seed_ids:
            for other, kind in self.neighbors(sid, direction="in"):
                if kind == "calls":
                    direct_callers.append(other)
                elif kind == "inherits":
                    inheritance.append(other)
            for other, kind in self.neighbors(sid, direction="out"):
                if kind == "calls":
                    direct_callees.append(other)
                elif kind == "inherits":
                    inheritance.append(other)

        def compact_symbols(items: list[SymbolRow], limit: int = 30) -> list[dict]:
            seen: set[int] = set()
            out: list[dict] = []
            for item in items:
                if item.id in seen:
                    continue
                seen.add(item.id)
                out.append(
                    {
                        "qualname": item.qualname,
                        "file": item.file,
                        "kind": item.kind,
                        "signature": item.signature,
                        "start_line": item.start_line,
                        "end_line": item.end_line,
                    }
                )
                if len(out) >= limit:
                    break
            return out

        test_symbols = [
            sym
            for sym in [*direct_callers, *direct_callees, *(item[0] for item in expanded.values())]
            if "test" in sym.file.lower() or sym.name.startswith("test_")
        ]
        touched_files = sorted(
            {
                sym.file
                for sym in [
                    *seed_symbols,
                    *direct_callers,
                    *direct_callees,
                    *(item[0] for item in expanded.values()),
                ]
            }
        )
        findings: list[QualityFinding] = []
        if symbol and seed_symbols:
            findings.extend(self.quality_findings(symbol_id=seed_symbols[0].id, limit=50))
        elif file:
            findings.extend(self.quality_findings(file=file, limit=50))

        high = sum(1 for f in findings if f.severity == "high")
        medium = sum(1 for f in findings if f.severity == "medium")
        caller_count = len({sym.id for sym in direct_callers})
        test_count = len({sym.id for sym in test_symbols})
        risk_score = min(100, caller_count * 10 + test_count * 6 + high * 25 + medium * 12)
        if risk_score >= 70:
            risk_level = "high"
        elif risk_score >= 35:
            risk_level = "medium"
        else:
            risk_level = "low"

        chunks: list[ChunkRow] = []
        if symbol and seed_symbols:
            chunks = self.chunks_touching_symbol(seed_symbols[0].id)
        elif file:
            chunks = self.chunks_for_file(file)

        return {
            "ok": True,
            "target": target,
            "risk": {
                "level": risk_level,
                "score": risk_score,
                "direct_callers": caller_count,
                "relevant_tests": test_count,
                "quality_high": high,
                "quality_medium": medium,
            },
            "direct_callers": compact_symbols(direct_callers),
            "direct_callees": compact_symbols(direct_callees),
            "inheritance_neighbors": compact_symbols(inheritance),
            "relevant_tests": compact_symbols(test_symbols),
            "touched_files": touched_files[:50],
            "expanded_neighbors": compact_symbols([item[0] for item in expanded.values()]),
            "chunks": [
                {
                    "id": chunk.id,
                    "file": chunk.file,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "tokens": chunk.tokens,
                    "quality_summary": chunk.quality_summary,
                    "symbols": chunk.symbols.split(",") if chunk.symbols else [],
                }
                for chunk in chunks[:12]
            ],
            "quality_findings": [finding.__dict__ for finding in findings],
        }

    def neighbors(self, symbol_id: int, direction: str = "both") -> list[tuple[SymbolRow, str]]:
        """Return (symbol, edge_kind) directly connected to symbol_id.

        direction: "out" (things this calls/inherits), "in" (things that call it),
        or "both".
        """
        out: list[tuple[SymbolRow, str]] = []
        if direction in ("out", "both"):
            rows = self.conn.execute(
                """SELECT s.*, e.kind AS ekind FROM edges e
                   JOIN symbols s ON s.id = e.dst_id WHERE e.src_id=?""",
                (symbol_id,),
            ).fetchall()
            out.extend((self._row_no_ekind(r), r["ekind"]) for r in rows)
        if direction in ("in", "both"):
            rows = self.conn.execute(
                """SELECT s.*, e.kind AS ekind FROM edges e
                   JOIN symbols s ON s.id = e.src_id WHERE e.dst_id=?""",
                (symbol_id,),
            ).fetchall()
            out.extend((self._row_no_ekind(r), r["ekind"]) for r in rows)
        return out

    def _row_no_ekind(self, r: sqlite3.Row) -> SymbolRow:
        keys = [k for k in r.keys() if k != "ekind"]
        return SymbolRow(**{k: r[k] for k in keys})

    def expand(self, seed_ids: Iterable[int], hops: int = 1) -> dict[int, tuple[SymbolRow, str]]:
        """BFS from seeds up to `hops`. Returns {id: (symbol, edge_kind_hint)}.

        Excludes the seed ids themselves.
        """
        seeds = set(seed_ids)
        frontier = set(seeds)
        found: dict[int, tuple[SymbolRow, str]] = {}
        for _ in range(hops):
            nxt: set[int] = set()
            for sid in frontier:
                for sym, kind in self.neighbors(sid):
                    if sym.id not in seeds and sym.id not in found:
                        found[sym.id] = (sym, kind)
                        nxt.add(sym.id)
            frontier = nxt
            if not frontier:
                break
        return found

    def dependents_of_file(self, file: str) -> list[SymbolRow]:
        """Symbols that call into symbols defined in `file` (reverse edges)."""
        rows = self.conn.execute(
            """SELECT DISTINCT src.* FROM edges e
               JOIN symbols dst ON dst.id = e.dst_id
               JOIN symbols src ON src.id = e.src_id
               WHERE dst.file=? AND e.kind='calls' AND src.file != ?""",
            (file, file),
        ).fetchall()
        return [self._row(r) for r in rows]

    def edges_touching_files(self, files: Iterable[str]) -> list[tuple[SymbolRow, SymbolRow, str]]:
        """Edges where either endpoint is in one of `files`."""
        files = list(files)
        if not files:
            return []
        placeholders = ",".join("?" for _ in files)
        rows = self.conn.execute(
            f"""SELECT src.id AS src_row_id, dst.id AS dst_row_id, e.kind AS ekind
                FROM edges e
                JOIN symbols src ON src.id = e.src_id
                JOIN symbols dst ON dst.id = e.dst_id
                WHERE src.file IN ({placeholders}) OR dst.file IN ({placeholders})
                ORDER BY e.kind, src.file, src.start_line""",
            (*files, *files),
        ).fetchall()
        out: list[tuple[SymbolRow, SymbolRow, str]] = []
        for row in rows:
            src = self.get(row["src_row_id"])
            dst = self.get(row["dst_row_id"])
            if src and dst:
                out.append((src, dst, row["ekind"]))
        return out

    def file_source(self, file: str, repo_root: str) -> str:
        import os

        try:
            with open(os.path.join(repo_root, file), "r", encoding="utf-8") as fh:
                return fh.read()
        except OSError:
            # fall back to concatenating known symbol sources
            rows = self.conn.execute(
                "SELECT source FROM symbols WHERE file=? ORDER BY start_line", (file,)
            ).fetchall()
            return "\n\n".join(r["source"] for r in rows)

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key,value) VALUES (?,?)", (key, value)
        )
        self.conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        r = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r["value"] if r else None

    def close(self) -> None:
        self.conn.close()
