"""Indexer: parse a Python repo into symbols + edges using the stdlib `ast`.

No tree-sitter, no network, no model. Pure static analysis. This is the exact,
deterministic foundation the rest of the system trusts.

Extracts:
  - symbols:  functions, async functions, classes, methods
              (name, qualname, kind, file, line span, signature, docstring, source)
  - calls:    caller-symbol -> callee-name  (resolved to symbols best-effort by name)
  - inherits: class -> base-class name       (resolved best-effort by name)
  - imports:  file -> (module, name)         (file-level, kept in its own table)
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class Symbol:
    qualname: str
    name: str
    kind: str  # "function" | "method" | "class"
    file: str
    start_line: int
    end_line: int
    signature: str
    docstring: str
    source: str
    # best-effort, resolved after the whole repo is parsed:
    calls: list[str] = field(default_factory=list)  # callee names seen in body
    bases: list[str] = field(default_factory=list)  # base class names (classes only)


@dataclass
class ImportRow:
    file: str
    module: str
    name: str
    alias: str


@dataclass
class SourceFile:
    path: str
    source: str
    line_count: int


@dataclass
class IndexResult:
    symbols: list[Symbol]
    imports: list[ImportRow]
    files: list[SourceFile] = field(default_factory=list)


def _signature(node: ast.AST) -> str:
    """Render a def signature. Uses ast.unparse when available (py3.9+)."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ""
    try:
        args = ast.unparse(node.args)
    except Exception:
        args = ", ".join(a.arg for a in node.args.args)
    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    ret = ""
    if node.returns is not None:
        try:
            ret = " -> " + ast.unparse(node.returns)
        except Exception:
            ret = ""
    return f"{prefix}{node.name}({args}){ret}"


def _expr_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _expr_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _called_names(node: ast.AST) -> list[str]:
    """Collect call expressions inside a function body.

    We keep the full expression (`client.session.refresh`). The graph resolver
    can use the receiver when import aliases are known, and can make a cautious
    terminal-name fallback for project-specific methods.
    """
    names: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = _expr_name(child.func)
            if not name:
                continue
            names.append(name)
    # de-dupe, preserve order
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _base_names(node: ast.ClassDef) -> list[str]:
    out: list[str] = []
    for b in node.bases:
        name = _expr_name(b)
        if name:
            out.append(name)
            terminal = name.rsplit(".", 1)[-1]
            if terminal != name:
                out.append(terminal)
    return out


def index_file(path: str, repo_root: str, source: str) -> IndexResult:
    """Parse a single file's source into symbols + imports."""
    rel = os.path.relpath(path, repo_root)
    symbols: list[Symbol] = []
    imports: list[ImportRow] = []
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError:
        return IndexResult(symbols=[], imports=[], files=[SourceFile(rel, source, len(source.splitlines()))])

    module_qual = rel[:-3].replace(os.sep, ".") if rel.endswith(".py") else rel

    def add_def(node: ast.AST, qual_prefix: str, kind: str) -> None:
        name = getattr(node, "name", "<anon>")
        qual = f"{qual_prefix}.{name}" if qual_prefix else name
        start = getattr(node, "lineno", 1)
        end = getattr(node, "end_lineno", start)
        seg = ast.get_source_segment(source, node) or ""
        sym = Symbol(
            qualname=qual,
            name=name,
            kind=kind,
            file=rel,
            start_line=start,
            end_line=end,
            signature=_signature(node) if kind != "class" else f"class {name}",
            docstring=ast.get_docstring(node) or "",
            source=seg,
        )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sym.calls = _called_names(node)
        if isinstance(node, ast.ClassDef):
            sym.bases = _base_names(node)
        symbols.append(sym)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import,)):
            for alias in node.names:
                imported_as = alias.asname or alias.name.split(".", 1)[0]
                imports.append(ImportRow(rel, alias.name, alias.name, imported_as))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                imports.append(ImportRow(rel, mod, alias.name, alias.asname or alias.name))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add_def(node, module_qual, "function")
        elif isinstance(node, ast.ClassDef):
            add_def(node, module_qual, "class")
            for sub in ast.iter_child_nodes(node):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add_def(sub, f"{module_qual}.{node.name}", "method")

    return IndexResult(
        symbols=symbols,
        imports=imports,
        files=[SourceFile(rel, source, len(source.splitlines()))],
    )


def _iter_source_files(repo_root: str, ignore: Iterable[str]) -> Iterable[str]:
    from .universal import SUPPORTED_EXTS

    ignore = set(ignore)
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in ignore and not d.startswith(".")]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in SUPPORTED_EXTS:
                yield os.path.join(dirpath, fn)


DEFAULT_IGNORE = {"__pycache__", "node_modules", "venv", ".venv", "build", "dist", ".git"}


def index_repo(repo_root: str, ignore: Iterable[str] = DEFAULT_IGNORE) -> IndexResult:
    """Walk a repo and index every supported source file.

    Python uses the exact `ast` indexer; all other languages use the universal
    regex indexer. Both yield the same IndexResult shape.
    """
    from .universal import index_file_generic

    all_symbols: list[Symbol] = []
    all_imports: list[ImportRow] = []
    all_files: list[SourceFile] = []
    for path in _iter_source_files(repo_root, ignore):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                source = fh.read()
        except (OSError, UnicodeDecodeError):
            continue
        if path.endswith(".py"):
            res = index_file(path, repo_root, source)
        else:
            res = index_file_generic(path, repo_root, source)
        all_symbols.extend(res.symbols)
        all_imports.extend(res.imports)
        all_files.extend(res.files)
    return IndexResult(symbols=all_symbols, imports=all_imports, files=all_files)
