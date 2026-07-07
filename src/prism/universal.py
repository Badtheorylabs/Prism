"""Universal (language-agnostic) indexer.

The Python path uses `ast` for exact structure. Every *other* language gets a
ctags-style regex indexer here: it extracts functions, classes/types, and
methods, plus best-effort call/inheritance/import facts. Precision is lower than
a real parser, but it produces the *same* Symbol/ImportRow/SourceFile shapes the
graph consumes — so retrieval, packing, impact, and the agent loop work on JS,
TS, Go, Rust, Java, C#, C/C++, Kotlin, Swift, Scala, PHP, Ruby, and more with no
tree-sitter grammars and no network.

Design: cross-file call edges resolve by *name* in the graph layer, which is
language-independent, so we only need to surface names + spans here.
"""

from __future__ import annotations

import os
import re

from .indexer import ImportRow, IndexResult, SourceFile, Symbol

# Names that look like calls/defs but never are user symbols.
_KEYWORDS = {
    "if", "for", "while", "switch", "catch", "return", "else", "do", "with",
    "function", "fn", "func", "def", "class", "struct", "enum", "interface",
    "trait", "impl", "new", "typeof", "await", "yield", "throw", "case",
    "default", "sizeof", "and", "or", "not", "in", "is", "use", "import",
    "from", "as", "pub", "const", "let", "var", "static", "public", "private",
    "protected", "void", "get", "set", "when", "where", "match", "select",
    "defer", "go", "print", "println", "printf",
}

_CALL_RE = re.compile(r"(?<![.\w])(?P<n>[A-Za-z_][A-Za-z0-9_]*)\s*\(")

# Lines starting with these are statements, never definitions — guards against
# `return foo(...)` / `if (foo())` being mistaken for a function definition.
_STATEMENT_STARTERS = {
    "return", "if", "for", "while", "switch", "else", "throw", "case", "do",
    "with", "await", "yield", "when", "assert", "break", "continue", "elif",
    "print", "println", "echo", "raise", "del", "pass",
}

_FIRST_TOKEN = re.compile(r"^\s*([A-Za-z_]\w*)")


def _shared(comment: str, funcs: list[str], classes: list[str], imports: list[str],
            style: str = "brace") -> dict:
    return {
        "comment": comment,
        "funcs": [re.compile(p) for p in funcs],
        "classes": [re.compile(p) for p in classes],
        "imports": [re.compile(p) for p in imports],
        "style": style,
    }


_JS = _shared(
    "//",
    funcs=[
        r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*(?P<name>[A-Za-z_$][\w$]*)",
        r"^\s*(?:export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)",
        r"^\s*(?:public\s+|private\s+|protected\s+|static\s+|async\s+|readonly\s+|get\s+|set\s+)*(?P<name>[A-Za-z_$][\w$]*)\s*\([^;{]*\)\s*\{",
    ],
    classes=[
        r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(?P<name>[A-Za-z_$][\w$]*)(?:\s+extends\s+(?P<bases>[A-Za-z_$][\w$.]*))?",
    ],
    imports=[
        r"""import\s+(?:.+?\s+from\s+)?['"](?P<mod>[^'"]+)['"]""",
        r"""require\(\s*['"](?P<mod>[^'"]+)['"]\s*\)""",
    ],
)

_GO = _shared(
    "//",
    funcs=[r"^\s*func\s*(?:\([^)]*\)\s*)?(?P<name>[A-Za-z_]\w*)\s*\("],
    classes=[r"^\s*type\s+(?P<name>[A-Za-z_]\w*)\s+(?:struct|interface)\b"],
    imports=[r'^\s*import\s+"(?P<mod>[^"]+)"', r'^\s*"(?P<mod>[^"]+)"\s*$'],
)

_RUST = _shared(
    "//",
    funcs=[r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(?P<name>[A-Za-z_]\w*)"],
    classes=[
        r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait|union)\s+(?P<name>[A-Za-z_]\w*)",
        r"^\s*impl(?:<[^>]*>)?\s+(?:[A-Za-z_][\w:<>]*\s+for\s+)?(?P<name>[A-Za-z_]\w*)",
    ],
    imports=[r"^\s*use\s+(?P<mod>[A-Za-z_][\w:]*)"],
)

_JVM = _shared(  # Java / C#
    "//",
    funcs=[
        r"^\s*(?:@\w+\s*)*(?:public|private|protected|internal|static|final|abstract|virtual|override|async|synchronized|\s)+[A-Za-z_][\w<>\[\].,\s]*\s+(?P<name>[A-Za-z_]\w*)\s*\([^;]*\)\s*(?:throws [\w.,\s]+)?\{?",
    ],
    classes=[
        r"^\s*(?:public|private|protected|internal|abstract|final|sealed|static|\s)*(?:class|interface|enum|record|struct)\s+(?P<name>[A-Za-z_]\w*)(?:\s*(?:extends|:)\s*(?P<bases>[\w.<>,\s]+?))?(?:\s+implements\s+[\w.,\s<>]+)?\s*\{?",
    ],
    imports=[r"^\s*import\s+(?:static\s+)?(?P<mod>[\w.]+)", r"^\s*using\s+(?:static\s+)?(?P<mod>[\w.]+)"],
)

_C = _shared(
    "//",
    funcs=[
        r"^\s*(?:static\s+|inline\s+|extern\s+|const\s+)*[A-Za-z_][\w<>:\*&,\s]*[\s\*&]+(?P<name>[A-Za-z_]\w*)\s*\([^;{]*\)\s*(?:const\s*)?\{?\s*$",
    ],
    classes=[r"^\s*(?:class|struct|union)\s+(?P<name>[A-Za-z_]\w*)"],
    imports=[r'^\s*#\s*include\s+[<"](?P<mod>[^>"]+)[>"]'],
)

_KOTLIN = _shared(
    "//",
    funcs=[r"^\s*(?:public|private|protected|internal|open|override|suspend|inline|\s)*fun\s+(?:<[^>]*>\s*)?(?P<name>[A-Za-z_]\w*)"],
    classes=[r"^\s*(?:public|private|protected|internal|open|abstract|sealed|data|\s)*(?:class|object|interface|enum class)\s+(?P<name>[A-Za-z_]\w*)(?:\s*:\s*(?P<bases>[\w.<>,\s()]+))?"],
    imports=[r"^\s*import\s+(?P<mod>[\w.]+)"],
)

_SWIFT = _shared(
    "//",
    funcs=[r"^\s*(?:public|private|internal|fileprivate|open|static|class|override|\s)*func\s+(?P<name>[A-Za-z_]\w*)"],
    classes=[r"^\s*(?:public|private|internal|open|final|\s)*(?:class|struct|enum|protocol|extension)\s+(?P<name>[A-Za-z_]\w*)(?:\s*:\s*(?P<bases>[\w.,\s]+))?"],
    imports=[r"^\s*import\s+(?P<mod>[\w.]+)"],
)

_SCALA = _shared(
    "//",
    funcs=[r"^\s*(?:override\s+|private\s+|protected\s+|final\s+|implicit\s+|\s)*def\s+(?P<name>[A-Za-z_]\w*)"],
    classes=[r"^\s*(?:sealed\s+|abstract\s+|final\s+|case\s+|\s)*(?:class|object|trait)\s+(?P<name>[A-Za-z_]\w*)(?:.*?extends\s+(?P<bases>[\w.]+))?"],
    imports=[r"^\s*import\s+(?P<mod>[\w.]+)"],
)

_PHP = _shared(
    "//",
    funcs=[r"^\s*(?:public|private|protected|static|final|abstract|\s)*function\s+(?P<name>[A-Za-z_]\w*)"],
    classes=[r"^\s*(?:abstract\s+|final\s+)*(?:class|interface|trait)\s+(?P<name>[A-Za-z_]\w*)(?:\s+extends\s+(?P<bases>[\w\\]+))?"],
    imports=[r"""^\s*(?:require|include|require_once|include_once)\s*\(?\s*['"](?P<mod>[^'"]+)['"]""", r"^\s*use\s+(?P<mod>[\w\\]+)"],
)

_RUBY = _shared(
    "#",
    funcs=[r"^\s*def\s+(?:self\.)?(?P<name>[A-Za-z_]\w*[!?=]?)"],
    classes=[r"^\s*(?:class|module)\s+(?P<name>[A-Z]\w*)(?:\s*<\s*(?P<bases>[\w:]+))?"],
    imports=[r"""^\s*require(?:_relative)?\s+['"](?P<mod>[^'"]+)['"]"""],
    style="keyword",
)

CONFIGS: dict[str, dict] = {}
for exts, cfg in [
    ((".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"), _JS),
    ((".go",), _GO),
    ((".rs",), _RUST),
    ((".java", ".cs"), _JVM),
    ((".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh", ".m", ".mm"), _C),
    ((".kt", ".kts"), _KOTLIN),
    ((".swift",), _SWIFT),
    ((".scala",), _SCALA),
    ((".php",), _PHP),
    ((".rb",), _RUBY),
]:
    for e in exts:
        CONFIGS[e] = cfg

SUPPORTED_EXTS = {".py"} | set(CONFIGS)


def _code_only(line: str, comment: str) -> str:
    """Strip strings and line comments so brace counting isn't fooled."""
    line = re.sub(r"'([^'\\]|\\.)*'|\"([^\"\\]|\\.)*\"|`([^`\\]|\\.)*`", '""', line)
    idx = line.find(comment)
    if idx != -1:
        line = line[:idx]
    return line


def _brace_end(lines: list[str], start: int, comment: str, max_scan: int = 4000) -> int:
    """Find the line index that closes the block opened at/after `start`."""
    depth = 0
    seen_open = False
    limit = min(len(lines), start + max_scan)
    for i in range(start, limit):
        code = _code_only(lines[i], comment)
        for ch in code:
            if ch == "{":
                depth += 1
                seen_open = True
            elif ch == "}":
                depth -= 1
        if seen_open and depth <= 0:
            return i
        if not seen_open and i > start + 3:
            # no opening brace shortly after the signature -> single-line decl
            return start
    return min(limit - 1, start)


def _keyword_end(lines: list[str], start: int, comment: str, max_scan: int = 4000) -> int:
    """Block end for def..end / do..end languages (Ruby), by keyword nesting."""
    depth = 0
    opener = re.compile(r"\b(def|class|module|do|if|unless|case|begin|while|until)\b")
    closer = re.compile(r"^\s*end\b")
    limit = min(len(lines), start + max_scan)
    for i in range(start, limit):
        code = _code_only(lines[i], comment)
        if i == start:
            depth += 1
            continue
        depth += len(opener.findall(code))
        if closer.search(code):
            depth -= len(closer.findall(code))
        if depth <= 0:
            return i
    return min(limit - 1, start)


def _docstring_above(lines: list[str], start: int, comment: str) -> str:
    out: list[str] = []
    i = start - 1
    while i >= 0 and len(out) < 4:
        s = lines[i].strip()
        if s.startswith(comment) or s.startswith("///") or s.startswith("*") or s.startswith("/*"):
            out.append(s.lstrip("/*# ").rstrip("*/ ").strip())
            i -= 1
        else:
            break
    out.reverse()
    return " ".join(x for x in out if x)[:200]


def _calls_in(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in _CALL_RE.finditer(text):
        n = m.group("n")
        if n in _KEYWORDS or n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def _bases_from(raw: str | None) -> list[str]:
    if not raw:
        return []
    parts = re.split(r"[,\s:()<>]+", raw)
    out: list[str] = []
    for p in parts:
        p = p.strip().strip(".")
        if p and p not in _KEYWORDS and p[0].isalpha() or (p and p[0] == "_"):
            terminal = p.rsplit(".", 1)[-1]
            if terminal:
                out.append(terminal)
    return out


def index_file_generic(path: str, repo_root: str, source: str) -> IndexResult:
    rel = os.path.relpath(path, repo_root)
    ext = os.path.splitext(path)[1].lower()
    cfg = CONFIGS.get(ext)
    lines = source.splitlines()
    src_file = SourceFile(rel, source, len(lines))
    if cfg is None:
        return IndexResult(symbols=[], imports=[], files=[src_file])

    comment = cfg["comment"]
    module_qual = os.path.splitext(rel)[0].replace(os.sep, ".")
    end_of = _keyword_end if cfg["style"] == "keyword" else _brace_end

    # First pass: classes (so methods can be attributed to their owner).
    raw_symbols: list[tuple[Symbol, int, int]] = []  # (sym, start_idx, end_idx)
    class_ranges: list[tuple[str, int, int]] = []

    for i, line in enumerate(lines):
        for pat in cfg["classes"]:
            m = pat.search(line)
            if not m:
                continue
            name = m.group("name")
            end = end_of(lines, i, comment)
            span = "\n".join(lines[i:end + 1])
            bases = _bases_from(m.groupdict().get("bases"))
            qual = f"{module_qual}.{name}"
            sym = Symbol(
                qualname=qual, name=name, kind="class", file=rel,
                start_line=i + 1, end_line=end + 1,
                signature=line.strip()[:200], docstring=_docstring_above(lines, i, comment),
                source=span, bases=bases,
            )
            raw_symbols.append((sym, i, end))
            class_ranges.append((name, i, end))
            break

    def enclosing_class(idx: int) -> str | None:
        best: tuple[int, str] | None = None
        for name, cs, ce in class_ranges:
            if cs < idx <= ce:
                span = ce - cs
                if best is None or span < best[0]:
                    best = (span, name)
        return best[1] if best else None

    for i, line in enumerate(lines):
        ft = _FIRST_TOKEN.match(line)
        if ft and ft.group(1) in _STATEMENT_STARTERS:
            continue  # this line is a statement, not a definition
        for pat in cfg["funcs"]:
            m = pat.search(line)
            if not m:
                continue
            name = m.group("name")
            if not name or name in _KEYWORDS:
                continue
            cls = enclosing_class(i)
            kind = "method" if cls else "function"
            qual = f"{module_qual}.{cls}.{name}" if cls else f"{module_qual}.{name}"
            end = end_of(lines, i, comment)
            span = "\n".join(lines[i:end + 1])
            sym = Symbol(
                qualname=qual, name=name, kind=kind, file=rel,
                start_line=i + 1, end_line=end + 1,
                signature=line.strip()[:200], docstring=_docstring_above(lines, i, comment),
                source=span, calls=_calls_in(span),
            )
            raw_symbols.append((sym, i, end))
            break

    imports: list[ImportRow] = []
    in_import_block = False
    for line in lines:
        if ext == ".go":
            if re.match(r"^\s*import\s*\($", line):
                in_import_block = True
                continue
            if in_import_block and re.match(r"^\s*\)", line):
                in_import_block = False
        for pat in cfg["imports"]:
            m = pat.search(line)
            if m:
                mod = m.group("mod")
                name = mod.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[-1]
                imports.append(ImportRow(rel, mod, name, name))
                break

    # Dedupe same-name definitions in a file (e.g. Rust `struct` + `impl`),
    # keeping the widest span so method attribution uses the real body.
    best: dict[tuple[str, str], Symbol] = {}
    order: list[tuple[str, str]] = []
    for sym in (s for (s, _a, _b) in raw_symbols):
        key = (sym.qualname, sym.kind)
        prev = best.get(key)
        if prev is None:
            best[key] = sym
            order.append(key)
        elif (sym.end_line - sym.start_line) > (prev.end_line - prev.start_line):
            # merge: widest span wins, but keep any calls/bases discovered
            sym.calls = list(dict.fromkeys([*prev.calls, *sym.calls]))
            sym.bases = list(dict.fromkeys([*prev.bases, *sym.bases]))
            best[key] = sym
    symbols = [best[k] for k in order]
    return IndexResult(symbols=symbols, imports=imports, files=[src_file])
