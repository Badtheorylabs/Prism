# Prism

**Split a whole codebase into just the slice that matters — a token-budget-aware
context layer that gives small models and agents whole-repo sight.**

> pip: `prism-context` · module: `prism` · CLI: `prism`

Big models "read your whole codebase" by brute force — huge context windows and
raw reasoning power. Small models can't. `prism-context` closes the gap by
moving the intelligence *out* of the model: it precomputes an exact code graph,
retrieves the relevant slice, and packs it to fit a fixed token budget — so a
3–8B model (or any agent) gets exactly the ~few-thousand tokens that matter,
pre-chewed.

The goal is a **virtual whole-codebase attention layer**: index the entire repo
once, then give small models stable handles for files, symbols, chunks, and line
windows so they can navigate huge codebases through tools instead of needing a
1M-token context window.

> The interface is the product. Everything behind `get_context` is swappable.

Core is **stdlib-only** (Python `ast` + `sqlite3` + a pure-Python BM25 hybrid
retriever) — no tree-sitter grammars, no embedding-model download, no network.
Runs on consumer hardware.

---

## The exoskeleton — Prism's layers

A frontier model is big weights **plus** a lot of scaffolding it does silently in
its head: it holds the whole repo in attention, plans multi-step, uses tools,
checks its own work. Prism externalizes that scaffolding into deterministic,
model-agnostic **layers** and orchestrates a small model through them. Frontier
behavior comes from the harness — and, where the research demands it, from a
**distilled reasoning model** underneath (not a vanilla base).

This design was **audited against the 2024–2026 literature** and redesigned where
the evidence said we were wrong — see [docs/research/frontier-vs-prism.md](docs/research/frontier-vs-prism.md).
The headline correction: reasoning is *trained into weights*, and self-critique
from a weak model is unreliable — so reasoning uses a distilled checkpoint and
**verification is grounded in execution, not self-talk.**

| Layer | Capability | Status |
|---|---|---|
| **1. Context** | budgeted whole-repo attention **+ agentic graph exploration + LSP-style navigation + line-level ranking** (BM25 + call-graph kept — anti-BM25 hype was refuted) | ✅ live |
| **2. Reasoning / Planning** | bounded steps as explicit state, **powered by a distilled reasoning checkpoint** (`deepseek-r1:7b`), plus adaptive test-time compute | ✅ live |
| **3. Tools** | hardened routing with **constrained/structured decoding** (fuzzy parse only as fallback), ReAct loop exposing Layers 1 & 2 as tools | ✅ live |
| **4. Verification** | **execution-grounded** (compile/tests/types on a repo overlay) as the primary signal; self-critique demoted to a *separate-verifier-only* hint; best-of-N + adaptive TTC ranked by execution | ✅ live |

Layer 2 runs **with no model at all** — it can derive a real, ordered plan from
the code graph alone (`prism plan "<task>"`), which is the thesis made literal:
*global reasoning is infrastructure; the model only does local reasoning.*
Layer 1's exploration (`prism explore "<task>"`) is likewise deterministic.

---

## The contract

The agent loop is now six composable tools: `build_attention_manifest` for the
whole-repo attention map, `search_code` for discovery, `get_context` for the
task dossier, `inspect_code` for exact file/symbol/line drill-down,
`analyze_impact` for blast radius, and `verify` for the
deterministic edit loop. See [`docs/agent-protocol.md`](docs/agent-protocol.md)
for the full plug-in contract.

### `build_attention_manifest(repo_root, task, token_budget) -> dict`
The first-call primitive for small-model agents:

```jsonc
{
  "mode": "virtual_whole_codebase_attention",
  "coverage": { "files": 420, "symbols": 18000, "lines": 900000 },
  "attention": {
    "symbols": [{ "handle": "symbol:api.login", "lines": [9, 16] }],
    "chunks": [{ "handle": "chunk:42", "file": "api.py" }],
    "files": [{ "handle": "file:api.py" }]
  },
  "navigation_protocol": [...]
}
```

It does not pretend to dump an infinite repo into a small prompt. It gives the
model a compact, navigable map of the entire indexed codebase.

### `search_code(repo_root, query, top_k=10) -> dict`
Discovery before a full context package:

```jsonc
{
  "symbols": [{ "qualname": "api.login", "score": 13.67 }],
  "chunks": [{ "file": "api.py", "start_line": 1, "end_line": 80 }],
  "files": [{ "file": "api.py", "score": 12.2 }]
}
```

### `get_context(repo_root, task, token_budget) -> dict`
The crown jewel. Returns the optimal use of the budget:

```jsonc
{
  "files_to_edit":    [{ "path": "...", "full_source": "...", "tokens": 812 }],
  "periphery":        [{ "qualname": "...", "signature": "...",
                         "one_line_summary": "...", "relation": "callee" }],
  "dependency_edges": [{ "from": "api.login", "to": "auth.py", "kind": "calls" }],
  "relevant_tests":   [{ "qualname": "tests.test_api.test_login_ok", ... }],
  "token_budget":     8000,
  "fits_in":          5231   // guaranteed <= token_budget
}
```

Budget policy: **full source** for edit targets (≤60%), **signatures + one-line
summaries** for the periphery (≤30%), tests (≤10%), quality signals near likely
edits, and exact dependency edges. Oversized files are returned as targeted
symbol excerpts instead of blowing the budget.

### `inspect_code(repo_root, file|symbol|lines) -> dict`
Fine-grained substrate access for agents that need more than the first dossier:

```jsonc
{
  "symbol": "api.login",
  "lines": [{ "line_no": 12, "role": "control_flow",
              "owner_symbol_id": 7, "quality_flags": "" }],
  "neighbors": [{ "qualname": "auth.verify_password", "kind": "calls" }],
  "chunks": [{ "file": "api.py", "start_line": 1, "end_line": 80 }],
  "quality_findings": []
}
```

This is how a small model zooms from repo-level context down to exact lines
without asking for a huge file or entire codebase dump.

### `analyze_impact(repo_root, file|symbol, hops=2) -> dict`
Pre-edit blast radius:

```jsonc
{
  "risk": { "level": "medium", "score": 48 },
  "direct_callers": [{ "qualname": "tests.test_api.test_login_ok" }],
  "direct_callees": [{ "qualname": "auth.verify_password" }],
  "relevant_tests": [...],
  "touched_files": ["api.py", "auth.py", "tests/test_api.py"]
}
```

### `verify(paths, run_tests=False) -> dict`
The verify hook — a free "big model." Compile/test-checks an edit and returns
structured errors to feed back to the small model:

```jsonc
{ "ok": false,
  "errors": [{ "file": "api.py", "line": 12, "message": "...", "kind": "syntax" }],
  "tests":  { "ran": true, "passed": 3, "failed": 1, "output": "..." } }
```

The agent loop is: **get_context → model edits → verify → feed errors back → retry.**

---

## Quick start

```bash
# no install needed for the core — stdlib only
cd prism

# instant wow path: ask a small-model-sized context server for a dossier
PYTHONPATH=src python -m prism.cli demo

# first-call primitive for a small coding agent
PYTHONPATH=src python -m prism.cli attention \
    "add rate limiting to login" --repo /path/to/repo --budget 8000

# 1. index a repo
PYTHONPATH=src python -m prism.cli index /path/to/repo

# 2. see the repo map the model does not have to infer
PYTHONPATH=src python -m prism.cli stats --repo /path/to/repo

# 3. ask for a budgeted context dossier
PYTHONPATH=src python -m prism.cli context "add rate limiting to the API" \
    --repo /path/to/repo --budget 8000 --format markdown

# 4. search and analyze impact before editing
PYTHONPATH=src python -m prism.cli search "rate limiting login" --repo /path/to/repo
PYTHONPATH=src python -m prism.cli impact --repo /path/to/repo --symbol api.login

# 5. drill into an exact symbol or line window
PYTHONPATH=src python -m prism.cli inspect --repo /path/to/repo --symbol api.login
PYTHONPATH=src python -m prism.cli inspect --repo /path/to/repo \
    --file api.py --start 20 --end 40

# 6. verify an edit
PYTHONPATH=src python -m prism.cli verify /path/to/repo/api.py

# 7. Layer 1: agentic graph exploration + line-level ranking (no model)
PYTHONPATH=src python -m prism.cli explore "add rate limiting to login" --repo /path/to/repo

# 8. Layer 2: decompose a task (distilled reasoning model recommended)
PYTHONPATH=src python -m prism.cli plan "add rate limiting" --repo /path/to/repo \
    --model deepseek-r1:7b --run

# end-to-end demo (stub model, no network):
PYTHONPATH=src python examples/reference_agent.py /path/to/repo "add caching"

# 9. drive a REAL local model through context -> verify -> retry (needs Ollama)
ollama serve && ollama pull qwen2.5-coder:7b
PYTHONPATH=src python -m prism.cli agent "add rate limiting to login" \
    --repo /path/to/repo --model qwen2.5-coder:7b --stream --diff

# 10. Layer 4: execution-grounded hardening (runs tests on a repo overlay)
PYTHONPATH=src python -m prism.cli harden "add rate limiting to login" \
    --repo /path/to/repo --model qwen2.5-coder:7b --ttc          # adaptive test-time compute
#   --best-of N        : sample N candidates, rank by EXECUTION (not self-score)
#   --verifier-model M : a SEPARATE model for advisory critique (never self-judges)
#   --no-tests         : skip the overlay test run (compile-check only)
#   --apply            : write the accepted files to the repo (off by default)

# north-star metric: retrieval recall replayed over real git history
PYTHONPATH=src python benchmarks/recall.py /path/to/some/git/repo --n 20
```

Install as a package (adds the `prism` command + optional extras):

```bash
pip install -e .                 # core
pip install -e ".[mcp,tokens]"   # MCP transport + tiktoken accuracy
```

### As an MCP tool for any agent
```bash
python -m prism.server /path/to/repo   # needs: pip install mcp
```
Exposes `get_code_context` and `verify_edit` as native MCP tools.

### What the demo proves

The built-in demo indexes a small login API and returns a compact dossier:

- full source for `api.py` and `limits.py`, the likely edit files
- signatures and summaries for password/session code as periphery
- tests that call the login flow
- dependency edges that show what calls into the edit surface
- budget accounting so an agent can fit the result into a small context window

That is the product thesis in one run: whole-repo awareness, packed into a
small-model prompt.

---

## Architecture

| Module | Role |
|---|---|
| `indexer` | `ast` → symbols (functions/classes/methods) + call/inherit/import facts |
| `graph` | SQLite persistence + symbols, lines, chunks, quality findings, traversal |
| `semantic` | pure-Python BM25 hybrid search (embedder is pluggable) |
| `packer` | **budget-aware packing** — the crown jewel |
| `verify` | compile/pytest checks → structured errors |
| `render` | JSON → agent-readable markdown dossier |
| `context` | orchestrator exposing `get_context` |
| `server` | optional MCP transport |

Data flow: `indexer → graph + semantic → packer → (agent edits) → verify → loop`.

---

## What's deliberately v1 (see the roadmap)

- **Universal language support.** Python uses the exact `ast` indexer; every
  other language (JS/TS, Go, Rust, Java, C#, C/C++, Kotlin, Swift, Scala, PHP,
  Ruby, …) uses a ctags-style regex indexer in [`universal.py`](src/prism/universal.py).
  Lower precision than a real parser, but same `Symbol`/edge shapes — so
  retrieval, packing, impact, and the agent loop work across all of them.
  Compile-level `verify` is Python-native today; other languages get a light
  balance check, with a per-language linter hook to plug in.
- **BM25, not embeddings.** Zero-dependency and offline; swap `semantic.search`
  for a small on-device embedder without touching anything else.
- **Full re-index on demand.** Incremental (LSP-style) re-index on file save is v2.
- **Single-file edit targets.** Coordinated cross-file edit sets are v2.

## North-star metric

Not "does it feel smart" — **file recall** replayed over real git commits
(`benchmarks/recall.py`): for a known change, did the packer surface every file
the commit actually touched, within budget? That number drives everything
downstream.

## Launch hook

> Big models read your whole repo by brute force. Small models need a map.
> `prism-context` builds that map, packs the right files into a fixed token
> budget, then lets compilers/tests close the loop.

See [`docs/launch.md`](docs/launch.md) for the demo script and launch copy.

## License
MIT
