<div align="center">

# 🔺 Prism

### The runtime around the weights.

**Give a small, local LLM the scaffolding a frontier product has around it — whole-repo context, planning, tools, and execution-grounded verification — and it does real work on _your_ machine.**

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Core](https://img.shields.io/badge/core-stdlib--only-orange)
![Local-first](https://img.shields.io/badge/local--first-Ollama-black)
![Model](https://img.shields.io/badge/model-agnostic-purple)
![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen)

</div>

---

> ### The bet
> Everyone is racing to scale models. But a huge share of what makes frontier AI *feel* frontier **isn't in the weights** — it's the **harness** around them: how they pull the right context, plan, call tools, and check their own work.
>
> **Prism extracts that harness, makes it pluggable, and points it at open models you run for free — on your laptop.** Not a bigger model. The missing runtime around a small one.

---

## ⚡ Why Prism

A frontier model "reads your whole codebase," plans multi-step, uses tools, and verifies its work — by brute-forcing it with enormous context and scale. A local 8B can't. **Prism gives it the missing runtime instead of the missing parameters.**

- 🧠 **Whole-repo awareness** — without a million-token context window
- ✅ **It runs your tests** — verification is grounded in *execution*, never a weak model judging itself
- 🔧 **Tools that don't fumble** — structured, schema-enforced tool-calling
- 🗺️ **Plans it can act on** — tasks decomposed into grounded, bounded steps
- 📈 **Every run is traced** — see exactly *which layer* moved the needle
- 🧩 **Not just code** — swap one component and the same harness does data, research, ops…
- 💻 **Local-first** — stdlib-only core, runs on consumer hardware via [Ollama](https://ollama.com)
- 🔌 **Model-agnostic** — Qwen today, whatever's best tomorrow. **The harness is yours.**

---

## 🧩 The exoskeleton

Prism decomposes "frontier behavior" into **pluggable, model-agnostic layers** and orchestrates a small model through them. Frontier behavior comes from the harness — not the weights.

| Layer | What it does | Status |
|---|---|:---:|
| 🧠 **Context** | Whole-repo code graph + retrieval + agentic exploration + LSP-style navigation, packed to a token budget | ✅ |
| 🗺️ **Reasoning** | Decompose a task into grounded, bounded steps the model can actually execute | ✅ |
| 🔧 **Tools** | Hardened tool routing — constrained decoding, schema validation, fuzzy repair | ✅ |
| ✅ **Verification** | **Execution-grounded** — runs tests/compile on a sandboxed copy; never self-judges | ✅ |
| 📈 **Trace** | Records every run — context, calls, checks, repairs — so you can see what helped | ✅ |
| 🔁 **Memory** | Learns across runs: what worked, which files change together | ✅ |
| 🌐 **Domains** | Swap the checker → the same loop works **beyond coding** | ✅ |

```
  task ─▶ CONTEXT ─▶ model ─▶ TOOLS ─▶ edit ─▶ VERIFY (run tests) ─▶ ✅ done
                       ▲                              │
                       └──────── repair ◀─────────────┘   (all recorded to TRACE)
```

---

## 📊 Does it actually work?

We benchmark the **layers in isolation** — so the model's raw coding ability is factored out and you see *Prism's* contribution, not the model's.

| Layer | What we measured | Bare model | **With Prism** |
|---|---|:---:|:---:|
| 🧠 Context | did it retrieve the right files? | — | **0.94** |
| ✅ Verify | accept good / reject bad / rank | — | **5 / 5** |
| 🗺️ Reasoning | picked the right files to change | 0.69 | **0.84 (+22%)** |

**On the same local model, Prism made its reasoning 22% better** — measured on planning, where the model writes no code, so it's *Prism's* gain, not the model's.

> 🔬 **Honest caveat:** early numbers (1 trial · 16 tasks · `qwen3:4b`). Directional, not a leaderboard — and fully reproducible: `python benchmarks/layer_bench.py`.

---

## 🚀 Quickstart

```bash
# 1. grab any local model
ollama pull qwen3:8b

# 2. clone — the core is stdlib-only, nothing to pip install to try it
git clone https://github.com/Badtheorylabs/Prism && cd Prism

# 3. see it instantly (indexes a tiny app, returns a budget-packed dossier)
PYTHONPATH=src python -m prism.cli demo

# 4. drive a REAL local model end-to-end:
#    context → edit → run the tests → repair → accept only if green
PYTHONPATH=src python -m prism.cli harden "add per-user rate limiting to login" \
    --repo examples/demo_repo --model qwen3:8b --ttc
```

**The toolbox** (`python -m prism.cli <cmd>`):

| Command | What it does |
|---|---|
| `demo` | Instant proof — packs a repo into a small-model-sized dossier |
| `context` / `explore` | Whole-repo context + agentic graph exploration (no model needed) |
| `plan` | Decompose a task into steps (graph-derived or model-refined) |
| `tools` | Drive a model through the hardened tool loop |
| `harden` | Generate → **run tests** → repair, with adaptive test-time compute |
| `trace` / `memory` | Inspect what happened and what the runtime learned |
| `data` | A **non-coding** task through the same harness (proof it generalizes) |

Optional install for the `prism` command + extras:
```bash
pip install -e ".[mcp,tokens]"
```

---

## 🌐 Not just code

The truth loop is **domain-agnostic**. For code it's tests. Swap the checker and the *same* harness — context, model, verify, repair, trace — does other work:

| Domain | The truth signal |
|---|---|
| 💻 Coding | tests / compile / type-check |
| 📊 Data | schema + reconciliation (do the numbers add up?) |
| 🔎 Research | citations resolve, sources agree |
| ⚙️ Ops | records reconcile, APIs return 200 |

> Already shipped: a data-reconciliation pack where a local model computes totals and the harness **rejects any answer that doesn't reconcile** — same loop, different checker.

---

## 🏗️ Architecture

Stdlib-only core. No tree-sitter grammars, no embedding downloads, no cloud.

```
src/prism/
├── context.py · graph.py · semantic.py · packer.py   # Layer 1 — Context
├── explorer.py · lsp.py                               #   + agentic search / nav
├── planner.py                                         # Layer 2 — Reasoning
├── tools.py                                           # Layer 3 — Tools
├── execution.py · verifier.py · ttc.py                # Layer 4 — Verification
├── trace.py · memory.py                               # Spine + Memory
├── checkers.py · domains.py                           # Beyond-coding
└── llm.py                                             # model backend (Ollama)
```

---

## 🔭 The thesis

> Frontier capability = **weights × harness.** The labs pour billions into the first term. Prism owns the second — and makes it local, inspectable, and model-agnostic.

Every time open models get better, Prism gets better *for free*. The model is a swappable part; **the runtime is the moat.**

---

## 🗺️ Status & roadmap

**Live:** Context · Reasoning · Tools · Verification · Trace · Memory · Domain packs
**Next:** more domain packs · multi-trial benchmarks · richer agentic exploration · MLX / vLLM backends

Contributions welcome — this is early, opinionated, and moving fast.

---

<div align="center">

**Built by [Bad Theory Labs](https://github.com/Badtheorylabs).** MIT licensed.

*Small models. Big scaffolding. Runs on your laptop.*

</div>
