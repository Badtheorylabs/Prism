<div align="center">

# Prism

### Frontier intelligence, assembled — not bought.

**A laptop-sized model behaves frontier-level not because it _is_ one — but because it's wearing the exoskeleton of one.**

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Core](https://img.shields.io/badge/core-stdlib--only-orange)
![Local-first](https://img.shields.io/badge/local--first-Ollama-black)
![Model](https://img.shields.io/badge/model-agnostic-purple)

</div>

---

## The principle

A frontier model isn't just big weights. It's big weights **plus** a huge amount of scaffolding it runs *silently in its head* — holding the whole repo in attention, planning multi-step, checking its own work, recalling facts, orchestrating tools, reasoning before it answers. Small models have thinner weights **and** they don't do that internal scaffolding well.

**Prism externalizes the scaffolding.** It takes each thing a big model does implicitly and makes it an explicit, deterministic, inspectable module *outside* the model — then runs a small, local model through the modules.

> Intelligence moves from inside the weights to inside the harness.

Not a context tool. Not a prompt pack. **An inference-time capability layer** — a small-model runtime where frontier behavior is *assembled* from pluggable, model-agnostic modules instead of *bought* in the weights. It's the bet the whole field is making — test-time compute over bigger models — productized as reusable plugins, so any open model on your own hardware inherits all of it.

---

## The exoskeleton

Each plate is one thing a frontier model does internally, pulled out and made explicit. Snap them onto a small model and it starts behaving like a big one.

| Plate | The frontier does it *in its head* | Prism makes it an explicit module | Status |
|---|---|---|:---:|
| **Context** | holds the whole repo in attention | code graph + retrieval + agentic exploration, packed to a token budget | Live |
| **Reasoning** | plans long chains internally | a decomposer that holds the plan as state and feeds the model one bounded step at a time | Live |
| **Tools** | orchestrates tools cleanly | a strict-schema router with constrained decoding + repair, so a weak model can't fumble the mechanics | Live |
| **Verification** | is "right more often" | compile / test / lint on a sandboxed copy — the compiler as a free, frontier-grade judge (never self-judging) | Live |
| **Critique** | catches its own errors | an adversarial reviewer feeding a repair loop | Live |
| **Trace + Memory** | remembers what worked | every run recorded; learns which moves and files win | Live |
| **Grounding** | "knows more" | retrieval over docs / APIs / the web — look it up instead of knowing it | Planned |
| **Ensemble** | one big brain | many small passes, fanned out and judged | Planned |

```
  task ──▶ CONTEXT ──▶ model ──▶ TOOLS ──▶ act ──▶ VERIFY (run it) ──▶ done
                         ▲                              │
                         └────────── repair ◀───────────┘   (every step → TRACE → MEMORY)
```

**Context was the first plate. The aim is the whole suit.**

---

## How we measure

Prism benchmarks each **plate in isolation** — so the model's raw ability is factored out and you see the *exoskeleton's* contribution, not the model's. Measuring a final patch mostly measures whether the model can code (which the harness can't change); measuring the plates shows what Prism actually adds.

Run it yourself:

```bash
# model-free plates (context, verification)
PYTHONPATH=src python benchmarks/layer_bench.py --layers context,verify

# model-dependent plates (needs Ollama)
PYTHONPATH=src python benchmarks/layer_bench.py --model qwen3:8b --layers reasoning,tools
```

---

## Quickstart

```bash
# 1. any local model
ollama pull qwen3:8b

# 2. clone — the core is stdlib-only, nothing to pip install to try it
git clone https://github.com/Badtheorylabs/Prism && cd Prism

# 3. see it instantly (indexes a tiny app, returns a budget-packed dossier)
PYTHONPATH=src python -m prism.cli demo

# 4. run a REAL local model through the exoskeleton:
#    context -> act -> run the tests -> repair -> accept only if green
PYTHONPATH=src python -m prism.cli harden "add per-user rate limiting to login" \
    --repo examples/demo_repo --model qwen3:8b --ttc
```

**The toolbox** (`python -m prism.cli <cmd>`):

| Command | Plate | What it does |
|---|---|---|
| `context` / `explore` | Context | whole-repo context + agentic graph exploration (no model needed) |
| `plan` | Reasoning | decompose a task into grounded, bounded steps |
| `tools` | Tools | drive a model through the hardened tool loop |
| `harden` | Verification | generate -> **run tests** -> repair, with adaptive test-time compute |
| `trace` / `memory` | Trace + Memory | inspect what happened and what the runtime learned |
| `data` | Domains | a **non-coding** task through the same harness |
| `demo` | — | instant proof on a bundled app |

Optional install for the `prism` command + extras: `pip install -e ".[mcp,tokens]"`

---

## Not just code

The exoskeleton is domain-agnostic. Verification for code is tests. Swap that one plate and the *same* runtime — context, model, verify, repair, trace — does other work:

| Domain | The truth signal |
|---|---|
| Coding | tests / compile / type-check |
| Data | schema + reconciliation (do the numbers add up?) |
| Research | citations resolve, sources agree |
| Ops | records reconcile, APIs return 200 |

> Already shipped: a data-reconciliation pack where a local model computes totals and the harness **rejects any answer that doesn't reconcile** — same loop, different checker.

---

## Architecture

Stdlib-only core. No tree-sitter grammars, no embedding downloads, no cloud.

```
src/prism/
├── context.py · graph.py · semantic.py · packer.py   # Context
├── explorer.py · lsp.py                               #   + agentic search / nav
├── planner.py                                         # Reasoning
├── tools.py                                           # Tools
├── execution.py · verifier.py · ttc.py                # Verification + Critique
├── trace.py · memory.py                               # Trace + Memory
├── checkers.py · domains.py                           # Beyond-coding
└── llm.py                                             # model backend (Ollama)
```

---

## The thesis

> Frontier capability = **weights × harness.** The labs pour billions into the first term. Prism owns the second — and makes it local, inspectable, and model-agnostic.

Every time open models get better, Prism gets better *for free*. The model is a swappable part; **the exoskeleton is the moat.**

---

## Status and roadmap

**Live:** Context, Reasoning, Tools, Verification, Critique, Trace, Memory, Domain packs.
**Next:** Grounding (docs/web retrieval), Ensemble (fan-out + judge), multi-trial benchmarks, MLX / vLLM backends.

Contributions welcome — this is early, opinionated, and moving fast.

---

<div align="center">

**Built by [Bad Theory Labs](https://github.com/Badtheorylabs).** MIT licensed.

*Small model. Whole exoskeleton. Runs on your laptop.*

</div>
