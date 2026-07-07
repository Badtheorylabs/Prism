"""Prism layer benchmark - measure the harness, not just final code writing.

The end-to-end benchmark asks: "did the model write a passing patch?"
That is useful, but it mostly measures the generator model.

This benchmark asks narrower questions:
  - context: did Prism retrieve the right edit files?
  - reasoning: did Prism's planning scaffold improve target-file planning?
  - tools: did structured tool scaffolding improve valid/appropriate tool use?
  - verify: did execution grounding accept/reject/rank candidates correctly?

Usage:
    PYTHONPATH=src python benchmarks/layer_bench.py --layers context,verify
    PYTHONPATH=src python benchmarks/layer_bench.py --model qwen3:4b --layers reasoning,tools
    PYTHONPATH=src python benchmarks/layer_bench.py --list
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prism.agent import ProposedFile  # noqa: E402
from prism.context import get_context  # noqa: E402
from prism.llm import DEFAULT_HOST, DEFAULT_MODEL, Ollama, call_chat  # noqa: E402
from prism.planner import PLAN_SYSTEM, Plan, Step, _parse_plan, make_plan  # noqa: E402
from prism.tools import run_tools  # noqa: E402
from prism.verifier import best_of_n, make_verdict  # noqa: E402

from tasks import TASKS, TASK_BY_NAME, _visible_source, materialize  # noqa: E402


EXPECTED_FILES: dict[str, set[str]] = {
    "impl_add": {"calc.py"},
    "fix_reverse": {"strutil.py"},
    "empty_mean": {"mathx.py"},
    "cross_file_hash": {"auth.py", "api.py"},
    "rate_limit": {"limits.py"},
    "clamp_edges": {"bounds.py"},
    "parse_int_default": {"parser.py"},
    "unique_preserve_order": {"dedupe.py"},
    "paginate_one_based": {"pages.py"},
    "expiry_boundary": {"tokens.py"},
    "moving_average_window": {"series.py"},
    "retry_backoff_cap": {"retry.py"},
    "inventory_transfer": {"ops.py", "inventory.py"},
    "slugify_punctuation": {"slug.py"},
    "cart_totals": {"cart.py"},
    "noisy_discount_context": {"billing/pricing.py", "billing/rules.py"},
}


@dataclass
class LayerResult:
    layer: str
    unit: str
    variant: str
    ok: bool
    score: float
    latency: float
    note: str


def _selected_tasks(names: str | None, band: str | None):
    tasks = [TASK_BY_NAME[n] for n in names.split(",")] if names else TASKS
    if band:
        tasks = [t for t in tasks if t.band == band]
    return tasks


def _coverage(expected: set[str], found: set[str]) -> tuple[float, str]:
    hits = sorted(expected & found)
    missing = sorted(expected - found)
    score = len(hits) / max(1, len(expected))
    note = f"hit={hits or '-'} missing={missing or '-'}"
    return score, note


def _file_terms(path: str) -> set[str]:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    return {path.lower(), base.lower(), stem.lower()}


def _plan_text(plan: Plan) -> str:
    return "\n".join(
        f"{s.title}\n{s.detail}\n{s.target}" for s in plan.steps
    ).lower()


def _plan_file_hits(plan: Plan, expected: set[str]) -> set[str]:
    text = _plan_text(plan)
    hits: set[str] = set()
    for path in expected:
        if any(term and term in text for term in _file_terms(path)):
            hits.add(path)
    return hits


def bench_context(tasks) -> list[LayerResult]:
    out: list[LayerResult] = []
    for task in tasks:
        expected = EXPECTED_FILES.get(task.name, set())
        if not expected:
            continue
        with tempfile.TemporaryDirectory() as root:
            materialize(task, root)
            t0 = time.time()
            payload = get_context(root, task.prompt, token_budget=6000)
            latency = time.time() - t0
        found = {f.get("path", "") for f in payload.get("files_to_edit", [])}
        score, note = _coverage(expected, found)
        out.append(LayerResult("context", task.name, "get_context", score == 1.0, score, latency, note))
    return out


BARE_PLAN_SYSTEM = (
    "You are a planning module. Do not write code. Given a repository and a task, "
    "respond ONLY with a JSON array of 2 to 6 steps. Each step must have "
    '{"title": "...", "detail": "...", "target": "file or symbol", "kind": "edit|investigate"}.'
)


# Reasoning bench holds THINKING OFF for both variants: thinking is the model's
# contribution, not Prism's. Disabling it isolates Prism's grounding as the delta
# and keeps calls fast/scorable on slow local hardware (base M2 ~8 tok/s).
def _bare_plan(root: str, prompt: str, client) -> Plan:
    user = f"Task: {prompt}\n\nRepository:\n{_visible_source(root)}"
    # Same output bounding as the Prism variant, so the comparison stays fair.
    raw = call_chat(client, BARE_PLAN_SYSTEM, user, think=False, format="json", num_predict=512)
    return Plan(prompt, _parse_plan(raw, prompt), "bare", ["bare model plan"])


def bench_reasoning(tasks, client) -> list[LayerResult]:
    out: list[LayerResult] = []
    for task in tasks:
        expected = EXPECTED_FILES.get(task.name, set())
        if not expected:
            continue
        with tempfile.TemporaryDirectory() as root:
            materialize(task, root)
            for variant, fn in (
                ("bare_plan", lambda: _bare_plan(root, task.prompt, client)),
                ("prism_plan", lambda: make_plan(root, task.prompt, token_budget=6000, client=client, think=False)),
            ):
                t0 = time.time()
                try:
                    plan = fn()
                    latency = time.time() - t0
                    hits = _plan_file_hits(plan, expected)
                    score, note = _coverage(expected, hits)
                    note = f"steps={len(plan.steps)} {note}"
                    out.append(LayerResult("reasoning", task.name, variant, score == 1.0, score, latency, note))
                except Exception as e:
                    out.append(LayerResult("reasoning", task.name, variant, False, 0.0, time.time() - t0, f"error: {e}"))
    return out


@dataclass
class ToolProbe:
    name: str
    repo_task: str
    prompt: str
    allowed_tools: set[str]


TOOL_PROBES = [
    ToolProbe(
        "find_relevant_files",
        "noisy_discount_context",
        "Find the files relevant to fixing VIP discount calculation.",
        {"search_code", "get_context", "plan"},
    ),
    ToolProbe(
        "read_specific_file",
        "noisy_discount_context",
        "Read billing/pricing.py around the final_price function.",
        {"read_file", "inspect_code"},
    ),
    ToolProbe(
        "impact_analysis",
        "noisy_discount_context",
        "Analyze the impact of changing final_price.",
        {"analyze_impact", "inspect_code"},
    ),
    ToolProbe(
        "make_plan",
        "noisy_discount_context",
        "Make a short plan for fixing the VIP discount bug.",
        {"plan", "get_context"},
    ),
]


def bench_tools(client) -> list[LayerResult]:
    out: list[LayerResult] = []
    for probe in TOOL_PROBES:
        task = TASK_BY_NAME[probe.repo_task]
        with tempfile.TemporaryDirectory() as root:
            materialize(task, root)
            for structured in (False, True):
                variant = "structured" if structured else "loose"
                t0 = time.time()
                try:
                    trace = run_tools(root, probe.prompt, client, max_steps=3, structured=structured)
                    latency = time.time() - t0
                    called = [s.tool for s in trace.steps]
                    ok_steps = [s for s in trace.steps if s.ok]
                    used = any(t in probe.allowed_tools for t in called)
                    ok = used and bool(ok_steps)
                    score = 1.0 if ok else 0.0
                    note = f"called={called or '-'} stopped={trace.stopped}"
                    out.append(LayerResult("tools", probe.name, variant, ok, score, latency, note))
                except Exception as e:
                    out.append(LayerResult("tools", probe.name, variant, False, 0.0, time.time() - t0, f"error: {e}"))
    return out


def bench_verify() -> list[LayerResult]:
    out: list[LayerResult] = []

    with tempfile.TemporaryDirectory() as root:
        task = TASK_BY_NAME["clamp_edges"]
        materialize(task, root)

        cases = [
            ("syntax_reject", [ProposedFile("bounds.py", "def clamp(:\n")], False),
            ("unsafe_path_reject", [ProposedFile("../escape.py", "def f():\n    return 1\n")], False),
        ]
        if importlib.util.find_spec("pytest") is not None:
            cases.extend([
                (
                    "logic_reject",
                    [ProposedFile("bounds.py", "def clamp(x, lo, hi):\n    if x > hi:\n        return hi\n    return x\n")],
                    False,
                ),
                (
                    "good_accept",
                    [ProposedFile("bounds.py", "def clamp(x, lo, hi):\n    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n    return x\n")],
                    True,
                ),
            ])

        for name, files, expected_ok in cases:
            t0 = time.time()
            verdict = make_verdict(root, files, task=task.prompt, run_tests=True)
            ok = verdict.ok is expected_ok
            out.append(LayerResult("verify", name, "make_verdict", ok, 1.0 if ok else 0.0, time.time() - t0,
                                   f"expected={expected_ok} got={verdict.ok} signal={verdict.signal}"))

        if importlib.util.find_spec("pytest") is not None:
            bad = [ProposedFile("bounds.py", "def clamp(x, lo, hi):\n    return x\n")]
            good = [ProposedFile("bounds.py", "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n")]

            def gen(_feedback, temperature):
                return bad if temperature < 0.2 else good

            t0 = time.time()
            winner, _all = best_of_n(root, task.prompt, client=None, n=2, run_tests=True, generate=gen)
            ok = winner.ok and winner.files and "max" in winner.files[0].code
            out.append(LayerResult("verify", "rank_good_candidate", "best_of_n", ok, 1.0 if ok else 0.0,
                                   time.time() - t0, f"winner_ok={winner.ok} signal={winner.signal}"))

    return out


def _print_results(results: list[LayerResult]) -> None:
    by_layer: dict[str, list[LayerResult]] = {}
    for r in results:
        by_layer.setdefault(r.layer, []).append(r)

    for layer, rows in by_layer.items():
        solved = sum(1 for r in rows if r.ok)
        avg = sum(r.score for r in rows) / max(1, len(rows))
        lat = sum(r.latency for r in rows) / max(1, len(rows))
        print(f"\n=== {layer} ===")
        print(f"pass={solved}/{len(rows)}  avg_score={avg:.2f}  avg_latency={lat:.2f}s")
        for r in rows:
            mark = "PASS" if r.ok else "fail"
            print(f"  {r.unit:24} {r.variant:14} {mark:4} score={r.score:.2f} {r.note}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--host", default=None)
    ap.add_argument("--layers", default="context,reasoning,tools,verify",
                    help="comma-separated: context,reasoning,tools,verify")
    ap.add_argument("--tasks", default=None, help="comma-separated task names")
    ap.add_argument("--band", choices=("basic", "recoverable", "context"), default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("Layers: context, reasoning, tools, verify")
        print("\nTasks:")
        for t in TASKS:
            gold = sorted(EXPECTED_FILES.get(t.name, []))
            print(f"  {t.name:22} [{t.band:11}] gold={gold}")
        print("\nTool probes:")
        for p in TOOL_PROBES:
            print(f"  {p.name:22} allowed={sorted(p.allowed_tools)}")
        return 0

    layers = [x.strip() for x in args.layers.split(",") if x.strip()]
    tasks = _selected_tasks(args.tasks, args.band)
    if not tasks:
        print("error: no tasks selected")
        return 2

    needs_model = any(layer in {"reasoning", "tools"} for layer in layers)
    client = None
    if needs_model:
        client = Ollama(model=args.model, host=args.host or DEFAULT_HOST)
        if not client.available():
            print(f"error: Ollama not reachable. `ollama serve` and `ollama pull {args.model}`.")
            return 2
        models = client.list_models()
        if models and args.model not in models:
            print(f"error: model {args.model!r} is not pulled. Try: `ollama pull {args.model}`.")
            print("available models:", ", ".join(models[:12]))
            return 2

    results: list[LayerResult] = []
    if "context" in layers:
        results.extend(bench_context(tasks))
    if "reasoning" in layers:
        results.extend(bench_reasoning(tasks, client))
    if "tools" in layers:
        results.extend(bench_tools(client))
    if "verify" in layers:
        results.extend(bench_verify())

    _print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
