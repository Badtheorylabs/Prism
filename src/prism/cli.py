"""Command-line interface for testing and scripting.

    prism index <repo>
    prism attention ["<task>"] [--repo .] [--budget 8000]
    prism stats [--repo .] [--reindex] [--format json|markdown]
    prism search "<query>" [--repo .] [--top-k 10]
    prism impact [--repo .] (--symbol qualname | --file path) [--hops 2]
    prism context "<task>" [--repo .] [--budget 8000] [--reindex]
                      [--format json|markdown] [--no-source]
    prism inspect [--repo .] (--symbol qualname | --file path [--start N --end M])
    prism verify <file.py> [file2.py ...] [--tests] [--target tests/]
    prism demo [--budget 1800] [--format markdown]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .attention import build_attention_manifest
from .context import (
    analyze_impact,
    build_repo_index,
    get_context,
    inspect_code,
    repo_stats,
    search_code,
)
from .render import render_markdown, render_stats
from .verify import verify


def _cmd_agent(args: argparse.Namespace) -> int:
    from .agent import run_agent
    from .llm import OllamaUnavailable

    def on_token(t: str) -> None:
        sys.stdout.write(t)
        sys.stdout.flush()

    try:
        result = run_agent(
            args.repo,
            args.task,
            model=args.model,
            token_budget=args.budget,
            max_attempts=args.max_attempts,
            apply=args.apply,
            host=args.host,
            stream=args.stream,
            show_diff=args.diff,
            on_event=lambda m: print(m),
            on_token=on_token if args.stream else None,
        )
    except OllamaUnavailable as e:
        print(f"error: {e}")
        return 2
    print(
        f"\nresult: {'OK' if result.ok else 'FAILED'} "
        f"after {result.attempts} attempt(s); "
        f"{result.context_tokens}/{result.budget} context tokens; "
        f"applied={result.applied}"
    )
    if result.ok and not result.applied:
        for pf in result.files:
            print(f"  proposed change: {pf.path} ({len(pf.code)} chars) — re-run with --apply to write")
    return 0 if result.ok else 1


def _cmd_data(args: argparse.Namespace) -> int:
    """Domain pack demo: a NON-coding task (data reconciliation) through the same harness."""
    from .domains import DEMO_ROWS, run_data_reconciliation
    from .llm import DEFAULT_HOST, Ollama
    from .trace import Tracer

    client = Ollama(model=args.model, host=args.host or DEFAULT_HOST)
    if not client.available():
        print(f"error: Ollama not reachable. `ollama serve` and `ollama pull {args.model}`.")
        return 2

    tracer = Tracer(root=args.repo, task="sum revenue by region", label="domain:data")
    print("Task: sum revenue by region (graded by reconciliation, not tests)")
    res = run_data_reconciliation(client, DEMO_ROWS, "region", "revenue",
                                  max_attempts=args.max_attempts, tracer=tracer,
                                  on_event=lambda m: print(m))
    tracer.close(resolved=res.resolved)
    print(f"\nresolved={res.resolved} in {res.attempts} attempt(s)")
    print(f"answer: {res.artifact}")
    if not res.resolved:
        print(f"remaining discrepancies: {[e['message'] for e in res.result.errors]}")
    print(f"trace: {tracer.run_dir}")
    return 0 if res.resolved else 1


def _cmd_memory(args: argparse.Namespace) -> int:
    import json

    from .memory import Memory

    mem = Memory(args.repo)
    if args.recall:
        print(mem.render_hint(args.recall) or "(no relevant past runs yet)")
        print("\n--- structured ---")
        print(json.dumps(mem.recall(args.recall), indent=2))
        return 0
    stats = mem.stats()
    print("=== Prism memory (from persisted runs) ===")
    print(json.dumps(stats, indent=2))
    cc = mem.co_changes()
    if cc:
        print("\nfiles that change together (successful runs):")
        for c in cc:
            print(f"  {c['count']}x  {c['files']}")
    return 0


def _cmd_trace(args: argparse.Namespace) -> int:
    import os

    from .trace import load_run

    runs_dir = os.path.join(args.repo, ".prism", "runs")
    if not os.path.isdir(runs_dir):
        print(f"no runs yet under {runs_dir} — run `prism harden ...` first")
        return 1
    runs = sorted(os.listdir(runs_dir))
    if args.list:
        print(f"runs under {runs_dir}:")
        for r in runs:
            data = load_run(os.path.join(runs_dir, r))
            s = data["summary"]
            print(f"  {r}  resolved={s.get('resolved')}  won_by={s.get('won_by')}  "
                  f"calls={s.get('model_calls')}  {s.get('label', '')}")
        return 0
    run = args.run or (runs[-1] if runs else None)
    if not run:
        print("no runs found")
        return 1
    data = load_run(os.path.join(runs_dir, run))
    s = data["summary"]
    print(f"=== trace {run} ===")
    print(f"task: {s.get('task')}  label: {s.get('label')}")
    print(f"resolved={s.get('resolved')}  won_by={s.get('won_by')}  attempts={s.get('attempts')}  "
          f"model_calls={s.get('model_calls')}  model_time={s.get('model_latency_s')}s")
    print("timeline:")
    for e in data["events"]:
        d = e.get("data", {})
        detail = {
            "context": lambda: f"fits {d.get('fits_in')}/{d.get('budget')} tok, edit={d.get('files_to_edit')}",
            "model_call": lambda: f"role={d.get('role')} think={d.get('think')} {d.get('latency')}s {d.get('output_chars')}ch",
            "check": lambda: f"{d.get('check_kind')} ok={d.get('ok')} signal={d.get('signal')} tests_ran={d.get('tests_ran')}",
            "repair": lambda: f"attempt {d.get('attempt')}",
            "outcome": lambda: f"resolved={d.get('resolved')} won_by={d.get('won_by')}",
        }.get(e["kind"], lambda: "")
        print(f"  +{e['t']:5.2f}s  {e['kind']:12} {detail()}")
    return 0


def _cmd_explore(args: argparse.Namespace) -> int:
    from .context import build_repo_index, index_path_for
    from .explorer import discovered_files, explore, rank_lines
    from .graph import CodeGraph
    import os

    db = index_path_for(args.repo)
    if args.reindex or not os.path.exists(db):
        build_repo_index(args.repo, db)
    graph = CodeGraph(db)
    discoveries = explore(graph, args.task, rounds=args.rounds)
    print(f"Agentic exploration for: {args.task}")
    print(f"anchors + {args.rounds}-hop expansion → {len(discoveries)} discoveries\n")
    print("Ranked files (retrieval anchors + structural discoveries):")
    files = discovered_files(discoveries)
    for f in files:
        tag = "anchor" if f["min_hops"] == 0 else f"hop-{f['min_hops']}"
        print(f"  {f['score']:6.3f}  {f['file']}  [{tag}] {f['why']}")
    if files:
        top = files[0]["file"]
        print(f"\nLine-level ranking within {top}:")
        for w in rank_lines(graph, top, args.task, top_k=3):
            print(f"  L{w['focus_line']} (score {w['score']}): {w['text'].splitlines()[0][:70] if w['text'] else ''}")
    graph.close()
    return 0


def _cmd_harden(args: argparse.Namespace) -> int:
    from .llm import DEFAULT_HOST, Ollama
    from .ttc import adaptive_solve
    from .verifier import best_of_n, verify_and_repair

    client = Ollama(model=args.model, host=args.host or DEFAULT_HOST)
    if not client.available():
        print(f"error: Ollama not reachable. Start `ollama serve` and `ollama pull {args.model}`.")
        return 2

    # A SEPARATE verifier model may critique — the generator never judges itself.
    verifier_client = None
    if args.verifier_model:
        verifier_client = Ollama(model=args.verifier_model, host=args.host or DEFAULT_HOST)
    adversarial = verifier_client is not None
    run_tests = not args.no_tests

    from .trace import Tracer

    mode = "ttc" if args.ttc else ("best_of_n" if args.best_of > 1 else "repair")
    tracer = Tracer(root=args.repo, task=args.task, label=f"harden:{mode}")

    if args.ttc:
        res = adaptive_solve(
            args.repo, args.task, client, max_samples=args.max_samples,
            token_budget=args.budget, run_tests=run_tests, on_event=lambda m: print(m),
            tracer=tracer,
        )
        winner = res.verdict
        print(f"\nadaptive TTC: {res.stopped} in {res.attempts} sample(s); "
              f"exec_ok={winner.ok} signal={winner.signal}")
    elif args.best_of > 1:
        winner, all_v = best_of_n(
            args.repo, args.task, client, n=args.best_of,
            token_budget=args.budget, adversarial=adversarial,
            verifier_client=verifier_client, run_tests=run_tests,
            on_event=lambda m: print(m), tracer=tracer,
        )
        print(f"\nbest-of-{args.best_of} (ranked by execution): "
              f"signal={winner.signal} exec_ok={winner.ok} accepted={winner.accepted}")
    else:
        winner = verify_and_repair(
            args.repo, args.task, client, max_attempts=args.max_attempts,
            token_budget=args.budget, adversarial=adversarial,
            verifier_client=verifier_client, run_tests=run_tests,
            on_event=lambda m: print(m), tracer=tracer,
        )
        print(f"\nrepair loop (execution-grounded): "
              f"signal={winner.signal} exec_ok={winner.ok} accepted={winner.accepted}")

    summary = tracer.close(resolved=winner.accepted)
    print(f"  trace: {tracer.run_dir}  ({summary['model_calls']} calls, "
          f"{summary['checks']} checks, {summary['model_latency_s']}s model time)")

    if winner.exec_report:
        er = winner.exec_report
        print(f"  execution: compiled={er.compiled} tests_ran={er.tests_ran} "
              f"tests_passed={er.tests_passed} types_ok={er.types_ok}")
    for f in winner.files:
        print(f"  file: {f.path} ({len(f.code)} chars)")
    if winner.critique and winner.critique.issues:
        print("  verifier-model hints (advisory only):")
        for issue in winner.critique.issues:
            print(f"    - {issue}")

    if args.apply and winner.accepted:
        import os

        from .pathsafe import safe_join

        wrote = 0
        for f in winner.files:
            dst = safe_join(args.repo, f.path)
            if dst is None:
                print(f"[apply] REJECTED unsafe path: {f.path}")
                continue
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            with open(dst, "w", encoding="utf-8") as fh:
                fh.write(f.code)
            wrote += 1
        print(f"[apply] wrote {wrote}/{len(winner.files)} verified file(s)")
    elif args.apply:
        print("[apply] skipped — winner was not accepted")
    return 0 if winner.accepted else 1


def _cmd_tools(args: argparse.Namespace) -> int:
    from .tools import build_default_registry, run_tools

    if args.list:
        reg = build_default_registry(args.repo)
        print("Prism tools (Layer 3):")
        print(reg.specs())
        return 0
    if not args.task:
        print("error: provide a task, or use --list to see tools")
        return 2
    if not args.model:
        print("error: tool loop needs --model (an Ollama model)")
        return 2

    from .llm import DEFAULT_HOST, Ollama

    client = Ollama(model=args.model, host=args.host or DEFAULT_HOST)
    if not client.available():
        print(f"error: Ollama not reachable. Start `ollama serve` and `ollama pull {args.model}`.")
        return 2

    trace = run_tools(
        args.repo, args.task, client,
        max_steps=args.max_steps, token_budget=args.budget,
        on_event=lambda m: print(m),
    )
    print(f"\nstopped: {trace.stopped}  ({len(trace.steps)} tool calls)")
    for i, s in enumerate(trace.steps, start=1):
        print(f"  {i}. {s.tool} {'ok' if s.ok else 'ERROR'}")
    if trace.answer:
        print(f"\nanswer:\n{trace.answer}")
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    from .planner import make_plan, run_plan

    client = None
    if args.model:
        from .llm import DEFAULT_HOST, Ollama

        client = Ollama(model=args.model, host=args.host or DEFAULT_HOST)
        if not client.available():
            print(
                f"error: Ollama not reachable. Start `ollama serve` and "
                f"`ollama pull {args.model}`, or omit --model for a graph-derived plan."
            )
            return 2

    def on_token(t: str) -> None:
        sys.stdout.write(t)
        sys.stdout.flush()

    plan = make_plan(args.repo, args.task, token_budget=args.budget, client=client)
    print(f"\nPlan for: {plan.task}")
    print(f"source: {plan.source}  ({len(plan.steps)} steps)")
    for note in plan.notes:
        print(f"  note: {note}")
    for s in plan.steps:
        tgt = f"  [{s.target}]" if s.target else ""
        print(f"  {s.index}. {s.title}{tgt}")
        print(f"       {s.detail}")

    if args.run and client is not None:
        print("\n--- executing plan ---")
        plan = run_plan(
            plan, args.repo, client, token_budget=args.budget,
            verify=not args.no_verify, stream=args.stream,
            on_event=lambda m: print(m),
            on_token=on_token if args.stream else None,
        )
        print("\n--- step results ---")
        for s in plan.steps:
            files = ",".join(f["path"] for f in s.proposed_files) or "-"
            print(f"  {s.index}. {s.status:9} {s.title}  files={files} ctx={s.context_tokens}tok")
    elif args.run:
        print("\n(--run needs --model; graph plan has nothing to execute)")
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    db = build_repo_index(args.repo)
    from .graph import CodeGraph

    g = CodeGraph(db)
    print(f"indexed {g.count()} symbols -> {db}")
    g.close()
    return 0


def _cmd_attention(args: argparse.Namespace) -> int:
    payload = build_attention_manifest(
        args.repo,
        task=args.task,
        token_budget=args.budget,
        reindex=args.reindex,
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("ok") else 1


def _cmd_context(args: argparse.Namespace) -> int:
    payload = get_context(
        args.repo, args.task, token_budget=args.budget, reindex=args.reindex
    )
    if args.format == "markdown":
        print(render_markdown(payload, include_source=not args.no_source), end="")
    else:
        indent = 2 if args.pretty else None
        print(json.dumps(payload, indent=indent))
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    stats = repo_stats(args.repo, reindex=args.reindex)
    if args.format == "markdown":
        print(render_stats(stats), end="")
    else:
        print(json.dumps(stats, indent=2))
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    payload = search_code(
        args.repo,
        args.query,
        top_k=args.top_k,
        reindex=args.reindex,
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("ok") else 1


def _cmd_impact(args: argparse.Namespace) -> int:
    payload = analyze_impact(
        args.repo,
        file=args.file,
        symbol=args.symbol,
        hops=args.hops,
        reindex=args.reindex,
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("ok") else 1


def _cmd_inspect(args: argparse.Namespace) -> int:
    payload = inspect_code(
        args.repo,
        file=args.file,
        symbol=args.symbol,
        start_line=args.start,
        end_line=args.end,
        radius=args.radius,
        reindex=args.reindex,
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("ok") else 1


def _cmd_verify(args: argparse.Namespace) -> int:
    result = verify(args.files, run_tests=args.tests, test_target=args.target)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


def _cmd_demo(args: argparse.Namespace) -> int:
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    repo = os.path.join(here, "examples", "demo_repo")
    task = "add per-user rate limiting to login without breaking password checks"
    payload = get_context(repo, task, token_budget=args.budget, reindex=True)
    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(render_markdown(payload, include_source=not args.no_source), end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="prism", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="index a repo into a code graph")
    pi.add_argument("repo", nargs="?", default=".")
    pi.set_defaults(func=_cmd_index)

    pa = sub.add_parser("attention", help="build virtual whole-codebase attention map")
    pa.add_argument("task", nargs="?", default=None)
    pa.add_argument("--repo", default=".")
    pa.add_argument("--budget", type=int, default=8000)
    pa.add_argument("--reindex", action="store_true")
    pa.set_defaults(func=_cmd_attention)

    ps = sub.add_parser("stats", help="print a compact architecture map")
    ps.add_argument("--repo", default=".")
    ps.add_argument("--reindex", action="store_true")
    ps.add_argument("--format", choices=("json", "markdown"), default="markdown")
    ps.set_defaults(func=_cmd_stats)

    pse = sub.add_parser("search", help="search symbols, chunks, and files")
    pse.add_argument("query")
    pse.add_argument("--repo", default=".")
    pse.add_argument("--top-k", type=int, default=10)
    pse.add_argument("--reindex", action="store_true")
    pse.set_defaults(func=_cmd_search)

    pim = sub.add_parser("impact", help="analyze blast radius for a file or symbol")
    pim.add_argument("--repo", default=".")
    pim.add_argument("--file", default=None)
    pim.add_argument("--symbol", default=None)
    pim.add_argument("--hops", type=int, default=2)
    pim.add_argument("--reindex", action="store_true")
    pim.set_defaults(func=_cmd_impact)

    pin = sub.add_parser("inspect", help="inspect exact file/symbol/line facts")
    pin.add_argument("--repo", default=".")
    pin.add_argument("--file", default=None)
    pin.add_argument("--symbol", default=None)
    pin.add_argument("--start", type=int, default=None)
    pin.add_argument("--end", type=int, default=None)
    pin.add_argument("--radius", type=int, default=2)
    pin.add_argument("--reindex", action="store_true")
    pin.set_defaults(func=_cmd_inspect)

    pc = sub.add_parser("context", help="get a budget-packed context dossier")
    pc.add_argument("task")
    pc.add_argument("--repo", default=".")
    pc.add_argument("--budget", type=int, default=8000)
    pc.add_argument("--reindex", action="store_true")
    pc.add_argument("--format", choices=("json", "markdown"), default="json")
    pc.add_argument("--no-source", action="store_true")
    pc.add_argument("--pretty", action="store_true")
    pc.set_defaults(func=_cmd_context)

    pv = sub.add_parser("verify", help="compile/test-check edited files")
    pv.add_argument("files", nargs="+")
    pv.add_argument("--tests", action="store_true")
    pv.add_argument("--target", default=None)
    pv.set_defaults(func=_cmd_verify)

    pd = sub.add_parser("demo", help="run the instant demo on a tiny Python app")
    pd.add_argument("--budget", type=int, default=1800)
    pd.add_argument("--format", choices=("json", "markdown"), default="markdown")
    pd.add_argument("--no-source", action="store_true")
    pd.set_defaults(func=_cmd_demo)

    pag = sub.add_parser("agent", help="drive a local Ollama model through the context+verify loop")
    pag.add_argument("task")
    pag.add_argument("--repo", default=".")
    pag.add_argument("--model", default="qwen2.5-coder:7b")
    pag.add_argument("--budget", type=int, default=8000)
    pag.add_argument("--max-attempts", type=int, default=3)
    pag.add_argument("--host", default=None)
    pag.add_argument("--apply", action="store_true", help="write verified files to the repo")
    pag.add_argument("--stream", action="store_true", help="stream model tokens live")
    pag.add_argument("--diff", action="store_true", help="show a unified diff before applying")
    pag.set_defaults(func=_cmd_agent)

    ppl = sub.add_parser("plan", help="Layer 2: decompose a task into bounded steps")
    ppl.add_argument("task")
    ppl.add_argument("--repo", default=".")
    ppl.add_argument("--model", default=None,
                     help="reasoning model; omit for a graph-derived plan. "
                          "Tip: a distilled reasoning checkpoint (deepseek-r1:7b) beats a vanilla base here")
    ppl.add_argument("--budget", type=int, default=8000)
    ppl.add_argument("--host", default=None)
    ppl.add_argument("--run", action="store_true", help="execute the plan step by step (needs --model)")
    ppl.add_argument("--stream", action="store_true")
    ppl.add_argument("--no-verify", action="store_true")
    ppl.set_defaults(func=_cmd_plan)

    pto = sub.add_parser("tools", help="Layer 3: drive a model through a hardened tool loop")
    pto.add_argument("task", nargs="?", default=None)
    pto.add_argument("--repo", default=".")
    pto.add_argument("--model", default=None, help="Ollama model")
    pto.add_argument("--budget", type=int, default=4000)
    pto.add_argument("--max-steps", type=int, default=8)
    pto.add_argument("--host", default=None)
    pto.add_argument("--list", action="store_true", help="list available tools and exit")
    pto.set_defaults(func=_cmd_tools)

    ph = sub.add_parser("harden", help="Layer 4: execution-grounded verify + repair / best-of-N / adaptive TTC")
    ph.add_argument("task")
    ph.add_argument("--repo", default=".")
    ph.add_argument("--model", required=True, help="generator Ollama model")
    ph.add_argument("--budget", type=int, default=6000)
    ph.add_argument("--best-of", type=int, default=1, help="sample N candidates, rank by execution")
    ph.add_argument("--ttc", action="store_true", help="adaptive test-time compute (escalate on difficulty)")
    ph.add_argument("--max-samples", type=int, default=6, help="max samples for --ttc")
    ph.add_argument("--max-attempts", type=int, default=3, help="repair-loop attempts (default mode)")
    ph.add_argument("--verifier-model", default=None,
                    help="SEPARATE model for advisory critique (generator never judges itself)")
    ph.add_argument("--no-tests", action="store_true", help="skip running the test suite in the overlay")
    ph.add_argument("--host", default=None)
    ph.add_argument("--apply", action="store_true", help="write the accepted files to the repo")
    ph.set_defaults(func=_cmd_harden)

    pex = sub.add_parser("explore", help="Layer 1: agentic graph exploration + line-level ranking (no model)")
    pex.add_argument("task")
    pex.add_argument("--repo", default=".")
    pex.add_argument("--rounds", type=int, default=2, help="graph-expansion hops from retrieval anchors")
    pex.add_argument("--reindex", action="store_true")
    pex.set_defaults(func=_cmd_explore)

    ptr = sub.add_parser("trace", help="inspect Eval/Trace runs (the runtime spine)")
    ptr.add_argument("--repo", default=".")
    ptr.add_argument("--run", default=None, help="run id (default: most recent)")
    ptr.add_argument("--list", action="store_true", help="list all runs")
    ptr.set_defaults(func=_cmd_trace)

    pmem = sub.add_parser("memory", help="runtime/skill memory learned from past runs")
    pmem.add_argument("--repo", default=".")
    pmem.add_argument("--recall", default=None, help="get a memory hint for a task")
    pmem.set_defaults(func=_cmd_memory)

    pda = sub.add_parser("data", help="domain pack demo: non-coding task via the same harness")
    pda.add_argument("--repo", default=".")
    pda.add_argument("--model", required=True, help="Ollama model")
    pda.add_argument("--max-attempts", type=int, default=3)
    pda.add_argument("--host", default=None)
    pda.set_defaults(func=_cmd_data)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
