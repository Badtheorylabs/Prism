"""Reference agent loop — proves the whole contract end to end.

This is the small-model story in miniature: get_context -> (model edits) ->
verify -> feed errors back -> retry. The "model" here is a stub so the example
runs with zero dependencies and no network; swap `fake_small_model` for a real
local model call (Qwen-Coder, etc.) and the loop is unchanged.

Run:  python examples/reference_agent.py <repo> "add rate limiting to the API"
"""

from __future__ import annotations

import sys

from prism.context import get_context
from prism.verify import verify_snippet


def fake_small_model(task: str, context: dict) -> str:
    """Stand-in for a real small model. Returns a trivial, valid snippet and
    demonstrates that it *received a budget-sized dossier* rather than the whole
    repo."""
    edit_files = [f["path"] for f in context["files_to_edit"]]
    periphery = [p["qualname"] for p in context["periphery"][:5]]
    return (
        f"# task: {task}\n"
        f"# model saw {context['fits_in']}/{context['token_budget']} tokens "
        f"of a {context['symbol_count']}-symbol repo\n"
        f"# would edit: {edit_files}\n"
        f"# aware of periphery: {periphery}\n"
        "def _placeholder_change():\n"
        "    return True\n"
    )


def run(repo: str, task: str, budget: int = 6000) -> int:
    print(f"[agent] task: {task!r}  budget: {budget} tokens\n")

    context = get_context(repo, task, token_budget=budget)
    print(f"[context] repo has {context['symbol_count']} symbols")
    print(f"[context] packed {context['fits_in']} tokens (<= {budget})")
    print(f"[context] files_to_edit: {[f['path'] for f in context['files_to_edit']]}")
    print(f"[context] periphery items: {len(context['periphery'])}")
    print(f"[context] dependency_edges: {len(context['dependency_edges'])}")
    print(f"[context] relevant_tests: {len(context['relevant_tests'])}\n")

    for attempt in range(1, 4):
        edit = fake_small_model(task, context)
        result = verify_snippet(edit)
        status = "OK" if result["ok"] else f"FAILED: {result['errors']}"
        print(f"[verify] attempt {attempt}: {status}")
        if result["ok"]:
            print("\n[agent] done — edit compiles clean.")
            return 0
    print("\n[agent] gave up after 3 attempts.")
    return 1


if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    task = sys.argv[2] if len(sys.argv) > 2 else "improve the code"
    raise SystemExit(run(repo, task))
