"""Run a real local model (via Ollama) through the prism-context loop.

Prereqs:
    ollama serve
    ollama pull qwen2.5-coder:7b      # ~4.7GB, comfortable on an M2 Pro

Usage:
    python examples/ollama_agent.py examples/demo_repo \
        "add per-user rate limiting to login" --model qwen2.5-coder:7b

Add --apply to write the verified files back to the repo (off by default so it's
a safe dry run).
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, __file__.rsplit("/examples/", 1)[0] + "/src")

from prism.agent import run_agent  # noqa: E402
from prism.llm import OllamaUnavailable  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("task")
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--budget", type=int, default=8000)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    try:
        result = run_agent(
            args.repo,
            args.task,
            model=args.model,
            token_budget=args.budget,
            apply=args.apply,
            on_event=print,
        )
    except OllamaUnavailable as e:
        print(f"\nOllama not available: {e}")
        return 2

    print("\n--- summary ---")
    print(f"ok={result.ok} attempts={result.attempts} "
          f"context={result.context_tokens}/{result.budget} tokens applied={result.applied}")
    for pf in result.files:
        print(f"  {pf.path}: {len(pf.code)} chars")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
