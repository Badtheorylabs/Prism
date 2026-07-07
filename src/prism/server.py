"""Optional MCP server exposing the public tools to any agent.

Requires the `mcp` package (`pip install mcp`). The core library does not
depend on it — this is just the transport that lets any MCP-speaking agent call
search/context/inspect/impact/verify as native tools.

Run:  python -m prism.server /path/to/repo
"""

from __future__ import annotations

import sys

from .attention import build_attention_manifest
from .context import analyze_impact, get_context, inspect_code, search_code
from .verify import verify


def serve(repo_root: str) -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        print(
            "MCP transport needs the 'mcp' package: pip install mcp\n"
            "The core library works without it — see prism.cli.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    app = FastMCP("prism-context")

    @app.tool()
    def whole_codebase_attention(task: str | None = None, token_budget: int = 8000) -> dict:
        """Return the compact handle-based map of the entire indexed repo."""
        return build_attention_manifest(repo_root, task=task, token_budget=token_budget)

    @app.tool()
    def search_codebase(query: str, top_k: int = 10) -> dict:
        """Search symbols, chunks, and files before asking for full context."""
        return search_code(repo_root, query, top_k=top_k)

    @app.tool()
    def get_code_context(task: str, token_budget: int = 8000) -> dict:
        """Return an optimally-packed codebase context dossier for a task,
        guaranteed to fit within token_budget."""
        return get_context(repo_root, task, token_budget=token_budget)

    @app.tool()
    def inspect_code_detail(
        file: str | None = None,
        symbol: str | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
        radius: int = 2,
    ) -> dict:
        """Return exact file, symbol, line-window, chunk, and quality facts."""
        return inspect_code(
            repo_root,
            file=file,
            symbol=symbol,
            start_line=start_line,
            end_line=end_line,
            radius=radius,
        )

    @app.tool()
    def analyze_change_impact(
        file: str | None = None,
        symbol: str | None = None,
        hops: int = 2,
    ) -> dict:
        """Return callers, callees, tests, quality findings, and risk score."""
        return analyze_impact(repo_root, file=file, symbol=symbol, hops=hops)

    @app.tool()
    def verify_edit(paths: list[str], run_tests: bool = False) -> dict:
        """Compile/test-check edited files; returns structured errors to feed back."""
        return verify(paths, run_tests=run_tests, cwd=repo_root)

    app.run()


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    serve(root)
