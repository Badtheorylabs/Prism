"""prism-context: a token-budget-aware code context server for small models & agents.

Public surface:
    build_attention_manifest(repo_db, task) -> dict     # virtual whole-repo attention
    search_code(repo_db, query) -> dict                 # discover symbols/chunks/files
    get_context(repo_db, task, token_budget) -> dict   # the crown-jewel tool
    inspect_code(repo_db, file/symbol/lines) -> dict    # fine-grained substrate
    analyze_impact(repo_db, file/symbol) -> dict        # pre-edit blast radius
    verify(paths) -> dict                               # the verify hook
"""

from .attention import build_attention_manifest
from .context import analyze_impact, get_context, inspect_code, search_code
from .verify import verify

__all__ = [
    "build_attention_manifest",
    "search_code",
    "get_context",
    "inspect_code",
    "analyze_impact",
    "verify",
]
__version__ = "0.1.0"
