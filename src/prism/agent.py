"""End-to-end agent loop: get_context -> small model (Ollama) -> verify -> retry.

This is the real small-model story, not a stub. It proves the thesis: an on-device
model (qwen2.5-coder etc.) makes a whole-repo-aware change from a budget-sized
dossier instead of a 1M-token context window.

Protocol with the model: it replies with one or more blocks of the form

    FILE: relative/path.py
    ```python
    <full new file contents>
    ```

We compile-check every proposed file. On failure we feed the structured errors
back and retry. With apply=True the verified files are written to the repo.
"""

from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass, field

from .context import get_context
from .llm import DEFAULT_MODEL, Ollama, OllamaUnavailable, call_chat
from .pathsafe import safe_join
from .render import render_markdown
from .verify import verify

SYSTEM_PROMPT = (
    "You are a precise coding agent working on a codebase. You are given a "
    "budget-sized context dossier (not the whole repo) produced by a code graph. "
    "Trust it: the dependency edges and periphery show what your change affects. "
    "Make the smallest change that satisfies the task and keeps the code valid. "
    "Respond ONLY with changed files, each as:\n\n"
    "FILE: <relative/path.ext>\n"
    "```<language>\n<full new file contents>\n```\n\n"
    "Keep each file's original language. Do not include prose, explanations, or "
    "unchanged files."
)

# Accept any fenced language tag (```python, ```js, ```go, ...).
_FILE_BLOCK = re.compile(
    r"FILE:\s*(?P<path>[^\n]+)\n```(?:[A-Za-z0-9_+-]*)?\n(?P<code>.*?)```",
    re.DOTALL,
)


@dataclass
class ProposedFile:
    path: str
    code: str


@dataclass
class AgentResult:
    ok: bool
    task: str
    attempts: int
    files: list[ProposedFile] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    applied: bool = False
    context_tokens: int = 0
    budget: int = 0
    symbol_count: int = 0
    raw_last_reply: str = ""
    diffs: list[dict] = field(default_factory=list)  # [{path, diff}]


def _parse_files(reply: str) -> list[ProposedFile]:
    out: list[ProposedFile] = []
    for m in _FILE_BLOCK.finditer(reply):
        out.append(ProposedFile(path=m.group("path").strip(), code=m.group("code")))
    return out


def _verify_files(repo_root: str, files: list[ProposedFile]) -> dict:
    """Write proposed files to a shadow dir and verify them.

    Python files are compile-checked with the stdlib. Other languages can't be
    compiled without their toolchains, so we only sanity-check that the code is
    non-empty and balanced-ish; a real per-language linter can plug in here.
    """
    import tempfile

    errors: list[dict] = []
    py_files = [pf for pf in files if pf.path.endswith(".py")]
    other_files = [pf for pf in files if not pf.path.endswith(".py")]

    with tempfile.TemporaryDirectory() as shadow:
        written: list[str] = []
        for pf in py_files:
            dst = os.path.join(shadow, os.path.basename(pf.path))
            with open(dst, "w", encoding="utf-8") as fh:
                fh.write(pf.code)
            written.append(dst)
        if written:
            result = verify(written)
            by_index = {os.path.join(shadow, os.path.basename(pf.path)): pf.path for pf in py_files}
            for err in result["errors"]:
                err["file"] = by_index.get(err["file"], err["file"])
                errors.append(err)

    for pf in other_files:
        if not pf.code.strip():
            errors.append({"file": pf.path, "line": None, "message": "empty file", "kind": "empty"})
        elif pf.code.count("{") != pf.code.count("}"):
            errors.append({"file": pf.path, "line": None,
                           "message": "unbalanced braces", "kind": "syntax"})
    return {"ok": not errors, "errors": errors}


def _unified_diff(repo_root: str, pf: ProposedFile) -> str:
    """Unified diff of a proposed file against what's currently on disk."""
    path = safe_join(repo_root, pf.path)  # never read outside the repo
    old = ""
    if path is not None:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                old = fh.read()
        except OSError:
            old = ""
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        pf.code.splitlines(keepends=True),
        fromfile=f"a/{pf.path}",
        tofile=f"b/{pf.path}",
    )
    return "".join(diff)


def run_agent(
    repo_root: str,
    task: str,
    model: str = DEFAULT_MODEL,
    token_budget: int = 8000,
    max_attempts: int = 3,
    apply: bool = False,
    host: str | None = None,
    on_event=None,
    stream: bool = False,
    show_diff: bool = False,
    on_token=None,
) -> AgentResult:
    """Drive an Ollama model through the get_context -> verify loop.

    stream=True prints model tokens live (via on_token, or on_event if unset).
    show_diff=True computes a unified diff of each verified file vs disk.
    """
    def emit(msg: str) -> None:
        if on_event:
            on_event(msg)

    token_sink = on_token or (lambda t: on_event(t) if on_event else None)

    client = Ollama(model=model, host=host or Ollama.host)
    if not client.available():
        raise OllamaUnavailable(
            f"Ollama not reachable. Start it with `ollama serve` and "
            f"`ollama pull {model}`."
        )

    emit(f"[context] packing dossier for: {task!r}")
    payload = get_context(repo_root, task, token_budget=token_budget)
    dossier = render_markdown(payload, include_source=True)
    result = AgentResult(
        ok=False,
        task=task,
        attempts=0,
        context_tokens=payload.get("fits_in", 0),
        budget=token_budget,
        symbol_count=payload.get("symbol_count", 0),
    )
    emit(f"[context] {result.context_tokens}/{token_budget} tokens, "
         f"{result.symbol_count} symbols indexed")

    feedback = ""
    for attempt in range(1, max_attempts + 1):
        result.attempts = attempt
        user = dossier if not feedback else f"{dossier}\n\n## Previous attempt failed\n{feedback}"
        emit(f"[model] {model} attempt {attempt}/{max_attempts} ...")
        # Edit generation is not reasoning -> disable thinking for speed.
        reply = call_chat(client, SYSTEM_PROMPT, user, think=False,
                          stream=stream, on_token=token_sink)
        result.raw_last_reply = reply
        files = _parse_files(reply)
        if not files:
            feedback = "No FILE blocks found. Reply strictly in the FILE: + code-fence format."
            emit("[model] no parseable file blocks; retrying")
            continue

        check = _verify_files(repo_root, files)
        if check["ok"]:
            result.ok = True
            result.files = files
            emit(f"[verify] OK — {len(files)} file(s) verified")
            if show_diff:
                for pf in files:
                    diff = _unified_diff(repo_root, pf)
                    result.diffs.append({"path": pf.path, "diff": diff})
                    if diff:
                        emit(f"[diff] {pf.path}\n{diff}")
            if apply:
                wrote = 0
                for pf in files:
                    dst = safe_join(repo_root, pf.path)
                    if dst is None:  # untrusted model path escaping the repo
                        emit(f"[apply] REJECTED unsafe path: {pf.path}")
                        continue
                    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
                    with open(dst, "w", encoding="utf-8") as fh:
                        fh.write(pf.code)
                    wrote += 1
                result.applied = wrote > 0
                emit(f"[apply] wrote {wrote}/{len(files)} file(s) to {repo_root}")
            return result

        result.errors = check["errors"]
        feedback = "Compile errors to fix:\n" + "\n".join(
            f"- {e['file']}:{e.get('line')}: {e['message']}" for e in check["errors"]
        )
        emit(f"[verify] FAILED: {feedback}")

    return result
