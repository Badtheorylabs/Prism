"""Layer 3 (Tools) tests. Parser, schema validation, fuzzy repair, and the
default registry are deterministic — no model needed. A fake client drives the
ReAct loop end to end without a network."""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prism.tools import (  # noqa: E402
    Param,
    Tool,
    ToolError,
    ToolRegistry,
    build_default_registry,
    parse_tool_call,
    run_tools,
)

SAMPLE = '''
def login(user, password):
    """Authenticate a user."""
    return check_quota(user)


def check_quota(user):
    """Rate-limit a user."""
    return True
'''


def _repo(root: str) -> None:
    with open(os.path.join(root, "api.py"), "w") as fh:
        fh.write(SAMPLE)


def test_parser_handles_messy_output():
    # plain json
    assert parse_tool_call('{"tool":"search_code","args":{"query":"x"}}')["tool"] == "search_code"
    # fenced json with surrounding prose
    call = parse_tool_call('Let me search.\n```json\n{"tool":"plan","args":{"task":"y"}}\n```\nok')
    assert call["tool"] == "plan" and call["args"]["task"] == "y"
    # function-call syntax
    call = parse_tool_call('I will call search_code(query="login", top_k=3)')
    assert call["tool"] == "search_code" and call["args"]["query"] == "login"
    # trailing object wins (models often end with the call)
    call = parse_tool_call('{"tool":"a","args":{}} then really {"tool":"final","args":{"answer":"done"}}')
    assert call["tool"] == "final"
    # nothing parseable
    assert parse_tool_call("just some prose, no call") is None


def test_validation_coerces_and_repairs():
    reg = ToolRegistry()
    reg.register(Tool("echo", "echo", [Param("n", "integer", True), Param("flag", "boolean", False, False)],
                      lambda n, flag: {"n": n, "flag": flag}))
    # string -> int coercion, boolean default
    out = reg.call("echo", {"n": "42"})
    assert out == {"n": 42, "flag": False}
    # fuzzy tool-name repair
    assert reg.resolve_name("ecko") == "echo"
    # missing required arg -> ToolError
    try:
        reg.call("echo", {})
        assert False, "expected ToolError"
    except ToolError:
        pass


def test_default_registry_tools_execute():
    with tempfile.TemporaryDirectory() as root:
        _repo(root)
        reg = build_default_registry(root)
        assert "search_code" in reg.names()
        # a real Layer-1 search through the tool interface
        res = reg.call("search_code", {"query": "rate limit user"})
        assert "files" in res
        # read_file tool
        rf = reg.call("read_file", {"path": "api.py", "start": 1, "end": 3})
        assert rf["path"] == "api.py" and "def login" in rf["text"]


class _FakeClient:
    """Emits a search call, then a final answer."""

    def __init__(self):
        self.turn = 0

    def chat(self, system, user):
        self.turn += 1
        if self.turn == 1:
            return 'thinking... {"tool":"search_code","args":{"query":"login"}}'
        return '{"tool":"final","args":{"answer":"found login in api.py"}}'


def test_react_loop_runs_and_finishes():
    with tempfile.TemporaryDirectory() as root:
        _repo(root)
        trace = run_tools(root, "where is login?", _FakeClient(), max_steps=5, on_event=lambda m: None)
        assert trace.stopped == "final"
        assert trace.answer == "found login in api.py"
        assert len(trace.steps) == 1
        assert trace.steps[0].tool == "search_code" and trace.steps[0].ok


if __name__ == "__main__":
    test_parser_handles_messy_output()
    test_validation_coerces_and_repairs()
    test_default_registry_tools_execute()
    test_react_loop_runs_and_finishes()
    print("all tools tests passed")
