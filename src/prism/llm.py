"""Minimal Ollama client — stdlib only (urllib), no extra deps.

Talks to a local Ollama daemon (default http://localhost:11434).

Per the research pivot (docs/research/frontier-vs-prism.md):
  - The REASONING layer should be powered by a *distilled reasoning checkpoint*
    (DeepSeek-R1-Distill), not a vanilla base — so DEFAULT_REASONING_MODEL points
    at `deepseek-r1:7b` (Ollama's R1-Distill-Qwen-7B). Raw code-generation still
    uses a coder model (DEFAULT_MODEL).
  - R1-distill models emit `<think>...</think>` reasoning traces. Those must NOT
    reach the FILE/JSON parsers, so chat() strips them from the returned content
    by default (while streaming still shows them live for visibility).
  - Layer 3 wants constrained decoding: chat() accepts `format` (Ollama's "json"
    or a JSON-schema dict) to force structured tool-call output.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5-coder:7b"          # raw code generation / edits
DEFAULT_REASONING_MODEL = "deepseek-r1:7b"  # planning / reasoning (R1-Distill-Qwen-7B)

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove `<think>...</think>` reasoning traces from a distilled model's output."""
    if not text:
        return text
    text = _THINK.sub("", text)
    if "</think>" in text:  # unbalanced (opening lost/truncated): keep post-reasoning
        text = text.split("</think>")[-1]
    return text.replace("<think>", "").strip()


def call_chat(client, system: str, user: str, think: bool | None = None,
              temperature: float = 0.1, stream: bool = False, on_token=None,
              format: object = None, num_predict: int | None = None) -> str:
    """Backend-tolerant chat that requests thinking mode when asked.

    - Real Ollama: passes `think` (planner -> True, edits -> False), and optional
      `format` (constrained decoding) + `num_predict` (output cap) to stop small
      models rambling to the context limit.
    - Test fakes / clients without these kwargs: degrade cleanly to chat(sys,usr).
    """
    base = {"temperature": temperature}
    if think is not None:
        base["think"] = think
    if format is not None:
        base["format"] = format
    if num_predict is not None:
        base["num_predict"] = num_predict

    if stream and hasattr(client, "chat_stream"):
        for kw in (dict(base, on_token=on_token), dict(on_token=on_token), {}):
            try:
                return client.chat_stream(system, user, **kw)
            except TypeError:
                continue
    for kw in (base, {"temperature": temperature}, {}):
        try:
            return client.chat(system, user, **kw)
        except TypeError:
            continue
    return client.chat(system, user)


class OllamaUnavailable(RuntimeError):
    pass


@dataclass
class Ollama:
    host: str = DEFAULT_HOST
    model: str = DEFAULT_MODEL
    timeout: int = 600

    def _post(self, path: str, body: dict) -> dict:
        url = f"{self.host}{path}"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise OllamaUnavailable(
                f"Could not reach Ollama at {self.host} ({e}). "
                f"Is it running? Try: `ollama serve` and `ollama pull {self.model}`."
            ) from e

    def available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()
            return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return [m.get("name", "") for m in payload.get("models", [])]
        except Exception:
            return []

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
        format: object = None,
        strip_reasoning: bool = True,
        think: bool | None = None,
        num_predict: int | None = None,
    ) -> str:
        """Single-turn chat. `format` enables constrained decoding (e.g. "json"
        or a JSON-schema dict). `think` toggles a hybrid model's reasoning mode
        (True for planning, False for fast edits); ignored/retried-without if the
        model doesn't support it. `num_predict` bounds output tokens (useful to
        cap thinking on slow hardware). Reasoning traces are stripped by default."""
        options = {"temperature": temperature}
        if num_predict is not None:
            options["num_predict"] = num_predict
        body = {
            "model": self.model,
            "stream": False,
            "options": options,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if format is not None:
            body["format"] = format
        if think is not None:
            body["think"] = think
        try:
            payload = self._post("/api/chat", body)
        except OllamaUnavailable:
            if think is None:
                raise
            body.pop("think", None)  # model may not support thinking; retry plain
            payload = self._post("/api/chat", body)
        msg = payload.get("message") or {}
        content = msg.get("content", "")
        # Resilience: on slow hardware a thinking model can spend its whole token
        # budget in <think> and return EMPTY content (done_reason=length). If we
        # asked to think but got no answer, retry WITHOUT thinking so callers
        # always get a usable, parseable response instead of "".
        if think and not (content or "").strip():
            body["think"] = False
            payload = self._post("/api/chat", body)
            content = (payload.get("message") or {}).get("content", "")
        return strip_think(content) if strip_reasoning else content

    def chat_stream(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
        on_token=None,
        format: object = None,
        strip_reasoning: bool = True,
        think: bool | None = None,
    ) -> str:
        """Streaming chat. on_token(text) fires per chunk (reasoning shown live);
        the RETURNED aggregate is the ANSWER only (thinking is separated out)."""
        body = {
            "model": self.model,
            "stream": True,
            "options": {"temperature": temperature},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if format is not None:
            body["format"] = format
        if think is not None:
            body["think"] = think
        url = f"{self.host}/api/chat"

        def _run(b: dict) -> str:
            data = json.dumps(b).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            parts: list[str] = []
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw in resp:  # Ollama streams newline-delimited JSON objects
                    line = raw.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg = obj.get("message") or {}
                    # native thinking mode returns reasoning under `thinking`
                    think_tok = msg.get("thinking", "")
                    if think_tok and on_token:
                        on_token(think_tok)
                    tok = msg.get("content", "")
                    if tok:
                        parts.append(tok)
                        if on_token:
                            on_token(tok)
                    if obj.get("done"):
                        break
            return "".join(parts)

        try:
            content = _run(body)
        except urllib.error.URLError as e:
            if think is not None:  # model may not support thinking; retry plain
                body.pop("think", None)
                try:
                    content = _run(body)
                except urllib.error.URLError as e2:
                    raise OllamaUnavailable(
                        f"Could not reach Ollama at {self.host} ({e2})."
                    ) from e2
            else:
                raise OllamaUnavailable(
                    f"Could not reach Ollama at {self.host} ({e}). "
                    f"Is it running? Try: `ollama serve` and `ollama pull {self.model}`."
                ) from e
        # Resilience: if thinking ate the whole budget and produced no answer,
        # retry without thinking so callers get usable content, not "".
        if think and not (content or "").strip():
            body["think"] = False
            content = _run(body)
        # In native thinking mode `content` is already answer-only; strip is a
        # no-op safety net for models that inline <think> tags into content.
        return strip_think(content) if strip_reasoning else content
