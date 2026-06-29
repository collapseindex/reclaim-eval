"""LLM clients: a real OpenRouter chat client, and a zero-cost DryRun fake that
simulates drift + a reclaim window so the whole pipeline can be validated for free.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# Optional markdown / quote / bracket noise a model may wrap the answer in
# ("ANSWER: **167**", "ANSWER: `Jo`"). Skipped between the colon and the value so a genuine
# commit that happens to be emphasised is not misread as an abstention.
_WRAP = r"[*_`'\"(\[]*\s*"


def parse_answer(text: str):
    """The number on the last 'ANSWER:' line, required to sit immediately after the marker
    (modulo markdown/$/whitespace). No adjacent number means the model did NOT commit a numeric
    answer -- return None rather than scrape a stray number out of refusal prose. The old loose
    fallback (grab any trailing number) did exactly that: it turned an abstention like "I can't
    recompute, though the earlier figure was 55" into a phantom confident-wrong commit, which
    inflated the blank-vs-lossy emit gap (see NOTE_parser_fix.md)."""
    if not text:
        return None
    m = re.findall(rf"ANSWER\s*:\s*{_WRAP}\$?\s*(-?\d[\d,]*\.?\d*)", text, flags=re.I)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


# Fallback filler/refusal blocklist for text problems that carry no explicit `options` set.
_NONANSWER = {
    "please", "unable", "cannot", "cant", "none", "nobody", "noone", "unknown", "unclear",
    "insufficient", "sorry", "i", "no", "not", "na", "n", "without", "need", "more",
    "there", "the", "a", "an", "sufficient", "determine", "provide", "details", "detail",
    "information", "info", "memory", "context", "unsure", "unspecified", "undetermined",
    "indeterminate", "ambiguous", "uncertain", "missing", "lacking", "given", "based",
}


def parse_answer_word(text: str, options=None):
    """The single-token answer on the last 'ANSWER:' line. The symbolic answer space is CLOSED:
    every valid answer is one of the problem's own candidate entities. When `options` is given we
    accept the parsed token only if it is one of them (returning the canonical-cased option) and
    treat anything else -- "Unable", "It", "I", any refusal/prose word -- as an abstention. That is
    a real validator against the known answer set, not a guess at what filler looks like; the
    blocklist path is only a fallback for problems with no declared options."""
    if not text:
        return None
    m = re.findall(rf"ANSWER\s*:\s*{_WRAP}([A-Za-z][A-Za-z/]*)", text, flags=re.I)
    if not m:
        return None
    w = m[-1]
    if options:
        for o in options:
            if w.lower() == str(o).lower():
                return o            # commit only to a recognised candidate
        return None                 # an unrecognised word == abstention, never a phantom commit
    return None if w.lower() in _NONANSWER else w


@dataclass
class OpenRouterLLM:
    model: str = "meta-llama/llama-3.1-8b-instruct"
    temperature: float = 0.0
    max_tokens: int = 600
    timeout: int = 60

    def __post_init__(self):
        self.key = os.environ.get("OPENROUTER_API_KEY")
        if not self.key:
            raise RuntimeError("OPENROUTER_API_KEY not set (put it in .env)")
        self.calls = 0
        self.prompt_tokens = 0       # measured from the API, for real cost reporting
        self.completion_tokens = 0   # includes reasoning tokens if the model emits them

    def chat(self, messages):
        body = {"model": self.model, "messages": messages,
                "temperature": self.temperature, "max_tokens": self.max_tokens}
        detail = "no attempt"
        for attempt in range(6):
            try:
                r = requests.post(OPENROUTER_URL, json=body, timeout=self.timeout,
                                  headers={"Authorization": f"Bearer {self.key}"})
                if r.status_code == 200:
                    data = r.json()
                    choices = data.get("choices") or []
                    content = (choices[0].get("message") or {}).get("content") if choices else None
                    if content:
                        self.calls += 1
                        usage = data.get("usage") or {}
                        self.prompt_tokens += usage.get("prompt_tokens", 0) or 0
                        self.completion_tokens += usage.get("completion_tokens", 0) or 0
                        return content
                    # 200 with no usable choices/content (provider error wrapped in 200,
                    # moderation, or an empty completion): treat as transient, retry.
                    detail = f"200 no content: {str(data.get('error') or data)[:160]}"
                elif r.status_code in (429, 500, 502, 503):
                    detail = f"http {r.status_code}"
                else:
                    raise RuntimeError(f"OpenRouter {r.status_code}: {r.text[:200]}")
            except requests.RequestException as e:
                detail = f"{type(e).__name__}: {str(e)[:120]}"
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"OpenRouter failed after retries ({detail})")


@dataclass
class AnthropicLLM:
    """Anthropic chat client with the same .chat(messages) -> str interface. Used for the
    frontier answering-model pass: hold the memory fixed, swap the model to Claude.
    Handles the API's system-message split and its strict role-alternation by merging
    consecutive same-role turns (our reclaim turn is [system, user(memory), user(reclaim)]).
    """
    model: str = "claude-sonnet-4-6"
    temperature: float = 0.0
    max_tokens: int = 600

    def __post_init__(self):
        self.key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.key:
            raise RuntimeError("ANTHROPIC_API_KEY not set (put it in .env)")
        import anthropic
        self._client = anthropic.Anthropic(api_key=self.key)
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def chat(self, messages):
        import anthropic
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        conv = []
        for m in messages:
            if m["role"] == "system":
                continue
            if conv and conv[-1]["role"] == m["role"]:
                conv[-1]["content"] += "\n" + m["content"]   # merge same-role runs
            else:
                conv.append({"role": m["role"], "content": m["content"]})
        send_temp = True   # some models (e.g. opus-4-8) reject the temperature param
        for attempt in range(6):
            try:
                kw = dict(model=self.model, max_tokens=self.max_tokens,
                          system=system or anthropic.NOT_GIVEN, messages=conv)
                if send_temp:
                    kw["temperature"] = self.temperature
                resp = self._client.messages.create(**kw)
                self.calls += 1
                self.prompt_tokens += resp.usage.input_tokens
                self.completion_tokens += resp.usage.output_tokens
                return "".join(b.text for b in resp.content
                               if getattr(b, "type", None) == "text")
            except anthropic.BadRequestError as e:
                if send_temp and "temperature" in str(e).lower():
                    send_temp = False   # retry immediately without it
                    continue
                raise
            except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
                code = getattr(e, "status_code", None)
                if code not in (None, 429, 500, 502, 503, 529):
                    raise
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("Anthropic failed after retries")


@dataclass
class OpenAILLM:
    """OpenAI direct client for gpt-5.x reasoning models: max_completion_tokens + reasoning_effort,
    NO temperature (the reasoning models reject it). The answer is in choices[0].message.content;
    the hidden reasoning is billed as completion tokens, so we accumulate usage even on an empty
    completion (a reasoning-token blowup that returns no content must still count against budget,
    and surface its finish_reason rather than look free)."""
    model: str = "gpt-5.4"
    reasoning_effort: str = "low"
    max_tokens: int = 2000
    timeout: int = 180

    def __post_init__(self):
        self.key = os.environ.get("OPENAI_API_KEY")
        if not self.key:
            raise RuntimeError("OPENAI_API_KEY not set (.env)")
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0   # includes reasoning tokens

    def chat(self, messages):
        body = {"model": self.model, "messages": messages,
                "max_completion_tokens": max(self.max_tokens, 2000),
                "reasoning_effort": self.reasoning_effort}
        detail = "no attempt"
        for attempt in range(6):
            try:
                r = requests.post("https://api.openai.com/v1/chat/completions", json=body,
                                  timeout=self.timeout,
                                  headers={"Authorization": f"Bearer {self.key}"})
                if r.status_code == 200:
                    data = r.json()
                    ch = data.get("choices") or []
                    content = (ch[0].get("message") or {}).get("content") if ch else None
                    usage = data.get("usage") or {}
                    self.prompt_tokens += usage.get("prompt_tokens", 0) or 0
                    self.completion_tokens += usage.get("completion_tokens", 0) or 0
                    if content:
                        self.calls += 1
                        return content
                    fr = ch[0].get("finish_reason") if ch else "?"
                    detail = f"200 no content (finish={fr}); raise max_completion_tokens if 'length'"
                elif r.status_code in (429, 500, 502, 503):
                    detail = f"http {r.status_code}"
                else:
                    raise RuntimeError(f"OpenAI {r.status_code}: {r.text[:200]}")
            except requests.RequestException as e:
                detail = f"{type(e).__name__}: {str(e)[:120]}"
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"OpenAI failed after retries ({detail})")


@dataclass
class XAILLM:
    """Official xAI API client (OpenAI-compatible) for grok-4.x. grok-4 is a reasoning model, so we
    run it deterministic (temperature=None -> not sent), matching how the other frontier readers are
    treated, and drop temperature on a 400 if the endpoint rejects it. Reasoning tokens count as
    completion tokens and are tracked for cost; usage is accumulated even on empty content."""
    model: str = "grok-4.3"
    temperature: object = None           # None -> deterministic, do not send temperature
    max_tokens: int = 2000
    timeout: int = 180

    def __post_init__(self):
        self.key = os.environ.get("XAI_API_KEY")
        if not self.key:
            raise RuntimeError("XAI_API_KEY not set (.env)")
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._send_temp = self.temperature is not None

    def chat(self, messages):
        body = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens}
        if self._send_temp:
            body["temperature"] = self.temperature
        detail = "no attempt"
        for attempt in range(6):
            try:
                r = requests.post("https://api.x.ai/v1/chat/completions", json=body,
                                  timeout=self.timeout,
                                  headers={"Authorization": f"Bearer {self.key}"})
                if r.status_code == 200:
                    data = r.json()
                    ch = data.get("choices") or []
                    content = (ch[0].get("message") or {}).get("content") if ch else None
                    usage = data.get("usage") or {}
                    self.prompt_tokens += usage.get("prompt_tokens", 0) or 0
                    self.completion_tokens += usage.get("completion_tokens", 0) or 0
                    if content:
                        self.calls += 1
                        return content
                    fr = ch[0].get("finish_reason") if ch else "?"
                    detail = f"200 no content (finish={fr})"
                elif r.status_code == 400 and self._send_temp and "temperature" in r.text.lower():
                    self._send_temp = False            # endpoint rejects temperature: drop and retry
                    body.pop("temperature", None)
                    continue
                elif r.status_code in (429, 500, 502, 503):
                    detail = f"http {r.status_code}"
                else:
                    raise RuntimeError(f"xAI {r.status_code}: {r.text[:200]}")
            except requests.RequestException as e:
                detail = f"{type(e).__name__}: {str(e)[:120]}"
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"xAI failed after retries ({detail})")


@dataclass
class DryRunLLM:
    """A free fake that drifts (uses the planted wrong value) and reclaims with a
    probability that FALLS with drift depth and is HIGHER for the directed arm, so a
    window appears and the harness/measurement can be validated end to end at no cost.
    """
    seed: int = 0

    def __post_init__(self):
        import numpy as np
        self._rng = __import__("numpy").random.default_rng(self.seed)
        self.calls = 0
        self._drift_val = None
        self._correct_val = None
        self._source = None
        self._locus = None

    def configure(self, drift, correct, source=None, locus=None):
        self._drift_val, self._correct_val = drift, correct
        self._source, self._locus = source, locus

    def _facts_present(self, messages) -> bool:
        """Is the recomputable source in context? Keyed on a distinctive prefix of the
        source string, which also appears verbatim in the original question (full
        transcript) and in the source_first memory note, but not in a lossy note."""
        if self._source:
            marker = self._source[:18].lower()
            return any(marker in m["content"].lower() for m in messages)
        # fallback for un-configured fakes
        return any(("facts were" in m["content"]) or (" buys " in m["content"])
                   for m in messages)

    def chat(self, messages):
        self.calls += 1
        last = messages[-1]["content"].lower()
        # reclaim turn? simulate the window: success prob decays with depth, +directed
        if "recheck" in last or "wrong" in last:
            # the broken sky: if the recomputable source is not in context, neither arm
            # can reclaim, no matter how directed (nothing left to recompute from).
            if not self._facts_present(messages):
                val = self._correct_val if self._rng.random() < 0.05 else self._drift_val
                return f"I am not sure I have enough to recompute. ANSWER: {val}"
            depth = self._depth_hint(messages)
            directed = bool(self._locus) and self._locus.lower()[:8] in last
            base = 0.95 if directed else 0.85
            decay = 0.06 if directed else 0.14            # generic forgets faster
            p = max(0.02, base - decay * depth)
            val = self._correct_val if self._rng.random() < p else self._drift_val
            return f"Rechecking... ANSWER: {val}"
        # normal/drift turn: commit to the wrong answer
        return f"Using what was given, ANSWER: {self._drift_val}"

    @staticmethod
    def _depth_hint(messages):
        # count assistant turns since the drift was planted = commitment depth
        return sum(1 for m in messages if m["role"] == "assistant")
