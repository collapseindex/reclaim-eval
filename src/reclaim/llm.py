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


def parse_answer(text: str):
    """Pull the last 'ANSWER: <number>' (or a trailing number) out of a reply."""
    if not text:
        return None
    m = re.findall(r"ANSWER:\s*\$?(-?\d[\d,]*\.?\d*)", text, flags=re.I)
    if not m:
        m = re.findall(r"\$?(-?\d[\d,]*\.\d+|-?\d[\d,]*)", text)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


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

    def chat(self, messages):
        body = {"model": self.model, "messages": messages,
                "temperature": self.temperature, "max_tokens": self.max_tokens}
        for attempt in range(4):
            try:
                r = requests.post(OPENROUTER_URL, json=body, timeout=self.timeout,
                                  headers={"Authorization": f"Bearer {self.key}"})
                if r.status_code == 200:
                    self.calls += 1
                    return r.json()["choices"][0]["message"]["content"]
                if r.status_code in (429, 500, 502, 503):
                    time.sleep(2 * (attempt + 1))
                    continue
                raise RuntimeError(f"OpenRouter {r.status_code}: {r.text[:200]}")
            except requests.RequestException:
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("OpenRouter failed after retries")


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

    def configure(self, drift, correct):
        self._drift_val, self._correct_val = drift, correct

    def chat(self, messages):
        self.calls += 1
        last = messages[-1]["content"].lower()
        # reclaim turn? simulate the window: success prob decays with depth, +directed
        if "recheck" in last or "wrong" in last:
            # the broken sky: if the recomputable source is not in context (the full
            # transcript's question uses "buys"; the rich memory note says "items
            # were"), neither arm can reclaim, no matter how directed.
            facts_present = any(("items were" in m["content"]) or (" buys " in m["content"])
                                for m in messages)
            if not facts_present:
                val = self._correct_val if self._rng.random() < 0.05 else self._drift_val
                return f"I am not sure I have enough to recompute. ANSWER: {val}"
            depth = self._depth_hint(messages)
            directed = "subtotal" in last or "tickets" in last  # locus named -> directed
            base = 0.95 if directed else 0.85
            decay = 0.06 if directed else 0.14            # generic forgets faster
            p = max(0.02, base - decay * depth)
            val = self._correct_val if self._rng.random() < p else self._drift_val
            return f"Rechecking... ANSWER: {val}"
        # normal/drift turn: commit to the wrong total
        return f"Using the figures given, ANSWER: {self._drift_val}"

    @staticmethod
    def _depth_hint(messages):
        # count assistant turns since the drift was planted = commitment depth
        return sum(1 for m in messages if m["role"] == "assistant")
