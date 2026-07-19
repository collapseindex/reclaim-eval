"""Provider factory for the cross-model frontier sweep. Reuses the paper's AnthropicLLM (opus) and
OpenAILLM (gpt-5.4 reasoning) unchanged; adds tiny OpenAI-compatible clients for xAI (grok) and
Google (gemini), both of which expose the standard /chat/completions schema.

Deterministic models (opus rejects temperature; gpt-5.4 reasoning takes none) are run at 1 seed,
because additional seeds would be identical -- sampling models (grok, gemini) get 3 seeds at temp 0.7.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import requests

from reclaim.llm import AnthropicLLM, OpenAILLM  # reused unchanged


@dataclass
class OpenAICompatLLM:
    """Minimal OpenAI-schema chat client (requests-based) for any /chat/completions endpoint."""
    base_url: str = ""
    key_env: str = ""
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 600
    timeout: int = 90

    def __post_init__(self):
        self.key = os.environ.get(self.key_env)
        if not self.key:
            raise RuntimeError(f"{self.key_env} not set (.env)")
        self.calls = self.prompt_tokens = self.completion_tokens = 0

    def chat(self, messages, response_format=None):
        body = {"model": self.model, "messages": messages,
                "temperature": self.temperature, "max_tokens": self.max_tokens}
        if response_format is not None:
            body["response_format"] = response_format
        detail = "no attempt"
        for attempt in range(6):
            try:
                r = requests.post(self.base_url, json=body, timeout=self.timeout,
                                  headers={"Authorization": f"Bearer {self.key}"})
                if r.status_code == 200:
                    data = r.json()
                    ch = data.get("choices") or []
                    content = (ch[0].get("message") or {}).get("content") if ch else None
                    if content:
                        self.calls += 1
                        usage = data.get("usage") or {}
                        self.prompt_tokens += usage.get("prompt_tokens", 0) or 0
                        self.completion_tokens += usage.get("completion_tokens", 0) or 0
                        return content
                    detail = f"200 no content: {str(data.get('error') or data)[:160]}"
                elif r.status_code in (429, 500, 502, 503):
                    detail = f"http {r.status_code}"
                else:
                    raise RuntimeError(f"{self.model} {r.status_code}: {r.text[:200]}")
            except requests.RequestException as e:
                detail = f"{type(e).__name__}: {str(e)[:120]}"
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"{self.model} failed after retries ({detail})")


# provider -> (factory, default_seeds, tag). Model ids are overridable via the MODEL env var.
_XAI = "https://api.x.ai/v1/chat/completions"
_GEMINI = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


def make_client(provider: str, model: str | None = None, temperature: float = 0.7):
    """Return (client, seeds, tag). seeds=1 for deterministic models."""
    p = provider.lower()
    if p == "grok":
        return OpenAICompatLLM(_XAI, "XAI_API_KEY", model or "grok-4.3", temperature), 3, "grok43"
    if p == "gemini":
        return OpenAICompatLLM(_GEMINI, "GOOGLE_API_KEY", model or "gemini-3.5-flash", temperature), 3, "gemini35flash"
    if p == "opus":
        return AnthropicLLM(model=model or "claude-opus-4-8", temperature=temperature), 1, "opus48"
    if p == "gpt5":
        return OpenAILLM(model=model or "gpt-5.4", reasoning_effort="low"), 1, "gpt54"
    raise ValueError(f"unknown provider {provider}")
