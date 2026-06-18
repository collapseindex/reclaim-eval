"""Real-world memory adapters.

Drop a published, off-the-shelf memory system into the exact seam where the paper's
hand-built `memory_note` sits, and measure the same Reclaim Rate against the same
session-1 trajectory. This is the benchmark wedge: an off-the-shelf memory either
keeps the recomputable SOURCE (and stays reclaimable) or compresses to the CONCLUSION
(and walls), and we just measure which.

Kept separate from `experiment.py` so the LangChain dependency is optional: only
importing this module pulls it in. Install with:
    pip install langchain langchain-classic langchain-openai
"""
from __future__ import annotations

import os

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _is_claude(model: str) -> bool:
    """Route claude-* writer models to the Anthropic API (uses ANTHROPIC_API_KEY)."""
    return model.lower().startswith("claude")


def _chat(model: str, api_key: str, temperature: float = 0.0, max_tokens: int = 600):
    """A LangChain chat model for the memory writer: Anthropic for claude-*, else
    OpenRouter (the harness's default backend)."""
    if _is_claude(model):
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temperature, max_tokens=max_tokens,
                             api_key=os.environ.get("ANTHROPIC_API_KEY"))
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=OPENROUTER_BASE,
        api_key=api_key,
    )


def _pairs(transcript):
    """Yield (human, ai) turn pairs from a harness message list, skipping system."""
    pending = None
    for m in transcript:
        role = m.get("role")
        if role == "user":
            pending = m["content"]
        elif role == "assistant" and pending is not None:
            yield pending, m["content"]
            pending = None


def langchain_summary(transcript, model: str, api_key: str | None = None,
                      temperature: float = 0.0) -> str:
    """Run LangChain's genuine ConversationSummaryMemory over a session-1 transcript
    and return the running summary it would carry into the next session.

    This is the actual library default: its `_DEFAULT_SUMMARIZER_TEMPLATE`, applied
    progressively (one summarization LLM call per turn pair), exactly as a deployed
    LangChain agent using summary memory would compress a long conversation.
    """
    from langchain_classic.memory import ConversationSummaryMemory

    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key and not _is_claude(model):
        raise RuntimeError("OPENROUTER_API_KEY not set (needed for the summary model).")
    mem = ConversationSummaryMemory(llm=_chat(model, api_key, temperature))
    for human, ai in _pairs(transcript):
        mem.save_context({"input": human}, {"output": ai})
    return mem.load_memory_variables({})["history"]


def mem0_memory(transcript, model: str, api_key: str | None = None,
                temperature: float = 0.0) -> str:
    """Run mem0 over a session-1 transcript and return the memories it would carry forward.

    mem0 is a retrieval memory: an LLM extracts salient ``facts'' from the conversation
    into a vector store, and a later turn retrieves the relevant ones. We run it genuinely,
    LLM routed to the same OpenRouter model, a local FastEmbed embedder (no extra key, no
    transformers), and an in-memory store. We carry forward all extracted facts (get_all),
    the generous upper bound on what its retrieval would surface: if even the full memory
    keeps the source only intermittently, top-k retrieval can do no better. A fresh store
    per call isolates problems.
    """
    from langchain_community.embeddings import FastEmbedEmbeddings
    from mem0 import Memory

    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key and not _is_claude(model):
        raise RuntimeError("OPENROUTER_API_KEY not set (needed for mem0's extractor).")
    emb = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")  # 384-dim, local
    if _is_claude(model):
        llm_cfg = {"provider": "anthropic", "config": {
            "model": model, "api_key": os.environ.get("ANTHROPIC_API_KEY"),
            "temperature": temperature}}
    else:
        llm_cfg = {"provider": "openai", "config": {
            "model": model, "openai_base_url": OPENROUTER_BASE,
            "api_key": api_key, "temperature": temperature}}
    cfg = {
        "llm": llm_cfg,
        "embedder": {"provider": "langchain", "config": {"model": emb}},
        "vector_store": {"provider": "qdrant", "config": {"embedding_model_dims": 384}},
    }
    mem = Memory.from_config(cfg)
    conv = [{"role": m["role"], "content": m["content"]} for m in transcript
            if m.get("role") in ("user", "assistant")]
    mem.add(conv, user_id="reclaim")
    got = mem.get_all(filters={"user_id": "reclaim"})
    facts = got.get("results", got) if isinstance(got, dict) else got
    return " ".join(f["memory"] for f in facts if isinstance(f, dict) and f.get("memory"))


def vector_rag(transcript, model: str, api_key: str | None = None, k: int = 4) -> str:
    """Vanilla retrieval memory: embed the raw session-1 turns and carry the top-k most
    relevant to a correction, with NO LLM rewriting. This is the control that isolates
    extraction: where mem0 lets an LLM rewrite the conversation into ``facts'' (and can
    corrupt the source doing it), this keeps the turns verbatim, so whatever source was
    said survives exactly. Retrieval uses a fixed correction-shaped query, so it is the
    same for both correction arms.
    """
    import numpy as np
    from fastembed import TextEmbedding

    chunks = [m["content"] for m in transcript if m.get("role") in ("user", "assistant")]
    if not chunks:
        return ""
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    embs = list(embedder.embed(chunks))
    q = "recheck the earlier conclusion and recompute the correct answer from the given facts"
    qemb = list(embedder.embed([q]))[0]

    def cos(a, b):
        return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))

    sims = [cos(qemb, e) for e in embs]
    top = sorted(range(len(chunks)), key=lambda i: sims[i], reverse=True)[:k]
    top.sort()  # restore chronological order
    return " ".join(chunks[i] for i in top)


SOURCE_FIRST_PROMPT = (
    "You are compressing a conversation into a short memory note for a FUTURE session that "
    "may need to CORRECT a mistake in it. Keep every given fact, quantity, and unit needed "
    "to recompute the answer from scratch (the source / the working). Do NOT assert the "
    "final answer or any derived conclusion as established fact, since it can be recomputed "
    "from the source. Be concise.\n\nConversation:\n{conv}\n\nSource-first memory note:")


def source_first_auto(transcript, model: str, api_key: str | None = None,
                      temperature: float = 0.0) -> str:
    """The paper's fix as a deployable policy: an LLM compresses the transcript toward the
    recomputable SOURCE and away from the conclusion. Unlike the hand-built source_first
    note (which is written per problem), this runs on arbitrary input through one prompt,
    so it is a drop-in memory-write policy, the product the paper's result implies.
    """
    conv = "\n".join(f'{m["role"]}: {m["content"]}' for m in transcript
                     if m.get("role") in ("user", "assistant"))
    if _is_claude(model):
        from .llm import AnthropicLLM
        llm = AnthropicLLM(model=model, temperature=temperature)
    else:
        from .llm import OpenRouterLLM
        llm = OpenRouterLLM(model=model, temperature=temperature)
    return llm.chat([{"role": "user", "content": SOURCE_FIRST_PROMPT.format(conv=conv)}])


# Registry of real-world memory builders. Each takes (transcript, model, api_key) and
# returns the session-2 carry-over text (NO envelope; the driver adds the framing so it
# matches the hand-built notes' "(Memory of an earlier session.)" prefix). The first three
# are off-the-shelf systems on distinct paradigms (LLM summary, LLM extraction + retrieval,
# raw retrieval); source_first_auto is the paper's fix made deployable.
BUILDERS = {
    "langchain_summary": langchain_summary,   # progressive LLM summary
    "mem0": mem0_memory,                       # LLM fact-extraction + vector retrieval
    "vector_rag": vector_rag,                  # raw-turn retrieval, no LLM rewriting
    "source_first_auto": source_first_auto,    # the fix, as a drop-in write policy
}
