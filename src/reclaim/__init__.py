"""Reclaim evaluation: does a compressed memory keep a model correctable?

A model drifts to a wrong answer; only a compressed memory of that session carries
forward. Reclaim evaluation measures whether a later correction can recover the *known*
answer, scored by exact match with no judge. The lever is what the memory kept: keep the
re-derivable conclusion and the error welds in; keep the recomputable source and it stays
fixable, at the same budget.

Quickstart::

    from reclaim import reclaim_rate, memory_note, DryRunLLM
    llm = DryRunLLM()  # free; swap for OpenRouterLLM(model=...) to test a real model
    for policy in ("lossy", "source_first"):
        rr = reclaim_rate(llm, lambda p, g: memory_note(p, g, policy), integrity=0.1)
        print(policy, rr)
"""
from .problems import PROBLEMS, PROBLEMS_LOGIC, Problem, FACTS
from .experiment import (
    SYSTEM, memory_note, reclaim_cross, score,
    run_problem, run_problem_crosssession, run_problem_distance,
)
from .llm import OpenRouterLLM, AnthropicLLM, DryRunLLM, parse_answer
from .probe import (
    classify_note, probe_policy, compare_policies, Verdict, ProbeReport,
)

__version__ = "0.3.0"

__all__ = [
    "reclaim_rate", "memory_note", "score", "Problem", "PROBLEMS", "PROBLEMS_LOGIC",
    "FACTS", "SYSTEM", "reclaim_cross", "OpenRouterLLM", "AnthropicLLM", "DryRunLLM",
    "parse_answer", "run_problem", "run_problem_crosssession", "run_problem_distance",
    "classify_note", "probe_policy", "compare_policies", "Verdict", "ProbeReport",
]


def reclaim_rate(llm, compress, problems=PROBLEMS, integrity=0.1, arm="directed"):
    """Reclaim Rate for a custom memory policy, judge-free.

    Args:
        llm: a client exposing ``.chat(messages) -> str`` (``OpenRouterLLM``,
            ``AnthropicLLM``, or the free deterministic ``DryRunLLM``).
        compress: callable ``(problem, integrity) -> str`` returning the session-2
            carry-over memory note for a drifted ``problem``. This is *your* policy; build
            the note from ``problem.wrong_premise``, ``problem.ask`` and
            ``FACTS[problem.pid]`` (the recomputable source). Compare against the built-ins
            with ``memory_note(problem, integrity, policy)`` for policy in
            ``{"lossy", "source_first", "lossy_padded", "blank"}``.
        problems: the task instances (default: the 8 arithmetic problems; pass
            ``PROBLEMS_LOGIC`` for the constraint-logic family).
        integrity: how compressed the memory is, in ``[0, 1]``; lower drops more source.
        arm: ``"directed"`` (names the error locus) or ``"generic"`` ("something is wrong").

    Returns:
        float: the fraction of problems whose true answer the correction recovers.
    """
    hits = 0
    for p in problems:
        note = compress(p, integrity)
        if hasattr(llm, "configure"):
            llm.configure(p.drift, p.correct, FACTS.get(p.pid), p.locus)
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": note},
                {"role": "user", "content": reclaim_cross(p, arm)}]
        hits += bool(score(llm.chat(msgs), p))
    return hits / len(problems)
