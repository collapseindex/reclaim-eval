"""Induce drift, let it commit over turns, then attempt reclaim at increasing depth
with a generic vs a directed (encoded) correction. Measure the window.
"""
from __future__ import annotations

from .problems import Problem, FOLLOWUPS
from .llm import parse_answer

SYSTEM = ("You are a careful assistant. Do the arithmetic step by step. End every "
          "reply with a line 'ANSWER: <number>'.")

DEPTHS = (1, 2, 4, 8)


def _plant(problem: Problem):
    """Opening user turn that injects the wrong premise to induce drift."""
    return (f"{problem.question}\nNote: {problem.wrong_premise}. Use that figure and "
            f"give the total before tax.")


def reclaim_message(problem: Problem, arm: str) -> str:
    base = ("Hold on. I think there is a mistake above. Please recheck and give the "
            "corrected total before tax as 'ANSWER: <number>'.")
    if arm == "generic":
        return base
    if arm == "directed":
        return (f"Hold on. I think {problem.locus} is wrong. Please recheck that and "
                f"give the corrected total before tax as 'ANSWER: <number>'.")
    raise ValueError(arm)


def build_trajectory(llm, problem: Problem):
    """Plant drift, then commit over FOLLOWUPS, capturing the message state at each
    checkpoint depth. Returns {depth: messages}."""
    if hasattr(llm, "configure"):
        llm.configure(problem.drift, problem.correct)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": _plant(problem)}]
    messages.append({"role": "assistant", "content": llm.chat(messages)})
    states = {}
    for i, fu in enumerate(FOLLOWUPS, start=1):
        messages = messages + [{"role": "user", "content": fu}]
        messages.append({"role": "assistant", "content": llm.chat(messages)})
        if i in DEPTHS:
            states[i] = list(messages)
    return states


def attempt_reclaim(llm, state, problem: Problem, arm: str):
    if hasattr(llm, "configure"):
        llm.configure(problem.drift, problem.correct)
    msgs = state + [{"role": "user", "content": reclaim_message(problem, arm)}]
    reply = llm.chat(msgs)
    ans = parse_answer(reply)
    ok = ans is not None and abs(ans - problem.correct) < 0.5
    return {"arm": arm, "answer": ans, "correct": ok}


def run_problem(llm, problem: Problem):
    """Full run for one problem: drift trajectory, then both arms at every depth."""
    states = build_trajectory(llm, problem)
    rows = []
    for depth in DEPTHS:
        if depth not in states:
            continue
        for arm in ("generic", "directed"):
            res = attempt_reclaim(llm, states[depth], problem, arm)
            rows.append({"pid": problem.pid, "depth": depth, **res})
    return rows
