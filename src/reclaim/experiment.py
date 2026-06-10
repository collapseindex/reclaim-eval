"""Induce drift, let it commit over turns, then attempt reclaim at increasing depth
with a generic vs a directed (encoded) correction. Measure the window.
"""
from __future__ import annotations

from .problems import Problem, FOLLOWUPS, FACTS
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


# ── channel degradation by distance: push the planted error back behind unrelated
#    filler so the model's grip on it dilutes. Canned, so it costs no extra calls. ──
FILLER = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Name a primary color.", "Red is a primary color."),
    ("How many days are in a week?", "There are seven days in a week."),
    ("What gas do plants take in?", "Plants take in carbon dioxide."),
    ("What is the freezing point of water in Celsius?", "Water freezes at 0 degrees Celsius."),
    ("Name a planet in our solar system.", "Mars is a planet in our solar system."),
    ("What is the opposite of hot?", "The opposite of hot is cold."),
    ("How many legs does a spider have?", "A spider has eight legs."),
    ("What language is spoken in Brazil?", "Portuguese is spoken in Brazil."),
    ("What is two plus two?", "Two plus two is four."),
    ("Name an ocean.", "The Pacific is an ocean."),
    ("What sound does a cat make?", "A cat says meow."),
    ("What is the largest mammal?", "The blue whale is the largest mammal."),
    ("How many sides does a triangle have?", "A triangle has three sides."),
    ("What is the boiling point of water in Celsius?", "Water boils at 100 degrees Celsius."),
    ("Name a day of the weekend.", "Saturday is a day of the weekend."),
]

DISTANCES = (0, 4, 8, 16)


def attempt_reclaim_distant(llm, state, problem: Problem, arm: str, n_filler: int):
    if hasattr(llm, "configure"):
        llm.configure(problem.drift, problem.correct)
    msgs = list(state)
    for i in range(n_filler):
        u, a = FILLER[i % len(FILLER)]
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs += [{"role": "user", "content": reclaim_message(problem, arm)}]
    reply = llm.chat(msgs)
    ans = parse_answer(reply)
    ok = ans is not None and abs(ans - problem.correct) < 0.5
    return {"arm": arm, "answer": ans, "correct": ok}


def run_problem_distance(llm, problem: Problem, distances=DISTANCES):
    """Fix commitment at max depth, then vary the DISTANCE the planted error sits
    behind unrelated filler (the channel diluting), both arms."""
    states = build_trajectory(llm, problem)
    deep = states[max(DEPTHS)]
    rows = []
    for nf in distances:
        for arm in ("generic", "directed"):
            res = attempt_reclaim_distant(llm, deep, problem, arm, nf)
            rows.append({"pid": problem.pid, "distance": nf, **res})
    return rows


# ── cross-session: the real broken sky. Session 1 drifts; only a COMPRESSED memory
#    carries into session 2. As integrity falls, the recomputable source leaves the
#    channel before the wrong conclusion does, so past a point even a directed
#    correction has nothing to recompute from. ───────────────────────────────────
INTEGRITY = (1.0, 0.6, 0.3, 0.1)


def memory_note(problem: Problem, integrity: float, policy: str = "lossy") -> str:
    """A compressed carry-over memory at the given integrity. Two POLICIES, matched
    roughly in length:
      - lossy: keep the salient CONCLUSION, shed the source as budget tightens (the
        realistic default; conclusion survives, the recomputable source does not).
      - source_first: keep the recomputable SOURCE FACTS at every budget, shed the
        (recomputable) conclusion. The conclusion can be re-derived from the source;
        the source can never be re-derived from the conclusion.
    """
    drift = f"{problem.drift:g}"
    facts = FACTS[problem.pid]
    if policy == "source_first":
        if integrity >= 0.5:
            return (f"(Memory of an earlier session.) The items were: {facts}. A note "
                    f"said {problem.wrong_premise}. You reached a pre-tax total of "
                    f"${drift}.")
        # tight budget: keep ONLY the source, drop the (re-derivable) conclusion
        return (f"(Memory of an earlier session.) The items were: {facts}. You were "
                f"computing the total before tax; the earlier figure may be off.")
    # lossy (default)
    if integrity >= 0.5:
        return (f"(Memory of an earlier session.) You were computing a store total. "
                f"The items were: {facts}. A note said {problem.wrong_premise}. You "
                f"concluded the total before tax was ${drift}.")
    if integrity >= 0.2:
        return (f"(Memory of an earlier session.) You concluded that "
                f"{problem.wrong_premise}, and that the total before tax was ${drift}.")
    return (f"(Memory of an earlier session.) You concluded the total before tax was "
            f"${drift}.")


def reclaim_cross(problem: Problem, arm: str) -> str:
    base = ("Looking back at that earlier conclusion, I think there is a mistake. "
            "Please recheck and give the corrected total before tax as 'ANSWER: <number>'.")
    if arm == "generic":
        return base
    return (f"Looking back at that earlier conclusion, I think {problem.locus} was "
            f"wrong. Please recheck that and give the corrected total before tax as "
            f"'ANSWER: <number>'.")


def run_problem_crosssession(llm, problem: Problem, integrities=INTEGRITY,
                             policy="lossy"):
    """Drift in session 1, then reclaim in a fresh session 2 whose only context is a
    memory compressed to `integrity` under the given `policy`. Both arms."""
    states = build_trajectory(llm, problem)
    transcript = states[max(DEPTHS)]
    rows = []
    for g in integrities:
        if g >= 0.99 and policy == "lossy":
            base = list(transcript)                      # full transcript survives
        else:
            base = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": memory_note(problem, g, policy)}]
        for arm in ("generic", "directed"):
            if hasattr(llm, "configure"):
                llm.configure(problem.drift, problem.correct)
            msgs = base + [{"role": "user", "content": reclaim_cross(problem, arm)}]
            reply = llm.chat(msgs)
            ans = parse_answer(reply)
            ok = ans is not None and abs(ans - problem.correct) < 0.5
            rows.append({"pid": problem.pid, "integrity": g, "arm": arm,
                         "policy": policy, "answer": ans, "correct": ok})
    return rows
