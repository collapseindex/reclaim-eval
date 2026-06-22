"""Induce drift, let it commit over turns, then attempt reclaim at increasing depth
with a generic vs a directed (encoded) correction. Measure the window.
"""
from __future__ import annotations

from .problems import Problem, FOLLOWUPS, FACTS, FOLLOWUPS_BY_KIND
from .llm import parse_answer, parse_answer_word

SYSTEM = ("You are a careful assistant. Work the problem step by step. End every "
          "reply with a line 'ANSWER: <answer>'.")

DEPTHS = (1, 2, 4, 8)


def score(reply: str, problem: Problem) -> bool:
    """Objective check, no judge. Numeric tasks: parsed number within tolerance.
    Text tasks: the first word after ANSWER: equals the correct single-word answer."""
    if problem.kind == "number":
        ans = parse_answer(reply)
        return ans is not None and abs(ans - float(problem.correct)) < 0.5
    tok = parse_answer_word(reply)
    return tok is not None and tok.lower() == str(problem.correct).lower()


def _logged_answer(reply: str, problem: Problem):
    """Best-effort parse purely for logging (not scoring)."""
    return parse_answer(reply) if problem.kind == "number" else parse_answer_word(reply)


def _configure(llm, problem: Problem):
    if hasattr(llm, "configure"):
        llm.configure(problem.drift, problem.correct, FACTS.get(problem.pid),
                      problem.locus)


def _plant(problem: Problem):
    """Opening user turn that injects the wrong premise to induce drift."""
    return (f"{problem.question}\nNote: {problem.wrong_premise}. Use that and "
            f"give {problem.ask}.")


def reclaim_message(problem: Problem, arm: str) -> str:
    base = (f"Hold on. I think there is a mistake above. Please recheck and give the "
            f"corrected {problem.ask} as 'ANSWER: <answer>'.")
    if arm == "generic":
        return base
    if arm == "directed":
        return (f"Hold on. I think {problem.locus} is wrong. Please recheck that and "
                f"give the corrected {problem.ask} as 'ANSWER: <answer>'.")
    raise ValueError(arm)


def build_trajectory(llm, problem: Problem):
    """Plant drift, then commit over the task's follow-ups, capturing the message
    state at each checkpoint depth. Returns {depth: messages}."""
    _configure(llm, problem)
    followups = FOLLOWUPS_BY_KIND[problem.kind]
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": _plant(problem)}]
    messages.append({"role": "assistant", "content": llm.chat(messages)})
    states = {}
    for i, fu in enumerate(followups, start=1):
        messages = messages + [{"role": "user", "content": fu}]
        messages.append({"role": "assistant", "content": llm.chat(messages)})
        if i in DEPTHS:
            states[i] = list(messages)
    return states


def attempt_reclaim(llm, state, problem: Problem, arm: str):
    _configure(llm, problem)
    msgs = state + [{"role": "user", "content": reclaim_message(problem, arm)}]
    reply = llm.chat(msgs)
    return {"arm": arm, "answer": _logged_answer(reply, problem),
            "correct": score(reply, problem)}


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
    _configure(llm, problem)
    msgs = list(state)
    for i in range(n_filler):
        u, a = FILLER[i % len(FILLER)]
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs += [{"role": "user", "content": reclaim_message(problem, arm)}]
    reply = llm.chat(msgs)
    return {"arm": arm, "answer": _logged_answer(reply, problem),
            "correct": score(reply, problem)}


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


def _concl(problem: Problem) -> str:
    """The carried conclusion, formatted by answer kind ($ for money, bare otherwise)."""
    if problem.kind == "number":
        return f"${problem.drift:g}"
    return f"{problem.drift}"


PAD = (" (For the record, this entry was retained from an archived session log; no "
       "additional working was preserved with it.)")


def memory_note(problem: Problem, integrity: float, policy: str = "lossy") -> str:
    """A compressed carry-over memory at the given integrity. Three POLICIES:
      - lossy: keep the salient CONCLUSION, shed the source as budget tightens (the
        realistic default; conclusion survives, the recomputable source does not).
      - source_first: keep the recomputable SOURCE at every budget, shed the
        (re-derivable) conclusion. The conclusion can be re-derived from the source;
        the source can never be re-derived from the conclusion.
      - lossy_padded: the budget-match CONTROL. Identical content to lossy, but padded
        with neutral filler up to the length of the source_first note, so any
        source_first advantage cannot be attributed to having more text.
      - blank: the EMPTY-MEMORY baseline. Keeps neither the source nor the conclusion,
        only that an earlier session was determining `ask`. There is no stale value to
        inherit, so a wrong value the model emits here is not an inherited attractor; this
        isolates whether lossy is worse than carrying nothing at all.
    """
    concl = _concl(problem)
    facts = FACTS[problem.pid]
    ask = problem.ask
    if policy == "source_first":
        if integrity >= 0.5:
            return (f"(Memory of an earlier session.) The facts were: {facts}. A note "
                    f"said {problem.wrong_premise}. You reached {ask} of {concl}.")
        # tight budget: keep ONLY the source, drop the (re-derivable) conclusion
        return (f"(Memory of an earlier session.) The facts were: {facts}. You were "
                f"determining {ask}; the earlier answer may be off.")
    if policy == "source_plus_conclusion":
        # keep BOTH the recomputable source AND the stale conclusion: tests whether the
        # conclusion re-introduces the attractor even when the source is present.
        return (f"(Memory of an earlier session.) The facts were: {facts}. You concluded "
                f"{ask} was {concl}.")
    if policy == "lossy_padded":
        base = memory_note(problem, integrity, "lossy")
        target = len(memory_note(problem, integrity, "source_first"))
        while len(base) < target:
            base += PAD
        return base
    if policy == "blank":
        # keep NEITHER source nor conclusion: nothing to inherit, nothing to recompute from
        return (f"(Memory of an earlier session.) You were earlier determining {ask}. "
                f"No figures or details from that session were retained.")
    # lossy (default)
    if integrity >= 0.5:
        return (f"(Memory of an earlier session.) The facts were: {facts}. A note said "
                f"{problem.wrong_premise}. You concluded {ask} was {concl}.")
    if integrity >= 0.2:
        return (f"(Memory of an earlier session.) You concluded that "
                f"{problem.wrong_premise}, and that {ask} was {concl}.")
    return (f"(Memory of an earlier session.) You concluded {ask} was {concl}.")


def reclaim_cross(problem: Problem, arm: str) -> str:
    base = (f"Looking back at that earlier conclusion, I think there is a mistake. "
            f"Please recheck and give the corrected {problem.ask} as 'ANSWER: <answer>'.")
    if arm == "generic":
        return base
    return (f"Looking back at that earlier conclusion, I think {problem.locus} was "
            f"wrong. Please recheck that and give the corrected {problem.ask} as "
            f"'ANSWER: <answer>'.")


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
            _configure(llm, problem)
            msgs = base + [{"role": "user", "content": reclaim_cross(problem, arm)}]
            reply = llm.chat(msgs)
            rows.append({"pid": problem.pid, "integrity": g, "arm": arm,
                         "policy": policy, "answer": _logged_answer(reply, problem),
                         "correct": score(reply, problem)})
    return rows
