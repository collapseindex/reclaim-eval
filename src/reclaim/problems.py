"""Verifiable multi-step arithmetic problems with a planted error site.

Each problem has a clean pre-tax total (the ground truth). We induce drift by
injecting a WRONG value for one named component (the locus), let the model build on
it over follow-up turns, then attempt to reclaim it back to the truth. Because the
answer is a number, "reclaim succeeded" is objective: no judge.

The follow-ups operate on the running pre-tax total (tax, coupon, restate, double),
so they deepen the model's commitment to the wrong total WITHOUT re-deriving the
corrupted component. Drift depth = how many follow-ups were applied before reclaim.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Problem:
    pid: str
    question: str          # the base question
    wrong_premise: str     # the planted-wrong fact injected to induce drift
    locus: str             # what is wrong, named for the directed reclaim (no value)
    correct: object        # the true answer (float for arithmetic, str for logic)
    drift: object          # the answer you get if you use the wrong premise
    ask: str = "the total before tax"   # what the reclaim asks the model to give
    kind: str = "number"   # "number" (numeric tolerance) or "text" (single-word match)


PROBLEMS: list[Problem] = [
    Problem("notebooks",
            "A store sells notebooks at $4 each and pens at $2 each. Maria buys 7 "
            "notebooks and 9 pens. What is her total before tax?",
            "a colleague already worked out that the pens come to $27",
            "the pens subtotal", correct=46.0, drift=55.0),
    Problem("tiles",
            "Tiles cost $6 each and grout bags cost $5 each. Sam buys 12 tiles and 4 "
            "grout bags. What is the total before tax?",
            "a colleague says the tiles come to $60", "the tiles subtotal",
            correct=92.0, drift=80.0),
    Problem("fruit",
            "Apples are $3 per bag and oranges are $4 per bag. Lee buys 5 bags of "
            "apples and 6 bags of oranges. Total before tax?",
            "a colleague computed the oranges at $30", "the oranges subtotal",
            correct=39.0, drift=45.0),
    Problem("tickets",
            "Adult tickets are $12 and child tickets are $8. A family buys 3 adult "
            "and 5 child tickets. Total before tax?",
            "a colleague worked out the child tickets at $48", "the child-ticket subtotal",
            correct=76.0, drift=84.0),
    Problem("paint",
            "Paint cans are $15 each and brushes are $7 each. Dana buys 4 cans and 3 "
            "brushes. Total before tax?",
            "a colleague says the brushes come to $35", "the brushes subtotal",
            correct=81.0, drift=95.0),
    Problem("coffee",
            "Bags of coffee are $9 each and filters are $3 each. Kim buys 6 bags and 4 "
            "filter packs. Total before tax?",
            "a colleague computed the coffee at $72", "the coffee subtotal",
            correct=66.0, drift=84.0),
    Problem("lumber",
            "Boards cost $11 each and screws cost $2 per box. Pat buys 8 boards and 7 "
            "boxes of screws. Total before tax?",
            "a colleague worked out the boards at $99", "the boards subtotal",
            correct=102.0, drift=113.0),
    Problem("flowers",
            "Roses are $5 per bunch and lilies are $6 per bunch. Noah buys 9 bunches "
            "of roses and 4 of lilies. Total before tax?",
            "a colleague says the roses come to $40", "the roses subtotal",
            correct=69.0, drift=64.0),
]

# the recomputable SOURCE for each problem: the original quantities and prices needed
# to get the correct answer. In the cross-session experiment this is what does or does
# not survive into the carried-over memory. When it is gone, even a directed correction
# cannot reclaim, because there is nothing left to recompute from (the broken sky).
FACTS: dict[str, str] = {
    "notebooks": "notebooks at $4 each (7 bought) and pens at $2 each (9 bought)",
    "tiles": "tiles at $6 each (12 bought) and grout bags at $5 each (4 bought)",
    "fruit": "apples at $3 per bag (5 bags) and oranges at $4 per bag (6 bags)",
    "tickets": "adult tickets at $12 (3) and child tickets at $8 (5)",
    "paint": "paint cans at $15 each (4) and brushes at $7 each (3)",
    "coffee": "coffee bags at $9 each (6) and filter packs at $3 each (4)",
    "lumber": "boards at $11 each (8) and boxes of screws at $2 each (7)",
    "flowers": "roses at $5 per bunch (9) and lilies at $6 per bunch (4)",
}


# follow-ups that build commitment on the running pre-tax total (never re-derive
# the corrupted component). Applied in order; depth = how many were used.
FOLLOWUPS: list[str] = [
    "Now apply an 8% sales tax to that pre-tax total. Give the post-tax amount.",
    "There is a $5 coupon off the post-tax amount. What is paid now?",
    "Round that final amount to the nearest cent.",
    "For the record, restate the pre-tax total you have been using.",
    "If the same order is placed again next week, what is the combined pre-tax total?",
    "What is half of the original pre-tax total?",
    "Confirm the original pre-tax total once more.",
    "Express the original pre-tax total in cents.",
]


# ── non-arithmetic task family: constraint-deduction puzzles. Same shape (a clean
#    source of clues fully determines a single-token answer; a planted wrong clue
#    drifts it), but the recomputation is LOGICAL, not numeric. This is the harder
#    case for source-first: the source is a set of relations, not a formula. Each
#    answer is one word, so scoring stays objective with no judge. ─────────────────
PROBLEMS_LOGIC: list[Problem] = [
    Problem("roles",
            "Three coworkers, Ana, Ben, and Cleo, each hold exactly one role: manager, "
            "designer, or auditor. Ben is not the manager. Cleo is the auditor. Who is "
            "the manager?",
            "a colleague noted that Ben is the manager", "Ben's role",
            correct="Ana", drift="Ben", ask="the manager", kind="text"),
    Problem("seating",
            "Four friends sit in a row in positions 1 to 4, left to right: Dee, Eve, "
            "Fia, Gus. Dee is at position 1. Eve is immediately to the right of Dee. Gus "
            "is at position 4. Who is at position 3?",
            "a colleague says Eve is at position 3", "Eve's position",
            correct="Fia", drift="Eve", ask="the person in position 3", kind="text"),
    Problem("race",
            "Five runners finished a race: Hal, Ira, Jo, Kit, and Lee. Hal finished "
            "ahead of Ira. Ira finished ahead of Jo. Jo finished ahead of Kit. Kit "
            "finished ahead of Lee. Who finished last?",
            "a colleague says Kit finished behind Lee", "the Kit and Lee ordering",
            correct="Lee", drift="Kit", ask="the runner who finished last", kind="text"),
    Problem("ages",
            "Three siblings are Mae, Ned, and Ola. Mae is older than Ned. Ola is younger "
            "than Ned. Who is the youngest?",
            "a colleague says Ned is younger than Ola", "the Ned and Ola age order",
            correct="Ola", drift="Ned", ask="the youngest sibling", kind="text"),
    Problem("pets",
            "Pam, Quincy, and Rosa each own exactly one pet: a cat, a dog, or a fish. "
            "Pam owns the dog. Quincy does not own the fish. Who owns the cat?",
            "a colleague says Quincy owns the fish", "Quincy's pet",
            correct="Quincy", drift="Rosa", ask="the cat's owner", kind="text"),
    Problem("days",
            "Three meetings are each on a different day, Monday, Tuesday, or Wednesday: "
            "the budget meeting, the design meeting, and the review meeting. The budget "
            "meeting is on Monday. The design meeting is not on Wednesday. Which day is "
            "the review meeting?",
            "a colleague says the design meeting is on Wednesday", "the design meeting's day",
            correct="Wednesday", drift="Tuesday", ask="the review meeting's day", kind="text"),
    Problem("height",
            "Four players are ranked by height: Sam, Tom, Uma, and Val. Sam is taller "
            "than Tom. Tom is taller than Uma. Uma is taller than Val. Who is the second "
            "tallest?",
            "a colleague says Uma is taller than Tom", "the Tom and Uma height order",
            correct="Tom", drift="Uma", ask="the second-tallest player", kind="text"),
    Problem("houses",
            "Three houses in a row are painted red, blue, and green, one colour each. "
            "The first house is red. The blue house is not the middle one. What colour "
            "is the third house?",
            "a colleague says the blue house is the middle one", "the blue house's position",
            correct="blue", drift="green", ask="the third house's colour", kind="text"),
]

FACTS_LOGIC: dict[str, str] = {
    "roles": "Ben is not the manager, and Cleo is the auditor (one each of manager, "
             "designer, auditor among Ana, Ben, Cleo)",
    "seating": "Dee is at position 1, Eve is immediately right of Dee, and Gus is at "
               "position 4 (positions 1 to 4 for Dee, Eve, Fia, Gus)",
    "race": "Hal ahead of Ira, Ira ahead of Jo, Jo ahead of Kit, and Kit ahead of Lee",
    "ages": "Mae is older than Ned, and Ola is younger than Ned",
    "pets": "Pam owns the dog, and Quincy does not own the fish (cat, dog, fish, one "
            "each among Pam, Quincy, Rosa)",
    "days": "the budget meeting is on Monday, and the design meeting is not on "
            "Wednesday (budget, design, review on Monday, Tuesday, Wednesday)",
    "height": "Sam taller than Tom, Tom taller than Uma, and Uma taller than Val",
    "houses": "the first house is red, and the blue house is not the middle one (red, "
              "blue, green, one each)",
}
FACTS.update(FACTS_LOGIC)

# commitment turns for the logic task: re-affirm the stated answer without re-solving.
FOLLOWUPS_LOGIC: list[str] = [
    "Restate your final answer in one word.",
    "How confident are you in that answer? One short sentence.",
    "If someone asked you in passing, what would you say the answer is?",
    "For the record, repeat the answer you settled on.",
    "Summarise your conclusion in a short sentence.",
    "Confirm that answer once more.",
    "What single word should I write down as the answer?",
    "State the answer again, just to be sure.",
]

# task registry and the follow-up set keyed by answer kind
TASKS: dict[str, list[Problem]] = {"arith": PROBLEMS, "logic": PROBLEMS_LOGIC}
FOLLOWUPS_BY_KIND: dict[str, list[str]] = {"number": FOLLOWUPS, "text": FOLLOWUPS_LOGIC}


# ── expand each family 8 -> 32 (n=24 -> n=96) via verified generators ──────────────
# Default on; set RECLAIM_EXPAND=0 to reproduce the original 8-problem (n=24) runs.
import os as _os  # noqa: E402
if _os.environ.get("RECLAIM_EXPAND", "1") == "1":
    from .problems_gen import (gen_arith, gen_logic, gen_assign,  # noqa: E402
                               validate_arith, validate_logic, validate_assign)
    _ga, _fa = gen_arith(24, seed=1); validate_arith(_ga, _fa)
    # logic expansion mirrors the canonical variety: 12 ordering + 12 assignment puzzles
    _gl, _fl = gen_logic(12, seed=2); validate_logic(_gl, _fl)
    _gs, _fs = gen_assign(12, seed=3); validate_assign(_gs, _fs)
    PROBLEMS.extend(_ga); PROBLEMS_LOGIC.extend(_gl); PROBLEMS_LOGIC.extend(_gs)
    FACTS.update(_fa); FACTS.update(_fl); FACTS.update(_fs)
