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
    question: str          # the base question (asks for the pre-tax total)
    wrong_premise: str     # the planted-wrong fact injected to induce drift
    locus: str             # what is wrong, named for the directed reclaim (no value)
    correct: float         # the true pre-tax total
    drift: float           # the total you get if you use the wrong premise


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
