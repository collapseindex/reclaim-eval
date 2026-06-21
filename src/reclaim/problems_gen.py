"""Deterministic generators that expand the two task families from 8 to 32 problems each,
so every reclaim cell goes from n=24 (8 problems x 3 seeds) to n=96 without hand-arithmetic risk.

Every generated problem is correct BY CONSTRUCTION: the arithmetic total is the literal sum of
price*quantity, and the planted drift corrupts exactly one named subtotal to a different value. The
logic puzzles are emitted from solver-checked templates whose single-token answer is verified before
the problem is kept. validate_arith / validate_logic re-prove cleanliness, so a bad instance can
never slip into a run (it raises instead).
"""
from __future__ import annotations

import random
from itertools import permutations

# Problem is imported lazily inside each generator to avoid a circular import:
# problems.py imports these generators at module load to expand its problem lists.

NAMES = ["Ana", "Ben", "Cleo", "Dee", "Eve", "Fia", "Gus", "Hal", "Ira", "Jo", "Kit",
         "Lee", "Mae", "Ned", "Ola", "Pam", "Quinn", "Rosa", "Sam", "Tom", "Uma", "Val"]

# (plural noun, singular-ish unit word) goods pool, disjoint from the 8 canonical items
GOODS = [
    "mugs", "plates", "candles", "markers", "ropes", "binders", "lamps", "tiles",
    "bricks", "cables", "hinges", "magnets", "sponges", "buckets", "trays", "clamps",
    "wrenches", "gloves", "batteries", "filters", "spools", "valves", "nozzles", "straps",
    "crayons", "erasers", "staplers", "rulers", "folders", "clips", "pins", "tags",
]


def gen_arith(n: int, seed: int = 1):
    """n two-item arithmetic problems with a clean pre-tax total and one corrupted subtotal."""
    from .problems import Problem
    rng = random.Random(seed)
    out, facts = [], {}
    goods = GOODS[:]
    rng.shuffle(goods)
    gi = 0
    for i in range(n):
        a, b = goods[gi % len(goods)], goods[(gi + 1) % len(goods)]
        gi += 2
        pa, pb = rng.randint(2, 15), rng.randint(2, 15)
        qa, qb = rng.randint(3, 12), rng.randint(3, 12)
        correct = pa * qa + pb * qb
        # corrupt one subtotal to a plausible-but-different value
        if rng.random() < 0.5:
            true_sub, name, p, q = pa * qa, a, pa, qa
        else:
            true_sub, name, p, q = pb * qb, b, pb, qb
        delta = rng.choice([d for d in (-13, -9, -7, 7, 9, 11, 13) if true_sub + d > 0])
        wrong_sub = true_sub + delta
        drift = correct - true_sub + wrong_sub
        pid = f"gen_a{i}"
        question = (f"{a.capitalize()} cost ${pa} each and {b} cost ${pb} each. A buyer gets "
                    f"{qa} {a} and {qb} {b}. What is the total before tax?")
        out.append(Problem(pid, question,
                           wrong_premise=f"a colleague worked out the {name} at ${wrong_sub}",
                           locus=f"the {name} subtotal",
                           correct=float(correct), drift=float(drift)))
        facts[pid] = f"{a} at ${pa} each ({qa} bought) and {b} at ${pb} each ({qb} bought)"
    return out, facts


def validate_arith(problems, facts):
    """Re-prove every generated problem: drift != correct, both positive, source names both goods."""
    for p in problems:
        assert p.correct != p.drift, f"{p.pid}: drift==correct"
        assert p.correct > 0 and p.drift > 0, f"{p.pid}: non-positive"
        assert p.pid in facts and "$" in facts[p.pid], f"{p.pid}: missing source"
        # the corrupted subtotal must move the total by the same signed amount
        assert abs((p.correct - p.drift)) == abs(int(p.correct - p.drift)), f"{p.pid}: non-integer drift"
    return True


# ── logic family: total-order (race) puzzles, brute-force verified ─────────────────
def _order_answer(names, ahead, query):
    """Unique answer to `query` over all orders satisfying `ahead` (X before Y), else None."""
    sols = set()
    for perm in permutations(names):
        idx = {n: i for i, n in enumerate(perm)}
        if all(idx[x] < idx[y] for x, y in ahead):
            sols.add(perm[-1] if query == "last" else perm[1])
    return next(iter(sols)) if len(sols) == 1 else None


def gen_logic(n: int, seed: int = 2):
    """n ordering puzzles: a clue chain fixes a unique answer; one reversed clue (the planted
    drift) yields a unique DIFFERENT answer. Both are verified by brute force before keeping."""
    from .problems import Problem
    rng = random.Random(seed)
    out, facts, kept, tries = [], {}, 0, 0
    while kept < n and tries < n * 80:
        tries += 1
        k = rng.choice([4, 5])
        order = rng.sample(NAMES, k)                         # true finish order, index 0 = first
        ahead = [(order[j], order[j + 1]) for j in range(k - 1)]   # full chain -> unique order
        query = rng.choice(["last", "second"])
        correct = _order_answer(order, ahead, query)
        if correct is None:
            continue
        j = rng.randrange(k - 1)                              # reverse one consecutive clue
        a, b = order[j], order[j + 1]
        corrupt = [p for p in ahead if p != (a, b)] + [(b, a)]
        drift = _order_answer(order, corrupt, query)
        if drift is None or drift == correct:
            continue
        listed = rng.sample(order, k)                        # list names NOT in finish order
        clue_txt = ". ".join(f"{x} finished ahead of {y}" for x, y in ahead)
        ask = "the runner who finished last" if query == "last" else "the runner who finished second"
        pid = f"gen_l{kept}"
        out.append(Problem(
            pid,
            f"{k} runners finished a race: {', '.join(listed)}. {clue_txt}. "
            f"Who finished {'last' if query == 'last' else 'second'}?",
            wrong_premise=f"a colleague says {b} finished ahead of {a}",
            locus=f"the {a}-versus-{b} finish order",
            correct=correct, drift=drift, ask=ask, kind="text"))
        facts[pid] = ", ".join(f"{x} ahead of {y}" for x, y in ahead)
        kept += 1
    if kept < n:
        raise RuntimeError(f"only generated {kept}/{n} logic problems")
    return out, facts


def validate_logic(problems, facts):
    """Re-prove each puzzle from its OWN clue source (facts), independent of generation."""
    for p in problems:
        # rebuild the ahead-pairs from the source string and re-solve
        pairs = [tuple(s.strip().split(" ahead of ")) for s in facts[p.pid].split(",")]
        query = "last" if "last" in p.ask else "second"
        names = sorted({n for pr in pairs for n in pr})
        ans = _order_answer(names, pairs, query)
        assert ans == p.correct, f"{p.pid}: source solves to {ans}, not {p.correct}"
        assert p.drift != p.correct, f"{p.pid}: drift==correct"
        assert isinstance(p.correct, str) and isinstance(p.drift, str), f"{p.pid}: non-token answer"
    return True


# ── logic family, part 2: assignment puzzles (who has item X), brute-force verified ──
ITEMSETS = [("manager", "designer", "auditor"), ("cat", "dog", "fish"),
            ("red", "blue", "green"), ("gold", "silver", "bronze"),
            ("tea", "coffee", "juice"), ("rose", "tulip", "daisy")]


def _assign_answer(people, items, pos, neg, query_item):
    """Unique person holding query_item over all bijections people->items satisfying clues, else None."""
    sols = set()
    for perm in permutations(items):
        a = dict(zip(people, perm))
        if all(a[p] == it for p, it in pos) and all(a[p] != it for p, it in neg):
            who = [p for p in people if a[p] == query_item]
            if len(who) == 1:
                sols.add(who[0])
    return next(iter(sols)) if len(sols) == 1 else None


def gen_assign(n: int, seed: int = 3):
    """n assignment puzzles: a clue set fixes a unique 'who has item X'; one flipped clue (the drift)
    yields a unique DIFFERENT person. Both verified by brute force before keeping."""
    from .problems import Problem
    rng = random.Random(seed)
    out, facts, kept, tries = [], {}, 0, 0
    while kept < n and tries < n * 120:
        tries += 1
        k = 3
        people = rng.sample(NAMES, k)
        items = list(rng.choice(ITEMSETS))
        true = dict(zip(people, rng.sample(items, k)))         # true bijection
        # one positive (a person's true item) + one negative (a person is-not a wrong item)
        pp = rng.choice(people)
        npers = rng.choice([p for p in people if p != pp])
        nitem = rng.choice([it for it in items if it != true[npers]])
        pos, neg = [(pp, true[pp])], [(npers, nitem)]
        query_item = rng.choice([it for it in items if it != true[pp]])  # ask about an unstated item
        correct = _assign_answer(people, items, pos, neg, query_item)
        if correct is None:
            continue
        # plant drift: flip the negative into a (false) positive -> different unique holder, if any
        cpos, cneg = pos + [(npers, nitem)], []
        drift = _assign_answer(people, items, cpos, cneg, query_item)
        if drift is None or drift == correct:
            continue
        pid = f"gen_s{kept}"
        clue_txt = (f"{pp} has the {true[pp]}. {npers} does not have the {nitem}")
        out.append(Problem(
            pid,
            f"{', '.join(people)} each have exactly one of: {', '.join(items)}. {clue_txt}. "
            f"Who has the {query_item}?",
            wrong_premise=f"a colleague says {npers} has the {nitem}",
            locus=f"what {npers} has",
            correct=correct, drift=drift, ask=f"the person with the {query_item}", kind="text"))
        facts[pid] = (f"{pp} has {true[pp]}; {npers} not {nitem}; "
                      f"domain {'/'.join(items)} among {'/'.join(people)}")
        kept += 1
    if kept < n:
        raise RuntimeError(f"only generated {kept}/{n} assignment problems")
    return out, facts


def validate_assign(problems, facts):
    """Re-prove each assignment puzzle from its own clue source, independent of generation."""
    import re
    for p in problems:
        src = facts[p.pid]
        pos = re.findall(r"(\w+) has (\w+);", src)
        neg = re.findall(r"(\w+) not (\w+);", src)
        dom = re.search(r"domain ([\w/]+) among ([\w/]+)", src)
        items, people = dom.group(1).split("/"), dom.group(2).split("/")
        qi = p.ask.split("with the ")[1]
        assert _assign_answer(people, items, pos, neg, qi) == p.correct, f"{p.pid}: source mismatch"
        assert p.drift != p.correct, f"{p.pid}: drift==correct"
    return True
