"""Validator: the pipeline measures a window when one exists, and can show its
absence. Runs entirely on the free DryRun LLM (no API). It can fail: if the harness
stopped distinguishing depth or arm, these assertions break.
"""
from collections import defaultdict

from reclaim.problems import PROBLEMS
from reclaim.experiment import DEPTHS, run_problem
from reclaim.llm import DryRunLLM, parse_answer


def _rates(seeds=3):
    succ = {a: defaultdict(list) for a in ("generic", "directed")}
    for s in range(seeds):
        llm = DryRunLLM(seed=s)
        for p in PROBLEMS:
            for row in run_problem(llm, p):
                succ[row["arm"]][row["depth"]].append(row["correct"])
    return {a: {d: sum(v) / len(v) for d, v in succ[a].items()} for a in succ}


def test_parse_answer():
    assert parse_answer("blah ANSWER: 46") == 46.0
    assert parse_answer("ANSWER: $92.0") == 92.0          # $/markdown tolerated after the marker
    assert parse_answer("the total is $92.0") is None     # no ANSWER: marker => not a commit (v2 strict parser)
    assert parse_answer("no number here") is None


def test_pipeline_measures_a_decaying_window():
    r = _rates()
    # generic reclaim should decay with depth (the window closing)
    assert r["generic"][1] > r["generic"][8]


def test_directed_holds_deeper_than_generic():
    r = _rates()
    # at the deepest point, directed should beat generic in the seeded fixture
    assert r["directed"][8] > r["generic"][8]
