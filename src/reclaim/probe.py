"""A standalone memory-correctability probe: a smoke alarm for agent memory.

Drop a compressed memory note in, get back a verdict the way a smoke alarm does, not
"is this memory good" but "has this memory gone *uncorrectable*, and will it fail loudly
or silently". The dangerous case is the silent one: the source needed to recompute the
answer is gone, but a stale wrong value is still carried, so the model emits it as a
confident answer with nothing to check it against. That is worse than an empty memory.

The probe has two layers:

  * a STATIC witness (``classify_note``) that reads the note alone, no model, instant,
    runnable at write time. It is exact: source-present -> correctable; source-gone but a
    value carried -> uncorrectable & SILENT; nothing carried -> uncorrectable but loud
    (the model will abstain). This is the deterministic core and the part you ship.
  * an optional BEHAVIORAL confirm (pass an ``llm``) that runs the reclaim battery and
    reports the Reclaim Rate, confirming the static verdict against a real or dry-run model.

Quickstart (free, no API key)::

    python -m reclaim.probe

Programmatic::

    from reclaim import classify_note, compare_policies, DryRunLLM
    # write-time, no model: is this one note safe to carry?
    v = classify_note("(Memory.) You concluded the total was $55.",
                      source="7 notebooks at $4, 9 pens at $2", conclusion="$55")
    print(v.regime, v.silent_failure)        # uncorrectable_silent True
    # behavioral confirm across the built-in policies:
    for r in compare_policies(llm=DryRunLLM()):
        print(r.policy, r.reclaim_rate, r.regime, r.verdict)
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .problems import PROBLEMS, Problem, FACTS
from .experiment import SYSTEM, memory_note, reclaim_cross, score, _concl
from .llm import DryRunLLM, parse_answer, parse_answer_word

# How many leading chars of the source string must survive for the source to count as
# present. Matches the anti-rig token test in DryRunLLM/the paper's validators: the source
# string appears verbatim in the full transcript and in a source-first note, never in a
# lossy one, so this is a measurement off the emitted string, not an assumption.
SRC_MARKER = 18

REGIME_CORRECTABLE = "correctable"
REGIME_SILENT = "uncorrectable_silent"
REGIME_LOUD = "uncorrectable_loud"


@dataclass
class Verdict:
    """A static, no-model witness on one carried memory note."""
    regime: str            # correctable | uncorrectable_silent | uncorrectable_loud
    silent_failure: bool   # the dangerous case: uncorrectable AND a wrong value carried
    correctable: bool      # the source survived, so a later correction can land
    detail: str            # one-line human-readable reason


@dataclass
class ProbeReport:
    """A probe verdict over a battery of notes built by one compression policy."""
    policy: str
    n: int
    reclaim_rate: float | None      # behavioral: fraction a correction recovers (None if no llm)
    abstain_rate: float | None      # behavioral: fraction that decline (None for dry-run/no llm)
    regime: str                     # modal static regime across the battery
    silent_failure_rate: float      # fraction of notes that are uncorrectable_silent
    silent_failure: bool            # any silent failure in the battery
    verdict: str                    # one-line human read


def classify_note(note: str, source: str | None, conclusion: str | None) -> Verdict:
    """Static write-time witness on a single carried memory note.

    Args:
        note: the compressed memory string that would cross into the next session.
        source: the recomputable source the answer depends on (line items, the user's
            verbatim utterance, the working). Pass what your memory was *supposed* to keep.
        conclusion: the committed answer/value (e.g. ``"$55"``). If this survives while the
            source does not, the memory is silently uncorrectable.

    Returns:
        Verdict: regime, whether it is a silent failure, whether it stays correctable.
    """
    n = (note or "").lower()
    src_present = bool(source) and source[:SRC_MARKER].lower() in n
    concl_present = bool(conclusion) and str(conclusion).lower() in n
    if src_present:
        return Verdict(REGIME_CORRECTABLE, False, True,
                       "source survived; a correction has something to recompute from")
    if concl_present:
        return Verdict(REGIME_SILENT, True, False,
                       "source dropped but a stale value is carried: the model will emit a "
                       "confident wrong answer with nothing to recompute from")
    return Verdict(REGIME_LOUD, False, False,
                   "neither source nor stale value: failures surface as abstentions, not "
                   "confident wrong answers")


_HEDGE = ("not sure", "cannot", "can't", "can not", "don't have", "do not have",
          "enough to recompute", "unable", "insufficient", "no way to", "need the original")


def _is_abstain(reply: str, problem: Problem) -> bool:
    """Did the model decline rather than commit a value? (Behavioral, real models only.)"""
    low = (reply or "").lower()
    ans = parse_answer(reply) if problem.kind == "number" else parse_answer_word(reply)
    return ans is None or any(h in low for h in _HEDGE)


def _verdict_line(regime: str, rr: float | None) -> str:
    rrs = "" if rr is None else f" (RR={rr:.2f})"
    if regime == REGIME_CORRECTABLE:
        return f"PASS{rrs}: source survives, the memory stays correctable."
    if regime == REGIME_SILENT:
        return (f"SILENT FAILURE{rrs}: a wrong value is carried with no source to recompute "
                f"from, worse than an empty memory.")
    return f"UNCORRECTABLE, LOUD{rrs}: no source, but it abstains rather than assert a value."


def probe_policy(compress, llm=None, problems=PROBLEMS, integrity=0.1, arm="directed",
                 source_of=None, conclusion_of=None, policy_name="custom") -> ProbeReport:
    """Run the probe over a battery of drifted problems compressed by ``compress``.

    Args:
        compress: ``(problem, integrity) -> note``, your memory-writing policy.
        llm: optional client with ``.chat(messages) -> str`` for the behavioral confirm
            (``DryRunLLM`` for free, ``OpenRouterLLM``/``AnthropicLLM`` for a real model).
            If ``None``, only the static witness runs (still exact).
        source_of / conclusion_of: how to read the true source / committed value off a
            ``problem`` (defaults to the built-in arithmetic fields).
        policy_name: label for the report.
    """
    source_of = source_of or (lambda p: FACTS.get(p.pid))
    conclusion_of = conclusion_of or _concl
    verdicts = [classify_note(compress(p, integrity), source_of(p), conclusion_of(p))
                for p in problems]
    n = len(problems)
    silent = sum(v.silent_failure for v in verdicts)
    regime = Counter(v.regime for v in verdicts).most_common(1)[0][0]

    rr = abst = None
    if llm is not None:
        hits = declined = 0
        for p in problems:
            note = compress(p, integrity)
            if hasattr(llm, "configure"):
                llm.configure(p.drift, p.correct, FACTS.get(p.pid), p.locus)
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": note},
                    {"role": "user", "content": reclaim_cross(p, arm)}]
            reply = llm.chat(msgs)
            ok = bool(score(reply, p))
            hits += ok
            if not ok:
                declined += _is_abstain(reply, p)
        rr = hits / n
        # The dry-run fake does not model the emit-vs-abstain split (that needs a real
        # model; see the paper's blank-vs-lossy table), so we report abstain only for real ones.
        abst = None if isinstance(llm, DryRunLLM) else declined / n

    return ProbeReport(policy_name, n, rr, abst, regime, silent / n, silent > 0,
                       _verdict_line(regime, rr))


def compare_policies(llm=None, policies=("source_first", "lossy", "lossy_padded", "blank"),
                     problems=PROBLEMS, integrity=0.1, arm="directed") -> list[ProbeReport]:
    """Probe the four built-in policies side by side: the fix, the wall, the length
    control, and the empty-memory baseline. The headline demonstration."""
    return [probe_policy(lambda p, g, _pol=pol: memory_note(p, g, _pol), llm=llm,
                         problems=problems, integrity=integrity, arm=arm, policy_name=pol)
            for pol in policies]


def _print_reports(reports, integrity, model_label) -> None:
    print(f"\nreclaim :: memory-correctability probe   "
          f"(integrity={integrity:.2f}, directed correction)")
    print(f"model: {model_label}\n")
    head = f"  {'policy':<14}{'RR':>6}  {'regime':<22}{'silent?':>8}"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in reports:
        rr = "n/a" if r.reclaim_rate is None else f"{r.reclaim_rate:.2f}"
        sil = "YES" if r.silent_failure else "no"
        print(f"  {r.policy:<14}{rr:>6}  {r.regime:<22}{sil:>8}")
    print()
    for r in reports:
        print(f"  {r.policy:<14} {r.verdict}")
    print("\n  The static regime/silent verdict reads the note alone, no model, at write "
          "time.\n  RR is the behavioral confirm. On a real model the silent row also shows "
          "the\n  emit-vs-abstain split (lossy emits a confident wrong value where blank "
          "abstains).\n")


if __name__ == "__main__":
    import os
    llm = DryRunLLM()
    label = "DryRunLLM (deterministic, free, no API key)"
    if os.environ.get("OPENROUTER_API_KEY"):
        try:
            from .llm import OpenRouterLLM
            llm = OpenRouterLLM()
            label = f"OpenRouterLLM ({llm.model})"
        except Exception:
            pass
    _print_reports(compare_policies(llm=llm), integrity=0.1, model_label=label)
