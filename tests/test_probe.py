"""The memory-correctability probe: the static witness is exact and model-free, and the
behavioral confirm agrees with it on the free dry-run fake."""
from reclaim import classify_note, compare_policies, DryRunLLM
from reclaim.probe import REGIME_CORRECTABLE, REGIME_SILENT, REGIME_LOUD


SOURCE = "7 notebooks at $4, 9 pens at $2"


def test_static_witness_silent_when_source_gone_value_kept():
    v = classify_note("(Memory of an earlier session.) You concluded the total was $55.",
                      source=SOURCE, conclusion="$55")
    assert v.regime == REGIME_SILENT
    assert v.silent_failure is True
    assert v.correctable is False


def test_static_witness_correctable_when_source_present():
    v = classify_note(f"(Memory.) The facts were: {SOURCE}. You were determining the total.",
                      source=SOURCE, conclusion="$55")
    assert v.regime == REGIME_CORRECTABLE
    assert v.silent_failure is False
    assert v.correctable is True


def test_static_witness_loud_when_nothing_carried():
    v = classify_note("(Memory.) You were determining the total. No figures were retained.",
                      source=SOURCE, conclusion="$55")
    assert v.regime == REGIME_LOUD
    assert v.silent_failure is False


def test_compare_policies_separates_fix_from_wall():
    by = {r.policy: r for r in compare_policies(llm=DryRunLLM())}
    # the fix keeps the memory correctable; the wall and its padded control fail silently;
    # the empty baseline fails but loudly (it has no stale value to assert).
    assert by["source_first"].regime == REGIME_CORRECTABLE
    assert by["source_first"].silent_failure is False
    assert by["source_first"].reclaim_rate > 0.8
    assert by["lossy"].silent_failure is True
    assert by["lossy"].reclaim_rate == 0.0
    assert by["lossy_padded"].silent_failure is True
    assert by["blank"].regime == REGIME_LOUD
    assert by["blank"].silent_failure is False
