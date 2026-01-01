"""The composed-pipeline result, and the one claim it rests on.

Appendix "Composing the Two Fixes" reports a negative result: chaining the
standing check in front of the specification-prompted merge, with either stage
able to reject, is WORSE than either stage alone. That holds only because both
stages catch every violation on this bank, so their errors are all freezes and
an OR-composition unions them. If a later change makes a stage miss a
violation, the negative result stops following from these numbers and the
appendix's reasoning has to be rewritten rather than quietly left standing.

These tests pin that structure and the arithmetic over it, recomputed from the
stored per-arm rates rather than trusted.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "final" / "composed_pipeline.json"
N_UPDATE_ITEMS = 48  # BranchMem-Hard: 12 keep, 48 update


@pytest.fixture(scope="module")
def payload() -> dict:
    if not RESULT.exists():
        pytest.skip(f"{RESULT.name} not generated yet")
    return json.loads(RESULT.read_text())


def test_all_four_arms_present(payload):
    assert set(payload["arms"]) == {
        "gate_only", "merge_only",
        "composed_either_rejects", "composed_both_must_reject"}
    assert payload["n_items"] == 60


def test_no_new_model_call_was_needed(payload):
    """Both stages are cache-served; an unparsed reply would change the numbers."""
    assert payload["gate"]["n_unparsed_replies"] == 0
    assert payload["merge"]["n_abstained"] == 0


def test_every_arm_catches_every_violation(payload):
    """The premise of the appendix's argument: all errors here are freezes."""
    for name, arm in payload["arms"].items():
        assert arm["violations_caught"] == 1.0, (
            f"{name} now misses a violation, so the OR-composition no longer "
            "merely unions freezes and the appendix's reasoning is stale")


def test_or_composition_unions_the_freezes(payload):
    """Either-rejects must freeze at least as much as its worse part."""
    a = payload["arms"]
    worse = max(a["gate_only"]["freezing_rate"], a["merge_only"]["freezing_rate"])
    assert a["composed_either_rejects"]["freezing_rate"] >= worse
    assert a["composed_either_rejects"]["balanced"] < min(
        a["gate_only"]["balanced"], a["merge_only"]["balanced"]), (
        "the paper's negative result is that chaining as a safety gate is worse "
        "than either part; that is no longer what the data shows")


def test_and_composition_intersects_the_freezes(payload):
    a = payload["arms"]
    better = min(a["gate_only"]["freezing_rate"], a["merge_only"]["freezing_rate"])
    assert a["composed_both_must_reject"]["freezing_rate"] <= better


def test_the_disagreement_accounting_adds_up(payload):
    """The complementarity claim is a partition of the two stages' errors."""
    s = payload["stage_agreement"]
    assert s["items_both_stages_agree"] + len(
        set(s["gate_wrong_merge_right"]) | set(s["merge_wrong_gate_right"])
    ) <= s["n_items"]
    overlap = set(s["gate_wrong_merge_right"]) & set(s["merge_wrong_gate_right"])
    assert not overlap, f"an item cannot be in both exclusive sets: {overlap}"
    assert set(s["both_wrong"]).isdisjoint(s["gate_wrong_merge_right"])
    assert set(s["both_wrong"]).isdisjoint(s["merge_wrong_gate_right"])


def test_the_appendix_numbers_are_the_stored_ones(payload):
    """Freeze counts the appendix quotes, as counts rather than rates."""
    a = payload["arms"]
    counts = {k: round(v["freezing_rate"] * N_UPDATE_ITEMS)
              for k, v in a.items()}
    assert counts == {"gate_only": 5, "merge_only": 4,
                      "composed_either_rejects": 8,
                      "composed_both_must_reject": 1}, counts
