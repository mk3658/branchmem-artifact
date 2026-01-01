"""The defeasible-ancestor bank must differ from the strict bank in one respect.

The ancestor-strength ablation's entire claim is that content is held fixed
while the ancestor's modal force changes. If a predicate or a branch value
drifted between the two banks, the comparison would be confounded and the
reported flip rates would mean nothing. These tests enforce that invariant.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STRICT = ROOT / "data" / "realistic_semantic_templates.json"
DEFEASIBLE = ROOT / "data" / "defeasible_ancestor_templates.json"

# Unambiguous markers of an absolute constraint. Words that merely quantify
# ("only a little", "all three of the kids") are not modal and are excluded on
# purpose: the invariant under test is modal force, not vocabulary.
_HARD_MODAL = re.compile(
    r"\bmust\b|\bcannot\b|\bcan't\b|\bnever\b|\bstrictly\b|\bcompletely avoid\b|"
    r"\bhard cap\b|\bno one\b|\bnot allowed\b|\bnot permitted\b|\bdocumented\b|"
    r"anaphyla|\bunder any circumstances\b|\bcontraindicated\b|\bwill only\b|"
    r"\bcompletely unavailable\b|\bmust not\b|\babsolute ceiling\b",
    re.I,
)


def _load(path: Path) -> list[dict]:
    return json.loads(path.read_text())["templates"]


@pytest.fixture(scope="module")
def banks() -> tuple[list[dict], list[dict]]:
    return _load(STRICT), _load(DEFEASIBLE)


def test_banks_are_the_same_size_and_order(banks):
    strict, defeasible = banks
    assert len(strict) == len(defeasible) == 60
    assert [t["predicate"] for t in strict] == [t["predicate"] for t in defeasible]


def test_only_the_ancestor_differs(banks):
    strict, defeasible = banks
    for s, d in zip(strict, defeasible):
        for field in ("predicate", "ancestor_predicate", "branch_a_value",
                      "branch_b_value", "correct_branch"):
            assert s[field] == d[field], f"{field} drifted for {s['predicate']}"
        assert s["ancestor_value"] != d["ancestor_value"], s["predicate"]


def test_every_defeasible_ancestor_drops_the_hard_modal(banks):
    """A rewritten ancestor that still says 'must never' is not defeasible."""
    strict, defeasible = banks
    offenders = [t["predicate"] for t in defeasible
                 if _HARD_MODAL.search(t["ancestor_value"])]
    assert offenders == [], offenders
    # The pattern has to be able to fire at all, or the test above is vacuous.
    strict_hits = sum(1 for t in strict if _HARD_MODAL.search(t["ancestor_value"]))
    assert strict_hits >= len(strict) // 2, strict_hits


def test_predicates_are_unique(banks):
    for bank in banks:
        predicates = [t["predicate"] for t in bank]
        assert len(set(predicates)) == len(predicates)


def test_bank_declares_its_provenance_and_the_label_caveat():
    note = json.loads(DEFEASIBLE.read_text())["note"].lower()
    assert "authored by the paper's authors" in note or "authored by" in note
    assert "flip" in note and "not a logical entailment" in note
