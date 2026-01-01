"""BranchMem-Hard must not leak the answer through surface wording.

The whole point of this bank is that an obligation-bearing ancestor no longer
tells you whether the update should be rejected. These tests enforce that
property, so an edit that reintroduces a lexical shortcut fails the suite
rather than quietly turning a detection result into a regex result.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "data" / "hard_standing_bank.json"
PIN = ROOT / "data" / "hard_standing_bank.sha256"

TYPES = ("standing_indefinite", "standing_expired",
         "standing_conditional_unmet", "standing_waived", "ordinary_value")
OBLIGATION_TYPES = TYPES[:4]

# The same marker the discrimination bank's validator uses. There it separates
# the two critical types perfectly (80/80 vs 0/80); here it must not.
MARKER = re.compile(
    r"\bonly\b|\bmust\b|\bcannot\b|\bnot permitted\b|\bnot allowed\b|\brequire[sd]?\b|"
    r"\bmaximum\b|\blimit\b|\bprohibit|\bforbid|\brule\b|\bpolicy\b|\bcontract|"
    r"\bban(s|ned)?\b|\bmandat|\bobliged\b|\bcapacity\b|\bdeadline\b|\bno more than\b",
    re.I)

# Material from earlier banks in this paper; BranchMem-Hard must not reuse it.
PRIOR_STEMS = (
    "allerg", "gluten", "peanut", "lactose", "celiac", "shellfish", "warfarin",
    "penicillin", "parking permit", "toll tag", "dishwasher", "thermostat",
    "router", "espresso", "banquet", "interlibrary", "stroller", "carpool",
)


@pytest.fixture(scope="module")
def bank() -> dict:
    return json.loads(BANK.read_text())


def test_bank_matches_its_frozen_hash():
    assert hashlib.sha256(BANK.read_bytes()).hexdigest() == PIN.read_text().split()[0], (
        "BranchMem-Hard has changed since it was frozen")


def test_shape(bank):
    assert bank["n_items"] == 60 and bank["n_families"] == 12
    items = [it for f in bank["families"] for it in f["items"]]
    assert len(items) == 60
    for t in TYPES:
        assert sum(1 for it in items if it["type"] == t) == 12


def test_every_family_has_one_item_of_each_type(bank):
    for fam in bank["families"]:
        assert [it["type"] for it in fam["items"]] == list(TYPES), fam["family"]


def test_predicates_unique(bank):
    preds = [it["predicate"] for f in bank["families"] for it in f["items"]]
    assert len(set(preds)) == len(preds)


def test_only_the_indefinite_type_rejects_the_update(bank):
    """The asymmetry the bank is built around."""
    for f in bank["families"]:
        for it in f["items"]:
            expected = "keep" if it["type"] == "standing_indefinite" else "update"
            assert it["correct_action"] == expected, (it["predicate"], it["type"])


def test_the_lexical_cue_does_not_separate_keep_from_update(bank):
    """THE load-bearing test.

    In the discrimination bank the marker fires on every reject-the-update
    ancestor and none of the accept-the-update ones. Here it must fire at
    essentially the same rate on all four obligation types, so no policy can
    recover the answer from the ancestor's register alone.
    """
    rate = {}
    for t in OBLIGATION_TYPES:
        items = [it for f in bank["families"] for it in f["items"] if it["type"] == t]
        rate[t] = sum(bool(MARKER.search(it["ancestor_value"])) for it in items) / len(items)
    keep_rate = rate["standing_indefinite"]
    for t in OBLIGATION_TYPES[1:]:
        assert abs(keep_rate - rate[t]) <= 0.15, (
            f"marker separates standing_indefinite ({keep_rate:.2f}) from "
            f"{t} ({rate[t]:.2f}); the lexical shortcut is back")
    # and it must genuinely fire on most of them, or the test is vacuous
    assert min(rate.values()) >= 0.75, rate


def test_a_modal_reading_detector_would_freeze_legitimate_updates(bank):
    """Quantifies the cost the paper reports, as a regression guard."""
    frozen = sum(
        1 for f in bank["families"] for it in f["items"]
        if it["correct_action"] == "update"
        and it["type"] != "ordinary_value"
        and MARKER.search(it["ancestor_value"]))
    assert frozen >= 30, f"only {frozen} legitimate updates would be frozen"


def test_no_value_mentions_provenance(bank):
    leak = re.compile(r"\bsource\b|\bconfidence\b|\breliab|\btrustworth|\binferred\b", re.I)
    for f in bank["families"]:
        for it in f["items"]:
            for field in ("ancestor_value", "update_value", "keep_value"):
                assert not leak.search(it[field]), (it["predicate"], field)


def test_branch_values_differ(bank):
    for f in bank["families"]:
        for it in f["items"]:
            assert it["update_value"].strip().lower() != it["keep_value"].strip().lower()


def test_domains_disjoint_from_earlier_banks(bank):
    blob = " ".join(
        f"{it['domain']} {it['predicate']} {it['ancestor_value']} {it['update_value']}"
        for f in bank["families"] for it in f["items"]).lower()
    hits = [w for w in PRIOR_STEMS if w in blob]
    assert hits == [], f"BranchMem-Hard reuses earlier material: {hits}"


def test_declares_its_authorship_and_purpose(bank):
    note = bank["note"].lower()
    assert "hand-authored" in note and "synthetic" in note
    assert "not generated by any model evaluated on it" in note
