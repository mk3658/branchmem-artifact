"""The prospective bank must stay exactly as it was frozen.

The discrimination experiment's claim depends on three properties of its bank:
that it has not changed since it was hashed, that the two critical types differ
only in the ancestor's standing force, and that no value leaks the metadata a
policy is supposed to be deciding without. These tests enforce all three, so a
later edit to the bank fails the suite rather than silently changing a
preregistered result.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "data" / "discrimination_bank.json"
PIN = ROOT / "data" / "discrimination_bank.sha256"
TYPES = ("immutable_constraint", "legitimate_override",
         "independent_observation", "ambiguous")

# Material used by banks that already existed when this one was authored; the
# prospective bank must not reuse it. Stems are matched as substrings, short
# words only on word boundaries, so that a book title like "The Silent Garden"
# is not mistaken for the religious season "Lent".
PRIOR_STEMS = (
    "allerg", "gluten", "peanut", "lactose", "celiac", "shellfish",
    "warfarin", "penicillin", "isotretinoin", "primaquine", "lisinopril",
    "shabbat", "ramadan", "gurdwara", "schengen", "wheelchair", "braille",
)
PRIOR_WORDS = ("vegan", "lent", "kosher", "passport", "visa", "hoyer", "asl")


@pytest.fixture(scope="module")
def bank() -> dict:
    return json.loads(BANK.read_text())


def test_bank_matches_its_frozen_hash():
    digest = hashlib.sha256(BANK.read_bytes()).hexdigest()
    pinned = PIN.read_text().split()[0]
    assert digest == pinned, (
        "the prospective bank has changed since it was frozen; a preregistered "
        "result cannot be reported against an edited bank")


def test_four_balanced_types(bank):
    items = [it for f in bank["families"] for it in f["items"]]
    assert len(items) == bank["n_items"] == 320
    for t in TYPES:
        assert sum(1 for it in items if it["type"] == t) == 80


def test_every_family_has_one_item_of_each_type(bank):
    for fam in bank["families"]:
        assert [it["type"] for it in fam["items"]] == list(TYPES), fam["family"]


def test_predicates_are_unique(bank):
    preds = [it["predicate"] for f in bank["families"] for it in f["items"]]
    assert len(set(preds)) == len(preds)


def test_domains_are_disjoint_from_earlier_banks(bank):
    blob = " ".join(
        f"{it['domain']} {it['predicate']} {it['ancestor_value']} "
        f"{it['update_value']} {it['keep_value']}"
        for f in bank["families"] for it in f["items"]).lower()
    hits = [w for w in PRIOR_STEMS if w in blob]
    hits += [w for w in PRIOR_WORDS if re.search(rf"\b{w}\b", blob)]
    assert hits == [], f"prospective bank reuses earlier material: {hits}"


def test_no_value_mentions_provenance(bank):
    leak = re.compile(r"\bsource\b|\bconfidence\b|\breliab|\btrustworth|\binferred\b", re.I)
    for f in bank["families"]:
        for it in f["items"]:
            for field in ("ancestor_value", "update_value", "keep_value"):
                assert not leak.search(it[field]), (it["predicate"], field)


def test_branch_values_differ_within_every_item(bank):
    for f in bank["families"]:
        for it in f["items"]:
            assert it["update_value"].strip().lower() != it["keep_value"].strip().lower()


def test_ambiguous_ancestor_does_not_favour_a_branch(bank):
    """An ambiguous item whose ancestor restates one branch is not ambiguous."""
    for f in bank["families"]:
        amb = next(it for it in f["items"] if it["type"] == "ambiguous")
        anc = set(re.findall(r"[a-z0-9]+", amb["ancestor_value"].lower()))
        for side in ("update_value", "keep_value"):
            toks = set(re.findall(r"[a-z0-9]+", amb[side].lower()))
            assert amb["ancestor_value"].strip().lower() != amb[side].strip().lower()
            assert len(anc & toks) / max(1, len(toks)) <= 0.8, (f["family"], side)


def test_critical_types_differ_in_the_ancestor_standing_force(bank):
    """The whole comparison rests on this: constraint ancestors carry standing
    force and override ancestors do not."""
    marker = re.compile(
        r"\bonly\b|\bmust\b|\bcannot\b|\bnot permitted\b|\bnot allowed\b|\brequire[sd]?\b|"
        r"\bmaximum\b|\blimit\b|\bprohibit|\bforbid|\brule\b|\bpolicy\b|\bcontract|"
        r"\bban(s|ned)?\b|\bmandat|\bobliged\b|\bcapacity\b|\bdeadline\b|\bno more than\b",
        re.I)
    for f in bank["families"]:
        items = {it["type"]: it for it in f["items"]}
        assert marker.search(items["immutable_constraint"]["ancestor_value"]), f["family"]
        assert not marker.search(items["legitimate_override"]["ancestor_value"]), f["family"]


def test_bank_declares_that_the_author_is_also_a_solver(bank):
    note = bank["note"].lower()
    assert "author" in note and "solver" in note
    assert bank["author_model"]
