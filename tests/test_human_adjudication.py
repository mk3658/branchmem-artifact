"""The collected human labels must stay the labels that were collected.

Three annotators adjudicated a 100-item sample of the discrimination bank, and
the paper's validation rests on those sheets. These tests pin the properties
the reported numbers depend on: that the sheets still line up with the sample
that was drawn, that every answer is in the allowed vocabulary, and that the
headline outcomes recompute. A later edit to a sheet fails here rather than
quietly moving a number in the paper.
"""
from __future__ import annotations

import csv
import glob
import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHEETS = ROOT / "annotation" / "discrimination_sheets"
KEY = SHEETS / "discrimination_key_HELD_BACK.csv"
SCORED = ROOT / "results" / "final" / "discrimination_human_adjudication.json"

VOCAB = {
    "q1_which_value_should_be_stored": {"a", "b", "neither_flag_for_review"},
    "q2_ancestor_is": {"standing_rule", "ordinary_value", "unclear"},
    "q3_who_could_lift_it": {"user_alone", "someone_else", "nobody_physical",
                             "unclear", ""},
}


@pytest.fixture(scope="module")
def key() -> dict:
    return {r["item_id"]: r for r in csv.DictReader(KEY.open())}


@pytest.fixture(scope="module")
def sheets() -> dict:
    out = {}
    for path in sorted(glob.glob(str(SHEETS / "discrimination_annotator*.csv"))):
        rows = list(csv.DictReader(Path(path).open()))
        out[Path(path).stem] = {r["item_id"]: r for r in rows}
    return out


def _g(sheet: dict, item: str, col: str) -> str:
    return (sheet.get(item, {}).get(col, "") or "").strip().lower()


def test_three_sheets_present_and_complete(sheets, key):
    assert len(sheets) == 3, f"expected 3 annotator sheets, found {len(sheets)}"
    for name, s in sheets.items():
        assert set(s) == set(key), f"{name} does not cover the drawn sample"
        for col in ("q1_which_value_should_be_stored", "q2_ancestor_is",
                    "q4_confidence_1to5"):
            missing = [i for i in key if not _g(s, i, col)]
            assert missing == [], f"{name} has {len(missing)} blanks in {col}"


def test_answers_are_in_vocabulary(sheets, key):
    for name, s in sheets.items():
        for col, allowed in VOCAB.items():
            bad = {_g(s, i, col) for i in key} - allowed
            assert bad == set(), f"{name} has out-of-vocabulary {col}: {bad}"
        for i in key:
            c = _g(s, i, "q4_confidence_1to5")
            assert c in {"1", "2", "3", "4", "5"}, (name, i, c)


def test_q3_answered_exactly_when_q2_is_standing_rule(sheets, key):
    """The brief asks for Q3 only on standing rules; a sheet that answers it
    elsewhere was filled against different instructions."""
    for name, s in sheets.items():
        for i in key:
            has_q3 = bool(_g(s, i, "q3_who_could_lift_it"))
            is_rule = _g(s, i, "q2_ancestor_is") == "standing_rule"
            assert has_q3 == is_rule, (name, i, _g(s, i, "q2_ancestor_is"))


def test_item_text_was_not_altered_in_the_round_trip(sheets, key):
    """A spreadsheet round-trip can mangle the item text; if it did, the
    annotators did not judge the items we think they judged."""
    for name, s in sheets.items():
        for i in key:
            assert s[i]["ancestor_value"].strip() == key[i]["ancestor_value"].strip(), (
                f"{name} item {i} text drifted")


def test_sheets_are_not_copies_of_each_other(sheets, key):
    """Independent annotation, not one sheet duplicated three times."""
    names = sorted(sheets)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            same = sum(
                _g(sheets[names[a]], i, "q1_which_value_should_be_stored")
                == _g(sheets[names[b]], i, "q1_which_value_should_be_stored")
                for i in key)
            assert same < len(key), f"{names[a]} and {names[b]} are identical on Q1"


def test_the_decisive_q3_outcome_still_holds(sheets, key):
    """0 of 25 constraint items judged the user's alone to lift. If a future
    edit changes this, the paper's normative claim changes with it."""
    con = [i for i in key if key[i]["type"] == "immutable_constraint"]
    user_alone = 0
    for i in con:
        votes = [_g(sheets[n], i, "q3_who_could_lift_it") for n in sorted(sheets)]
        top = Counter([v for v in votes if v]).most_common()
        maj = None if (len(top) > 1 and top[0][1] == top[1][1]) else top[0][0]
        user_alone += int(maj == "user_alone")
    assert user_alone == 0, (
        f"{user_alone}/{len(con)} constraints now judged user_alone; the paper "
        "reports 0 and its normative claim depends on it")


def test_the_take_update_gap_still_separates_people_from_policies(sheets, key):
    """Every annotator's gap must stay above every policy's. The paper reports
    non-overlapping ranges (+.640..+1.000 vs +.000..+.280)."""
    for name, s in sheets.items():
        rates = {}
        for ty in ("immutable_constraint", "legitimate_override"):
            sub = [i for i in key if key[i]["type"] == ty]
            rates[ty] = sum(
                _g(s, i, "q1_which_value_should_be_stored") == key[i]["update_is_branch"]
                for i in sub) / len(sub)
        gap = rates["legitimate_override"] - rates["immutable_constraint"]
        assert gap >= 0.6, f"{name} gap fell to {gap:.3f}; paper reports >= +0.640"


def test_scored_output_matches_the_sheets(sheets):
    """The released JSON must be the one these sheets produce."""
    if not SCORED.exists():
        pytest.skip("scored output not generated yet")
    d = json.loads(SCORED.read_text())
    assert d["n_annotators"] == len(sheets)
    assert d["convention_refuted_by_q3"] is False
    assert d["q3_user_alone_items"] == 0
    assert d["inter_annotator_q1"]["fleiss_kappa"] is not None
