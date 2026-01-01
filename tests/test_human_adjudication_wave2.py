"""The second adjudication wave, and the claim that the bank is now covered.

Wave 1 adjudicated a 100-item sample; wave 2 adjudicated the remaining 220, and
the paper now says every item in the bank has been judged by people who did not
author it. That sentence is only true if the two waves are disjoint and together
exhaustive, and the pooled numbers are only meaningful if the three wave-2
sheets posed the SAME questions -- a majority vote over sheets showing different
inputs would be pooling answers to different items.

Wave 2's sheets were issued by the study organizer rather than generated here,
so its key is derived from the returned sheets and the frozen bank. These tests
pin that derivation: the key must still describe the sheets it was built from,
the waves must still partition the bank, and the outcomes the paper quotes must
recompute from the CSVs rather than from the stored JSON.
"""
from __future__ import annotations

import csv
import glob
import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WAVE1 = ROOT / "annotation" / "discrimination_sheets"
WAVE2 = ROOT / "annotation" / "discrimination_sheets_wave2"
KEY2 = WAVE2 / "discrimination_key_HELD_BACK.csv"
BANK = ROOT / "data" / "discrimination_bank.json"
SCORED_FULL = ROOT / "results" / "final" / "discrimination_human_adjudication_full_bank.json"
SCORED_W2 = ROOT / "results" / "final" / "discrimination_human_adjudication_wave2.json"

INPUT_COLS = ("ancestor_value",
              "branch_a_value", "branch_a_source", "branch_a_confidence",
              "branch_b_value", "branch_b_source", "branch_b_confidence")
VOCAB = {
    "q1_which_value_should_be_stored": {"a", "b", "neither_flag_for_review"},
    "q2_ancestor_is": {"standing_rule", "ordinary_value", "unclear"},
    "q3_who_could_lift_it": {"user_alone", "someone_else", "nobody_physical",
                             "unclear", ""},
}
CRIT = ("immutable_constraint", "legitimate_override")


@pytest.fixture(scope="module")
def key2() -> dict:
    return {r["item_id"]: r for r in csv.DictReader(KEY2.open())}


@pytest.fixture(scope="module")
def sheets2() -> dict:
    out = {}
    for path in sorted(glob.glob(str(WAVE2 / "discrimination_annotator*.csv"))):
        out[Path(path).stem] = {r["item_id"]: r
                                for r in csv.DictReader(Path(path).open())}
    return out


def _majority(labels: list[str]) -> str | None:
    vals = [v for v in labels if v]
    if not vals:
        return None
    top = Counter(vals).most_common()
    return None if len(top) > 1 and top[0][1] == top[1][1] else top[0][0]


def test_three_sheets_present_and_complete(sheets2, key2):
    assert len(sheets2) == 3, f"expected three wave-2 sheets, got {sorted(sheets2)}"
    assert len(key2) == 220, f"wave 2 should hold 220 items, key has {len(key2)}"
    for name, rows in sheets2.items():
        assert set(rows) == set(key2), f"{name} does not cover the keyed items"
        blank = [i for i, r in rows.items()
                 if not r["q1_which_value_should_be_stored"].strip()]
        assert not blank, f"{name} leaves Q1 blank on {blank[:5]}"


def test_the_three_sheets_pose_the_same_questions(sheets2):
    """A majority vote is only defined if all three saw identical inputs."""
    names = sorted(sheets2)
    first = sheets2[names[0]]
    for name in names[1:]:
        bad = [i for i in first
               if any(sheets2[name][i][c] != first[i][c] for c in INPUT_COLS)]
        assert not bad, f"{name} shows different inputs from {names[0]} on {bad[:5]}"


def test_answers_are_in_vocabulary(sheets2):
    for name, rows in sheets2.items():
        for col, allowed in VOCAB.items():
            bad = {r[col].strip() for r in rows.values()} - allowed
            assert not bad, f"{name}.{col} holds unexpected answers {bad}"
        for i, r in rows.items():
            c = r["q4_confidence_1to5"].strip()
            assert c.isdigit() and 1 <= int(c) <= 5, f"{name} {i} confidence {c!r}"


def test_q3_answered_exactly_when_q2_is_standing_rule(sheets2):
    for name, rows in sheets2.items():
        for i, r in rows.items():
            rule = r["q2_ancestor_is"].strip() == "standing_rule"
            got = bool(r["q3_who_could_lift_it"].strip())
            assert rule == got, f"{name} {i}: q2={r['q2_ancestor_is']!r} q3={r['q3_who_could_lift_it']!r}"


def test_sheets_are_not_copies_of_each_other(sheets2):
    names = sorted(sheets2)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            x, y = sheets2[names[a]], sheets2[names[b]]
            same = sum(x[i]["q1_which_value_should_be_stored"]
                       == y[i]["q1_which_value_should_be_stored"] for i in x)
            assert same < len(x), f"{names[a]} and {names[b]} answer Q1 identically"


def test_the_waves_partition_the_bank(key2):
    """The paper says every bank item has been adjudicated; this is that claim."""
    key1 = {r["item_id"]: r for r in
            csv.DictReader((WAVE1 / "discrimination_key_HELD_BACK.csv").open())}
    assert not set(key1) & set(key2), "the two waves share item ids"

    def triples(key):
        out = set()
        for r in key.values():
            upd_a = r["update_is_branch"] == "a"
            upd = (r["branch_a_value"] if upd_a else r["branch_b_value"]).strip()
            keep = (r["branch_b_value"] if upd_a else r["branch_a_value"]).strip()
            out.add((r["ancestor_value"].strip(), upd, keep))
        return out

    covered = triples(key1) | triples(key2)
    assert len(covered) == len(key1) + len(key2), "an item is adjudicated in both waves"
    bank = {(i["ancestor_value"].strip(), i["update_value"].strip(),
             i["keep_value"].strip())
            for fam in json.loads(BANK.read_text())["families"] for i in fam["items"]}
    assert covered == bank, (
        f"waves cover {len(covered)} items, bank has {len(bank)}; "
        "the paper's 'every item' claim depends on these being equal")


def test_the_key_still_describes_the_sheets_it_was_derived_from(sheets2, key2):
    """The key is derived, so a sheet edit must invalidate it rather than slide by."""
    rows = sheets2[sorted(sheets2)[0]]
    for i, k in key2.items():
        for c in INPUT_COLS:
            assert rows[i][c] == k[c], f"{i}.{c} differs between sheet and key"
        strong = (k["branch_a_source"], k["branch_a_confidence"]) == ("user", "0.95")
        assert k["update_is_branch"] == ("a" if strong else "b"), (
            f"{i}: update_is_branch disagrees with which branch carries user/0.95")


def test_the_decisive_q3_outcome_still_holds(sheets2, key2):
    """0 of the wave-2 constraints judged the user's alone to lift."""
    names = sorted(sheets2)
    con = [i for i, k in key2.items() if k["type"] == "immutable_constraint"]
    assert len(con) == 55
    user_alone = sum(
        _majority([sheets2[n][i]["q3_who_could_lift_it"].strip() for n in names])
        == "user_alone" for i in con)
    assert user_alone == 0, (
        f"{user_alone}/{len(con)} constraints judged the user's alone to lift; "
        "the paper reports none, in either wave")


def test_the_take_update_gap_still_separates_people_from_policies(sheets2, key2):
    """Within wave 2 alone, every annotator's gap exceeds every policy's."""
    names = sorted(sheets2)
    gaps = []
    for n in names:
        rates = {}
        for ty in CRIT:
            sub = [i for i, k in key2.items() if k["type"] == ty]
            rates[ty] = sum(
                sheets2[n][i]["q1_which_value_should_be_stored"].strip()
                == key2[i]["update_is_branch"] for i in sub) / len(sub)
        gaps.append(rates["legitimate_override"] - rates["immutable_constraint"])
    assert min(gaps) > 0.9, f"wave-2 annotator gaps fell to {sorted(gaps)}"
    scored = json.loads(SCORED_W2.read_text())
    policy = [v["gap"] for k, v in scored["take_update_rate_by_type"].items()
              if not k.startswith("discrimination_annotator")]
    assert max(policy) < min(gaps), (
        f"policy gaps {sorted(policy)} now reach the annotator range {sorted(gaps)}")


def test_the_pooled_run_records_that_the_briefs_differed(sheets2):
    """The waves used different briefs; the artifact must not lose that."""
    scored = json.loads(SCORED_FULL.read_text())
    assert scored["n_items"] == 320
    assert scored["briefs_identical_across_waves"] is False
    assert scored["brief_difference_note"], (
        "the pooled result must carry the brief-difference disclosure, since the "
        "flagging rate is not comparable across the two waves")
    b1 = (WAVE1 / "INSTRUCTIONS_FOR_ANNOTATORS.md").read_text()
    b2 = (WAVE2 / "INSTRUCTIONS_FOR_ANNOTATORS.md").read_text()
    assert b1 != b2, (
        "the briefs are now identical, so the disclosure in the paper and in the "
        "scorer is stale and should be removed")
