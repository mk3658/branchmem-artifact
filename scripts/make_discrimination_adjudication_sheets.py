#!/usr/bin/env python3
"""Build blank human-adjudication sheets for the discrimination bank.

The discrimination claim rests on a labelling convention -- a standing entry
survives a confident, user-sourced update -- which is a normative choice rather
than a fact, and which for a long time no human had checked. This script builds
everything needed to check it: a stratified, order-randomised sample of 100
items, three blank annotator sheets, a held-back key, and the questions that
actually test the objection. Nothing it writes for annotators contains our
labels.

The questionnaire asks four things per item. Q1 is the merge decision. Q2 asks
whether the stored entry is a standing rule or an ordinary value, which is the
classification the two-stage mitigation performs. Q3 asks WHO could lift the
rule, which states the objection as a measurable question: if annotators say
the user may lift most of these constraints, our label convention is wrong and
the paper's claim does not stand. Q4 records confidence.

THE STUDY HAS SINCE BEEN RUN. Three annotators filled these sheets; the filled
copies are in this same directory and the scored outcome is
`results/final/discrimination_human_adjudication.json`. Re-running this script
would overwrite those answers with blanks, so it refuses to touch a sheet that
has any answer in it unless `--force` is passed. Use `--manifest-only` to
regenerate `manifest.json` alone.

Presentation is blinded in three ways: our label is never shown, which physical
branch carries the stronger provenance is randomised per item, and the four
conflict types are interleaved so the type is not inferable from position.

Usage: PYTHONPATH=. python scripts/make_discrimination_adjudication_sheets.py \
           [--manifest-only] [--force]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "data" / "discrimination_bank.json"
OUT = ROOT / "annotation" / "discrimination_sheets"
SEED = 9091          # fixed before the sample was drawn
N_PER_TYPE = 25      # 4 types x 25 = 100 items
TYPES = ("immutable_constraint", "legitimate_override",
         "independent_observation", "ambiguous")

# The provenance assignment used by the primary arm of the experiment: the
# update side always carries the strong signal, so the metadata cannot
# distinguish the two critical types. Annotators see exactly this.
STRONG = ("user", 0.95)
WEAK = ("inference", 0.55)

ANNOTATOR_COLUMNS = [
    "item_id",
    "ancestor_value",
    "branch_a_value", "branch_a_source", "branch_a_confidence",
    "branch_b_value", "branch_b_source", "branch_b_confidence",
    # Q1 merge decision
    "q1_which_value_should_be_stored",      # a | b | neither_flag_for_review
    # Q2 kind of entry
    "q2_ancestor_is",                       # standing_rule | ordinary_value | unclear
    # Q3 who may lift it (only if q2 = standing_rule)
    "q3_who_could_lift_it",                 # user_alone | someone_else | nobody_physical | unclear
    # Q4 confidence
    "q4_confidence_1to5",
    "notes",
]


def _filled_sheets() -> list[Path]:
    """The annotator sheets that already contain answers."""
    return [p for p in sorted(OUT.glob("discrimination_annotator*.csv"))
            if _n_answers(p)]


def _n_answers(path: Path) -> int:
    """How many answer cells are filled in, across the four answer columns."""
    if not path.exists():
        return 0
    cols = ("q1_which_value_should_be_stored", "q2_ancestor_is",
            "q3_who_could_lift_it", "q4_confidence_1to5")
    with path.open() as fh:
        return sum(bool((r.get(c) or "").strip())
                   for r in csv.DictReader(fh) for c in cols)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="overwrite annotator sheets even if they hold answers")
    ap.add_argument("--manifest-only", action="store_true",
                    help="regenerate manifest.json only; touch no sheet")
    args = ap.parse_args()
    bank = json.loads(BANK.read_text())
    digest = hashlib.sha256(BANK.read_bytes()).hexdigest()
    pinned = (BANK.parent / "discrimination_bank.sha256").read_text().split()[0]
    if digest != pinned:
        raise SystemExit("bank hash mismatch; refusing to build sheets from an edited bank")

    by_type: dict[str, list[dict]] = {t: [] for t in TYPES}
    for fam in bank["families"]:
        for it in fam["items"]:
            by_type[it["type"]].append(it)

    rng = random.Random(SEED)
    # Sample families first, so the 25 items of a type come from 25 distinct
    # families and no family is over-represented.
    chosen: list[dict] = []
    for t in TYPES:
        pool = sorted(by_type[t], key=lambda i: i["predicate"])
        fams = sorted({i["family"] for i in pool})
        pick_fams = rng.sample(fams, N_PER_TYPE)
        for f in pick_fams:
            chosen.append(next(i for i in pool if i["family"] == f))

    rng.shuffle(chosen)

    master, blank = [], []
    for n, it in enumerate(chosen, 1):
        item_id = f"d{n:03d}"
        update_on_a = rng.random() < 0.5
        a_val, b_val = ((it["update_value"], it["keep_value"]) if update_on_a
                        else (it["keep_value"], it["update_value"]))
        a_src, a_conf = STRONG if update_on_a else WEAK
        b_src, b_conf = WEAK if update_on_a else STRONG
        row = {
            "item_id": item_id,
            "ancestor_value": it["ancestor_value"],
            "branch_a_value": a_val, "branch_a_source": a_src, "branch_a_confidence": a_conf,
            "branch_b_value": b_val, "branch_b_source": b_src, "branch_b_confidence": b_conf,
        }
        blank.append({**row, **{c: "" for c in ANNOTATOR_COLUMNS if c not in row}})
        master.append({
            **row,
            "type": it["type"],
            "predicate": it["predicate"],
            "family": it["family"],
            "domain": it["domain"],
            "update_is_branch": "a" if update_on_a else "b",
            "construction_correct_value": (
                "" if it["type"] == "ambiguous"
                else (it["update_value"] if it["type"] != "immutable_constraint"
                      else it["keep_value"])),
            "construction_correct_branch": (
                "neither" if it["type"] == "ambiguous"
                else ("a" if (update_on_a) == (it["type"] != "immutable_constraint") else "b")),
            "why_correct": it["why_correct"],
        })

    OUT.mkdir(parents=True, exist_ok=True)
    if not args.manifest_only:
        filled = [s for s in _filled_sheets() if s]
        if filled and not args.force:
            raise SystemExit(
                "refusing to overwrite collected human labels.\n  "
                + "\n  ".join(f"{s.name} has {_n_answers(s)} answers" for s in filled)
                + "\n\nThese are the annotations the paper reports. Pass --force only "
                  "if you intend to discard them, or --manifest-only to regenerate "
                  "manifest.json without touching any sheet.")
        for i in (1, 2, 3):
            path = OUT / f"discrimination_annotator{i}.csv"
            with path.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=ANNOTATOR_COLUMNS)
                w.writeheader()
                w.writerows(blank)
        key_path = OUT / "discrimination_key_HELD_BACK.csv"
        with key_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(master[0]))
            w.writeheader()
            w.writerows(master)

    counts: dict[str, int] = {}
    for m in master:
        counts[m["type"]] = counts.get(m["type"], 0) + 1
    meta = {
        "note": (
            "The 100-item stratified sample drawn for human adjudication of the "
            "discrimination bank. This script generates the BLANK sheets and the "
            "held-back key; the key is withheld from annotators and released only so "
            "the sample is auditable. Q3 is the decisive question: if annotators judge "
            "that the user alone could lift most of these constraints, the paper's "
            "labelling convention is wrong. THE STUDY HAS SINCE BEEN RUN -- the filled "
            "sheets are in this directory and the scored outcome is in "
            "results/final/discrimination_human_adjudication.json, which upheld the "
            "convention (0 of 25 constraints judged the user's alone to lift). "
            "A SECOND WAVE has since adjudicated the remaining 220 items; it lives in "
            "annotation/discrimination_sheets_wave2/ and the two waves together cover "
            "the whole bank, so this sample is not all the human data there is. "
            "Re-running this generator refuses to overwrite a filled sheet unless "
            "--force is passed."
        ),
        "bank_sha256": digest,
        "seed": SEED,
        "n_items": len(master),
        "items_per_type": counts,
        "n_distinct_families": len({m["family"] for m in master}),
        "provenance_shown": {"strong": STRONG, "weak": WEAK,
                             "note": "update side always strong, as in the primary arm"},
        "files": sorted(p.name for p in OUT.iterdir()),
    }
    (OUT / "manifest.json").write_text(json.dumps(meta, indent=2))
    print(f"{len(master)} items, per type {counts}, "
          f"{meta['n_distinct_families']} distinct families")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
