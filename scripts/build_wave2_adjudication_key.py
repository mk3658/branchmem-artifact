#!/usr/bin/env python3
"""Derive the held-back key for the second adjudication wave.

Wave 1 covered a stratified 100-item sample of the discrimination bank
(`annotation/discrimination_sheets/`, ids d001--d100), drawn and keyed by
`make_discrimination_adjudication_sheets.py` at seed 9091. Wave 2 covers the
remaining 220 items, ids d101--d320, so that between them the bank is
adjudicated in full.

The wave-2 sheets were issued by the study organizer, not by this repository,
so their id assignment and their per-item branch ordering are not reproducible
from a seed here. The key is therefore DERIVED from the returned sheets and the
frozen bank rather than generated alongside them, which is a deterministic
function with no free parameters:

  * each sheet row shows the ancestor and the two branch values, with the
    update side carrying `user`/0.95 and the stored side `inference`/0.55, as
    in the experiment's primary arm;
  * so (ancestor_value, update_value, keep_value) identifies the bank item,
    and all 320 of those triples are distinct;
  * the matched bank item supplies the type, predicate, family and domain, and
    the source column says which branch is the update.

Every one of those steps is checked, and any failure stops the script rather
than producing a key that would silently mis-score the study:

  - the bank must match its pinned hash;
  - the three sheets must show IDENTICAL inputs for every id, or a majority
    vote would be pooling answers to different questions;
  - every row must match exactly one bank item;
  - the provenance pattern must be one of the two the convention allows;
  - the matched items must be exactly the bank minus the wave-1 sample.

Usage: PYTHONPATH=. python scripts/build_wave2_adjudication_key.py
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "data" / "discrimination_bank.json"
WAVE1 = ROOT / "annotation" / "discrimination_sheets"
WAVE2 = ROOT / "annotation" / "discrimination_sheets_wave2"
KEY = WAVE2 / "discrimination_key_HELD_BACK.csv"

INPUT_COLS = ("ancestor_value",
              "branch_a_value", "branch_a_source", "branch_a_confidence",
              "branch_b_value", "branch_b_source", "branch_b_confidence")
STRONG = ("user", "0.95")
WEAK = ("inference", "0.55")

KEY_COLS = ["item_id", "ancestor_value",
            "branch_a_value", "branch_a_source", "branch_a_confidence",
            "branch_b_value", "branch_b_source", "branch_b_confidence",
            "type", "predicate", "family", "domain", "update_is_branch",
            "construction_correct_value", "construction_correct_branch",
            "why_correct"]


def _bank_items() -> dict[tuple[str, str, str], dict]:
    digest = hashlib.sha256(BANK.read_bytes()).hexdigest()
    pinned = (BANK.parent / "discrimination_bank.sha256").read_text().split()[0]
    if digest != pinned:
        raise SystemExit("bank hash mismatch; refusing to key a wave against an edited bank")
    out: dict[tuple[str, str, str], dict] = {}
    for fam in json.loads(BANK.read_text())["families"]:
        for it in fam["items"]:
            k = (it["ancestor_value"].strip(),
                 it["update_value"].strip(), it["keep_value"].strip())
            if k in out:
                raise SystemExit(f"bank holds two items with identical values: {k[0]!r}")
            out[k] = it
    return out


def _sheets() -> dict[str, dict[str, str]]:
    """The wave-2 rows, after checking the three annotators were shown the same."""
    paths = sorted(WAVE2.glob("discrimination_annotator*.csv"))
    if len(paths) < 2:
        raise SystemExit(f"expected the returned annotator sheets in {WAVE2}")
    loaded = [{r["item_id"]: r for r in csv.DictReader(p.open())} for p in paths]
    first = loaded[0]
    for path, other in zip(paths[1:], loaded[1:]):
        if set(other) != set(first):
            raise SystemExit(f"{path.name} covers different items from the first sheet")
        for i in first:
            if any(other[i][c] != first[i][c] for c in INPUT_COLS):
                raise SystemExit(
                    f"{path.name} shows different inputs for {i}; the sheets are not "
                    "the same questionnaire and must not be pooled")
    return first


def main() -> None:
    bank = _bank_items()
    rows = _sheets()

    wave1_triples = set()
    for r in csv.DictReader((WAVE1 / "discrimination_key_HELD_BACK.csv").open()):
        upd_a = r["update_is_branch"] == "a"
        upd = (r["branch_a_value"] if upd_a else r["branch_b_value"]).strip()
        keep = (r["branch_b_value"] if upd_a else r["branch_a_value"]).strip()
        wave1_triples.add((r["ancestor_value"].strip(), upd, keep))

    master, seen = [], set()
    for item_id in sorted(rows):
        r = rows[item_id]
        a = (r["branch_a_source"].strip(), r["branch_a_confidence"].strip())
        b = (r["branch_b_source"].strip(), r["branch_b_confidence"].strip())
        if (a, b) == (STRONG, WEAK):
            update_on_a = True
        elif (a, b) == (WEAK, STRONG):
            update_on_a = False
        else:
            raise SystemExit(f"{item_id}: provenance {a}/{b} is not the primary arm's")
        upd = (r["branch_a_value"] if update_on_a else r["branch_b_value"]).strip()
        keep = (r["branch_b_value"] if update_on_a else r["branch_a_value"]).strip()
        triple = (r["ancestor_value"].strip(), upd, keep)
        it = bank.get(triple)
        if it is None:
            raise SystemExit(f"{item_id}: no bank item has these values; sheet was edited")
        if triple in seen:
            raise SystemExit(f"{item_id}: duplicates an earlier row")
        if triple in wave1_triples:
            raise SystemExit(f"{item_id}: already adjudicated in wave 1; waves must be disjoint")
        seen.add(triple)
        master.append({
            "item_id": item_id,
            "ancestor_value": r["ancestor_value"],
            "branch_a_value": r["branch_a_value"],
            "branch_a_source": r["branch_a_source"],
            "branch_a_confidence": r["branch_a_confidence"],
            "branch_b_value": r["branch_b_value"],
            "branch_b_source": r["branch_b_source"],
            "branch_b_confidence": r["branch_b_confidence"],
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
                else ("a" if update_on_a == (it["type"] != "immutable_constraint") else "b")),
            "why_correct": it["why_correct"],
        })

    if len(seen) + len(wave1_triples) != len(bank):
        raise SystemExit(
            f"waves cover {len(seen) + len(wave1_triples)} of {len(bank)} bank items; "
            "wave 2 is meant to be exactly the complement of wave 1")

    with KEY.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=KEY_COLS)
        w.writeheader()
        w.writerows(master)

    counts: dict[str, int] = {}
    for m in master:
        counts[m["type"]] = counts.get(m["type"], 0) + 1
    meta = {
        "note": (
            "Held-back key for the second adjudication wave: the 220 bank items not "
            "covered by the 100-item wave-1 sample, so that the two waves together "
            "adjudicate all 320. The sheets were issued by the study organizer, so "
            "this key is DERIVED from the returned sheets and the frozen bank by "
            "build_wave2_adjudication_key.py, which refuses to write a key unless the "
            "bank hash matches, the three sheets show identical inputs per item, every "
            "row matches exactly one bank item under the primary arm's provenance, and "
            "the two waves are disjoint and exhaustive."),
        "bank_sha256": hashlib.sha256(BANK.read_bytes()).hexdigest(),
        "derived_from_returned_sheets": True,
        "n_items": len(master),
        "items_per_type": counts,
        "n_distinct_families": len({m["family"] for m in master}),
        "provenance_shown": {"strong": list(STRONG), "weak": list(WEAK),
                             "note": "update side always strong, as in the primary arm"},
        "wave1_items": len(wave1_triples),
        "bank_items": len(bank),
        "files": sorted(p.name for p in WAVE2.iterdir()),
    }
    (WAVE2 / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{len(master)} items, per type {counts}, "
          f"{meta['n_distinct_families']} distinct families")
    print(f"waves cover {len(seen) + len(wave1_triples)}/{len(bank)} bank items")
    print("wrote", KEY.relative_to(ROOT))


if __name__ == "__main__":
    main()
